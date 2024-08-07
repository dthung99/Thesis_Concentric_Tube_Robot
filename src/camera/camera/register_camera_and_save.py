import cv2
import numpy as np
import time
import pickle
import os
import sys
# ROS pkg
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from camera.camera_register_module import *
from camera.video_process_func import *

def calibrate_and_validate_one_camera(camera_device_name,
                                      checker_board_size=[10,7],
                                      checker_board_square_edge_length=25,
                                      number_of_frame_for_registering=50):
    # Calibrate the camera
    ## Show the camera
    show_camera(camera_device_name)
    ## Detect the corners from the checker board
    rclpy.logging.get_logger('my_logger').info(f"Registering camera {camera_device_name}...")
    calibrate_result = calibrate_one_camera_and_get_checker_board_vertices(camera_device_name=camera_device_name,
                                                                           checker_board_size=checker_board_size,
                                                                           checker_board_square_edge_length=checker_board_square_edge_length,
                                                                           number_of_frame_for_registering=number_of_frame_for_registering,
                                                                           show=True)
    assert len(calibrate_result[0])>0, f"Failed to calibrate camera {camera_device_name}"
    rclpy.logging.get_logger('my_logger').info(f"succeeded to calibrate camera {camera_device_name} with {len(calibrate_result[0])} frames")
    ## Validate the calibration
    detected_corners, world_points, mtx, dist, rvecs, tvecs = calibrate_result
    # Because the camera is fixed, the rvecs and tvecs should be the same for all frame
    rvecs = np.stack(rvecs, axis=0)
    rclpy.logging.get_logger('my_logger').info(f"Std of camera {camera_device_name} of rotation vector {rvecs.std()}")
    rvecs = rvecs.mean(axis=0)
    tvecs = np.stack(tvecs, axis=0)
    rclpy.logging.get_logger('my_logger').info(f"Std of camera {camera_device_name} of rotation vector {tvecs.std()}")
    tvecs = tvecs.mean(axis=0)
    # Validate the camera
    validate_camera_registration(camera_device_name, rvecs, tvecs, mtx, dist, checker_board_square_edge_length)
    # Calculate the projection matrix
    projection_matrix = compute_projection_matrix(mtx, rvec=rvecs, tvec=tvecs)
    return projection_matrix, mtx, rvecs, tvecs

def registering_background_and_colors_one_camera(camera_device_name,
                                                 number_of_frames_for_registering_background=100,
                                                 number_of_frames_for_registering_colors=100,
                                                 background_varThreshold=128,
                                                 color_varThreshold=16):
    # Show camera
    show_camera(camera_device_name=camera_device_name)
    # Start registering and segmentation
    rclpy.logging.get_logger('my_logger').info(f"Registering background {camera_device_name}... Please wait!!!")
    background_subtractor = register_background(camera_device_name=camera_device_name,
                                                number_of_register_frame=number_of_frames_for_registering_background,
                                                varThreshold=background_varThreshold)
    rclpy.logging.get_logger('my_logger').info(f"Registering color {camera_device_name}... Please select your colors!!!")
    list_of_color_detectors = register_gmm_colors(camera_device_name=camera_device_name,
                                                  number_of_register_frame=number_of_frames_for_registering_colors,
                                                  varThreshold=color_varThreshold)
    return background_subtractor, list_of_color_detectors

class ColorExtractorNode(Node):
    def __init__(self, camera_device_name, background_subtractor, list_of_color_detectors):
        super().__init__('color_extractor_node')
        self.publisher_ = self.create_publisher(String, 'topic_name', 1)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.camera_device_name = camera_device_name
        self.background_subtractor = background_subtractor
        self.list_of_color_detectors = list_of_color_detectors
        # Open the video capture
        self.cap = cv2.VideoCapture(self.camera_device_name)

    def timer_callback(self):
        """Read the camera and segment the pixels periodically"""
        # msg = String()
        # msg.data = "Hello from ROS2 publisher!"
        # self.publisher_.publish(msg)
        # self.get_logger().info('Published: "%s"' % msg.data)

        start_time = time.perf_counter()
        # Read the camera
        ret, frame = self.cap.read()
        # First segment the background and get the mask
        fgmask = self.background_subtractor.apply(frame, learningRate=0)==255 # The mask have 3 value 0 127 for shadow and 255 for foreground
        # Second segment the colors by applying each detector on the frame and get the mask
        fgmask_color = np.zeros(shape=frame.shape[0:2], dtype=np.bool)
        for detector in self.list_of_color_detectors:
            fgmask_i = detector.apply(frame, learningRate=0)
            fgmask_i = ((fgmask_i==0) | (fgmask_i==127))
            # Combine with previous masks
            fgmask_color = fgmask_i | fgmask_color
        # Combine with background masks
        fgmask = fgmask & fgmask_color
        # Get the non-zero coordinate
        coordinates = extract_non_zero_pixel_in_black_white_image(image=fgmask)
        print(len(coordinates))
        # NOTE: Execute time for previous code ~ 0.025-0.050 ~ 20-40Hz
        # if len(coordinates) < 1000:
        #     adjacency_matrix = get_adjacency_matrix(coordinates=coordinates)
        # connected_components = get_connected_components(adjacency_matrix=adjacency_matrix)
        # Show the foreground
        """Please select only one of these two options: binary image or masked image"""
        # Create a binary image
        fgmask = fgmask.astype(np.uint8)*255
        # # Apply the mask to the image
        # fgmask = (fgmask[:,:,None])*frame
        cv2.imshow('frame', fgmask)
        cv2.waitKey(1)
        rclpy.logging.get_logger('my_logger').info(f"Execute time {round(time.perf_counter()-start_time,3)}")

def main(args=sys.argv):
    # Some parameters
    camera_device_name = args[1]
    checker_board_size=[10,7]
    checker_board_square_edge_length=25
    number_of_frames_for_calibrating=10
    number_of_register_frame=100
    number_of_register_frame=100

    # Configurate the camera
    configurate_result = configurate_camera(camera_device_name)
    assert configurate_result==True, "Failed to configurate camera"
    # Test the cameras
    rclpy.logging.get_logger('my_logger').info(f"Testing camera {camera_device_name}...")
    test_camera(camera_device_name)
    rclpy.logging.get_logger('my_logger').info(f"Finish testing camera {camera_device_name}!!!")
    # Calibrate camera:
    rclpy.logging.get_logger('my_logger').info(f"Move the checker board into center and press 'esc' to start calibrating {camera_device_name}!!!")
    main_calibrate_result = calibrate_and_validate_one_camera(camera_device_name=camera_device_name,
                                                              checker_board_size=checker_board_size,
                                                              checker_board_square_edge_length=checker_board_square_edge_length,
                                                              number_of_frame_for_registering=number_of_frames_for_calibrating)
    projection_matrix, mtx, rvec, tvec = main_calibrate_result

    # registering background and colors
    rclpy.logging.get_logger('my_logger').info(f"Press 'esc' to start registering background and colors {camera_device_name}")
    main_register_result = registering_background_and_colors_one_camera(camera_device_name,
                                                                        number_of_frames_for_registering_background=number_of_register_frame,
                                                                        number_of_frames_for_registering_colors=number_of_register_frame,
                                                                        background_varThreshold=128,
                                                                        color_varThreshold=8)
    background_subtractor, list_of_color_detectors = main_register_result
    rclpy.logging.get_logger('my_logger').info(f"Finish registering background and colors {camera_device_name}")
    # Save the camera variables to the .pkl file
    # Get the path to the current package
    package_path = os.path.dirname(os.path.abspath(__file__))
    # Define the file path for the .pkl file
    file_path = os.path.join(package_path, f"camera_variables_{camera_device_name[-1]}.pkl")
    # Save
    camera_variables = {
        "camera_device_name":camera_device_name,
        "projection_matrix":projection_matrix,
        "mtx":mtx,
        "rvec":rvec,
        "tvec":tvec,
        "background_subtractor":background_subtractor,
        "list_of_color_detectors":list_of_color_detectors,
    }
    with open(file_path, 'wb') as f:
        pickle.dump(camera_variables, f)
    # Clean exit (Optional as Python garbage collectors will handle them)
    cv2.destroyAllWindows()
    rclpy.logging.get_logger('my_logger').info(f"Succeed to save camera variables {camera_device_name}")

if __name__ == '__main__':
    main(sys.argv)
    # show_camera("/dev/video4")