import cv2
import numpy as np
import time
# # import os
# # import subprocess
# import psutil
import sys
# ROS pkg
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from camera.camera_register_module import *
from camera.video_process_func import *

def configurate_camera(camera_device_name):
    exposure=0
    white_balance=4600
    # Turn auto mode off for exposure and white balance
    cap = cv2.VideoCapture(camera_device_name)
    result = True
    result = result & cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
    result = result & cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
    result = result & cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
    result = result & cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    result = result & cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    # Set the exposure and white balance manually
    # result = result & cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    # result = result & cap.set(cv2.CAP_PROP_WB_TEMPERATURE, white_balance)
    return result

def calibrate_and_validate_one_camera(camera_device_name,
                                      checker_board_size=[10,7],
                                      checker_board_square_edge_length=25,
                                      number_of_frame_for_registering=50):
    # Calibrate the camera
    ## Show the camera
    # show_camera(camera_device_name)
    ## Detect the corners from the checker board
    rclpy.logging.get_logger('my_logger').info(f"Registering camera {camera_device_name}...")
    calibrate_result = calibrate_one_camera_and_get_checker_board_vertices_test(camera_device_name=camera_device_name,
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
    # validate_camera_registration(camera_device_name, rvecs, tvecs, mtx, dist, checker_board_square_edge_length)
    # Calculate the projection matrix
    projection_matrix = compute_projection_matrix(mtx, rvec=rvecs, tvec=tvecs)
    return world_points, detected_corners, mtx, dist, rvecs, tvecs, projection_matrix

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

def read_and_apply_subtractors_for_one_camera(cap, background_subtractor, list_of_color_detectors):
    """Read from one camera and segment"""
    # Read the camera
    ret, frame = cap.read()
    # First segment the background and get the mask
    fgmask = background_subtractor.apply(frame, learningRate=0)==255 # The mask have 3 value 0 127 for shadow and 255 for foreground
    # Second segment the colors by applying each detector on the frame and get the mask
    fgmask_color = np.zeros(shape=frame.shape[0:2], dtype=np.bool)
    for detector in list_of_color_detectors:
        fgmask_i = detector.apply(frame, learningRate=0)
        fgmask_i = ((fgmask_i==0) | (fgmask_i==127))
        # Combine with previous masks
        fgmask_color = fgmask_i | fgmask_color
    # Combine with background masks
    fgmask = fgmask & fgmask_color
    return fgmask

class ColorExtractorNode(Node):
    def __init__(self, camera_device_names, background_subtractors, lists_of_color_detectors, rectify_parameters):
        super().__init__('color_extractor_node')
        self.publisher_ = self.create_publisher(String, 'topic_name', 1)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.camera_device_names = camera_device_names
        self.background_subtractor_0 = background_subtractors[0]
        self.background_subtractor_1 = background_subtractors[1]
        self.list_of_color_detectors_0 = lists_of_color_detectors[0]
        self.list_of_color_detectors_1 = lists_of_color_detectors[1]
        self.cam_1_map1 = rectify_parameters[0]
        self.cam_1_map2 = rectify_parameters[1]
        self.cam_2_map1 = rectify_parameters[2]
        self.cam_2_map2 = rectify_parameters[3]
        self.Q = rectify_parameters[4]
        # Open the video capture
        self.cap_0 = cv2.VideoCapture(self.camera_device_names[0])
        self.cap_1 = cv2.VideoCapture(self.camera_device_names[1])

    def timer_callback(self):
        """Read the camera and segment the pixels periodically"""
        # msg = String()
        # msg.data = "Hello from ROS2 publisher!"
        # self.publisher_.publish(msg)
        # self.get_logger().info('Published: "%s"' % msg.data)

        start_time = time.perf_counter()
        # fgmask_0 = read_and_apply_subtractors_for_one_camera(self.cap_0, self.background_subtractor_0, self.list_of_color_detectors_0)
        # fgmask_1 = read_and_apply_subtractors_for_one_camera(self.cap_1, self.background_subtractor_1, self.list_of_color_detectors_1)
        # # Show the foreground (Convert binary to image)
        # fgmask_0 = fgmask_0.astype(np.uint8)*255
        # fgmask_1 = fgmask_1.astype(np.uint8)*255
        ret, fgmask_0 = self.cap_0.read()
        ret, fgmask_1 = self.cap_1.read()

        # Rectify the images
        fgmask_0 = cv2.remap(src=fgmask_0,
                            map1=self.cam_1_map1,
                            map2=self.cam_1_map2,
                            interpolation=cv2.INTER_LINEAR)
        fgmask_1 = cv2.remap(src=fgmask_1,
                            map1=self.cam_2_map1,
                            map2=self.cam_2_map2,
                            interpolation=cv2.INTER_LINEAR)
        # # Convert images to grayscale for stereo matching
        # img1_rect = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
        # img2_rect = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)
        # Show
        cv2.imshow('frame_0', fgmask_0)
        cv2.imshow('frame_1', fgmask_1)
        cv2.waitKey(1)
        rclpy.logging.get_logger('my_logger').info(f"Execute time {round(time.perf_counter()-start_time,3)}")

def main(args=sys.argv):
    """Some parameters"""
    camera_device_name_0 = '/home/dangthehung/Downloads/3_2_Stereo_Vision/image/myleft.jpg'
    camera_device_name_1 = '/home/dangthehung/Downloads/3_2_Stereo_Vision/image/myright.jpg'
    checker_board_size=[10,7]
    checker_board_square_edge_length=20
    number_of_frames_for_calibrating=1
    number_of_frames_for_registering_background=25
    number_of_frames_for_registering_colors=25
    background_varThreshold=128
    color_varThreshold=8
    # """Configurate the cameras"""
    # configurate_result = configurate_camera(camera_device_name_0)
    # assert configurate_result==True, "Failed to configurate camera"
    # configurate_result = configurate_camera(camera_device_name_1)
    # assert configurate_result==True, "Failed to configurate camera"
    # # show_cameras([camera_device_name_0,camera_device_name_1])
    # """Test the cameras"""
    # rclpy.logging.get_logger('my_logger').info(f"Testing camera...")
    # image_shape_0=test_camera(camera_device_name_0)
    # image_shape_1=test_camera(camera_device_name_1)
    # rclpy.logging.get_logger('my_logger').info(f"Finish testing camera!!!")
    """Calibrate cameras"""
    main_calibrate_result = calibrate_and_validate_one_camera(camera_device_name=camera_device_name_0,
                                                              checker_board_size=checker_board_size,
                                                              checker_board_square_edge_length=checker_board_square_edge_length,
                                                              number_of_frame_for_registering=number_of_frames_for_calibrating)
    world_points_0, detected_corners_0, mtx_0, dist_0, rvec_0, tvec_0, projection_matrix_0 = main_calibrate_result
    main_calibrate_result = calibrate_and_validate_one_camera(camera_device_name=camera_device_name_1,
                                                              checker_board_size=checker_board_size,
                                                              checker_board_square_edge_length=checker_board_square_edge_length,
                                                              number_of_frame_for_registering=number_of_frames_for_calibrating)
    world_points_1, detected_corners_1, mtx_1, dist_1, rvec_1, tvec_1, projection_matrix_1 = main_calibrate_result
    print(world_points_1[0][0])
    print(world_points_1[0][-1])
    """Calibrate the stereo parameters of the cameras"""   
    assert (world_points_0[0] == world_points_1[0]).all(), "Two cameras must identify the same world points"
    # assert (image_shape_0 == image_shape_1), "Expecting the camera to have same image shape"
    img1 = cv2.imread('/home/dangthehung/Downloads/3_2_Stereo_Vision/image/myleft.jpg')  #queryimage # left image
    img2 = cv2.imread('/home/dangthehung/Downloads/3_2_Stereo_Vision/image/myright.jpg') #trainimage # right image
    image_shape_0=img1.shape
    
    dist_0 = np.array([[ 0.0, 0, 0, 0, 0]])
    dist_1 = np.array([[ 0.0, 0, 0, 0, 0]])

    # mtx_0 = np.array([[1.32402549e+03, 0.00000000e+00, 9.98625786e+02], [0.00000000e+00, 1.33954539e+03, 7.43163899e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # mtx_1 = np.array([[1.32402549e+03, 0.00000000e+00, 9.98625786e+02], [0.00000000e+00, 1.33954539e+03, 7.43163899e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


    rectify_parameters = get_rectify_map_and_Q(world_points_list=world_points_0,
                                               detected_corners_list_1=detected_corners_0,
                                               detected_corners_list_2=detected_corners_1,
                                               mtx1=mtx_0,
                                               dist1=dist_0,
                                               mtx2=mtx_1,
                                               dist2=dist_1,
                                               image_shape=image_shape_0[0:2],
                                               alpha=1)
    cam_1_map1, cam_1_map2, cam_2_map1, cam_2_map2, Q = rectify_parameters
    print(cam_1_map1)
    print(cam_1_map2)

    # Rectify the image
    img1_rect = cv2.remap(src=img1,
                        map1=cam_1_map1,
                        map2=cam_1_map2,
                        interpolation=cv2.INTER_NEAREST)
    img2_rect = cv2.remap(src=img2,
                        map1=cam_2_map1,
                        map2=cam_2_map2,
                        interpolation=cv2.INTER_NEAREST)

    img1_rect=cv2.resize(img1_rect,dsize=(640,480))
    img2_rect=cv2.resize(img2_rect,dsize=(640,480))

    # Convert images to grayscale for stereo matching
    img1_rect = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
    img2_rect = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("1", img1)
    # cv2.imshow("2", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imshow("1", img1_rect)
    cv2.imshow("2", img2_rect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # background_subtractor_0=None
    # background_subtractor_1=None
    # list_of_color_detectors_0=None
    # list_of_color_detectors_1=None
    # # """Registering background"""
    # # rclpy.logging.get_logger('my_logger').info(f"Registering background {camera_device_name_0}... Please wait!!!")
    # # background_subtractor_0 = register_background(camera_device_name=camera_device_name_0,
    # #                                               number_of_register_frame=number_of_frames_for_registering_background,
    # #                                               varThreshold=background_varThreshold)
    # # rclpy.logging.get_logger('my_logger').info(f"Registering background {camera_device_name_1}... Please wait!!!")
    # # background_subtractor_1 = register_background(camera_device_name=camera_device_name_1,
    # #                                               number_of_register_frame=number_of_frames_for_registering_background,
    # #                                               varThreshold=background_varThreshold)
    # # """Registering colors"""
    # # rclpy.logging.get_logger('my_logger').info(f"Registering color {camera_device_name_0}... Please select your colors!!!")
    # # list_of_color_detectors_0 = register_gmm_colors(camera_device_name=camera_device_name_0,
    # #                                                 number_of_register_frame=number_of_frames_for_registering_colors,
    # #                                                 varThreshold=color_varThreshold)
    # # rclpy.logging.get_logger('my_logger').info(f"Registering color {camera_device_name_1}... Please select your colors!!!")
    # # list_of_color_detectors_1 = register_gmm_colors(camera_device_name=camera_device_name_1,
    # #                                                 number_of_register_frame=number_of_frames_for_registering_colors,
    # #                                                 varThreshold=color_varThreshold)
    # # rclpy.logging.get_logger('my_logger').info(f"Finish registering background and colors")
    # """Main ROS publisher"""
    # rclpy.init(args=args)
    # color_extractor_node = ColorExtractorNode(camera_device_names=[camera_device_name_0, camera_device_name_1],
    #                                           background_subtractors=[background_subtractor_0, background_subtractor_1],
    #                                           lists_of_color_detectors=[list_of_color_detectors_0, list_of_color_detectors_1],
    #                                           rectify_parameters=rectify_parameters)
    # try:
    #     rclpy.spin(color_extractor_node)
    # except KeyboardInterrupt:
    #     pass
    # rclpy.logging.get_logger('my_logger').info(f"Exiting")
    # # Clean exit (Optional as Python garbage collectors will handle them)
    # color_extractor_node.cap.release()
    # cv2.destroyAllWindows()
    # color_extractor_node.destroy_node()
    # rclpy.shutdown()

    # #region code storage
    # # # Show the cameras
    # # show_cameras(camera_device_names=[camera_device_name_0,])


    # # # Create 2D -> 3D converter
    # # inverse_projector = Inverse_Projector()
    # # inverse_projector.register_projection_matrix(main_calibrate_result_0, main_calibrate_result_1)
    # # rclpy.logging.get_logger('my_logger').info(inverse_projector)


    # #endregion

if __name__ == '__main__':
    main(sys.argv)
    # show_camera("/dev/video4")