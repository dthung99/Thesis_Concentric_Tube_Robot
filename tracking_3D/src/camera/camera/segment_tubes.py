# General pkg
import cv2
import numpy as np
import sys
import time
# ROS pkg
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
# My pkg
from camera.camera_register_module import *
from camera.video_process_func import *

test_point = None

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

def drawline(image,line):
    ''' image - image on which we draw the line
        line - corresponding line '''
    img = image.copy()
    if len(line) == 0:
        return img
    height, width = img.shape[0:2]
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r = line
    color = tuple(np.random.randint(0,255,3).tolist())
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [width, -(r[2]+r[0]*width)/r[1]])
    img = cv2.line(img, (x0,y0), (x1,y1), color,1)
    return img

def draw_points(image, points, color=255, radius=1, thickness=1):
    # Draw points on an image
    img = image.copy()
    try:
        for x, y in points:
            cv2.circle(img, (int(x), int(y)), radius=radius, color=color, thickness=thickness)
        return img
    except Exception as e:
        print(f"An error occurred: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return img

def create_and_configurate_cloud_points():
    # Create PointCloud2 message
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = "map"
    msg.height = 1
    # Please assure that each x y z is a float32 data
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.is_dense = True
    return msg

class ColorExtractorNode(Node):
    def __init__(self, camera_device_names, background_subtractors, lists_of_color_detectors, calibration_parameters, points_for_epilines=None):
        super().__init__('color_extractor_node')
        # Create the publisher and timer
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/tubes_point_cloud', 1)
        self.point_cloud_msg = create_and_configurate_cloud_points()
        self.timer = self.create_timer(0.1, self.timer_callback)
        # Store important variables
        self.camera_device_names = camera_device_names
        self.background_subtractor_0 = background_subtractors[0]
        self.background_subtractor_1 = background_subtractors[1]
        self.list_of_color_detectors_0 = lists_of_color_detectors[0]
        self.list_of_color_detectors_1 = lists_of_color_detectors[1]
        self.projection_matrix_0 = calibration_parameters[0]
        self.projection_matrix_1 = calibration_parameters[1]
        self.F = calibration_parameters[2]
        # Open the video capture
        self.cap_0 = cv2.VideoCapture(self.camera_device_names[0])
        self.cap_1 = cv2.VideoCapture(self.camera_device_names[1])
        # Create a black image with the same type and shape as the frame
        ret, frame = self.cap_1.read()
        self.back_image = np.zeros(shape=frame.shape[0:2], dtype=np.uint8)
        # Plot the epilines on the image
        if points_for_epilines is not None:
            lines_1 = cv2.computeCorrespondEpilines(points_for_epilines.reshape(-1,1,2),
                                                    whichImage=1,
                                                    F=self.F)
            lines_1=lines_1.reshape(-1,3)
            for line in lines_1:
                self.back_image = drawline(image=self.back_image, line=line)

    def timer_callback(self):
        """Read the camera and segment the pixels periodically"""
        start_time = time.perf_counter()
        # Segment the color
        fgmask_0 = read_and_apply_subtractors_for_one_camera(self.cap_0, self.background_subtractor_0, self.list_of_color_detectors_0)
        fgmask_1 = read_and_apply_subtractors_for_one_camera(self.cap_1, self.background_subtractor_1, self.list_of_color_detectors_1)
        # Extract the non-zero pixels
        """NOTE THE COORDINATES returned are (height, width) while I need (width, height)"""
        coordinates_0 = extract_non_zero_pixel_in_black_white_image(image=fgmask_0)[:,::-1]
        if len(coordinates_0)==0:
            self.simple_plot(fgmask_0, fgmask_1, self.back_image)
            return
        coordinates_1 = extract_non_zero_pixel_in_black_white_image(image=fgmask_1)[:,::-1]
        # adjacency_matrix_0 =  get_adjacency_matrix(coordinates=coordinates_0, neighbour_square_distance_cut_off=100)
        # adjacency_matrix_1 =  get_adjacency_matrix(coordinates=coordinates_1, neighbour_square_distance_cut_off=100)
        # connected_components_0=get_connected_components(adjacency_matrix_0)
        # connected_components_1=get_connected_components(adjacency_matrix_1)
        # Find epilines corresponding to points in left image (first image) and draw its lines on right image
        lines_1 = cv2.computeCorrespondEpilines(coordinates_0.reshape(-1,1,2),
                                                whichImage=1,
                                                F=self.F)
        # Find the matching points
        matched_result=find_points_nearest_to_lines_and_return_one_on_those_lines(points=coordinates_1.reshape(-1,2),
                                                                                  lines=lines_1.reshape(-1,3),
                                                                                  square_of_cut_off_for_near=1.5,
                                                                                  std_cut_off=20)
        matched_coordinates_1, matched_mask = matched_result
        if matched_mask.sum()==0:
            self.simple_plot(fgmask_0, fgmask_1, self.back_image)
            return
        if len(matched_mask)!=len(matched_coordinates_1):
            self.simple_plot(fgmask_0, fgmask_1, self.back_image)
            return # TODO later: I have no idea why sometimes the length of these two do not match
        # Get the points that have correspondence and project to 3D
        proj_point_0=coordinates_0[matched_mask].astype(np.float32)
        proj_point_1=matched_coordinates_1[matched_mask]
        point_in_3D = cv2.triangulatePoints(projMatr1=self.projection_matrix_0,
                                            projMatr2=self.projection_matrix_1,
                                            projPoints1=proj_point_0.squeeze().T,
                                            projPoints2=proj_point_1.squeeze().T)
        # The output shape is (4,n) in homogeneous coordinate => transform it to (n,3)
        point_in_3D[1:3]=-point_in_3D[1:3] #Flip the y and z axis for easier visualization
        point_in_3D = (point_in_3D[0:3]/point_in_3D[3]).T
        point_in_3D = point_in_3D/10 #Scale down to cm for easier visulization
        # Set up the point clouds to send
        self.point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        self.point_cloud_msg.width = point_in_3D.shape[0]
        self.point_cloud_msg.row_step = self.point_cloud_msg.point_step * point_in_3D.shape[0]
        self.point_cloud_msg.data = point_in_3D.tobytes()
        # Publish the point cloud to RVIZ
        self.point_cloud_publisher.publish(self.point_cloud_msg)
        self.get_logger().info(f"Published: {point_in_3D.shape} points cloud")
        # Show the foreground (Convert binary to image)
        fgmask_0 = fgmask_0.astype(np.uint8)*255
        fgmask_1 = fgmask_1.astype(np.uint8)*255
        fgmask_1 = drawline(fgmask_1, lines_1.reshape(-1,3)[0])
        img = draw_points(image=self.back_image, points=proj_point_1)
        # Show
        cv2.imshow('frame_0', fgmask_0)
        cv2.imshow('frame_1', fgmask_1)
        cv2.imshow('validation', img)
        # Read the camera
        ret, frame = self.cap_0.read()
        cv2.imshow('frame_0_o', frame)
        ret, frame = self.cap_1.read()
        cv2.imshow('frame_1_o', frame)
        cv2.waitKey(1)
        rclpy.logging.get_logger('my_logger').info(f"Execute time {round(time.perf_counter()-start_time,3)}")
    def simple_plot(self, fgmask_0, fgmask_1, back_image):
        """"Helper function to plot the fgmask"""
        fgmask_0 = fgmask_0.astype(np.uint8)*255
        fgmask_1 = fgmask_1.astype(np.uint8)*255
        cv2.imshow('frame_0', fgmask_0)
        cv2.imshow('frame_1', fgmask_1)
        cv2.imshow('validation', back_image)
        cv2.waitKey(1)
        return        

def main(args=sys.argv):
    """Some parameters"""
    camera_device_name_0 = args[1]
    camera_device_name_1 = args[2]
    checker_board_size=[6,4]
    checker_board_square_edge_length=15
    number_of_frames_for_calibrating=10
    number_of_frames_for_registering_background=25
    number_of_frames_for_registering_colors=25
    background_varThreshold=128
    color_varThreshold=8
    """Configurate the cameras"""
    configurate_result = configurate_camera(camera_device_name_0)
    assert configurate_result==True, "Failed to configurate camera"
    configurate_result = configurate_camera(camera_device_name_1)
    assert configurate_result==True, "Failed to configurate camera"
    # show_cameras([camera_device_name_0,camera_device_name_1])
    """Test the cameras"""
    rclpy.logging.get_logger('my_logger').info(f"Testing camera...")
    image_shape_0=test_camera(camera_device_name_0)
    image_shape_1=test_camera(camera_device_name_1)
    rclpy.logging.get_logger('my_logger').info(f"Finish testing camera!!!")
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
    # Get the fundamental matrix
    F, mask = cv2.findFundamentalMat(points1=detected_corners_0[0],
                                     points2=detected_corners_1[0],
                                     method=cv2.FM_LMEDS)
    # global test_point
    # test_point = detected_corners_0[0][0]
    # print(test_point)
    background_subtractor_0=None
    background_subtractor_1=None
    list_of_color_detectors_0=None
    list_of_color_detectors_1=None
    """Registering background"""
    rclpy.logging.get_logger('my_logger').info(f"Registering background {camera_device_name_0}... Please wait!!!")
    background_subtractor_0 = register_background(camera_device_name=camera_device_name_0,
                                                  number_of_register_frame=number_of_frames_for_registering_background,
                                                  varThreshold=background_varThreshold)
    rclpy.logging.get_logger('my_logger').info(f"Registering background {camera_device_name_1}... Please wait!!!")
    background_subtractor_1 = register_background(camera_device_name=camera_device_name_1,
                                                  number_of_register_frame=number_of_frames_for_registering_background,
                                                  varThreshold=background_varThreshold)
    """Registering colors"""
    rclpy.logging.get_logger('my_logger').info(f"Registering color {camera_device_name_0}... Please select your colors!!!")
    list_of_color_detectors_0 = register_gmm_colors(camera_device_name=camera_device_name_0,
                                                    number_of_register_frame=number_of_frames_for_registering_colors,
                                                    varThreshold=color_varThreshold)
    rclpy.logging.get_logger('my_logger').info(f"Registering color {camera_device_name_1}... Please select your colors!!!")
    list_of_color_detectors_1 = register_gmm_colors(camera_device_name=camera_device_name_1,
                                                    number_of_register_frame=number_of_frames_for_registering_colors,
                                                    varThreshold=color_varThreshold)
    rclpy.logging.get_logger('my_logger').info(f"Finish registering background and colors")
    """Main ROS publisher"""
    rclpy.init(args=args)
    calibration_parameters = [projection_matrix_0, projection_matrix_1, F]
    color_extractor_node = ColorExtractorNode(camera_device_names=[camera_device_name_0, camera_device_name_1],
                                              background_subtractors=[background_subtractor_0, background_subtractor_1],
                                              lists_of_color_detectors=[list_of_color_detectors_0, list_of_color_detectors_1],
                                              calibration_parameters=calibration_parameters,
                                              points_for_epilines=detected_corners_0[0])
    try:
        rclpy.spin(color_extractor_node)
    except KeyboardInterrupt:
        pass
    rclpy.logging.get_logger('my_logger').info(f"Exiting")
    # Clean exit (Optional as Python garbage collectors will handle them)
    color_extractor_node.cap.release()
    cv2.destroyAllWindows()
    color_extractor_node.destroy_node()
    rclpy.shutdown()

    #region code storage
    # # Show the cameras
    # show_cameras(camera_device_names=[camera_device_name_0,])


    # # Create 2D -> 3D converter
    # inverse_projector = Inverse_Projector()
    # inverse_projector.register_projection_matrix(main_calibrate_result_0, main_calibrate_result_1)
    # rclpy.logging.get_logger('my_logger').info(inverse_projector)


    #endregion

if __name__ == '__main__':
    main(sys.argv)
    # show_camera("/dev/video4")