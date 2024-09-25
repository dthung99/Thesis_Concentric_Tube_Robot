import cv2
import numpy as np
import time
import pyautogui
# import os
# import subprocess
from camera.video_process_func import *
import psutil

def print_memory_usage():
    """Used to manage memory"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_used = memory_info.rss / (1024.0 ** 2)  # Memory used in MB
    print(f"Current memory usage: {memory_used:.2f} MB")

def configurate_camera(camera_device_name):
    exposure=0
    white_balance=4600
    # Turn auto mode off for exposure and white balance
    cap = cv2.VideoCapture(camera_device_name)
    result = True
    result = result & cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
    result = result & cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
    result = result & cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
    # Set the exposure and white balance manually
    # result = result & cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    # result = result & cap.set(cv2.CAP_PROP_WB_TEMPERATURE, white_balance)
    return result

def test_camera(camera_device_name):
    """Test the camera to have the correct format"""
    # Open the camera
    cap = cv2.VideoCapture(camera_device_name)
    ret, frame = cap.read()
    assert ret == 1, "Can not open camera"
    assert len(frame.shape) == 3, "The functions are designed to work with color videos"
    assert frame.dtype == np.uint8, "The functions are designed to work with uint8 data type, comment out this line if you still want to work with other data type"
    return frame.shape

def show_camera(camera_device_name):
    # Open the camera
    cap = cv2.VideoCapture(camera_device_name)
    ret, frame = cap.read()
    while True:
        # start_time = time.perf_counter()
        # Capture a frame from the camera
        ret, frame = cap.read()
        # Display the frame
        cv2.imshow(f'Camera Feed {camera_device_name}', frame)
        # Press 'esc' to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # print(time.perf_counter()-start_time)
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

def show_cameras(camera_device_names):
    # Open the camera
    caps = []
    for camera_device_name in camera_device_names:
        caps.append(cv2.VideoCapture(camera_device_name))
    while True:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            # Display the frame
            cv2.imshow(f'Camera Feed {i}', frame)
        # Press 'esc' to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # Release the camera and close all windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

def calibrate_one_camera_and_get_checker_board_vertices_test(camera_device_name,
                                                             checker_board_size=[10,7],
                                                             checker_board_square_edge_length=25,
                                                             number_of_frame_for_registering=50,
                                                             show=False):
    # Open the camera
    frame = cv2.imread(camera_device_name)
    ret = 1
    # Some parameter for tuning
    expected_image_size = np.array(frame.shape)
    corner_refinement_window = max(int(expected_image_size.max()/50), 11) #The window size for subpixel refinement
    # Termination criteria for checker board detection refinement
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # Number of u and v corner (u = width, v = height)
    number_of_u_corners = int(checker_board_size[0])
    number_of_v_corners = int(checker_board_size[1])
    # Prepare object points (3D points), like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((number_of_v_corners*number_of_u_corners,3), np.float32)
    objp[:,:2] = np.mgrid[0:number_of_u_corners,0:number_of_v_corners].T.reshape(-1,2) #Someone else code. It just create a grid of (-1,3) coordinate
    objp = objp*checker_board_square_edge_length
    # Create a counter to keep track of number of frames that is registered
    # objpoints and imgpoints to store the points detected in multiple frame
    counter = 0
    world_points_list = []
    detected_corners_list = []
    while counter<number_of_frame_for_registering:
        counter+=1
        print(f"Counter: {counter}/{number_of_frame_for_registering}")
        # Get the image from tha camera
        frame = cv2.imread(camera_device_name)
        if ret == False:
            continue
        # Convert the image to gray scale
        if(frame.shape[-1] == 3):
            input_image_in_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        else:
            input_image_in_gray = frame
        # Find the chess board corners
        find_corner_result, detected_corners = cv2.findChessboardCorners(input_image_in_gray,
                                                                         (number_of_u_corners, number_of_v_corners),
                                                                         flags=None)
        if find_corner_result==False:
            # This show help when show=True, the visualization is smoother
            if show:
                cv2.imshow(f'Detecting corners {camera_device_name}',frame)
                # Press 'esc' to exit the loop
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue
        # Refine the points to get better ones
        # This code modify the original corners too!!!
        detected_corners = cv2.cornerSubPix(input_image_in_gray,
                                            detected_corners,
                                            (corner_refinement_window, corner_refinement_window),
                                            (-1,-1),
                                            criteria)
        # Store the points
        world_points_list.append(objp)
        detected_corners_list.append(detected_corners)
        # Show the corners detected
        if show:
            # Draw and display the corners
            frame = cv2.drawChessboardCorners(frame,
                                              (number_of_u_corners,number_of_v_corners),
                                              detected_corners,
                                              1)
            cv2.imshow(f'Detecting corners {camera_device_name}',frame)
            # Press 'esc' to exit the loop
            if cv2.waitKey(1) & 0xFF == 27:
                break
        # return find_corner_result, detected_corners, objp
    # Release the camera and close all windows
    cv2.destroyAllWindows()
    # End if no corner is detected
    if len(world_points_list) == 0:
        return detected_corners_list, world_points_list        
    # Calibrate the camera and get the projection matrix
    print(f"Calibrating {camera_device_name}")
    image_shape = expected_image_size[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points_list, detected_corners_list, image_shape[::-1], None, None)
    ##Calculate and print projection Error
    mean_error = 0
    for i in range(len(world_points_list)):
        projected_imgpoints, _ = cv2.projectPoints(world_points_list[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(detected_corners_list[i],projected_imgpoints, cv2.NORM_L2)/len(projected_imgpoints)
        mean_error += error
    print(f"Average projection error for camera {camera_device_name}: {round(mean_error/len(world_points_list),4)} pixels")    
    return detected_corners_list, world_points_list, mtx, dist, rvecs, tvecs


def calibrate_one_camera_and_get_checker_board_vertices(camera_device_name,
                                                        checker_board_size=[10,7],
                                                        checker_board_square_edge_length=25,
                                                        number_of_frame_for_registering=50,
                                                        show=False):
    # Open the camera
    cap = cv2.VideoCapture(camera_device_name)
    ret, frame = cap.read()
    # Some parameter for tuning
    expected_image_size = np.array(frame.shape)
    corner_refinement_window = max(int(expected_image_size.max()/50), 11) #The window size for subpixel refinement
    # Termination criteria for checker board detection refinement
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # Number of u and v corner (u = width, v = height)
    number_of_u_corners = int(checker_board_size[0])
    number_of_v_corners = int(checker_board_size[1])
    # Prepare object points (3D points), like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((number_of_v_corners*number_of_u_corners,3), np.float32)
    objp[:,:2] = np.mgrid[0:number_of_u_corners,0:number_of_v_corners].T.reshape(-1,2) #Someone else code. It just create a grid of (-1,3) coordinate
    objp = objp*checker_board_square_edge_length
    # Create a counter to keep track of number of frames that is registered
    # objpoints and imgpoints to store the points detected in multiple frame
    counter = 0
    world_points_list = []
    detected_corners_list = []
    while counter<number_of_frame_for_registering:
        counter+=1
        print(f"Counter: {counter}/{number_of_frame_for_registering}")
        # Get the image from tha camera
        ret, frame = cap.read()
        if ret == False:
            continue
        # Convert the image to gray scale
        if(frame.shape[-1] == 3):
            input_image_in_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        else:
            input_image_in_gray = frame
        # Find the chess board corners
        find_corner_result, detected_corners = cv2.findChessboardCorners(input_image_in_gray,
                                                                         (number_of_u_corners, number_of_v_corners),
                                                                         flags=None)
        if find_corner_result==False:
            # This show help when show=True, the visualization is smoother
            if show:
                cv2.imshow(f'Detecting corners {camera_device_name}',frame)
                # Press 'esc' to exit the loop
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue
        # Refine the points to get better ones
        # This code modify the original corners too!!!
        detected_corners = cv2.cornerSubPix(input_image_in_gray,
                                            detected_corners,
                                            (corner_refinement_window, corner_refinement_window),
                                            (-1,-1),
                                            criteria)
        # Store the points
        world_points_list.append(objp)
        detected_corners_list.append(detected_corners)
        # Show the corners detected
        if show:
            # Draw and display the corners
            frame = cv2.drawChessboardCorners(frame,
                                              (number_of_u_corners,number_of_v_corners),
                                              detected_corners,
                                              1)
            cv2.imshow(f'Detecting corners {camera_device_name}',frame)
            # Press 'esc' to exit the loop
            if cv2.waitKey(1) & 0xFF == 27:
                break
        # return find_corner_result, detected_corners, objp
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    # End if no corner is detected
    if len(world_points_list) == 0:
        return detected_corners_list, world_points_list        
    # Calibrate the camera and get the projection matrix
    print(f"Calibrating {camera_device_name}")
    image_shape = expected_image_size[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points_list, detected_corners_list, image_shape[::-1], None, None)
    ##Calculate and print projection Error
    mean_error = 0
    for i in range(len(world_points_list)):
        projected_imgpoints, _ = cv2.projectPoints(world_points_list[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(detected_corners_list[i],projected_imgpoints, cv2.NORM_L2)/len(projected_imgpoints)
        mean_error += error
    print(f"Average projection error for camera {camera_device_name}: {round(mean_error/len(world_points_list),4)} pixels")    
    return detected_corners_list, world_points_list, mtx, dist, rvecs, tvecs

# Warning!!!: no unit test for this
def get_rectify_map_and_Q(world_points_list,
                          detected_corners_list_1,
                          detected_corners_list_2,
                          mtx1,
                          dist1,
                          mtx2,
                          dist2,
                          image_shape,
                          alpha=1):
    """Get the neccessary map for rectification and the Q matrix
    world_points_list: list of world coordinates,
    detected_corners_list_1: list of checkerboard corners coordinates,
    detected_corners_list_2: list of checkerboard corners coordinates,
    mtx1: camera matrix 1,
    dist1: distortion coeff 1,
    mtx2: camera matrix 2,
    dist2: distortion coeff 2,
    image_shape: MUST BE NUMPY SHAPE (HEIGHT, WIDTH), NO CHANNEL"""
    #  stereo Calibrate
    stereo_calibrate_results = cv2.stereoCalibrate(objectPoints=world_points_list,
                                                   imagePoints1=detected_corners_list_1,
                                                   imagePoints2=detected_corners_list_2,
                                                   cameraMatrix1=mtx1,
                                                   distCoeffs1=dist1,
                                                   cameraMatrix2=mtx2,
                                                   distCoeffs2=dist2,
                                                   imageSize=image_shape[::-1])
    retval, mtx1, dist1, mtx2, dist2, R, T, E, F = stereo_calibrate_results
    # Get the Rectification coeff
    rectify_coeff_results = cv2.stereoRectify(cameraMatrix1=mtx1,
                                              distCoeffs1=dist1,
                                              cameraMatrix2=mtx2,
                                              distCoeffs2=dist2,
                                              imageSize=image_shape[::-1],
                                              R=R,
                                              T=T,
                                              alpha=alpha)

    R1, R2, P1, P2, Q, ROI1, ROI2 = rectify_coeff_results

    # Calculate the Rectification map
    cam_1_map1, cam_1_map2 = cv2.initUndistortRectifyMap(cameraMatrix=mtx1,
                                                        distCoeffs=dist1,
                                                        R=R1,
                                                        newCameraMatrix=P1,
                                                        size=image_shape[::-1],
                                                        m1type=cv2.CV_32FC1)

    cam_2_map1, cam_2_map2 = cv2.initUndistortRectifyMap(cameraMatrix=mtx2,
                                                        distCoeffs=dist2,
                                                        R=R2,
                                                        newCameraMatrix=P2,
                                                        size=image_shape[::-1],
                                                        m1type=cv2.CV_32FC1)
    
    return cam_1_map1, cam_1_map2, cam_2_map1, cam_2_map2, Q

def draw_line(img, corners, imgpts, line_width=2):
    assert imgpts.dtype == np.int_, "imgpts need to be np.int_"
    corner = tuple(np.round(corners[0].ravel()).astype(np.int_))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), line_width)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), line_width)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), line_width)
    return img

def validate_camera_registration(camera_device_name, rvecs, tvecs, mtx, dist, axis_length):
    axis = np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)*axis_length
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    imgpts = np.round(imgpts).astype(np.int_)
    # Open the camera
    cap = cv2.VideoCapture(camera_device_name)
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        # Draw on the image
        frame = draw_line(frame,imgpts[0],imgpts[1:])
        # Display the frame
        cv2.imshow(f'Validate camera calibration {camera_device_name}', frame)
        # Press 'esc' to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return

def compute_projection_matrix(camera_matrix, rvec, tvec):
    """compute_projection_matrix from the camera_matrix (initial matrix)
    and rotation (rvec) and tranlation of (tvec) camera to world
    NOTE: The matrix notaion is different from what I learned in Robotic theory
    homogenous transformation should be performed with care!!!
    camera_matrix: 3x4
    rvec: 3x1
    tvec: 3x1
    """
    R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to matrix
    extrinsic = np.hstack((R, tvec))  # Combine rotation and translation
    return camera_matrix @ extrinsic # Compute the projection matrix

"""WARNING: NOT TESTED REGIONS!!!!! - The test for code in this region is not provided"""
#region NOT TESTED CODES!!!!
def get_3D_point_from_homogeneous_coordinate(x):
    x = x/x[-1]
    return x[:3]

class Inverse_Projector:
    """Used to get 3D coordinate from 2 pictures"""
    def register_projection_matrix(self, projection_matrix_0, projection_matrix_1):
        """Store the projection matrix of 2 cameras"""
        self.projection_matrix_0=projection_matrix_0
        self.projection_matrix_1=projection_matrix_1
    def get_3D_from_2D_nx2(self, point_array_0: np.ndarray((2,100)), point_array_1: np.ndarray((2,100))) -> np.ndarray((3,100)):
        """Calculate the 3D coordinate
        WARNING!!!
        The input needs to be np array of shape (2,n)
        The output is (3,n)"""
        point_in_3D = cv2.triangulatePoints(self.projection_matrix_0,
                                            self.projection_matrix_1,
                                            point_array_0,
                                            point_array_1)
        results = np.apply_along_axis(get_3D_point_from_homogeneous_coordinate, axis=0, arr=point_in_3D)
        return results
#endregion

def register_background(camera_device_name,
                        number_of_register_frame=500,
                        varThreshold=128):
    """Register the background for background removal"""
    # Open the video capture and create the background subtractors
    cap = cv2.VideoCapture(camera_device_name)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=varThreshold)
    # Create a counter to keep track of number of frames that is registered
    counter = 0
    while(counter<number_of_register_frame):
        counter+=1
        print(f"Counter: {counter}/{number_of_register_frame}")
        # Read the camera
        ret, frame = cap.read()
        # applying to get the subtractor and background image
        if ret:
            fgmask = background_subtractor.apply(frame, learningRate=-1)
        # Display the frame
        cv2.imshow(f'Registering background {camera_device_name}', frame)
        # Press 'esc' to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # Clean the environment
    cap.release()
    cv2.destroyAllWindows()
    return background_subtractor

def register_gmm_colors(camera_device_name,
                        number_of_register_frame: int = 500,
                        varThreshold=16):
    # Create storages for the detectors, interator for the cameras
    registered_point_coordinates = []
    list_of_color_detector = []
    number_of_frame_registered_for_each_detector = []
    # Create a window to display the image
    cv2.namedWindow(f'Register colors for {camera_device_name}')
    # Define a function to store the color at mouse clicks position
    def get_ggm_color_subtractor(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:            
            print(f"Registering color at ({x}, {y}) for {camera_device_name}, color {frame[y,x]}")
            # Change the image
            cv2.circle(clicked_point_image, (x, y), 1, (0, 0, 255), -1)
            # Update the points
            registered_point_coordinates.append([y,x])
            # Create new cv2.BackgroundSubtractorMOG2
            list_of_color_detector.append(cv2.createBackgroundSubtractorMOG2(varThreshold=varThreshold))
            number_of_frame_registered_for_each_detector.append(0)
    # Set the mouse callback function for the window
    cv2.setMouseCallback(f'Register colors for {camera_device_name}', get_ggm_color_subtractor, camera_device_name)
    # Open the video capture
    cap = cv2.VideoCapture(camera_device_name)
    # Image to store the clicked point
    ret, frame = cap.read()
    image_shape = frame.shape
    clicked_point_image = np.zeros(frame.shape, dtype=np.uint8)
    mouse_step = 1 # Mouse step to control with keyboard
    while True:
        # Read the camera
        ret, frame = cap.read()
        # Register the detector
        for i, (coordinate, detector) in enumerate(zip(registered_point_coordinates, list_of_color_detector)):
            if number_of_frame_registered_for_each_detector[i] < number_of_register_frame:
                # Broadcast the current point to image shape
                colors_image = frame[*coordinate]
                colors_image = np.tile(colors_image[None,None,:],[*image_shape[0:2],1])
                # Update the detector
                detector.apply(colors_image, learningRate=-1)
                number_of_frame_registered_for_each_detector[i] +=1
                print(f"Registering: {number_of_frame_registered_for_each_detector[i]}/{number_of_register_frame}")
        # Identify the clicked points
        # MAKE SURE IT'S AFTER COLOR REGISTERING PLEASE
        frame = np.where(clicked_point_image==0, frame, clicked_point_image)
        cv2.imshow(f'Register colors for {camera_device_name}', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # Keys to move the mouse
        if k == ord("w"):
            pyautogui.moveRel(0, -mouse_step, duration=0)
        if k == ord("s"):
            pyautogui.moveRel(0, mouse_step, duration=0)
        if k == ord("a"):
            pyautogui.moveRel(-mouse_step, 0, duration=0)
        if k == ord("d"):
            pyautogui.moveRel(mouse_step, 0, duration=0)
        if k == ord(" "):
            pyautogui.click()

    for detector in list_of_color_detector:
        final_color = detector.getBackgroundImage()
        cv2.imshow(f"Final colors for {camera_device_name}", final_color)
        while(1):
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release() 
    cv2.destroyAllWindows()
    return list_of_color_detector

def segment_color_and_background(camera_device_name,
                                 background_subtractor,
                                 list_of_color_detectors):
    # Open the video capture
    cap = cv2.VideoCapture(camera_device_name)
    while(1):
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
        # Apply the mask to the image
        # # Apply the mask to the image
        # fgmask = (fgmask[:,:,None])*frame # The mask have 3 value 0 127 for shadow and 255 for foreground
        # # Get the non-zero coordinate
        # coordinates = extract_non_zero_pixel(image=fgmask)
        # adjacency_matrix = get_adjacency_matrix(coordinates=coordinates)
        # connected_components = get_connected_components(adjacency_matrix=adjacency_matrix)
        # Show the foreground
        cv2.imshow('frame', fgmask.astype(np.uint8)*255)
        # Exit when press "esc"
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
    # Clean the environment
    cap.release() 
    cv2.destroyAllWindows() 

#region NOT USING
def remove_background(camera_device_name, background_subtractor):
    # Open the video capture
    cap = cv2.VideoCapture(camera_device_name)
    while(1):
        # Read the camera
        ret, frame = cap.read()
        # Applying on each frame and get the mask
        fgmask = background_subtractor.apply(frame, learningRate=0)
        # Apply the mask to the image
        fgmask = (fgmask[:,:,None]==255)*frame # The mask have 3 value 0 127 for shadow and 255 for foreground
        # # Get the non-zero coordinate
        # coordinates = extract_non_zero_pixel(image=fgmask)
        # adjacency_matrix = get_adjacency_matrix(coordinates=coordinates)
        # connected_components = get_connected_components(adjacency_matrix=adjacency_matrix)
        # Show the foreground
        cv2.imshow('frame', fgmask)
        # Exit when press "esc"
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
    # Clean the environment
    cap.release() 
    cv2.destroyAllWindows() 


def segment_ggm_color(camera_device_name, list_of_color_detectors):
    # Open the video capture
    cap = cv2.VideoCapture(camera_device_name)
    while(1):
        # Read the camera
        ret, frame = cap.read()
        # Applying on each frame and get the mask
        fgmask = np.zeros(shape=frame.shape[0:2], dtype=np.uint8)
        for detector in list_of_color_detectors:
            fgmask_i = detector.apply(frame, learningRate=0)
            fgmask_i = ((fgmask_i==0) | (fgmask_i==127))
            fgmask = fgmask_i | fgmask
        # # Apply the mask to the image
        # fgmask = (fgmask[:,:,None])*frame # The mask have 3 value 0 127 for shadow and 255 for foreground
        # # Get the non-zero coordinate
        # coordinates = extract_non_zero_pixel(image=fgmask)
        # adjacency_matrix = get_adjacency_matrix(coordinates=coordinates)
        # connected_components = get_connected_components(adjacency_matrix=adjacency_matrix)
        # Show the foreground
        cv2.imshow('frame', fgmask*255)
        # Exit when press "esc"
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
    # Clean the environment
    cap.release() 
    cv2.destroyAllWindows() 
#endregion

if __name__ == '__main__':
    """Run this script to test the camera on webcam
    """
    # """Test calibration"""
    # # Some parameters
    # camera_device = '/dev/video0'
    # checker_board_size=[10,7]
    # checker_board_square_edge_length=25
    # number_of_frame_for_registering=50
    # # Check configurate_camera
    # configurate_result = configurate_camera(camera_device)
    # assert configurate_result==True, "configurate_camera failed"
    # # Test the cameras and get camera output shape
    # print(f"Testing camera {camera_device}...")
    # camera_output_shape_0 = test_camera(camera_device)
    # print(f"Finished testing camera!!!")
    # # Calibrate the camera
    # ## Show the camera
    # show_camera(camera_device)
    # ## Detect the corners from the checker board
    # print(f"Registering camera {camera_device}...")
    # calibrate_result_0 = calibrate_one_camera_and_get_checker_board_vertices(camera_device_name=camera_device,
    #                                                                          checker_board_size=checker_board_size,
    #                                                                          checker_board_square_edge_length=checker_board_square_edge_length,
    #                                                                          number_of_frame_for_registering=number_of_frame_for_registering,
    #                                                                          show=True)
    # assert len(calibrate_result_0[0])>0, f"Failed to calibrate camera {camera_device}"
    # print(f"succeeded to calibrate camera {camera_device} with {len(calibrate_result_0[0])} frames")
    # ## Validate the calibration
    # detected_corners, world_points, mtx, dist, rvecs, tvecs = calibrate_result_0
    # # Because the camera is fixed, the rvecs and tvecs should be the same for all frame
    # rvecs = np.stack(rvecs, axis=0)
    # print(f"Std of camera {camera_device} of rotation vector {rvecs.std()}")
    # rvecs = rvecs.mean(axis=0)
    # tvecs = np.stack(tvecs, axis=0)
    # print(f"Std of camera {camera_device} of rotation vector {tvecs.std()}")
    # tvecs = tvecs.mean(axis=0)
    # # Validate the camera
    # validate_camera_registration(camera_device, rvecs, tvecs, mtx, dist, checker_board_square_edge_length)
    # # Calculate the projection matrix
    # projection_matrix = compute_projection_matrix(mtx, rvec=rvecs, tvec=tvecs)

    """Test segmentation"""
    # camera_device_name = '/dev/video0'
    # number_of_frames_for_registering=100
    # background_subtractor = register_background(camera_device_name=camera_device_name,
    #                                             number_of_register_frame=number_of_frames_for_registering,
    #                                             varThreshold=128)
    # remove_background(camera_device_name=camera_device_name,
    #                   background_subtractor=background_subtractor)

    # number_of_frames_for_registering=100
    # list_of_color_detectors = register_gmm_colors(camera_device_name=camera_device_name,
    #                                               number_of_register_frame=number_of_frames_for_registering,
    #                                               varThreshold=16)
    # segment_ggm_color(camera_device_name=camera_device_name,
    #                   list_of_color_detectors=list_of_color_detectors)

    # segment_color_and_background(camera_device_name=camera_device_name,
    #                              background_subtractor=background_subtractor,
    #                              list_of_color_detectors=list_of_color_detectors)

