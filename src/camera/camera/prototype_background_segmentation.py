import cv2
import numpy as np
import time
import os
import subprocess
from camera.video_process_func import *

import psutil
# Image shape (480, 640, 3)
# time_point = time.perf_counter()
# print(f"Time 1 {time.perf_counter()-time_point}")

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_used = memory_info.rss / (1024.0 ** 2)  # Memory used in MB
    print(f"Current memory usage: {memory_used:.2f} MB")

camera_device = '/dev/video0'

def test_camera():
    """Test the camera to have the correct format"""
    # Open the camera
    cap = cv2.VideoCapture(camera_device)
    ret, frame = cap.read()
    assert ret == 1, "Can not open camera"
    assert len(frame.shape) == 3, "The functions are designed to work with color videos"
    assert frame.dtype == np.uint8, "The functions are designed to work with uint8 data type, comment out this line if you still want to work with other data type"

def show_camera():
    # Open the camera
    cap = cv2.VideoCapture(camera_device)
    ret, frame = cap.read()
    while True:
        # start_time = time.perf_counter()
        # Capture a frame from the camera
        ret, frame = cap.read()
        # Display the frame
        cv2.imshow('Camera Feed', frame)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print(time.perf_counter()-start_time)
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

def register_background(background_subtractor: cv2.BackgroundSubtractorMOG2,
                        register_period: float = 2.0):
    # Open the video capture
    cap = cv2.VideoCapture(camera_device)
    # Start the timer
    start_time = time.time()
    while(time.time()-start_time<register_period):
        # read the camera
        ret, frame = cap.read()
        # applying to get the subtractor and background image
        if ret:
            fgmask = background_subtractor.apply(frame, learningRate=-1)
    cap.release()

def remove_background(background_subtractor):
    # Open the video capture
    cap = cv2.VideoCapture(camera_device)
    while(1):
        print_memory_usage()
        # Read the camera
        ret, frame = cap.read()
        # Applying on each frame and get the mask
        time_point = time.perf_counter()
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


def register_colors(registered_colors):
    # Create a window to display the image
    cv2.namedWindow('Register colors')
    # Define a function to store the color at mouse clicks position
    def get_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            
            print(f"Registering color at ({x}, {y}), color {frame[y,x]}")
            # Change the image
            cv2.circle(clicked_point_image, (x, y), 3, (0, 0, 255), -1)
            # Update the points
            registered_colors.append(frame[y,x])
    # Set the mouse callback function for the window
    cv2.setMouseCallback('Register colors', get_coordinates)
    # Display the image and wait for user input
    # Open the video capture
    cap = cv2.VideoCapture(camera_device)
    # Image to store the clicked point
    ret, frame = cap.read()
    clicked_point_image = np.zeros((frame.shape), dtype=np.uint8)

    while True:
        # Read the camera
        ret, frame = cap.read()
        frame = np.where(clicked_point_image==0, frame, clicked_point_image)
        cv2.imshow('Register colors', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # Close all windows
    cap.release()
    cv2.destroyAllWindows()
    return registered_colors

def segment_color_from_camera(registered_colors):
    # Early stop if no color is registered
    if len(registered_colors) == 0:
        return
    # Open the video capture
    cap = cv2.VideoCapture(camera_device)

    # Convert registered color to hsv for segmentation
    registered_colors = cv2.cvtColor(registered_colors[None,:,:], cv2.COLOR_BGR2HSV).reshape(-1,3)
    while(1):
        print_memory_usage()
        # Read the camera and convert to hsv for segmentation
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Get the colored coordinate
        # coordinates = extract_color_pixel(frame, registered_colors=registered_colors, color_distance=64)
        coordinates = extract_hsv_pixel_h(frame, registered_colors=registered_colors, color_distance=16)
        # Create and apply the mask
        mask = np.zeros((frame.shape), dtype=np.uint8)
        mask[coordinates[:,0],coordinates[:,1]]=1
        frame = mask*frame
        # coordinates = extract_non_zero_pixel(image=fgmask)
        # adjacency_matrix = get_adjacency_matrix(coordinates=coordinates)
        # connected_components = get_connected_components(adjacency_matrix=adjacency_matrix)
        # Show the foreground
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame', frame)

        # print(fgmask.shape)
        # Exit when press "esc"
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
    # Clean the environment
    cap.release() 
    cv2.destroyAllWindows() 


def register_gmm_colors(register_number_of_frame: int = 500,
                        varThreshold=16):
    registered_point_coordinates = []
    list_of_color_detector = []
    number_of_frame_registered_for_each_detector = []
    # Create a window to display the image
    cv2.namedWindow('Register GMM colors')
    # Define a function to store the color at mouse clicks position
    def get_ggm_color_subtractor(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:            
            print(f"Registering color at ({x}, {y}), color {frame[y,x]}")
            # Change the image
            cv2.circle(clicked_point_image, (x, y), 3, (0, 0, 255), -1)
            # Update the points
            registered_point_coordinates.append([y,x])
            # Create new cv2.BackgroundSubtractorMOG2
            list_of_color_detector.append(cv2.createBackgroundSubtractorMOG2(varThreshold=varThreshold))
            number_of_frame_registered_for_each_detector.append(0)

    # Set the mouse callback function for the window
    cv2.setMouseCallback('Register GMM colors', get_ggm_color_subtractor)
    # Display the image and wait for user input
    # Open the video capture
    cap = cv2.VideoCapture(camera_device)

    # Image to store the clicked point
    ret, frame = cap.read()
    image_shape = frame.shape
    clicked_point_image = np.zeros(frame.shape, dtype=np.uint8)

    while True:
        # Read the camera
        ret, frame = cap.read()
        # Register the detector
        for i, (coordinate, detector) in enumerate(zip(registered_point_coordinates, list_of_color_detector)):
            if number_of_frame_registered_for_each_detector[i] < register_number_of_frame:
                # Broadcast the current point to image shape
                colors_image = frame[*coordinate]
                colors_image = np.tile(colors_image[None,None,:],[*image_shape[0:2],1])
                # Update the detector
                detector.apply(colors_image, learningRate=-1)
                number_of_frame_registered_for_each_detector[i] +=1
                print(number_of_frame_registered_for_each_detector[i])
        # Identify the clicked points
        # MAKE SURE IT'S AFTER COLOR REGISTERING PLEASE
        frame = np.where(clicked_point_image==0, frame, clicked_point_image)
        cv2.imshow('Register GMM colors', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    for detector in list_of_color_detector:
        final_color = detector.getBackgroundImage()
        cv2.imshow("Final color", final_color)
        while(1):
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release() 
    cv2.destroyAllWindows()
    return list_of_color_detector

def segment_ggm_color(list_of_color_detector):
    # Open the video capture
    cap = cv2.VideoCapture(camera_device)
    while(1):
        print_memory_usage()
        # Read the camera
        ret, frame = cap.read()
        # Applying on each frame and get the mask
        time_point = time.perf_counter()
        fgmask = np.zeros(shape=frame.shape[0:2], dtype=np.uint8)
        for detector in list_of_color_detector:
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


if __name__ == '__main__':
    """There are multiple combination that I use for segmentation:
    register_background() + remove_background(): Identify the back ground with gmm and remove it
    register_colors() + segment_color_from_camera(): A naive approach for color segmentation. Not working well anyway
    register_gmm_colors() + segment_ggm_color(): A gmm approach for color segmentation. Working well
    """
    test_camera()
    
    """Combination 1"""
    # initializing subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=128)
    register_background(background_subtractor=background_subtractor,
                        register_period=2.0)
    remove_background(background_subtractor=background_subtractor)

    """Combination 2"""
    # Register multiple color
    registered_colors = []
    register_colors(registered_colors=registered_colors)
    registered_colors=np.stack(registered_colors, axis=0)
    segment_color_from_camera(registered_colors)

    """Combination 3"""
    # Register multiple color with gmm
    list_of_color_detector = register_gmm_colors(register_number_of_frame=100,
                                                 varThreshold=16)
    segment_ggm_color(list_of_color_detector)



# def register_backgrounds(camera_device_names,
#                         number_of_register_frame=500,
#                         varThreshold=128):
#     # Open the video capture and create the background subtractors
#     caps = []
#     background_subtractors = []
#     for camera_device_name in camera_device_names:
#         caps.append(cv2.VideoCapture(camera_device_name))
#         background_subtractors.append(cv2.createBackgroundSubtractorMOG2(varThreshold=varThreshold))
#     # Create a counter to keep track of number of frames that is registered
#     counter = 0
#     while(counter<number_of_register_frame):
#         counter+=1
#         print(f"Counter: {counter}/{number_of_register_frame}")
#         # Read the camera
#         for cap, background_subtractor, camera_device_name in zip(caps, background_subtractors, camera_device_names):
#             ret, frame = cap.read()
#             # applying to get the subtractor and background image
#             if ret:
#                 fgmask = background_subtractor.apply(frame, learningRate=-1)
#             # Display the frame
#             cv2.imshow(f'Registering background {camera_device_name}', frame)
#         # Press 'esc' to exit the loop
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
#     # Clean the environment
#     for cap in caps:
#         cap.release()
#     cv2.destroyAllWindows()
#     return background_subtractors
