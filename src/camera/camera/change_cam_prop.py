import cv2
import time

# CAP_PROP_AUTOFOCUS: 1.0
# CAP_PROP_AUTO_EXPOSURE: 3.0
# CAP_PROP_AUTO_WB: 1.0
# CAP_PROP_BACKEND: 200.0
# CAP_PROP_BACKLIGHT: 0.0
# CAP_PROP_BRIGHTNESS: 0.0
# CAP_PROP_BUFFERSIZE: 4.0
# CAP_PROP_CONTRAST: 50.0
# CAP_PROP_CONVERT_RGB: 1.0
# CAP_PROP_EXPOSURE: 83.0
# CAP_PROP_FOCUS: 68.0
# CAP_PROP_FORMAT: 16.0
# CAP_PROP_FOURCC: 1448695129.0
# CAP_PROP_FPS: 30.0
# CAP_PROP_FRAME_HEIGHT: 480.0
# CAP_PROP_FRAME_WIDTH: 640.0
# CAP_PROP_GAMMA: 300.0
# CAP_PROP_HUE: 0.0
# CAP_PROP_MODE: 0.0
# CAP_PROP_ORIENTATION_AUTO: 1.0
# CAP_PROP_PAN: 0.0
# CAP_PROP_POS_MSEC: 0.0
# CAP_PROP_SATURATION: 70.0
# CAP_PROP_SHARPNESS: 60.0
# CAP_PROP_TEMPERATURE: 4600.0
# CAP_PROP_TILT: 0.0
# CAP_PROP_WB_TEMPERATURE: 4600.0
# CAP_PROP_ZOOM: 0.0


# focus 68.0
# exposure 83.0
# white balance 4600.0


# # Get all attributes of cv2 that start with "CAP_PROP_"
# prop_list = [attr for attr in dir(cv2) if attr.startswith('CAP_PROP_')]
# attribute_code = [getattr(cv2, prop) for prop in prop_list]
# attribute_code.sort()
# cap = cv2.VideoCapture("/dev/video2")
# for prop in prop_list:
#     value = cap.get(getattr(cv2, prop))
#     if value!=-1:
#         print(f"{prop}: {value}")

cap = cv2.VideoCapture("/dev/video0")
result = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
# result = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
# print(result)

result = cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
# result = cap.get(cv2.CAP_PROP_AUTO_WB)
# print(result)

# cap.set(cv2.CAP_PROP_FOCUS,-1)
# cap.set(cv2.CAP_PROP_EXPOSURE,-1)
# cap.set(cv2.CAP_PROP_WB_TEMPERATURE,-1)


# print(f"focus {cap.get(cv2.CAP_PROP_FOCUS)}")
# print(f"exposure {cap.get(cv2.CAP_PROP_EXPOSURE)}")
# print(f"white balance {cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}")

# # Set the desired frame size
# desired_width = 400
# desired_height = 300

# # Set the frame size
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# # Get the actual frame size
# actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print(f"Desired frame size: {desired_width}x{desired_height}")
# print(f"Actual frame size: {actual_width}x{actual_height}")

# Read and display the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    # Display the frame
    cv2.imshow("Video", frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print(f"focus {cap.get(cv2.CAP_PROP_FOCUS)}")
    # print(f"exposure {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    # print(f"white balance {cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}")
# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()












        