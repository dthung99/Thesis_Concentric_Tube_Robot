import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('image/myleft.jpg')  #queryimage # left image
img2 = cv2.imread('image/myright.jpg') #trainimage # right image

# Get the matched points on the two images
def detect_checker_board(input_image, checker_board_size = [10, 7], square_size = 20, show = True):
    """Detect a checkerboard and return the pixels corresponding to the corners
    checker_board_size: number of corners of checker board. Should be int
    square_size: length (mm) of the checkerboard square

    """
    # Deep copy to avoid changing orginal image
    input_image = input_image.copy()
    print(f"Start detect_checker_board")
    # Some parameter for tuning
    expected_image_size = np.array(input_image.shape)
    corner_refinement_window = max(int(expected_image_size.max()/50), 11) #The window size for subpixel refinement
    print(f"Image size: {expected_image_size}")
    print(f"corner_refinement_window: {corner_refinement_window}")
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # Number of u and v cor1ner
    number_of_u_corners = int(checker_board_size[0])
    number_of_v_corners = int(checker_board_size[1])
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((number_of_v_corners*number_of_u_corners,3), np.float32)
    objp[:,:2] = np.mgrid[0:number_of_u_corners,0:number_of_v_corners].T.reshape(-1,2) #Someone else code. It just create a grid of (-1,3) coordinate
    objp = objp*square_size
    # Convert the image to gray scale
    if(input_image.shape[-1] == 3):
        input_image_in_gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
    else:
        input_image_in_gray = input_image
    # Find the chess board corners
    find_corner_result, detected_corners = cv2.findChessboardCorners(input_image_in_gray,
                                                            (number_of_u_corners, number_of_v_corners),
                                                            flags=None)
    # If found, add object points, image points (after refining them)
    if find_corner_result == True:
        # Refine the points to get better ones
        # This code modify the original corners too!!!
        detected_corners = cv2.cornerSubPix(input_image_in_gray,
                                   detected_corners,
                                   (corner_refinement_window, corner_refinement_window),
                                   (-1,-1),
                                   criteria)
        # Draw and display the corners
        input_image = cv2.drawChessboardCorners(input_image,
                                                (number_of_u_corners,number_of_v_corners),
                                                detected_corners,
                                                find_corner_result)
        resized_image = cv2.resize(input_image, (1000, 750))  
        if show:
            cv2.imshow('img',resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"End detect_checker_board")
        return find_corner_result, detected_corners, objp
    return False, False, False

show=False
# Detect the corners from two image and find the fundamental matrix
find_corner_result, detected_corners_1, world_points_1 = detect_checker_board(input_image = img1,
                                                                              checker_board_size = [10, 7],
                                                                              square_size = 20,
                                                                              show=show
                                                                              )
assert find_corner_result, "Could not find corners in image"
find_corner_result, detected_corners_2, world_points_2 = detect_checker_board(input_image = img2,
                                                                              checker_board_size = [10, 7],
                                                                              square_size = 20,
                                                                              show=show
                                                                              )
assert find_corner_result, "Could not find corners in image"

# Variables for camera calibration
objpoints = [world_points_1, world_points_2]
imgpoints = [detected_corners_1, detected_corners_2]
# Condition variable to check if camera internal matrix exist
have_internal_matrix = False

if have_internal_matrix:
    # TODO later:
    # Load camera matrix
    # Get image points for world points
    pass
else:
    assert img1.shape == img2.shape, "For internal calibration, image size should be the same"
    image_shape = img1.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[::-1], None, None)

# Get projection matrices for each view
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


projection_matrix_1 = compute_projection_matrix(mtx, rvec=rvecs[0], tvec=tvecs[0])
projection_matrix_2 = compute_projection_matrix(mtx, rvec=rvecs[1], tvec=tvecs[1])

# The detected_corners have shape (n,1,2), need to change it to (2,n)
detected_corners_1 = detected_corners_1.squeeze().T
detected_corners_2 = detected_corners_2.squeeze().T
point_in_3D = cv2.triangulatePoints(projection_matrix_1,
                                    projection_matrix_2,
                                    detected_corners_1,
                                    detected_corners_2)

def get_3D_point_from_homogeneous_coordinate(x):
    x = x/x[-1]
    return x[:3]

results = np.apply_along_axis(get_3D_point_from_homogeneous_coordinate, axis=0, arr=point_in_3D)
print(results.T)





















# # find the fundamental matrix
# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]

# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     height, width = img1.shape[0:2]
#     if img1.shape[-1] == 1:
#         img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     if img2.shape[-1] == 1:
#         img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [width, -(r[2]+r[0]*width)/r[1]])
#         img1 = cv2.line(img1, (x0,y0), (x1,y1), color,3)
#         img1 = cv2.circle(img1,tuple(pt1),10,color,-1)
#         img2 = cv2.circle(img2,tuple(pt2),10,color,-1)
#     return img1,img2

# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2),
#                                        whichImage=2,
#                                        F = F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),
#                                        whichImage=1,
#                                        F=F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

# if show:
#     plt.subplot(121),plt.imshow(img5)
#     plt.subplot(122),plt.imshow(img3)
#     plt.show()
