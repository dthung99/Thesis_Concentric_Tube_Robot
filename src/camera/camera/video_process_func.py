import numpy as np
import networkx as nx
import time

def extract_non_zero_pixel_in_black_white_image(image):
    """Input x is a np array (h, w) of uint8
    """
    # Sum over channels and find those >0
    condition = image != 0
    # Extract the coordinate of non-zero pixels
    coordinate = np.stack(np.where(condition), axis=1)
    return coordinate
def extract_non_zero_pixel(image):
    """Input x is a np array (h, w, c) of uint8
    """
    # Sum over channels and find those >0
    condition = image.sum(axis=2)
    condition = condition != 0
    # Extract the coordinate of non-zero pixels
    coordinate = np.stack(np.where(condition), axis=1)
    return coordinate
def get_adjacency_matrix(coordinates, neighbour_square_distance_cut_off = 10):
    """Input coordinate is a np array (n, 2) of int
    neighbour_square_distance_cut_off is the square of euclidean distance cut-off
    above which the points are no longer considered adjacent
    """
    # get the adjacency matrix
    adjacency_matrix=coordinates[:,None,:]-coordinates[None,:,:]
    adjacency_matrix = np.sum(adjacency_matrix**2, axis=-1)
    adjacency_matrix = adjacency_matrix<neighbour_square_distance_cut_off
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix
def get_connected_components(adjacency_matrix):
    """Input adjacency_matrix is a np array (n, n) of 0 and 1
    """
    # Get the connected components from adjacency matrix
    graph = nx.from_numpy_array(adjacency_matrix)
    return nx.connected_components(graph)

def extract_color_pixel(image, registered_colors, color_distance = 10):
    """Input image is a np array (h, w, c) of uint8
    registered_colors is a np array (n, c) of uint8
    color_square_distance is the square of euclidean distance between two colors
    above which the points are no longer considered the same color
    """
    # Find pixel wise difference between image colors and registered colors
    distance_image = image[:,:,None,:] - registered_colors[None,None,:,:]
    distance_image = (np.abs(distance_image)) < color_distance
    # Find those nearest to registered color
    condition = distance_image.all(axis=-1)
    condition = condition.any(axis=-1)
    # Extract the coordinate of non-zero pixels
    coordinate = np.stack(np.where(condition), axis=1)
    return coordinate

def extract_hsv_pixel_h(image, registered_colors, color_distance = 10):
    """Input image is a np array (h, w, c) in hsv
    registered_colors is a np array (n, c) in hsv
    color_square_distance is the square of euclidean distance between two colors
    above which the points are no longer considered the same color
    """
    # Find pixel wise difference between image colors and registered colors
    distance_image = image[:,:,None,0] - registered_colors[None,None,:,0]
    distance_image = (np.abs(distance_image)) < color_distance
    # Find those nearest to registered color
    condition = distance_image.any(axis=-1)
    # Extract the coordinate of non-zero pixels
    coordinate = np.stack(np.where(condition), axis=1)
    return coordinate

def find_points_nearest_to_lines_and_return_one_on_those_lines(points, lines, square_of_cut_off_for_near, std_cut_off):
    """
    points: list of points of shape (p,2). THE POINTS MUST BE (x,y)
    lines: list of lines of shape (n,3) (ax+by+c=0)
    distance: (n, p): is calculated as |ax0 + by0 + c| / sqrt(a^2 + b^2)
    return one point on the line for each line (n, p', 2)
    INPUT OUTPUT are NP ARRAY
    """
    num_point = len(points)
    num_line = len(lines)
    points=points[None,:,:] # (1, p, 2)
    points=np.tile(points,(num_line,1,1))
    lines=lines[:,None,:] # (n, 1, 3)
    # lines=np.tile(lines,(1,num_point,1)) ## I didn't tile lines cause it's not necessary 
    a=lines[:,:,0]
    b=lines[:,:,1]
    c=lines[:,:,2]
    x=points[:,:,0]
    y=points[:,:,1]
    # Get the distances
    distance=(a*x+b*y+c)**2/(a**2+b**2)
    # Extract only the nearest x and get the median x
    result_x = np.ma.masked_where(distance>square_of_cut_off_for_near, x)
    # Mask where std is greater than a cut off
    std_mask = np.ma.std(result_x, axis=1)>std_cut_off
    std_mask=np.ma.masked_where(std_mask,std_mask)
    std_mask=~std_mask.mask
    # Get the median of x value
    result_x = np.ma.median(result_x, axis=1)[:,None]
    # Get one point on each line
    result_y = (-c - a*result_x)/b
    mask = ~result_x.mask*std_mask.reshape((-1,1))
    result = np.concatenate((result_x.data, result_y.data), axis=-1)*mask
    return result, mask.reshape((-1))

if __name__ == '__main__':
    # Unit test for each function
    test_array = np.array([[[2,4,1], [1,2,3], [0,0,0]],[[3,2,1], [0,0,0], [1,2,3]],[[0,0,0], [3,0,0], [2,3,4]]])
    coordinates = extract_non_zero_pixel(test_array)
    adjacency_matrix =  get_adjacency_matrix(coordinates=coordinates, neighbour_square_distance_cut_off=1.3)
    connected_components=get_connected_components(adjacency_matrix)
    # Start testing
    target = np.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]])
    assert (coordinates == target).all(), "extract_non_zero_pixel not working"

    target = np.array([[0, 1, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 1, 0]])
    assert (adjacency_matrix == target).all(), "get_adjacency_matrix not working"
    target = [{0,1,2},{3,4,5}]
    
    assert all([components==target_i for components, target_i in zip(connected_components, target)]), "get_connected_components not working"

    registered_colors = np.array([[1,2,3],[1,3,4],[2,0,0]])
    coordinates = extract_color_pixel(test_array, registered_colors=registered_colors, color_distance=1.3)
    target = np.array([[0, 1], [1, 2], [2, 1], [2, 2]])    
    assert (coordinates==target).all(), "extract_color_pixel not working"

    coordinates = extract_hsv_pixel_h(test_array, registered_colors=registered_colors, color_distance=0.5)
    target = np.array([[0, 0], [0, 1], [1, 2], [2, 2]])
    assert (coordinates==target).all(), "extract_hsv_pixel_h not working"

    """Unit tests for funtions that are added later"""
    # Additional unit test extract_non_zero_pixel_in_black_white_image
    test_array_black_white = np.array([[2, 3, 0],[2, 0, 0], [1, 2, 3]])
    target_black_white = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [2, 2]])
    coordinates_black_white = extract_non_zero_pixel_in_black_white_image(test_array_black_white)
    assert (coordinates_black_white == target_black_white).all(), "extract_non_zero_pixel not working"

    points = np.array([[1, 1], [-12, 9]])
    lines = np.array([[0, 1, -9], [1, 0, 100], [1, -1, 0]])
    result, mask = find_points_nearest_to_lines_and_return_one_on_those_lines(points=points, lines=lines, square_of_cut_off_for_near=100, std_cut_off=5)
    assert (result == np.array([[0, 0], [0, 0], [1, 1]])).all(), "find_points_nearest_to_lines_and_return_one_on_those_lines error"
    assert (mask == np.array([False, False, True])).all(), "find_points_nearest_to_lines_and_return_one_on_those_lines error"

    print("All test passed")

    