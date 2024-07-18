import numpy as np
import networkx as nx
import time

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
    np.fill_diagonal(adjacency_matrix, 0.0)
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

    print("All test passed")

    