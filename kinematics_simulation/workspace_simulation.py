import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import homogeneous_matrix_handler as mh
import time


"""Unit of the plots here are mm"""
list_of_object = dict()
# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
wire_3_plot, = ax.plot([], [], [], color='r', linewidth=2)
wire_2_plot, = ax.plot([], [], [], color='g', linewidth=2)
wire_1_plot, = ax.plot([], [], [], color='b', linewidth=2)
curve_portions = torch.linspace(0, 1, 100)
update_period = 0.1 #Time period to update plot
discretize_number = 100 #How much you discretize the environment for taskspace analysis


# Function definitions
def My_Function():
    #region Prepare environment
    # Plot the axis of the plot
    plot_axis([0,0,0], 50)
    # Plot a sphere
    x2, y2, z2 = np.random.uniform(-2, 2, size=(3, 50))
    r = 0.5
    # ax.scatter(x2, y2, z2, s=200, color='r', marker='o', alpha=0.5)
    # Set axis labels and limits and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-50, 50])
    #endregion
    
    for i in range(discretize_number):
        r3 = 20/np.random.uniform(0.2, 1)
        r1 = 20/np.random.uniform(0.2, 1)
        l3 = np.random.uniform(0, 10)
        l1 = np.random.uniform(0, 10)
        a3 = np.random.uniform(-np.pi/2, np.pi/2)
        a1 = np.random.uniform(-np.pi/2, np.pi/2)

        update_plot_with_new_wire_configurations(wire_3_radius = r3,
                                                 wire_2_radius = 10,
                                                 wire_1_radius = r1,
                                                 wire_3_rotate_angle = a3,
                                                 wire_2_rotate_angle = torch.pi*0,
                                                 wire_1_rotate_angle = a1,
                                                 arc_length_3 = 20,
                                                 arc_length_2 = 0 ,
                                                 arc_length_1 = 20,
                                                 straight_start_length_3 = l3,
                                                 straight_start_length_2 = 0,
                                                 straight_start_length_1 = l1,)

    # update_plot_with_new_wire_configurations(wire_3_radius = 10,
    #                                             wire_2_radius = 20,
    #                                             wire_1_radius = 20,
    #                                             wire_3_rotate_angle = torch.pi*0,
    #                                             wire_2_rotate_angle = torch.pi*0,
    #                                             wire_1_rotate_angle = torch.pi*1,
    #                                             arc_length_3 = 0,
    #                                             arc_length_2 = 10,
    #                                             arc_length_1 = 10,
    #                                             straight_start_length_3 = 80,
    #                                             straight_start_length_2 = 0,
    #                                             straight_start_length_1 = 0,)

    # Timer
    def simplier_robot(link_1_rotate_angle, link_1_straight_start_length, link_1_radius, link_3_radius,
                       fixed_link_1_arc_length=20, fixed_link_3_arc_length=20, fixed_link_3_rotate_angle=torch.pi*0):
        """
        link_1_rotate_angle: from -pi to pi
        link_1_straight_start_length: from -inf to +inf
        link_1_radius: from 20 to 100
        link_3_radius: from 20 to 100
        fixed_link_1_arc_length: fixed, used for optimization
        fixed_link_3_arc_length: fixed, used for optimization
        fixed_link_3_rotate_angle: from -pi to pi fixed, used for optimization       
        """
        results = get_end_effector_matrix(link_1_radius = link_1_radius, #TODO Variable
                                          link_2_radius = 10, #We only simulate 2 wires, and we set its length to 0
                                          link_3_radius = link_3_radius, #TODO Variable
                                          link_1_arc_length = fixed_link_1_arc_length, #Fixed length of link 1
                                          link_2_arc_length_2 = 0, #Zero cause we only simulate 2 wires
                                          link_3_arc_length = fixed_link_3_arc_length, #Fixed length of link 3
                                          link_1_straight_start_length = link_1_straight_start_length, #TODO Variable for translation 
                                          link_2_straight_start_length = 0, #Zero cause we only simulate curve wires
                                          link_3_straight_start_length = 0, #Zero cause we only simulate curve wires
                                          link_1_rotate_angle = link_1_rotate_angle, #TODO Variable
                                          link_2_rotate_angle = torch.pi*0, #We only simulate 2 wires, and we set its length to 0
                                          link_3_rotate_angle = fixed_link_3_rotate_angle, #Fixed angle
                                          )
        return results

    # ax.view_init(elev=0, azim=-90)
    # plt.show()

    #region test gradient
    #endregion

    return 0

def update_plot_with_new_wire_configurations(wire_3_radius = 10,
                                             wire_2_radius = 10,
                                             wire_1_radius = 10,
                                             wire_3_rotate_angle = torch.pi*0,
                                             wire_2_rotate_angle = torch.pi*0,
                                             wire_1_rotate_angle = torch.pi*0,
                                             arc_length_3 = 20,
                                             arc_length_2 = 0 ,
                                             arc_length_1 = 20,
                                             straight_start_length_3 = 0,
                                             straight_start_length_2 = 0,
                                             straight_start_length_1 = 0,):
    # Declare the wire
    wire_3 = Curve_Wire(arc_length = arc_length_3,
                        radius = wire_3_radius,
                        straight_start_length = straight_start_length_3,
                        name = "wire_3")
    wire_2 = Curve_Wire(arc_length = arc_length_2,
                        radius = wire_2_radius,
                        straight_start_length = straight_start_length_2,
                        name = "wire_2")
    wire_1 = Curve_Wire(arc_length = arc_length_1,
                        radius = wire_1_radius,
                        straight_start_length = straight_start_length_1,
                        name = "wire_1")
    # Plotting
    current_matrix = mh.identity()
    # Wire 3
    # Get the points on the wires
    current_matrix = current_matrix@mh.rotate(wire_3_rotate_angle, [0,0,1])
    plot_points_3 = [(current_matrix@wire_3.get_a_point_on_curve_matrix(portion))[0:3,3] for portion in curve_portions]
    plot_points_3 = torch.stack(plot_points_3)
    current_matrix = current_matrix@wire_3.curve_end_point_matrix
    # Plot the points on the wires
    wire_3_plot.set_data_3d(plot_points_3[:,0], plot_points_3[:,1], plot_points_3[:,2])

    # Wire 2
    # Get the points on the wires
    current_matrix = current_matrix@mh.rotate(wire_2_rotate_angle, [0,0,1])
    plot_points_2 = [(current_matrix@wire_2.get_a_point_on_curve_matrix(portion))[0:3,3] for portion in curve_portions]
    plot_points_2 = torch.stack(plot_points_2)
    current_matrix = current_matrix@wire_2.curve_end_point_matrix
    # Plot the points on the wires
    wire_2_plot.set_data_3d(plot_points_2[:,0], plot_points_2[:,1], plot_points_2[:,2])

    # Wire 1
    # Get the points on the wires
    current_matrix = current_matrix@mh.rotate(wire_1_rotate_angle, [0,0,1])
    plot_points_1 = [(current_matrix@wire_1.get_a_point_on_curve_matrix(portion))[0:3,3] for portion in curve_portions]
    plot_points_1 = torch.stack(plot_points_1)
    current_matrix = current_matrix@wire_1.curve_end_point_matrix
    # Plot the points on the wires
    wire_1_plot.set_data_3d(plot_points_1[:,0], plot_points_1[:,1], plot_points_1[:,2])

    ax.scatter(current_matrix[0,3], current_matrix[1,3], current_matrix[2,3], s=2, color='purple', marker='o', alpha=0.5)
    list_of_object = dict()
    plt.pause(update_period)

def get_end_effector_matrix(link_1_radius = 10,
                            link_2_radius = 10,
                            link_3_radius = 10,
                            link_1_arc_length = 20,
                            link_2_arc_length_2 = 0 ,
                            link_3_arc_length = 20,
                            link_1_straight_start_length = 0,
                            link_2_straight_start_length = 0,
                            link_3_straight_start_length = 0,
                            link_1_rotate_angle = torch.pi*0,
                            link_2_rotate_angle = torch.pi*0,
                            link_3_rotate_angle = torch.pi*0,):
    """Get the homogeneous matrix of the end effector
    The commented code are all for plotting for validation"""
    # Declare the wire
    wire_3 = Curve_Wire(arc_length = link_1_arc_length,
                        radius = link_1_radius,
                        straight_start_length = link_1_straight_start_length,
                        name = "wire_3")
    wire_2 = Curve_Wire(arc_length = link_2_arc_length_2,
                        radius = link_2_radius,
                        straight_start_length = link_2_straight_start_length,
                        name = "wire_2")
    wire_1 = Curve_Wire(arc_length = link_3_arc_length,
                        radius = link_3_radius,
                        straight_start_length = link_3_straight_start_length,
                        name = "wire_1")
    # Plotting
    current_matrix = mh.identity()
    # Wire 3
    # Get the points on the wires
    current_matrix = current_matrix@mh.rotate(link_1_rotate_angle, [0,0,1])
    # plot_points_3 = [(current_matrix@wire_3.get_a_point_on_curve_matrix(portion))[0:3,3] for portion in curve_portions]
    # plot_points_3 = torch.stack(plot_points_3)
    current_matrix = current_matrix@wire_3.curve_end_point_matrix
    # Plot the points on the wires
    # wire_3_plot.set_data_3d(plot_points_3[:,0], plot_points_3[:,1], plot_points_3[:,2])

    # Wire 2
    # Get the points on the wires
    current_matrix = current_matrix@mh.rotate(link_2_rotate_angle, [0,0,1])
    # plot_points_2 = [(current_matrix@wire_2.get_a_point_on_curve_matrix(portion))[0:3,3] for portion in curve_portions]
    # plot_points_2 = torch.stack(plot_points_2)
    current_matrix = current_matrix@wire_2.curve_end_point_matrix
    # Plot the points on the wires
    # wire_2_plot.set_data_3d(plot_points_2[:,0], plot_points_2[:,1], plot_points_2[:,2])

    # Wire 1
    # Get the points on the wires
    current_matrix = current_matrix@mh.rotate(link_3_rotate_angle, [0,0,1])
    # plot_points_1 = [(current_matrix@wire_1.get_a_point_on_curve_matrix(portion))[0:3,3] for portion in curve_portions]
    # plot_points_1 = torch.stack(plot_points_1)
    current_matrix = current_matrix@wire_1.curve_end_point_matrix
    # Plot the points on the wires
    # wire_1_plot.set_data_3d(plot_points_1[:,0], plot_points_1[:,1], plot_points_1[:,2])

    # ax.scatter(current_matrix[0,3], current_matrix[1,3], current_matrix[2,3], s=2, color='purple', marker='o', alpha=0.5)
    list_of_object = dict()
    # plt.pause(update_period)

    return current_matrix




def plot_axis(origin = [0, 0, 0], vector_scale = 50):
    """Add coordinate vector"""
    Ox = [1, 0, 0]; Oy = [0, 1, 0]; Oz = [0, 0, 1]
    ax.quiver(origin, origin, origin, Ox, Oy, Oz, color='k', length=vector_scale, normalize=True)


# Class to create a curve wire
class Curve_Wire():
    """The kinematics of the curve wires"""
    def __init__(self, arc_length, radius, straight_start_length = 0, name = "default_name") -> None:
        """
        curve_length: the length of the curve
        radius: the radius of the curve
        straight_start_length: length of the straight part at the beginning before curving
        name: name of the part, it should be unique
        The curve lie on Oxz plane, curve in a positive direction around Oy
        """
        if not name in list_of_object:
            list_of_object[name] = self
        else:
            warnings.warn(f"Dupplicate object {self.__class__.__name__}")

        assert radius > 0, "Curve_Wire only accept positive radius for radius"

        self.arc_length = arc_length
        self.radius = radius
        self.straight_start_length = straight_start_length
        self.get_curve_end_point_matrix()

    def get_origin_matrix(self):
        """The origin of the wire"""
        self.origin_matrix = mh.identity()
        return self.origin_matrix
    def get_curve_start_point_matrix(self):
        """The start point of the curve part of the wire"""       
        self.curve_start_point_matrix = mh.translate(0, 0, self.straight_start_length)@self.get_origin_matrix()
        return self.curve_start_point_matrix
    def get_curve_center_matrix(self):
        """The center of the curve part of the wire"""
        self.curve_center_matrix = mh.translate(self.radius, 0, 0)@self.get_curve_start_point_matrix()
        return self.curve_center_matrix
    def get_curve_angle(self):
        """The radius of curve part of the wire"""
        self.curve_angle = mh.num_to_tensor(self.arc_length)/self.radius
        return self.curve_angle
    def get_curve_end_point_matrix(self):
        """The end point/tip of curve part of the wire"""
        # Rotate and traslate in the body frame, not original frame (world frame)
        self.curve_end_point_matrix = self.get_curve_center_matrix()@mh.rotate(self.get_curve_angle(), [0,1,0])
        self.curve_end_point_matrix = self.curve_end_point_matrix@mh.translate(-self.radius, 0, 0)
        return self.curve_end_point_matrix
    def get_a_point_on_curve_matrix(self, curve_fraction = 0.5):
        """Get a point on the curve at the curve_fraction portion of whole curve"""
        if self.arc_length + self.straight_start_length == 0:
            return self.origin_matrix
        straight_fraction = self.straight_start_length/(self.arc_length + self.straight_start_length)
        if straight_fraction == 1:
            length_fraction = curve_fraction/straight_fraction
            result = self.origin_matrix*(1-length_fraction) + self.curve_start_point_matrix*length_fraction
            return result
        if curve_fraction >= straight_fraction:
            angle = self.curve_angle*(curve_fraction-straight_fraction)/(1-straight_fraction)
            result = self.curve_center_matrix@mh.rotate(angle, [0,1,0])@mh.translate(-self.radius, 0, 0)        
        else:
            length_fraction = curve_fraction/straight_fraction
            result = self.origin_matrix*(1-length_fraction) + self.curve_start_point_matrix*length_fraction
        return result

# Test functions
class My_Function_Test():
    """Use for testing function"""        
    def run_test(self):
        """Loop through all test function and run it"""
        for member in dir(self):
            if member.startswith('test'):
                self.control_test_to_show(self, member)
                
    def control_test_to_show(self, obj, member):
        """Customize what to print when testing"""
        getattr(obj, member)()
        print(f"{member}: Pass")

    def test_homogeneous_matrix_handler(self):
        # Test create identity tensor
        condition = (mh.identity() == torch.eye(4)).all()
        assert condition, "homogeneous_matrix_handler create Identity matrix have error!!!"
        # Test create translation tensor
        condition = (mh.translate(1,2,3) == torch.tensor([[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]])).all()
        assert condition, "homogeneous_matrix_handler create translation matrix have error in equation!!!"
        condition = mh.translate(1,torch.tensor(2.0, requires_grad=True),3).requires_grad == True
        assert condition, "homogeneous_matrix_handler create translation matrix, the result do not keep track of gradient!!!"
        # Test create rotation matrix
        condition_1 = (mh.rotate(torch.pi/2, [0, 0, 3]) == mh.rotate(torch.pi/2, [0, 0, 1])).all()
        angle = torch.tensor(torch.pi/4)
        condition_2 = (mh.rotate(torch.pi/4, [0, 0, 1]) == torch.tensor([[torch.cos(angle),-torch.sin(angle),0, 0],
                                                                        [torch.sin(angle),torch.cos(angle),0, 0],
                                                                        [0,0,1, 0],
                                                                        [0, 0, 0, 1]])).all()
        condition_3 = (mh.rotate(torch.pi/2, [0, 1, 0]) == torch.tensor([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])).all()
        condition_4 = (mh.rotate(torch.pi/2, [1, 0, 0]) == torch.tensor([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])).all()
        condition = condition_1 and condition_2 and condition_3 and condition_4
        assert condition, "homogeneous_matrix_handler create rotation matrix have error in equation!!!"
        condition_1 = mh.rotate(torch.tensor(2.0, requires_grad=True), [1, 2, 3]).requires_grad == True
        condition_2 = mh.rotate(2, torch.tensor([1.0, 2, 3], requires_grad=True)).requires_grad == True
        condition = condition_1 and condition_2
        assert condition, "homogeneous_matrix_handler create rotation matrix, the result do not keep track of gradient!!!"

    def test_curve_wire(self):
        """Test the Curve_Wire class function."""
        test = Curve_Wire(arc_length = torch.pi/2,
                          radius = 1,
                          straight_start_length = 4,
                          name = "test_curve_wire_1")
        condition_1 = (test.origin_matrix == torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])).all()
        condition_2 = (test.curve_start_point_matrix == torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,4],[0,0,0,1]])).all()
        condition_3 = (test.curve_center_matrix == torch.tensor([[1,0,0,1],[0,1,0,0],[0,0,1,4],[0,0,0,1]])).all()
        condition_4 = (test.curve_angle == torch.pi/2)
        condition_5 = (test.curve_end_point_matrix == torch.tensor([[0,0,1,1],[0,1,0,0],[-1,0,0,5],[0,0,0,1]])).all()
        condition = condition_1 and condition_2 and condition_3 and condition_4 and condition_5
        assert condition, "Curve_Wire calculation is wrong!!!"
        test = Curve_Wire(arc_length = torch.pi/2,
                          radius = 1,
                          straight_start_length = torch.tensor(2.0, requires_grad=True),
                          name = "test_curve_wire_2")
        condition_1 = test.curve_end_point_matrix.requires_grad == True
        test = Curve_Wire(arc_length = torch.tensor(2.0, requires_grad=True),
                          radius = 1,
                          straight_start_length = 2,
                          name = "test_curve_wire_3")
        condition_2 = test.curve_end_point_matrix.requires_grad == True
        test = Curve_Wire(arc_length = torch.pi/2,
                          radius = torch.tensor(2.0, requires_grad=True),
                          straight_start_length = 2,
                          name = "test_curve_wire_4")
        condition_3 = test.curve_end_point_matrix.requires_grad == True
        condition = condition_1 and condition_2 and condition_3
        assert condition, "Curve_Wire do not keep track of gradient!!!"
        test = Curve_Wire(arc_length = torch.pi,
                          radius = 1,
                          straight_start_length = torch.pi,
                          name = "test_curve_wire_5")
        condition_1 = (test.get_a_point_on_curve_matrix(1/4) - torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,torch.pi/2],[0,0,0,1]])).norm() < 0.001
        condition_2 = (test.get_a_point_on_curve_matrix(3/4) - torch.tensor([[0,0,1,1],[0,1,0,0],[-1,0,0,torch.pi+1],[0,0,0,1]])).norm() < 0.001
        condition = condition_1 and condition_2
        assert condition, "Curve_Wire get a point on the curve is wrong!!!"

    def test_1(self):
        """Test the my_function() function."""
        # print(test.curve_start_point_matrix)
        # print(test.curve_center_matrix)
        return

# Main entry point
if __name__ == "__main__":
    """Main execution"""
    # Call test functions
    My_Function_Test().run_test()
    assert My_Function() == 0, "Wrong!!!"
