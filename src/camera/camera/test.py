import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('pointcloud_publisher')
        self.publisher = self.create_publisher(PointCloud2, 'pointcloud', 10)
        self.timer = self.create_timer(1.0, self.publish_pointcloud)
        # Create PointCloud2 message
        self.msg = PointCloud2()
        self.msg.header = Header()
        self.msg.header.frame_id = "map"
        self.msg.height = 1
        # Please assure that each x y z is a float32 data
        self.msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        self.msg.is_bigendian = False
        self.msg.point_step = 12
        self.msg.is_dense = True

    def publish_pointcloud(self):
        # Create a sample point cloud (replace this with your actual point cloud data)
        points = np.array([[0,0,0],[1,1,1]]).astype(np.float32)

        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.width = points.shape[0]
        self.msg.row_step = self.msg.point_step * points.shape[0]
        self.msg.data = points.tobytes()

        # Publish the message
        self.publisher.publish(self.msg)
        self.get_logger().info('Publishing pointcloud')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()