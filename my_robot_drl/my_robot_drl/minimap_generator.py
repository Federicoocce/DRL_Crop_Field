import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import numpy as np
from sensor_msgs_py import point_cloud2

# You will need to install open3d: pip install open3d
import open3d as o3d

class MinimapGenerator(Node):
    def __init__(self):
        super().__init__('minimap_generator')
        
        # Declare and get the radius for the minimap
        self.declare_parameter('minimap_radius', 30.0) # Radius in meters
        self.MINIMAP_RADIUS = self.get_parameter('minimap_radius').get_parameter_value().double_value
        self.get_logger().info(f"Minimap radius set to {self.MINIMAP_RADIUS} meters.")

        # Subscriber to the full point cloud map from FAST-LIO
        self.full_map_sub = self.create_subscription(
            PointCloud2,
            '/cloud_registered',  # Default topic for the registered map from FAST-LIO
            self.full_map_callback,
            rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        )
        
        # Subscriber to the odometry from FAST-LIO
        self.odom_sub = self.create_subscription(
            Odometry,
            '/Odometry', # Default topic for odometry from FAST-LIO
            self.odom_callback,
            10)

        # Publisher for the local minimap
        self.minimap_pub = self.create_publisher(PointCloud2, '/minimap', 10)

        self.full_map_o3d = o3d.geometry.PointCloud()
        self.current_position = None
        self.map_updated = False

    def odom_callback(self, msg):
        """Stores the robot's current position."""
        self.current_position = msg.pose.pose.position

    def full_map_callback(self, msg):
        """Converts the incoming full map to an Open3D point cloud and triggers processing."""
        # Convert ROS PointCloud2 to a NumPy array
        points = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'))
        
        # Update the Open3D point cloud object
        self.full_map_o3d.points = o3d.utility.Vector3dVector(points)
        self.map_updated = True
        
        # Process and publish the minimap immediately upon receiving a new map
        if self.current_position is not None:
            self.generate_and_publish_minimap()

    def generate_and_publish_minimap(self):
        """Crops the full map around the robot's position and publishes it."""
        if not self.map_updated or self.current_position is None:
            return

        robot_pos = np.array([self.current_position.x, self.current_position.y, self.current_position.z])
        
        # Define a bounding box centered on the robot
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=robot_pos - self.MINIMAP_RADIUS,
            max_bound=robot_pos + self.MINIMAP_RADIUS
        )
        
        # Crop the full point cloud
        minimap_o3d = self.full_map_o3d.crop(bounding_box)
        
        if not minimap_o3d.has_points():
            return # Don't publish if there are no points in the vicinity
            
        # Convert the cropped Open3D point cloud back to a NumPy array
        points_np = np.asarray(minimap_o3d.points)
        
        # Create the ROS PointCloud2 message header
        header = self.get_clock().now().to_msg()
        # The map and odometry from FAST-LIO are in the 'camera_init' frame
        header.frame_id = 'camera_init' 

        # Create the PointCloud2 message
        minimap_msg = point_cloud2.create_cloud_xyz32(header, points_np)
        
        self.minimap_pub.publish(minimap_msg)
        self.map_updated = False # Reset flag after processing

def main(args=None):
    rclpy.init(args=args)
    node = MinimapGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()