#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import os
from ament_index_python.packages import get_package_share_directory

# --- MODIFIED IMPORT ---
# Import the exact same waypoint generation function the DRL environment uses.
# Make sure this path is correct for your package structure.
from my_robot_drl.dense_waypoint import get_dense_lane_waypoints

class GazeboWaypointVisualizer(Node):
    def __init__(self):
        super().__init__('gazebo_waypoint_visualizer')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gazebo /spawn_entity service not available, waiting again...')
        
        self.get_logger().info('Gazebo Waypoint Visualizer Node started and connected to /spawn_entity.')
        
        # This part for loading the box model is correct.
        pkg_share = get_package_share_directory('my_robot_drl')
        self.model_path = os.path.join(pkg_share, 'models', 'waypoint_box', 'model.sdf')
        
        self.model_xml = ""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'r') as f:
                self.model_xml = f.read()
            self.get_logger().info(f'Successfully loaded waypoint model SDF from: {self.model_path}')
        else:
            self.get_logger().error(f"Waypoint model SDF not found at: {self.model_path}")

        # Use a short timer to ensure Gazebo is fully ready before spawning.
        self.one_shot_timer = self.create_timer(2.0, self.generate_and_spawn_waypoints)

    def generate_and_spawn_waypoints(self):
        # Stop the timer so this only runs once.
        self.destroy_timer(self.one_shot_timer)

        if not self.model_xml:
            self.get_logger().error("Cannot spawn waypoints without a model SDF. Shutting down.")
            rclpy.shutdown()
            return

        # --- MODIFIED FUNCTION CALL ---
        # Call the correct function.
        waypoints = get_dense_lane_waypoints()

        if not waypoints:
            self.get_logger().warn('Failed to generate waypoints. Nothing to spawn.')
            return

        self.get_logger().info(f'Generated {len(waypoints)} waypoints. Spawning them in Gazebo as boxes...')

        for i, wp in enumerate(waypoints):
            # --- MODIFIED NAMING ---
            # The new waypoint function returns a simpler dictionary {'x': ..., 'y': ...}.
            # We'll use a simple index for the name.
            waypoint_name = f"viz_wp_{i}"
            self.spawn_waypoint_model(waypoint_name, wp['x'], wp['y'])
        
        self.get_logger().info("Finished sending all spawn requests. Check Gazebo.")
        # We can shutdown after spawning, as the models will remain in Gazebo.
        self.get_logger().info("Shutting down visualizer node.")
        rclpy.shutdown()


    def spawn_waypoint_model(self, name, x, y):
        """Spawns a single waypoint model at the given x, y and a fixed high z."""
        request = SpawnEntity.Request()
        request.name = name
        request.xml = self.model_xml
        
        pose = Pose()
        pose.position.x = float(x) # Ensure they are float
        pose.position.y = float(y)
        # Spawn them slightly in the air so they are clearly visible above the ground.
        pose.position.z = 0.5 
        request.initial_pose = pose
        
        self.get_logger().info(f"Requesting spawn for '{name}' at (x={x:.2f}, y={y:.2f})")
        
        # Send the request but don't wait. Let them spawn asynchronously.
        future = self.spawn_client.call_async(request)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboWaypointVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()