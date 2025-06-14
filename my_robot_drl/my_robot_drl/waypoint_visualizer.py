#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import os
from ament_index_python.packages import get_package_share_directory

# Import your waypoint generation function
from my_robot_drl.get_field_data import get_precise_row_waypoints

class GazeboWaypointVisualizer(Node):
    def __init__(self):
        super().__init__('gazebo_waypoint_visualizer')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gazebo /spawn_entity service not available, waiting again...')
        
        self.get_logger().info('Gazebo Waypoint Visualizer Node started and connected to /spawn_entity.')
        
        pkg_share = get_package_share_directory('my_robot_drl')
        # --- MODIFIED PATH TO USE THE BOX MODEL ---
        self.model_path = os.path.join(pkg_share, 'models', 'waypoint_box', 'model.sdf')
        
        self.model_xml = ""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'r') as f:
                self.model_xml = f.read()
            self.get_logger().info(f'Successfully loaded waypoint model SDF from: {self.model_path}')
        else:
            self.get_logger().error(f"Waypoint model SDF not found at: {self.model_path}")

        self.one_shot_timer = self.create_timer(2.0, self.generate_and_spawn_waypoints)

    def generate_and_spawn_waypoints(self):
        self.destroy_timer(self.one_shot_timer)

        if not self.model_xml:
            self.get_logger().error("Cannot spawn waypoints without a model SDF. Shutting down.")
            rclpy.shutdown()
            return

        waypoints = get_precise_row_waypoints()

        if not waypoints:
            self.get_logger().warn('Failed to generate waypoints. Nothing to spawn.')
            return

        self.get_logger().info(f'Generated {len(waypoints)} waypoints. Spawning them in Gazebo...')

        for i, wp in enumerate(waypoints):
            # Using a more descriptive name for the waypoint entities
            waypoint_name = f"viz_wp_{wp['type'].replace('lane_','l').replace('_start','s').replace('_mid','m').replace('_end','e')}_{i}"
            self.spawn_waypoint_model(waypoint_name, wp['x'], wp['y'])
        
        self.get_logger().info("Finished sending all spawn requests.")

    def spawn_waypoint_model(self, name, x, y):
        """Spawns a single waypoint model at the given x, y and a fixed high z."""
        request = SpawnEntity.Request()
        request.name = name
        request.xml = self.model_xml
        
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 1.0 # Adjusted Z height for visibility, ensure it's above ground/crops
        request.initial_pose = pose
        
        self.get_logger().info(f"Spawning '{name}' at (x={x:.2f}, y={y:.2f}, z={pose.position.z:.2f})")
        
        future = self.spawn_client.call_async(request)
        # rclpy.spin_until_future_complete(self, future, timeout_sec=1.0) # Optional: wait for each spawn
        # if future.result() is not None and not future.result().success:
        #     self.get_logger().error(f"Failed to spawn {name}: {future.result().status_message}")


def main(args=None):
    rclpy.init(args=args)
    node = GazeboWaypointVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Gazebo Waypoint Visualizer.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()