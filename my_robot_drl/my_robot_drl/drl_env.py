# drl_env.py

import rclpy
from rclpy.node import Node
import gymnasium
from gymnasium import spaces
import numpy as np
import math
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from .dense_waypoint import get_dense_lane_waypoints

class MaizeNavigationEnv(gymnasium.Env, Node):
    """Custom Gymnasium Environment for Maize Field Navigation."""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        gymnasium.Env.__init__(self)
        Node.__init__(self, 'maize_drl_environment')

        # --- NEW: LOAD WAYPOINTS ONCE DURING INITIALIZATION ---
        self.get_logger().info("Attempting to load waypoints for the environment...")
        self.master_waypoints = get_dense_lane_waypoints()
        if not self.master_waypoints:
            self.get_logger().fatal("CRITICAL: Failed to load waypoints during environment initialization. Cannot proceed.")
            # In a real application, you might raise an exception here to halt everything.
            # For simplicity, we'll let it continue but it will fail on reset.
            # raise RuntimeError("Failed to load waypoints for the environment.")
        else:
            self.get_logger().info(f"Successfully loaded a master set of {len(self.master_waypoints)} waypoints.")
        # --- END NEW ---

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32
        )
        
        obs_shape = (362,)
        low_obs = np.full(obs_shape, -np.inf, dtype=np.float32)
        high_obs = np.full(obs_shape, np.inf, dtype=np.float32)
        low_obs[0:360] = 0.0
        high_obs[0:360] = 2.0
        low_obs[360] = 0.0
        high_obs[360] = 50.0
        low_obs[361] = -math.pi
        high_obs[361] = math.pi
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, shape=obs_shape, dtype=np.float32
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/laser_controller/out', self.scan_callback, 10)
        self.reset_sim_client = self.create_client(Empty, '/reset_simulation')
        
        self.current_odom = None
        self.current_scan = np.full(360, 2.0, dtype=np.float32)
        self.min_lidar_range = 0.12
        self.collision_threshold = 0.14
        self.too_far_lidar_threshold = 2.0 
        
        # This will hold the waypoints for the CURRENT episode
        self.waypoints = []
        self.visited_waypoints = []
        self.num_waypoints_total = 0
        self.num_waypoints_visited_current_episode = 0
        self.target_waypoint_index = None
        self.last_distance_to_target = 0.0
        self.REWARD_FACTOR_DISTANCE = 15.0

        self.episode_done = False
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.waypoint_reach_threshold = 0.3

        self.get_logger().info("MaizeNavigationEnv initialized with pre-loaded waypoints.")
        self.debug_counter = 0

    def odom_callback(self, msg):
        self.current_odom = msg

    def scan_callback(self, msg):
        scan_data = np.array(msg.ranges, dtype=np.float32)
        scan_data[np.isinf(scan_data)] = msg.range_max
        scan_data[scan_data < self.min_lidar_range] = msg.range_max
        self.current_scan = scan_data

    # --- MODIFIED: RESET FUNCTION ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.get_logger().info("Resetting environment...")

        # --- Check if master waypoints were loaded successfully in __init__ ---
        if not self.master_waypoints:
            self.get_logger().error("Cannot reset: Master waypoint list is empty. Was there an error on startup?")
            # Return a dummy observation and info, but the training will likely crash or be invalid.
            return self._get_observation(), self._get_info()

        # The reset loop for the simulation service remains the same
        while True:
            while not self.reset_sim_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Reset service not available, waiting again...')
            
            reset_future = self.reset_sim_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, reset_future, timeout_sec=2.0)
            if not reset_future.done() or reset_future.result() is None:
                self.get_logger().warn("Reset service call failed or timed out. Retrying...")
                time.sleep(0.5)
                continue
            
            time.sleep(0.5)

            self.current_odom = None
            self.current_scan = np.full(360, 2.0, dtype=np.float32)
            
            max_retries_sensor = 10
            for i in range(max_retries_sensor):
                rclpy.spin_once(self, timeout_sec=0.2)
                if self.current_odom is not None and not np.all(self.current_scan == 2.0):
                    break
                if i == max_retries_sensor - 1:
                     self.get_logger().warn("Failed to get fresh odom/scan after reset. Retrying full reset...")
            
            if self.current_odom is None or np.all(self.current_scan == 2.0):
                continue

            # --- NO LONGER CALLS get_dense_lane_waypoints() ---
            # Instead, we just copy the master list.
            self.waypoints = list(self.master_waypoints) # Use list() for a shallow copy
            
            self.num_waypoints_total = len(self.waypoints)
            self.visited_waypoints = [False] * self.num_waypoints_total
            self.num_waypoints_visited_current_episode = 0
            
            self.target_waypoint_index = self._find_closest_unvisited_waypoint()
            if self.target_waypoint_index is not None:
                target_wp = self.waypoints[self.target_waypoint_index]
                robot_pos = self.current_odom.pose.pose.position
                self.last_distance_to_target = math.sqrt(
                    (target_wp['x'] - robot_pos.x)**2 + (target_wp['y'] - robot_pos.y)**2
                )
                self.get_logger().info(f"Initial target waypoint: #{self.target_waypoint_index} at distance {self.last_distance_to_target:.2f}m")
            else:
                self.last_distance_to_target = 0.0

            self.episode_done = False
            self.last_action = np.array([0.0, 0.0], dtype=np.float32)
            self.debug_counter = 0
            
            min_initial_scan = np.min(self.current_scan)
            if min_initial_scan < self.collision_threshold:
                self.get_logger().warn(f"Spawned in collision (min_scan: {min_initial_scan:.3f}). Retrying reset...")
                time.sleep(0.5) 
                continue
            
            break 

        observation = self._get_observation()
        info = self._get_info()
        self.get_logger().info(f"Reset complete. Using master set of {self.num_waypoints_total} waypoints. Visited: 0")
        return observation, info

    # The rest of the file (step, _calculate_reward, _get_observation, etc.) remains unchanged.
    # I am omitting the rest of the file for brevity, as no other changes are needed.
    # ... (all other methods remain the same) ...

    def step(self, action):
        if self.episode_done:
            self.get_logger().warn("Step called on an already completed episode. Resetting.")
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, True, info

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action

        twist_msg = Twist()
        twist_msg.linear.x = float(action[0])
        twist_msg.angular.z = float(action[1])
        self.cmd_vel_pub.publish(twist_msg)

        rclpy.spin_once(self, timeout_sec=0.1) 

        reward = self._calculate_reward() 
        observation = self._get_observation()
        
        terminated = False
        truncated = False
        min_current_scan = np.min(self.current_scan)

        if min_current_scan < self.collision_threshold:
            terminated = True
            reward = -100.0
            self.get_logger().info(f"Episode terminated due to COLLISION. Min scan: {min_current_scan:.3f}. Reward: {reward}")
        
        elif self.episode_done and self.num_waypoints_visited_current_episode == self.num_waypoints_total:
            terminated = True 
            self.get_logger().info(f"Episode terminated successfully (ALL WAYPOINTS VISITED). Reward: {reward}")

        if not terminated and min_current_scan >= self.too_far_lidar_threshold:
            truncated = True
            reward = -10.0
            self.get_logger().info(f"Episode TRUNCATED due to being TOO FAR. Min scan: {min_current_scan:.3f}. Reward: {reward}")

        self.episode_done = terminated or truncated
        self.debug_counter += 1
        return observation, reward, terminated, truncated, self._get_info()

    def _find_closest_unvisited_waypoint(self):
        if self.current_odom is None:
            return None
        
        robot_pos = self.current_odom.pose.pose.position
        closest_dist_sq = float('inf')
        closest_idx = None
        
        for i, visited in enumerate(self.visited_waypoints):
            if not visited:
                wp = self.waypoints[i]
                dist_sq = (wp['x'] - robot_pos.x)**2 + (wp['y'] - robot_pos.y)**2
                if dist_sq < closest_dist_sq:
                    closest_dist_sq = dist_sq
                    closest_idx = i
                    
        return closest_idx

    def _calculate_reward(self):
        REWARD_WAYPOINT_REACHED = 75.0
        REWARD_ALL_WAYPOINTS_VISITED_BONUS = 200.0
        TIME_PENALTY_PER_STEP = -0.1 
        FORWARD_SPEED_REWARD_FACTOR = 1.0 

        current_reward = 0.0
        current_reward += float(self.last_action[0]) * FORWARD_SPEED_REWARD_FACTOR
        current_reward += TIME_PENALTY_PER_STEP

        if self.current_odom is None or self.target_waypoint_index is None:
            return current_reward 
        
        robot_pos = self.current_odom.pose.pose.position
        min_scan_for_reward_check = np.min(self.current_scan)

        if min_scan_for_reward_check >= self.collision_threshold and \
           min_scan_for_reward_check < self.too_far_lidar_threshold:

            target_wp = self.waypoints[self.target_waypoint_index]
            current_distance = math.sqrt((target_wp['x'] - robot_pos.x)**2 + (target_wp['y'] - robot_pos.y)**2)
            
            distance_diff = self.last_distance_to_target - current_distance
            distance_reward = distance_diff * self.REWARD_FACTOR_DISTANCE
            current_reward += distance_reward
            self.last_distance_to_target = current_distance

            if current_distance < self.waypoint_reach_threshold:
                self.visited_waypoints[self.target_waypoint_index] = True
                self.num_waypoints_visited_current_episode += 1
                current_reward += REWARD_WAYPOINT_REACHED
                self.get_logger().info( 
                    f"REWARD_FN: Visited waypoint #{self.target_waypoint_index}. "
                    f"Total visited: {self.num_waypoints_visited_current_episode}/{self.num_waypoints_total}"
                )

                if self.num_waypoints_visited_current_episode == self.num_waypoints_total:
                    current_reward += REWARD_ALL_WAYPOINTS_VISITED_BONUS
                    self.episode_done = True
                    self.target_waypoint_index = None
                    self.get_logger().info(f"REWARD_FN: ALL WAYPOINTS VISITED! Bonus added.")
                else:
                    new_target_idx = self._find_closest_unvisited_waypoint()
                    self.target_waypoint_index = new_target_idx
                    if new_target_idx is not None:
                        new_target_wp = self.waypoints[new_target_idx]
                        self.last_distance_to_target = math.sqrt(
                            (new_target_wp['x'] - robot_pos.x)**2 + (new_target_wp['y'] - robot_pos.y)**2
                        )
                        self.get_logger().info(f"REWARD_FN: New target is #{new_target_idx}.")
        
        return current_reward
    
    def _get_observation(self):
        dist_to_goal = 0.0
        angle_to_goal = 0.0

        if self.current_odom is not None and self.target_waypoint_index is not None:
            robot_pos = self.current_odom.pose.pose.position
            _roll, _pitch, robot_yaw = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
            
            target_wp = self.waypoints[self.target_waypoint_index]
            
            dist_to_goal = math.sqrt((target_wp['x'] - robot_pos.x)**2 + (target_wp['y'] - robot_pos.y)**2)
            
            angle_rad = math.atan2(target_wp['y'] - robot_pos.y, target_wp['x'] - robot_pos.x)
            
            angle_to_goal = angle_rad - robot_yaw
            angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi
            
        goal_obs = np.array([dist_to_goal, angle_to_goal], dtype=np.float32)
        full_obs = np.concatenate([self.current_scan, goal_obs])
        
        return full_obs.astype(np.float32)

    def _get_info(self):
        min_scan_val = -1.0
        if self.current_scan is not None and len(self.current_scan) > 0:
            min_scan_val = float(np.min(self.current_scan))

        info_dict = {
            "waypoints_visited": self.num_waypoints_visited_current_episode,
            "waypoints_total": self.num_waypoints_total,
            "current_target_wp_idx": self.target_waypoint_index if self.target_waypoint_index is not None else -1,
            "distance_to_target": self.last_distance_to_target,
            "collision_sensor_min_range": min_scan_val
        }
        if self.episode_done:
            if min_scan_val < self.collision_threshold:
                info_dict["termination_reason"] = "collision"
            elif self.num_waypoints_visited_current_episode == self.num_waypoints_total:
                info_dict["termination_reason"] = "all_waypoints_visited"
            elif min_scan_val >= self.too_far_lidar_threshold:
                 info_dict["termination_reason"] = "too_far"
            else:
                 info_dict["termination_reason"] = "unknown"
        return info_dict
        
    def euler_from_quaternion(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        t0 = +2.0 * (w * x + y * z); t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x); t2 = +1.0 if t2 > +1.0 else t2; t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y); t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z

    def render(self, mode='human'): pass
    def close(self):
        self.get_logger().info("Closing MaizeNavigationEnv.")
        self.destroy_node()