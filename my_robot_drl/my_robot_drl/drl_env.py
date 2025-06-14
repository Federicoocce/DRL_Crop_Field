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

from .get_field_data import get_precise_row_waypoints

class MaizeNavigationEnv(gymnasium.Env, Node):
    """Custom Gymnasium Environment for Maize Field Navigation."""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        gymnasium.Env.__init__(self)
        Node.__init__(self, 'maize_drl_environment')

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32
        )
        obs_shape = (360,)
        low_obs = np.full(obs_shape, 0.0, dtype=np.float32)
        high_obs = np.full(obs_shape, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, shape=obs_shape, dtype=np.float32
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/laser_controller/out', self.scan_callback, 10)
        self.reset_sim_client = self.create_client(Empty, '/reset_simulation')
        
        self.current_odom = None
        self.current_scan = np.full(360, 10.0, dtype=np.float32)
        self.min_lidar_range = 0.12
        self.collision_threshold = 0.14
        # --- NEW PARAMETER ---
        self.too_far_lidar_threshold = 3.0 # Terminate if min LiDAR reading is >= this value
        
        self.waypoints = []
        self.visited_waypoints = [] 
        self.num_waypoints_total = 0
        self.num_waypoints_visited_current_episode = 0

        self.episode_done = False # Master flag for any termination/truncation
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.waypoint_reach_threshold = 0.3 

        self.get_logger().info("MaizeNavigationEnv initialized.")
        self.debug_counter = 0

    def odom_callback(self, msg):
        self.current_odom = msg

    def scan_callback(self, msg):
        scan_data = np.array(msg.ranges, dtype=np.float32)
        scan_data[np.isinf(scan_data)] = msg.range_max
        # CRITICAL FIX: Ignore self-hits. Any reading below the sensor's minimum
        #    range is invalid and should be treated as seeing nothing (max range).
        scan_data[scan_data < self.min_lidar_range] = msg.range_max # Use msg.range_max for invalid
        self.current_scan = scan_data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.get_logger().info("Resetting environment...")

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
            self.current_scan = np.full(360, 10.0, dtype=np.float32)
            
            max_retries_sensor = 10
            for i in range(max_retries_sensor):
                rclpy.spin_once(self, timeout_sec=0.2)
                if self.current_odom is not None and not np.all(self.current_scan == 10.0):
                    break
                if i == max_retries_sensor -1:
                     self.get_logger().warn("Failed to get fresh odom/scan after reset. Retrying full reset...")
            
            if self.current_odom is None or np.all(self.current_scan == 10.0):
                continue

            all_waypoints = get_precise_row_waypoints()
            if not all_waypoints:
                self.get_logger().error("Failed to get waypoints. Retrying reset...")
                time.sleep(1.0)
                continue
            self.waypoints = all_waypoints 
            
            self.num_waypoints_total = len(self.waypoints)
            self.visited_waypoints = [False] * self.num_waypoints_total
            self.num_waypoints_visited_current_episode = 0

            self.episode_done = False
            self.last_action = np.array([0.0, 0.0], dtype=np.float32)
            self.debug_counter = 0
            
            # Check for immediate collision or being too far at spawn
            min_initial_scan = np.min(self.current_scan)
            if min_initial_scan < self.collision_threshold:
                self.get_logger().warn(f"Spawned in collision (min_scan: {min_initial_scan:.3f}). Retrying reset...")
                time.sleep(0.5) 
                continue
            # elif min_initial_scan >= self.too_far_lidar_threshold: # Unlikely at spawn if field exists
            #     self.get_logger().warn(f"Spawned too far (min_scan: {min_initial_scan:.3f}). Retrying reset...")
            #     time.sleep(0.5)
            #     continue
            
            break 

        observation = self._get_observation()
        info = self._get_info()
        self.get_logger().info(f"Reset complete. {self.num_waypoints_total} waypoints loaded. Visited: 0")
        return observation, info
    
    def step(self, action):
        if self.episode_done:
            self.get_logger().warn("Step called on an already completed episode. Resetting.")
            # Consistent with SB3, return last observation, 0 reward, terminated=True, truncated=True, info
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, True, info


        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action

        twist_msg = Twist()
        twist_msg.linear.x = float(action[0])
        twist_msg.angular.z = float(action[1])
        self.cmd_vel_pub.publish(twist_msg)

        rclpy.spin_once(self, timeout_sec=0.1) # Allow ROS callbacks (odom, scan) to update

        observation = self._get_observation()
        # _calculate_reward might set self.episode_done to True if all waypoints are visited
        reward = self._calculate_reward() 
        
        # --- TERMINATION AND TRUNCATION LOGIC ---
        terminated = False # Due to failure (collision) or success (all waypoints)
        truncated = False  # Due to other reasons like time limit or being "too far"

        min_current_scan = np.min(self.current_scan)

        # 1. Check for collision (failure)
        if min_current_scan < self.collision_threshold:
            terminated = True
            reward = -100.0 # Override reward for collision
            self.get_logger().info(f"Episode terminated due to COLLISION. Min scan: {min_current_scan:.3f}. Reward: {reward}")
        
        # 2. Check if all waypoints visited (success)
        # self.episode_done might have been set to True by _calculate_reward()
        elif self.episode_done and self.num_waypoints_visited_current_episode == self.num_waypoints_total:
            terminated = True 
            # Reward is already set by _calculate_reward for success
            self.get_logger().info(f"Episode terminated successfully (ALL WAYPOINTS VISITED). Reward: {reward}")

        # 3. Check for being too far (truncation) - only if not already terminated by collision/success
        if not terminated and min_current_scan >= self.too_far_lidar_threshold:
            truncated = True
            # Assign a specific reward for being too far, e.g., small penalty or neutral
            # Let's use a small penalty to discourage wandering.
            reward = -10.0 # Or 0.0 if you prefer neutral
            self.get_logger().info(f"Episode TRUNCATED due to being TOO FAR. Min scan: {min_current_scan:.3f}. Reward: {reward}")

        # Update master episode_done flag
        self.episode_done = terminated or truncated

        self.debug_counter += 1
        return observation, reward, terminated, truncated, self._get_info()

    def _calculate_reward(self):
        REWARD_WAYPOINT_REACHED = 75.0
        REWARD_ALL_WAYPOINTS_VISITED_BONUS = 200.0
        # PENALTY_COLLISION is handled in step() now for override
        TIME_PENALTY_PER_STEP = -0.1 
        FORWARD_SPEED_REWARD_FACTOR = 1.0 

        current_reward = 0.0
        current_reward += float(self.last_action[0]) * FORWARD_SPEED_REWARD_FACTOR
        current_reward += TIME_PENALTY_PER_STEP

        # Collision check is now primarily handled in step() for termination and reward override.
        # We can skip explicit collision reward here if step() handles it.
        # However, if other parts of reward calc depend on no collision, keep a check.
        # For simplicity now, assuming step() handles the collision penalty.

        if self.current_odom is None:
            # self.get_logger().warn("REWARD_FN: No odom data, cannot check waypoints.")
            return current_reward 

        robot_pos = self.current_odom.pose.pose.position
        newly_visited_this_step = 0

        # Only check waypoints if not already in a terminal state determined by step()
        # (e.g. collision or too_far would already be caught by step)
        # This prevents awarding waypoint rewards if a collision happens simultaneously
        min_scan_for_reward_check = np.min(self.current_scan)
        if min_scan_for_reward_check >= self.collision_threshold and \
           min_scan_for_reward_check < self.too_far_lidar_threshold:

            for i in range(self.num_waypoints_total):
                if not self.visited_waypoints[i]:
                    wp = self.waypoints[i]
                    dist_sq = (wp['x'] - robot_pos.x)**2 + (wp['y'] - robot_pos.y)**2
                    
                    if dist_sq < self.waypoint_reach_threshold**2:
                        self.visited_waypoints[i] = True
                        self.num_waypoints_visited_current_episode += 1
                        newly_visited_this_step +=1
                        current_reward += REWARD_WAYPOINT_REACHED
                        self.get_logger().info( 
                            f"REWARD_FN: Visited waypoint #{i} ('{wp['type']}'). "
                            f"Total visited: {self.num_waypoints_visited_current_episode}/{self.num_waypoints_total}"
                        )

            if newly_visited_this_step > 0 and self.num_waypoints_visited_current_episode == self.num_waypoints_total:
                current_reward += REWARD_ALL_WAYPOINTS_VISITED_BONUS
                self.episode_done = True # Signal successful completion for step()
                self.get_logger().info(f"REWARD_FN: ALL WAYPOINTS VISITED! Bonus: {REWARD_ALL_WAYPOINTS_VISITED_BONUS}. Episode success.")
        
        return current_reward
    
    def _get_observation(self):
        return self.current_scan.astype(np.float32)

    def _get_info(self):
        min_scan_val = -1.0
        if self.current_scan is not None and len(self.current_scan) > 0:
            min_scan_val = float(np.min(self.current_scan))

        info_dict = {
            "waypoints_visited": self.num_waypoints_visited_current_episode,
            "waypoints_total": self.num_waypoints_total,
            "collision_sensor_min_range": min_scan_val
        }
        # Add reason for termination if episode is done
        if self.episode_done:
            if min_scan_val < self.collision_threshold:
                info_dict["termination_reason"] = "collision"
            elif self.num_waypoints_visited_current_episode == self.num_waypoints_total:
                info_dict["termination_reason"] = "all_waypoints_visited"
            elif min_scan_val >= self.too_far_lidar_threshold:
                 info_dict["termination_reason"] = "too_far"
            else:
                 info_dict["termination_reason"] = "unknown" # Should ideally not happen
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