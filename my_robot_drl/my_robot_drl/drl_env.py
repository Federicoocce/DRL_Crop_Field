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

### MODIFIED ###
# Import the new utility functions and specific sensor message types
from sensor_msgs.msg import Image, PointCloud2
import cv2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from .transfuser_util import scale_and_crop_image, lidar_to_histogram_features, render_sensor_data, close_windows

# Make sure you have this library installed: pip install dubins or pip install dubins-py
import dubins 

from .dense_waypoint import get_dense_lane_waypoints

class MaizeNavigationEnv(gymnasium.Env, Node):
    """Custom Gymnasium Environment for Maize Field Navigation."""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        gymnasium.Env.__init__(self)
        Node.__init__(self, 'maize_drl_environment')
        
        ### MODIFIED ###
        # --- Parameters based on the TransFuser Paper ---
        self.IMAGE_CROP_SIZE = 256 # From the paper's 'crop' parameter
        self.LIDAR_CROP_SIZE = 256 # From the paper's 'crop' parameter

        self.get_logger().info("Attempting to load waypoints for the environment...")
        self.master_waypoints = get_dense_lane_waypoints()
        if not self.master_waypoints:
            self.get_logger().fatal("CRITICAL: Failed to load waypoints. Cannot proceed.")
        else:
            self.get_logger().info(f"Successfully loaded a master set of {len(self.master_waypoints)} waypoints.")

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32
        )
        
        ### MODIFIED: Expanded state space for local and distant goals ###
        # State vector: [local_goal1_x, local_goal1_y, ..., local_goal4_x, local_goal4_y, 
        #                distant_goal_x, distant_goal_y, linear_vel, angular_vel]
        STATE_VECTOR_SIZE = 4 

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.IMAGE_CROP_SIZE, self.IMAGE_CROP_SIZE, 3), dtype=np.uint8),
            'lidar_bev': spaces.Box(low=0, high=255, shape=(self.LIDAR_CROP_SIZE, self.LIDAR_CROP_SIZE, 2), dtype=np.uint8),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_VECTOR_SIZE,), dtype=np.float32),
            # This contains the ground truth for the auxiliary loss.
            # It is NOT part of the policy's direct state input.
            'gt_waypoints': spaces.Box(low=-np.inf, high=np.inf, shape=(4, 2), dtype=np.float32)
        })

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.reset_sim_client = self.create_client(Empty, '/reset_simulation')
        
        self.camera_sub = self.create_subscription(Image, '/tracked_robot/rgb_camera/image_raw', self.camera_callback, 10)
        self.lidar_3d_sub = self.create_subscription(PointCloud2, '/points', self.lidar_3d_callback, 10)
        self.bridge = CvBridge()
        
        self.current_odom = None
        self.current_scan = np.full(360, 2.0, dtype=np.float32)
        self.current_image = np.zeros((self.IMAGE_CROP_SIZE, self.IMAGE_CROP_SIZE, 3), dtype=np.uint8)
        self.current_lidar_bev = np.zeros((self.LIDAR_CROP_SIZE, self.LIDAR_CROP_SIZE, 2), dtype=np.uint8)

        self.min_lidar_range = 0.14
        self.collision_threshold = 0.16
        self.too_far_lidar_threshold = 1.8
        
        self.waypoints, self.visited_waypoints = [], []
        self.num_waypoints_total, self.num_waypoints_visited_current_episode = 0, 0
        self.target_waypoint_index, self.previous_waypoint_index = None, None 
        self.last_distance_to_target, self.REWARD_FACTOR_DISTANCE = 0.0, 15.0
        self.episode_done, self.last_action = False, np.array([0.0, 0.0], dtype=np.float32)
        self.waypoint_reach_threshold = 0.3

        self.turning_radius, self.turn_wp_step_distance = 0.7, 0.2
        self.original_target_after_turn_idx = None
        self.local_goal_waypoints = []
        self.distant_goal_world_coords = None
        self.boundary_crossing_counter = 0

        self.get_logger().info("MaizeNavigationEnv initialized with TransFuser data processing.")
        self.debug_counter = 0

    def odom_callback(self, msg):
        self.current_odom = msg

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max
        ranges[ranges < self.min_lidar_range] = msg.range_max
        self.current_scan = ranges

    ### MODIFIED ###
    def camera_callback(self, msg):
        """Processes incoming camera images using the paper's function."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8") # Use rgb8 to match PIL
            self.current_image = scale_and_crop_image(cv_image, crop=self.IMAGE_CROP_SIZE)
        except Exception as e:
            self.get_logger().error(f"Error in camera_callback: {e}")

    def lidar_3d_callback(self, msg):
        """Processes 3D LiDAR PointCloud2 using the paper's function."""
        try:
            point_generator = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            
            # Manually build a list of simple Python tuples.
            # This is the most robust way to strip the complex dtype.
            points_list = []
            for p in point_generator:
                points_list.append((p[0], p[1], p[2]))

            if not points_list:
                return # Exit if the point cloud is empty
            
            # Now, convert the clean list of tuples to a NumPy array.
            # This will succeed because the input data is no longer structured.
            point_cloud_np = np.array(points_list, dtype=np.float32)

            self.current_lidar_bev = lidar_to_histogram_features(point_cloud_np, crop=self.LIDAR_CROP_SIZE)
            
        except Exception as e:
            self.get_logger().error(f"Error in lidar_3d_callback: {e}")
            import traceback
            traceback.print_exc()

    # The rest of the file (reset, step, _get_observation, etc.) is identical to the
    # previous version that already handles the Dict observation space.
    # No changes are needed there.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.get_logger().info("Resetting environment...")

        if not self.master_waypoints:
            self.get_logger().error("Cannot reset: Master waypoint list is empty.")
            return self._get_observation(), self._get_info()

        while rclpy.ok():
            while not self.reset_sim_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Reset service not available, waiting again...')
            
            reset_future = self.reset_sim_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, reset_future, timeout_sec=5.0)
            
            if not reset_future.done() or reset_future.result() is None:
                self.get_logger().warn("Reset service call failed or timed out. Retrying...")
                time.sleep(1.0)
                continue

            self.current_odom = None
            self.current_scan = np.full(360, 2.0, dtype=np.float32)
            self.current_image = np.zeros((self.IMAGE_CROP_SIZE, self.IMAGE_CROP_SIZE, 3), dtype=np.uint8)
            self.current_lidar_bev = np.zeros((self.LIDAR_CROP_SIZE, self.LIDAR_CROP_SIZE, 2), dtype=np.uint8)
            
            start_time, timeout_seconds, got_fresh_data = time.time(), 10.0, False

            self.get_logger().info(f"Waiting up to {timeout_seconds}s for fresh odom, scan, image, and point cloud...")
            while time.time() - start_time < timeout_seconds:
                time.sleep(1.0) 
                rclpy.spin_once(self, timeout_sec=0.05)
                if (self.current_odom is not None and not np.all(self.current_scan == 2.0) and
                    not np.all(self.current_image == 0) and not np.all(self.current_lidar_bev == 0)):
                    self.get_logger().info("Successfully received all fresh sensor data.")
                    got_fresh_data = True
                    break
            
            if not got_fresh_data:
                self.get_logger().warn(f"Timed out waiting for sensor data. Retrying full reset process.")
                continue 
            self.boundary_crossing_counter = 0
            self.waypoints = [wp.copy() for wp in self.master_waypoints]
            self.num_waypoints_total = len(self.waypoints)
            self.visited_waypoints = [False] * self.num_waypoints_total
            self.num_waypoints_visited_current_episode = 0
            time.sleep(1.0)
            self.target_waypoint_index = self._find_closest_unvisited_waypoint()
            self._initialize_local_goals()
            ### MODIFIED: Set initial distant goal ###
            if self.target_waypoint_index is not None:
                initial_lane_idx = self.waypoints[self.target_waypoint_index].get('original_lane_index')
                if initial_lane_idx is not None:
                    self.distant_goal_world_coords = self._find_end_of_lane(initial_lane_idx)
                    if self.distant_goal_world_coords:
                         self.get_logger().info(f"Initial distant goal set to end of lane {initial_lane_idx}.")
                else: self.distant_goal_world_coords = None
            else: self.distant_goal_world_coords = None
            self.previous_waypoint_index = None
            self.original_target_after_turn_idx = None

            if self.target_waypoint_index is not None:
                target_wp = self.waypoints[self.target_waypoint_index]
                robot_pos = self.current_odom.pose.pose.position
                self.last_distance_to_target = math.sqrt((target_wp['x'] - robot_pos.x)**2 + (target_wp['y'] - robot_pos.y)**2)
            else:
                self.last_distance_to_target = 0.0

            if (self.current_scan < self.min_lidar_range).any():
                self.get_logger().warn("Received scan with values below minimum range. Retrying full reset process.")
                continue

            self.episode_done = False
            self.last_action = np.array([0.0, 0.0], dtype=np.float32)
            self.render()
            break

        observation = self._get_observation()
        info = self._get_info()
        self.get_logger().info(f"Reset complete. Initial target: #{self.target_waypoint_index}")
        return observation, info
    
    def render(self, mode='human'):
        """
        Delegates the rendering task to the utility function.
        """
        # This function now simply passes the relevant data to the helper.
        render_sensor_data(self.current_image, self.current_lidar_bev)

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        # If the episode is already marked as done, return the last observation without taking a new action.
        if self.episode_done:
            return self._get_observation(), 0.0, True, True, self._get_info()

        # 1. Action Execution
        # Clip the action from the agent to ensure it's within the valid range defined by the action space.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action

        # Create and publish the Twist message to move the robot.
        twist_msg = Twist()
        twist_msg.linear.x = float(action[0])
        twist_msg.angular.z = float(action[1])
        self.cmd_vel_pub.publish(twist_msg)

        # 2. State Update
        # Spin ROS once to process all incoming sensor messages and update the environment's state.
        # This is crucial for getting the latest odom, scan, image, and point cloud data.
        rclpy.spin_once(self, timeout_sec=0.1) 

        # 3. Calculate Reward and Get New Observation
        # Based on the new state after moving, calculate the reward.
        reward = self._calculate_reward() 
        # Assemble the new observation for the agent.
        observation = self._get_observation()
        
        # 4. Check for Termination or Truncation
        terminated = False  # Episode ends due to a terminal state (win/loss)
        truncated = False   # Episode ends due to a time limit or other condition
        
        # Get the minimum reading from the 2D laser scan for collision checking.
        min_current_scan = np.min(self.current_scan)

        # --- CONDITION 1: COLLISION (Termination) ---
        # If the closest object is nearer than the collision threshold, the episode is over.
        if min_current_scan < self.collision_threshold:
            terminated = True
            reward = -50.0  # Assign a large negative reward for crashing.
            self.get_logger().info(f"Episode terminated: COLLISION. Min scan: {min_current_scan:.3f}. Reward: {reward}")
        
        # --- CONDITION 2: SUCCESS (Termination) ---
        # Check if the reward function determined that all waypoints have been visited.
        elif self.episode_done and self.num_waypoints_visited_current_episode >= self.num_waypoints_total:
            terminated = True 
            self.get_logger().info(f"Episode terminated: SUCCESS. Reward: {reward}")

        # --- CONDITION 3: LOST/STUCK (Truncation) ---
        # If not already terminated, check if the robot is too far from any obstacles.
        # This prevents the agent from wandering endlessly in open space.
        if not terminated and min_current_scan >= self.too_far_lidar_threshold:
            truncated = True
            reward = -20.0 # Assign a penalty for getting lost.
            self.get_logger().info(f"Episode TRUNCATED: TOO FAR. Min scan: {min_current_scan:.3f}. Reward: {reward}")

        # Update the master 'done' flag for the environment.
        self.episode_done = terminated or truncated
        
        # Increment a debug counter (optional)
        self.debug_counter += 1
        self.render()
        # 5. Return the standard 5-tuple for Gymnasium environments
        return observation, reward, terminated, truncated, self._get_info()
    
    def _get_observation(self):
        # 1. Get ground truth waypoints for the loss function
        gt_waypoints_rel = self._get_relative_local_goals()

        # 2. Get relative distant goal
        distant_goal_rel = self._get_local_coords_from_world_point(self.distant_goal_world_coords)

        # 3. Get current velocity
        linear_vel = self.last_action[0]
        angular_vel = self.last_action[1]

        # 4. Concatenate into the lean state vector for the policy
        state_obs = np.concatenate([
            np.array(distant_goal_rel, dtype=np.float32),
            np.array([linear_vel, angular_vel], dtype=np.float32)
        ])
        
        obs_dict = {
            'image': self.current_image, 
            'lidar_bev': self.current_lidar_bev, 
            'state': state_obs,
            'gt_waypoints': gt_waypoints_rel 
        }
        
        return obs_dict
        
    def _find_closest_unvisited_waypoint(self):
        if self.current_odom is None: 
            return None
        
        robot_pos = self.current_odom.pose.pose.position
        closest_dist_sq = float('inf')
        closest_idx = None
        
        candidate_indices = []
        
        for i, visited in enumerate(self.visited_waypoints):
            if not visited:
                # First, get the waypoint data
                wp = self.waypoints[i]
                # THEN, use it to calculate the distance
                dist_sq = (wp['x'] - robot_pos.x)**2 + (wp['y'] - robot_pos.y)**2
                
                if dist_sq < closest_dist_sq - 1e-6:
                    closest_dist_sq = dist_sq
                    candidate_indices = [i]
                    
                elif abs(dist_sq - closest_dist_sq) < 1e-6:
                    candidate_indices.append(i)
        
        if candidate_indices:
            return min(candidate_indices)
        return closest_idx

    def _calculate_reward(self):
        REWARD_WAYPOINT_REACHED, REWARD_ALL_WAYPOINTS_VISITED_BONUS, TIME_PENALTY_PER_STEP = 25.0, 200.0, -0.1
        REWARD_FACTOR_FORWARD_VELOCITY = 1.0
        current_reward = TIME_PENALTY_PER_STEP + (self.last_action[0] * REWARD_FACTOR_FORWARD_VELOCITY)
        
        if self.current_odom is None or self.target_waypoint_index is None:
            return current_reward
            
        robot_pos = self.current_odom.pose.pose.position
        min_scan = np.min(self.current_scan)
        
        if min_scan >= self.collision_threshold and min_scan < self.too_far_lidar_threshold:
            target_wp = self.waypoints[self.target_waypoint_index]
            current_distance = math.sqrt((target_wp['x'] - robot_pos.x)**2 + (target_wp['y'] - robot_pos.y)**2)
            
            distance_diff = self.last_distance_to_target - current_distance
            current_reward += distance_diff * self.REWARD_FACTOR_DISTANCE
            self.last_distance_to_target = current_distance
            
            # --- Waypoint Reached Logic ---
            if current_distance < self.waypoint_reach_threshold:
                wp_reached_idx = self.target_waypoint_index
                wp_just_reached = self.waypoints[wp_reached_idx]
                is_boundary_wp = wp_just_reached.get('is_lane_boundary', False)

                self.visited_waypoints[wp_reached_idx] = True
                self.num_waypoints_visited_current_episode += 1
                current_reward += REWARD_WAYPOINT_REACHED
                self.get_logger().info(f"REWARD_FN: Reached waypoint #{wp_reached_idx}. Visited: {self.num_waypoints_visited_current_episode}/{self.num_waypoints_total}")
                
                # --- Update Planning Goals (Local and Distant) ---
                self._update_local_goals()

                # *** NEW COUNTER-BASED DISTANT GOAL LOGIC ***
                if is_boundary_wp:
                    self.boundary_crossing_counter += 1
                    self.get_logger().info(f"Boundary crossed. Counter is now: {self.boundary_crossing_counter}")
                    
                    # If this is the FIRST boundary after a turn was initiated (counter was reset to 0),
                    # it means we are at the START of the new lane.
                    if self.boundary_crossing_counter == 1:
                        current_lane_idx = wp_just_reached.get('original_lane_index')
                        if current_lane_idx is not None:
                            new_distant_goal = self._find_end_of_lane(current_lane_idx)
                            if new_distant_goal:
                                self.distant_goal_world_coords = new_distant_goal
                                self.get_logger().info(f"COUNTER=1: At start of lane {current_lane_idx}. New distant goal is END of this lane. {self.distant_goal_world_coords} with index {self.target_waypoint_index}")
                
                # --- Update Target Waypoint for Agent ---
                if self.num_waypoints_visited_current_episode >= self.num_waypoints_total:
                    current_reward += REWARD_ALL_WAYPOINTS_VISITED_BONUS
                    self.episode_done = True
                    self.target_waypoint_index = None
                else:
                    is_assist_wp_just_reached = wp_just_reached.get('is_turn_assist_wp', False)
                    if is_assist_wp_just_reached:
                        # Follow the chain of turn-assist waypoints
                        next_sequential_idx = wp_reached_idx + 1
                        if next_sequential_idx < len(self.waypoints) and self.waypoints[next_sequential_idx].get('is_turn_assist_wp', False):
                            self.target_waypoint_index = next_sequential_idx
                        else:
                            self.target_waypoint_index = self.original_target_after_turn_idx
                            self.original_target_after_turn_idx = None
                    else:
                        # We reached a normal (non-assist) waypoint
                        potential_next_target_idx = self._find_closest_unvisited_waypoint()
                        if potential_next_target_idx is not None:
                            lane_reached = wp_just_reached.get('original_lane_index', -1)
                            lane_next = self.waypoints[potential_next_target_idx].get('original_lane_index', -2)
                            
                            # Check for a lane change, which triggers a U-turn.
                            # This happens when we are at the END of a lane.
                            if is_boundary_wp and lane_reached != lane_next and self.previous_waypoint_index is not None:
                                self.get_logger().info(f"END OF LANE {lane_reached}: Initiating turn.")
                                
                                # 1. RESET the counter. This signifies a turn has started.
                                self.boundary_crossing_counter = 0
                                
                                # 2. SET the distant goal to be the START of the next lane.
                                self.distant_goal_world_coords = self.waypoints[potential_next_target_idx]
                                self.get_logger().info(f"New distant goal is START of lane {lane_next}.")

                                self.original_target_after_turn_idx = potential_next_target_idx
                                num_turn_wps_added = self._generate_dubins_uturn_waypoints(self.previous_waypoint_index, wp_reached_idx, potential_next_target_idx)
                                
                                self.target_waypoint_index = wp_reached_idx + 1 if num_turn_wps_added > 0 else self.original_target_after_turn_idx
                                if num_turn_wps_added <= 0: self.original_target_after_turn_idx = None

                            else: # No lane change, just proceed to the next closest waypoint
                                self.target_waypoint_index = potential_next_target_idx
                                self.original_target_after_turn_idx = None
                        else:
                            self.target_waypoint_index = None # No more waypoints left

                    # Update the distance-to-target for the next step's reward calculation
                    if self.target_waypoint_index is not None:
                        new_target_wp = self.waypoints[self.target_waypoint_index]
                        self.last_distance_to_target = math.sqrt((new_target_wp['x'] - robot_pos.x)**2 + (new_target_wp['y'] - robot_pos.y)**2)

                self.previous_waypoint_index = wp_reached_idx
        return current_reward
    
    def _generate_dubins_uturn_waypoints(self, prev_wp_idx, end_of_lane_wp_idx, start_of_next_actual_lane_wp_idx):
        try:
            if prev_wp_idx is None: return 0
            wp_prev, wp_end_lane, wp_start_next_lane_proper = self.waypoints[prev_wp_idx], self.waypoints[end_of_lane_wp_idx], self.waypoints[start_of_next_actual_lane_wp_idx]
            x0, y0 = wp_end_lane['x'], wp_end_lane['y']
            dx_start, dy_start = wp_end_lane['x'] - wp_prev['x'], wp_end_lane['y'] - wp_prev['y']
            if abs(dx_start) < 1e-6 and abs(dy_start) < 1e-6:
                 if self.current_odom: _, _, yaw_start = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
                 else: return 0
            else: yaw_start = math.atan2(dy_start, dx_start)
            q0 = (x0, y0, yaw_start)
            wp_start_next_lane_proper_next = self.waypoints[start_of_next_actual_lane_wp_idx + 1]
            x1, y1 = wp_start_next_lane_proper['x'], wp_start_next_lane_proper['y']
            dx_next, dy_next = wp_start_next_lane_proper_next['x'] - wp_start_next_lane_proper['x'], wp_start_next_lane_proper_next['y'] - wp_start_next_lane_proper['y']
            yaw_end = math.atan2(dy_next, dx_next)
            if yaw_end - yaw_start < 20 * math.pi / 180.0:
                wp_start_next_lane_proper_prev = self.waypoints[start_of_next_actual_lane_wp_idx - 1]
                dx_next, dy_next = wp_start_next_lane_proper_prev['x'] - wp_start_next_lane_proper['x'], wp_start_next_lane_proper_prev['y'] - wp_start_next_lane_proper['y']
                yaw_end = math.atan2(dy_next, dx_next)
            q1 = (x1, y1, yaw_end)
            path = dubins.shortest_path(q0, q1, self.turning_radius)
            configurations, _ = path.sample_many(self.turn_wp_step_distance)
            if not configurations or len(configurations) < 2: return 0
            new_turn_waypoints, lane_id_of_turn = [], wp_end_lane.get('original_lane_index', -1)
            if len(configurations) > 2:
                for i in range(1, len(configurations) - 1): new_turn_waypoints.append({'x': float(configurations[i][0]), 'y': float(configurations[i][1]), 'original_lane_index': lane_id_of_turn, 'is_turn_assist_wp': True})
            if not new_turn_waypoints: return 0
            insertion_point_idx = end_of_lane_wp_idx + 1
            for i, turn_wp in enumerate(new_turn_waypoints): self.waypoints.insert(insertion_point_idx + i, turn_wp), self.visited_waypoints.insert(insertion_point_idx + i, False)
            self.num_waypoints_total += len(new_turn_waypoints)
            self.get_logger().info(f"Dubins: Inserted {len(new_turn_waypoints)} U-turn waypoints. New total: {self.num_waypoints_total}. Path type: {path.path_type()}")
            return len(new_turn_waypoints)
        except Exception as e: self.get_logger().error(f"Error during Dubins U-turn generation: {e}"); import traceback; traceback.print_exc(); return 0

    def _get_info(self):
        min_scan_val = float(np.min(self.current_scan)) if self.current_scan is not None and len(self.current_scan) > 0 else -1.0
        current_target_info = "None"
        if self.target_waypoint_index is not None and 0 <= self.target_waypoint_index < len(self.waypoints):
            wp_data = self.waypoints[self.target_waypoint_index]
            current_target_info = f"#{self.target_waypoint_index} (Assist: {wp_data.get('is_turn_assist_wp', False)}) @ ({wp_data['x']:.2f},{wp_data['y']:.2f})"
        ### MODIFIED: Added distant goal info ###
        distant_goal_info = "None"
        if self.distant_goal_world_coords:
            distant_goal_info = f"({self.distant_goal_world_coords.get('x', 0.0):.2f}, {self.distant_goal_world_coords.get('y', 0.0):.2f})"

        info_dict = {
            "waypoints_visited": self.num_waypoints_visited_current_episode, 
            "waypoints_total": self.num_waypoints_total, 
            "current_target_wp_info": current_target_info, 
            "distant_goal_world_coords": distant_goal_info,
            "distance_to_target": self.last_distance_to_target, 
            "collision_sensor_min_range": min_scan_val, 
            "original_target_after_turn_idx": self.original_target_after_turn_idx if self.original_target_after_turn_idx is not None else -1
        }
        if self.episode_done:
            if min_scan_val < self.collision_threshold: info_dict["termination_reason"] = "collision"
            elif self.num_waypoints_visited_current_episode >= self.num_waypoints_total: info_dict["termination_reason"] = "all_waypoints_visited"
            elif min_scan_val >= self.too_far_lidar_threshold: info_dict["termination_reason"] = "too_far_from_obstacles"
            else: info_dict["termination_reason"] = "unknown_or_logic_end"
        return info_dict
        
    def euler_from_quaternion(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w # <--- This was the typo (w vs q.w)
        
        # Broke the one-liner into multiple lines for clarity
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z 

    def close(self):
        """
        Cleans up resources, including calling the utility function to close windows.
        """
        self.get_logger().info("Closing MaizeNavigationEnv.")
        # Call the helper function to destroy all OpenCV windows.
        close_windows()
        # ... rest of the cleanup ...
        if self.cmd_vel_pub: self.destroy_publisher(self.cmd_vel_pub)
        if self.odom_sub: self.destroy_subscription(self.odom_sub)
        if self.scan_sub: self.destroy_subscription(self.scan_sub)
        if self.camera_sub: self.destroy_subscription(self.camera_sub)
        if self.lidar_3d_sub: self.destroy_subscription(self.lidar_3d_sub)
        if self.reset_sim_client: self.destroy_client(self.reset_sim_client)
        if rclpy.ok(): super().destroy_node()

    ### NEW HELPER FUNCTIONS ###

    def _find_next_waypoint_from_ref(self, reference_pos: dict, exclusion_indices: set) -> int | None:
        """
        Finds the master index of the single closest unvisited waypoint relative to a given
        reference position, excluding any indices already in the plan.
        """
        closest_dist_sq = float('inf')
        closest_idx = None

        for i, wp in enumerate(self.waypoints):
            if self.visited_waypoints[i] or i in exclusion_indices:
                continue

            dist_sq = (wp['x'] - reference_pos['x'])**2 + (wp['y'] - reference_pos['y'])**2
            if dist_sq < closest_dist_sq:
                closest_dist_sq = dist_sq
                closest_idx = i
                
        return closest_idx

    def _initialize_local_goals(self):
        """
        Initializes the 4-waypoint plan at the start of an episode.
        """
        self.local_goal_waypoints.clear()

        # 1. Use your existing function to find the first waypoint closest to the robot.
        first_wp_idx = self._find_closest_unvisited_waypoint()
        
        if first_wp_idx is None: return

        # 2. Add it to the plan and set it as the reference for the next search.
        ref_wp = self.waypoints[first_wp_idx].copy()
        ref_wp['master_index'] = first_wp_idx
        self.local_goal_waypoints.append(ref_wp)
        
        # 3. Chain the next 3 waypoints.
        for _ in range(3):
            current_plan_indices = {wp['master_index'] for wp in self.local_goal_waypoints}
            next_wp_idx = self._find_next_waypoint_from_ref(ref_wp, current_plan_indices)

            if next_wp_idx is not None:
                ref_wp = self.waypoints[next_wp_idx].copy()
                ref_wp['master_index'] = next_wp_idx
                self.local_goal_waypoints.append(ref_wp)
            else:
                break

    def _update_local_goals(self):
        """
        Removes the reached waypoint and adds the next one to the end of the plan.
        """
        # 1. Remove the first waypoint (the one that was just reached).
        if self.local_goal_waypoints:
            self.local_goal_waypoints.pop(0)

        # 2. Find the next waypoint relative to the end of the current plan.
        if self.local_goal_waypoints:
            ref_wp = self.local_goal_waypoints[-1]
            current_plan_indices = {wp['master_index'] for wp in self.local_goal_waypoints}
            next_wp_idx = self._find_next_waypoint_from_ref(ref_wp, current_plan_indices)

            if next_wp_idx is not None:
                new_wp = self.waypoints[next_wp_idx].copy()
                new_wp['master_index'] = next_wp_idx
                self.local_goal_waypoints.append(new_wp)
    ### NEW/MODIFIED HELPER FUNCTIONS ###
    
    def _get_local_coords_from_world_point(self, world_point: dict | None) -> tuple[float, float]:
        """
        Computes the coordinates of a single world point relative to the robot's
        current position and orientation. Returns (0.0, 0.0) if data is unavailable.
        """
        if self.current_odom is None or world_point is None:
            return (0.0, 0.0)

        robot_pos = self.current_odom.pose.pose.position
        _roll, _pitch, robot_yaw = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
        
        cos_yaw, sin_yaw = math.cos(robot_yaw), math.sin(robot_yaw)
        dx, dy = world_point['x'] - robot_pos.x, world_point['y'] - robot_pos.y
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
            
        return (local_x, local_y)
    
    def _get_relative_local_goals(self) -> np.ndarray:
        """
        Computes the coordinates of the next 4 local waypoints relative to the
        robot's current position and orientation using the generic helper function.
        """
        relative_coords = []
        for wp in self.local_goal_waypoints:
            # Use the new generic helper function for cleaner code
            local_coords = self._get_local_coords_from_world_point(wp)
            relative_coords.append(list(local_coords))

        # Pad the list with [0.0, 0.0] pairs if there are fewer than 4 waypoints
        for _ in range(len(relative_coords), 4):
            relative_coords.append([0.0, 0.0])
        self.get_logger().debug(f"Relative local goals (4): {relative_coords}")

        return np.array(relative_coords, dtype=np.float32)
    
    def _find_end_of_lane(self, lane_index: int) -> dict | None:
        """
        Finds the waypoint marked as the end-of-lane boundary for a specific lane,
        ensuring it has not been visited yet.
        """
        end_of_lane_wp = None
        for i, wp in enumerate(self.waypoints):
            if (wp.get('original_lane_index') == lane_index and
                wp.get('is_lane_boundary', False) and
                not self.visited_waypoints[i]):
                # Because waypoints are ordered, the last one found for a given lane will be its end point.
                # We create a copy to avoid modifying the master list with temporary data
                end_of_lane_wp = wp.copy()
        return end_of_lane_wp