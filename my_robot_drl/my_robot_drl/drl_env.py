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

# Make sure you have this library installed: pip install dubins or pip install dubins-py
import dubins 

from .dense_waypoint import get_dense_lane_waypoints

class MaizeNavigationEnv(gymnasium.Env, Node):
    """Custom Gymnasium Environment for Maize Field Navigation."""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        gymnasium.Env.__init__(self)
        Node.__init__(self, 'maize_drl_environment')

        self.get_logger().info("Attempting to load waypoints for the environment...")
        self.master_waypoints = get_dense_lane_waypoints()
        if not self.master_waypoints:
            self.get_logger().fatal("CRITICAL: Failed to load waypoints. Cannot proceed.")
        else:
            self.get_logger().info(f"Successfully loaded a master set of {len(self.master_waypoints)} waypoints.")

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32
        )
        
        obs_shape = (362,)
        low_obs = np.full(obs_shape, -np.inf, dtype=np.float32)
        high_obs = np.full(obs_shape, np.inf, dtype=np.float32)
        low_obs[0:360] = 0.0 # LIDAR ranges
        high_obs[0:360] = 2.0 # LIDAR ranges (clamped for observation, actual max_range can be higher)
        low_obs[360] = 0.0 # Distance to goal
        high_obs[360] = 10.0 # Distance to goal (max expected field dimension)
        low_obs[361] = -math.pi # Angle to goal
        high_obs[361] = math.pi # Angle to goal
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, shape=obs_shape, dtype=np.float32
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.reset_sim_client = self.create_client(Empty, '/reset_simulation')
        
        self.current_odom = None
        self.current_scan = np.full(360, 2.0, dtype=np.float32) # Initialize with a typical far value
        self.min_lidar_range = 0.14 # Physical minimum range of the LIDAR
        self.collision_threshold = 0.155 # If min_scan < this, it's a collision
        self.too_far_lidar_threshold = 1.5 # If min_scan > this, considered too far from obstacles (potential issue)
        
        self.waypoints = []
        self.visited_waypoints = []
        self.num_waypoints_total = 0
        self.num_waypoints_visited_current_episode = 0
        
        self.target_waypoint_index = None
        self.previous_waypoint_index = None 
        
        self.last_distance_to_target = 0.0
        self.REWARD_FACTOR_DISTANCE = 15.0

        self.episode_done = False
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.waypoint_reach_threshold = 0.25 # Meters


        # --- Parameters for Dubins U-Turns ---
        self.turning_radius = 0.4  # Meters
        self.turn_wp_step_distance = 0.2 # Meters, density of waypoints on the turn
        self.original_target_after_turn_idx = None # Stores the index of the true next lane WP
        # ---

        self.get_logger().info("MaizeNavigationEnv initialized.")
        self.debug_counter = 0

    def odom_callback(self, msg):
        if msg.header.frame_id != "odom":
            self.get_logger().warn("Unexpected odom frame!")
        self.current_odom = msg

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max
        ranges[ranges < self.min_lidar_range] = msg.range_max
        self.current_scan = ranges

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.get_logger().info("Resetting environment...")

        if not self.master_waypoints:
            self.get_logger().error("Cannot reset: Master waypoint list is empty.")
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return dummy_obs, self._get_info()

        # This outer loop will retry the entire Gazebo reset if sensor data isn't received.
        while rclpy.ok():
            # 1. Call Gazebo reset service
            while not self.reset_sim_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Reset service not available, waiting again...')
            
            reset_future = self.reset_sim_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, reset_future, timeout_sec=5.0)
            
            if not reset_future.done() or reset_future.result() is None:
                self.get_logger().warn("Reset service call failed or timed out. Retrying...")
                time.sleep(1.0)
                continue # Retry the outer while loop

            # 2. Robustly wait for fresh sensor data
            # This replaces the old logic with a more patient while loop.
            self.current_odom = None
            self.current_scan = np.full(360, 2.0, dtype=np.float32)
            
            start_time = time.time()
            timeout_seconds = 10.0
            got_fresh_data = False

            self.get_logger().info(f"Waiting up to {timeout_seconds}s for fresh odom and scan data...")
            while time.time() - start_time < timeout_seconds:
                time.sleep(1.0) # Sleep to avoid busy-waiting
                rclpy.spin_once(self, timeout_sec=0.05) # Actively process callbacks
                # Check if we have received new data since clearing it
                if self.current_odom is not None and not np.all(self.current_scan == 2.0):
                    self.get_logger().info("Successfully received fresh odom and scan.")
                    got_fresh_data = True
                    break
            
            if not got_fresh_data:
                self.get_logger().warn(f"Timed out waiting for sensor data. Retrying full reset process.")
                continue # Retry the outer while loop

            # 3. Initialize episode state (only runs if data was received)
            self.waypoints = [wp.copy() for wp in self.master_waypoints]
            self.num_waypoints_total = len(self.waypoints)
            self.visited_waypoints = [False] * self.num_waypoints_total
            self.num_waypoints_visited_current_episode = 0
            time.sleep(1.0) # Allow time for the system to stabilize after reset
            self.target_waypoint_index = self._find_closest_unvisited_waypoint()
            self.previous_waypoint_index = None
            self.original_target_after_turn_idx = None

            if self.target_waypoint_index is not None:
                target_wp = self.waypoints[self.target_waypoint_index]
                robot_pos = self.current_odom.pose.pose.position
                self.last_distance_to_target = math.sqrt(
                    (target_wp['x'] - robot_pos.x)**2 + (target_wp['y'] - robot_pos.y)**2
                )
                self.get_logger().info(f"current_odom: {robot_pos.x:.2f}, {robot_pos.y:.2f}")
                self.get_logger().info(f"Initial target waypoint #{self.target_waypoint_index} at {target_wp['x']:.2f}, {target_wp['y']:.2f}, distance {self.last_distance_to_target:.2f}m")
            else:
                self.last_distance_to_target = 0.0

            if (self.current_scan == 2.0).all():
                self.get_logger().warn("Received scan with all values at max range. Retrying full reset process.")
                continue
            if (self.current_scan < self.min_lidar_range).any():
                self.get_logger().warn("Received scan with values below minimum range. Retrying full reset process.")
                continue

            self.episode_done = False
            self.last_action = np.array([0.0, 0.0], dtype=np.float32)
            
            # If we've reached here, the reset was successful. Break the outer loop.
            break

        observation = self._get_observation()
        info = self._get_info()
        self.get_logger().info(f"Reset complete. Initial target: #{self.target_waypoint_index}")
        return observation, info

    def step(self, action):
        if self.episode_done:
            return self._get_observation(), 0.0, True, True, self._get_info()

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
            reward = -50.0
            self.get_logger().info(f"Episode terminated: COLLISION. Min scan: {min_current_scan:.3f}. Reward: {reward}")
        
        elif self.episode_done and self.num_waypoints_visited_current_episode >= self.num_waypoints_total:
            terminated = True 
            self.get_logger().info(f"Episode terminated: SUCCESS. Reward: {reward}")

        if not terminated and min_current_scan >= self.too_far_lidar_threshold:
            truncated = True
            reward = -20.0
            self.get_logger().info(f"Episode TRUNCATED: TOO FAR. Min scan: {min_current_scan:.3f}. Reward: {reward}")

        self.episode_done = terminated or truncated
        self.debug_counter += 1

        return observation, reward, terminated, truncated, self._get_info()
    
    def _find_closest_unvisited_waypoint(self):
        if self.current_odom is None: 
            return None
        
        robot_pos = self.current_odom.pose.pose.position
        closest_dist_sq = float('inf')
        closest_idx = None
        
        # Store candidates when distances are nearly equal
        candidate_indices = []
        
        for i, visited in enumerate(self.visited_waypoints):
            if not visited:
                wp = self.waypoints[i]
                dist_sq = (wp['x'] - robot_pos.x)**2 + (wp['y'] - robot_pos.y)**2
                
                # Check if distance is significantly closer
                if dist_sq < closest_dist_sq - 1e-6:  # Tolerance for floating point
                    closest_dist_sq = dist_sq
                    candidate_indices = [i]  # Reset candidates
                    
                # If distance is approximately equal, add to candidates
                elif abs(dist_sq - closest_dist_sq) < 1e-6:
                    candidate_indices.append(i)
        
        # Select smallest index if multiple candidates
        if candidate_indices:
            return min(candidate_indices)
        return closest_idx

    def _calculate_reward(self):
        REWARD_WAYPOINT_REACHED = 25.0
        REWARD_ALL_WAYPOINTS_VISITED_BONUS = 200.0
        TIME_PENALTY_PER_STEP = -0.1


        REWARD_FACTOR_FORWARD_VELOCITY = 1.0 # MODIFIED: Added factor for new forward motion reward

        
        current_reward = TIME_PENALTY_PER_STEP + (self.last_action[0] * REWARD_FACTOR_FORWARD_VELOCITY)
        if self.current_odom is None or self.target_waypoint_index is None:
            self.get_logger().warn("REWARD_FN: No odom or target_waypoint_index. Returning base reward.")
            return current_reward 
        
        robot_pos = self.current_odom.pose.pose.position
        min_scan = np.min(self.current_scan) # Already checked for collision in step, but good for context

        # Only calculate distance reward if not in immediate collision danger (already handled)
        # and not too far (also handled in step, but this reward encourages staying near features)
        if min_scan >= self.collision_threshold and min_scan < self.too_far_lidar_threshold:
            target_wp = self.waypoints[self.target_waypoint_index]
            current_distance = math.sqrt((target_wp['x'] - robot_pos.x)**2 + (target_wp['y'] - robot_pos.y)**2)
            
            # Reward for getting closer to the target waypoint
            distance_diff = self.last_distance_to_target - current_distance
            distance_reward = distance_diff * self.REWARD_FACTOR_DISTANCE
            current_reward += distance_reward
            self.last_distance_to_target = current_distance
            

            # Check if the current target waypoint is reached
            if current_distance < self.waypoint_reach_threshold:
                wp_reached_idx = self.target_waypoint_index
                self.visited_waypoints[wp_reached_idx] = True
                self.num_waypoints_visited_current_episode += 1
                current_reward += REWARD_WAYPOINT_REACHED
                self.get_logger().info(f"REWARD_FN: Reached waypoint #{wp_reached_idx} ({self.waypoints[wp_reached_idx].get('is_turn_assist_wp', False)}). Total visited: {self.num_waypoints_visited_current_episode}/{self.num_waypoints_total}")
                
                wp_reached_data = self.waypoints[wp_reached_idx]
                is_assist_wp_just_reached = wp_reached_data.get('is_turn_assist_wp', False)

                # Check if all waypoints (including dynamically added turn waypoints) are visited
                if self.num_waypoints_visited_current_episode >= self.num_waypoints_total:
                    current_reward += REWARD_ALL_WAYPOINTS_VISITED_BONUS
                    self.episode_done = True # Mark episode as done (success)
                    self.target_waypoint_index = None # No more targets
                    self.get_logger().info(f"REWARD_FN: ALL WAYPOINTS VISITED! Bonus added. Episode ends.")
                else:
                    # Logic for selecting the NEXT target waypoint
                    
                    # ---- NEW TARGET SELECTION LOGIC ----
                    if is_assist_wp_just_reached:
                        # We just reached a turn assist waypoint.
                        # The next target MUST be the next sequential assist waypoint,
                        # or the original_target_after_turn_idx if this was the last assist WP.
                        next_sequential_idx_in_list = wp_reached_idx + 1
                        if next_sequential_idx_in_list < len(self.waypoints) and \
                           self.waypoints[next_sequential_idx_in_list].get('is_turn_assist_wp', False):
                            # The next WP in the list is also an assist WP for this turn
                            self.target_waypoint_index = next_sequential_idx_in_list
                            self.get_logger().info(f"REWARD_FN: Continuing U-Turn. Next assist target: #{self.target_waypoint_index}")
                        else:
                            # This was the last assist WP for the current turn.
                            # Force target to the stored original next lane WP.
                            if self.original_target_after_turn_idx is not None:
                                self.target_waypoint_index = self.original_target_after_turn_idx
                                self.get_logger().info(f"REWARD_FN: U-Turn completed. Next target is original next lane WP: #{self.target_waypoint_index}")
                                self.original_target_after_turn_idx = None # Clear it after use
                            else:
                                # Should not happen if logic is correct, but fallback
                                self.get_logger().warn("REWARD_FN: Last assist WP reached, but no original_target_after_turn_idx. Finding closest.")
                                self.target_waypoint_index = self._find_closest_unvisited_waypoint()
                    else:
                        # We just reached a REGULAR waypoint (not an assist one).
                        # Check for lane change and potentially generate a U-turn.
                        potential_next_actual_lane_target_idx = self._find_closest_unvisited_waypoint() # This finds the next *non-assist* WP in a general sense

                        if potential_next_actual_lane_target_idx is not None:
                            wp_next_actual_lane_data = self.waypoints[potential_next_actual_lane_target_idx]
                            lane_reached = wp_reached_data.get('original_lane_index', -1)
                            lane_next_actual = wp_next_actual_lane_data.get('original_lane_index', -2)

                            if lane_reached != lane_next_actual and self.previous_waypoint_index is not None:
                                self.get_logger().info(f"REWARD_FN: Lane change detected (from {lane_reached} to {lane_next_actual}). Generating U-turn to connect WP#{wp_reached_idx} to WP#{potential_next_actual_lane_target_idx}.")
                                
                                # Store the intended next lane target *before* Dubins WPs are inserted
                                self.original_target_after_turn_idx = potential_next_actual_lane_target_idx
                                
                                num_turn_wps_added = self._generate_dubins_uturn_waypoints(
                                    self.previous_waypoint_index, 
                                    wp_reached_idx, # End of current lane
                                    potential_next_actual_lane_target_idx # Start of next actual lane
                                )
                                if num_turn_wps_added > 0:
                                    # First new Dubins WP is at wp_reached_idx + 1
                                    self.target_waypoint_index = wp_reached_idx + 1
                                    self.get_logger().info(f"REWARD_FN: U-turn generated. First assist target: #{self.target_waypoint_index}")
                                else:
                                    # Dubins generation failed or added no points.
                                    # Revert to the original target, clear stored index.
                                    self.target_waypoint_index = self.original_target_after_turn_idx
                                    self.original_target_after_turn_idx = None 
                                    self.get_logger().warn("REWARD_FN: Dubins U-turn failed. Targeting original next lane WP: #{self.target_waypoint_index}")
                            else:
                                # No lane change, or not enough info for U-turn.
                                # Proceed to the closest unvisited (which is potential_next_actual_lane_target_idx).
                                self.target_waypoint_index = potential_next_actual_lane_target_idx
                                self.original_target_after_turn_idx = None # Ensure it's clear
                        else:
                             # No more unvisited waypoints found by _find_closest_unvisited_waypoint
                             self.get_logger().info("REWARD_FN: No more unvisited waypoints found by _find_closest_unvisited_waypoint after reaching regular WP.")
                             self.target_waypoint_index = None # This might trigger episode end if num_visited < total
                    # ---- END OF NEW TARGET SELECTION LOGIC ----

                    # Update last_distance_to_target for the new target
                    if self.target_waypoint_index is not None:
                        new_target_wp = self.waypoints[self.target_waypoint_index]
                        self.last_distance_to_target = math.sqrt((new_target_wp['x'] - robot_pos.x)**2 + (new_target_wp['y'] - robot_pos.y)**2)
                        self.get_logger().info(f"REWARD_FN: New target is #{self.target_waypoint_index} ({self.waypoints[self.target_waypoint_index].get('is_turn_assist_wp', False)}) at {self.waypoints[self.target_waypoint_index]['x']:.2f},{self.waypoints[self.target_waypoint_index]['y']:.2f}, dist {self.last_distance_to_target:.2f}m")
                    else:
                        # This case handles if _find_closest_unvisited_waypoint returns None after all WPs visited.
                        # It's also hit if the logic above results in no next target.
                        self.get_logger().info("REWARD_FN: No next target waypoint determined.")
                        if self.num_waypoints_visited_current_episode < self.num_waypoints_total:
                           self.get_logger().error("REWARD_FN: Logic error - episode not done but no next target.")
                           self.episode_done = True # Force end if stuck
                
                # Always update previous_waypoint_index to the one just reached
                self.previous_waypoint_index = wp_reached_idx 
        
        return current_reward

    def _generate_dubins_uturn_waypoints(self, prev_wp_idx, end_of_lane_wp_idx, start_of_next_actual_lane_wp_idx):
        """
        Generates U-turn waypoints using Dubins paths to connect the end of
        the current lane to the start of the next actual (non-assist) lane.
        The `start_of_next_actual_lane_wp_idx` is the index of the *original* waypoint
        that starts the next lane, *before* any Dubins points are inserted for *this* turn.
        Returns the number of waypoints added.
        """
        try:
            # 1. Get Waypoint Data
            if prev_wp_idx is None:
                self.get_logger().warn("Dubins: Previous waypoint index is None. Cannot determine start heading. Skipping turn generation.")
                return 0
            
            wp_prev = self.waypoints[prev_wp_idx]
            wp_end_lane = self.waypoints[end_of_lane_wp_idx]
            # `start_of_next_actual_lane_wp_idx` refers to an existing WP in self.waypoints
            wp_start_next_lane_proper = self.waypoints[start_of_next_actual_lane_wp_idx]
       
            # self.get_logger().debug(f"Dubins Input WPs: prev_idx={prev_wp_idx}, end_lane_idx={end_of_lane_wp_idx}, start_next_actual_idx={start_of_next_actual_lane_wp_idx}")
            # self.get_logger().debug(f"WP Data: prev={wp_prev}, end_lane={wp_end_lane}, start_next_proper={wp_start_next_lane_proper}")


            # 2. Define Start Pose (q0) for Dubins Path
            # Position is at wp_end_lane
            # Heading is from wp_prev to wp_end_lane
            x0, y0 = wp_end_lane['x'], wp_end_lane['y']
            dx_start = wp_end_lane['x'] - wp_prev['x']
            dy_start = wp_end_lane['y'] - wp_prev['y']
            if abs(dx_start) < 1e-6 and abs(dy_start) < 1e-6: 
                 self.get_logger().warn("Dubins: Start waypoints for heading (q0) are too close. Using robot's current yaw.")
                 if self.current_odom:
                     _ , _, yaw_start = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
                 else: # Should not happen if called from _calculate_reward
                     self.get_logger().error("Dubins: No odom for fallback yaw_start.")
                     return 0
            else:
                yaw_start = math.atan2(dy_start, dx_start)
            q0 = (x0, y0, yaw_start)

            # 3. Define End Pose (q1) for Dubins Path
            # Position is at wp_start_next_lane_proper
            # Heading will be reversed from yaw_start for a simple U-turn.
            x1, y1 = wp_start_next_lane_proper['x'], wp_start_next_lane_proper['y']
            yaw_end = yaw_start + math.pi # Standard U-turn aims for 180 deg from entry
            yaw_end = (yaw_end + math.pi) % (2 * math.pi) - math.pi # Normalize to [-pi, pi]
            q1 = (x1, y1, yaw_end)
    
            self.get_logger().info(f"Dubins U-Turn Input: q0=({x0:.2f},{y0:.2f},{math.degrees(yaw_start):.1f}deg), q1=({x1:.2f},{y1:.2f},{math.degrees(yaw_end):.1f}deg), R={self.turning_radius:.2f}m")

            # 4. Generate Dubins Path
            path = dubins.shortest_path(q0, q1, self.turning_radius)
            configurations, _ = path.sample_many(self.turn_wp_step_distance)
            # self.get_logger().debug(f"Dubins Path Configurations: {len(configurations)} points sampled: {configurations}")

            if not configurations or len(configurations) < 2: # Need at least start and end
                self.get_logger().warn("Dubins: Path sampling returned no or too few points (less than 2). Skipping turn generation.")
                return 0

            # 5. Prepare and Insert New Waypoints
            new_turn_waypoints = []
            lane_id_of_turn = wp_end_lane.get('original_lane_index', -1) 

            # Insert intermediate points from the Dubins path.
            # configurations[0] is q0, configurations[-1] is q1.
            # We want to insert points *between* wp_end_lane and wp_start_next_lane_proper.
            # The waypoints list will look like: ... wp_end_lane, [Dubins_pts...], wp_start_next_lane_proper ...
            
            # Iterate over configurations[1] up to configurations[-2]
            # These are the actual new points to insert.
            if len(configurations) > 2: # Only if there are intermediate points
                for i in range(1, len(configurations) - 1): 
                    config = configurations[i]
                    turn_wp = {
                        'x': float(config[0]),
                        'y': float(config[1]),
                        'original_lane_index': lane_id_of_turn, 
                        'is_turn_assist_wp': True 
                    }
                    new_turn_waypoints.append(turn_wp)
            
            if not new_turn_waypoints:
                self.get_logger().info("Dubins: No intermediate turn waypoints generated (start/end too close or path too short for step_distance).")
                return 0

            # The new waypoints are inserted *after* end_of_lane_wp_idx.
            # And *before* the original start_of_next_actual_lane_wp_idx.
            # Since start_of_next_actual_lane_wp_idx's position in the list will shift
            # after insertions, we use end_of_lane_wp_idx + 1 as the consistent insertion point.
            insertion_point_idx = end_of_lane_wp_idx + 1 
            for i, turn_wp in enumerate(new_turn_waypoints):
                self.waypoints.insert(insertion_point_idx + i, turn_wp)
                self.visited_waypoints.insert(insertion_point_idx + i, False)
            
            self.num_waypoints_total += len(new_turn_waypoints)
            self.get_logger().info(f"Dubins: Inserted {len(new_turn_waypoints)} U-turn waypoints. New total waypoints in episode: {self.num_waypoints_total}. Path type: {path.path_type()}")
            return len(new_turn_waypoints)

        except Exception as e:
            self.get_logger().error(f"Error during Dubins U-turn generation: {e}")
            import traceback; traceback.print_exc()
            return 0
    
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

        current_target_info = "None"
        if self.target_waypoint_index is not None and 0 <= self.target_waypoint_index < len(self.waypoints):
            wp_data = self.waypoints[self.target_waypoint_index]
            is_assist = wp_data.get('is_turn_assist_wp', False)
            current_target_info = f"#{self.target_waypoint_index} (Assist: {is_assist}) @ ({wp_data['x']:.2f},{wp_data['y']:.2f})"


        info_dict = {
            "waypoints_visited": self.num_waypoints_visited_current_episode,
            "waypoints_total": self.num_waypoints_total,
            "current_target_wp_info": current_target_info,
            "distance_to_target": self.last_distance_to_target,
            "collision_sensor_min_range": min_scan_val,
            "original_target_after_turn_idx": self.original_target_after_turn_idx if self.original_target_after_turn_idx is not None else -1,
        }
        if self.episode_done:
            if min_scan_val < self.collision_threshold: info_dict["termination_reason"] = "collision"
            elif self.num_waypoints_visited_current_episode >= self.num_waypoints_total: info_dict["termination_reason"] = "all_waypoints_visited"
            elif min_scan_val >= self.too_far_lidar_threshold: info_dict["termination_reason"] = "too_far_from_obstacles"
            else: info_dict["termination_reason"] = "unknown_or_logic_end" # e.g. if target_waypoint_index becomes None unexpectedly
        return info_dict
        
    def euler_from_quaternion(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        t0 = +2.0 * (w * x + y * z); t1 = +1.0 - 2.0 * (x * x + y * y); roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x); t2 = +1.0 if t2 > +1.0 else t2; t2 = -1.0 if t2 < -1.0 else t2; pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y); t4 = +1.0 - 2.0 * (y * y + z * z); yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z

    def render(self, mode='human'): pass # Placeholder
    def close(self):
        self.get_logger().info("Closing MaizeNavigationEnv.")
        # Cleanly shut down ROS node resources
        if self.cmd_vel_pub: self.destroy_publisher(self.cmd_vel_pub)
        if self.odom_sub: self.destroy_subscription(self.odom_sub)
        if self.scan_sub: self.destroy_subscription(self.scan_sub)
        if self.reset_sim_client: self.destroy_client(self.reset_sim_client)
        # Call Node's destroy_node if it's not automatically handled by context manager
        if rclpy.ok():
            super().destroy_node() # Call Node's destroy_node if gymnasium.Env doesn't handle it