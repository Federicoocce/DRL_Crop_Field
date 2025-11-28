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
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
### MODIFIED ###
# Import the new utility functions and specific sensor message types
from sensor_msgs.msg import Image, PointCloud2
import cv2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from .transfuser_util import  lidar_to_histogram_features, render_sensor_data, close_windows
from .spawn_point_calculator import get_spawn_points
# Make sure you have this library installed: pip install dubins or pip install dubins-py
import dubins 
import random
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from .realtime_mapper import RealTimeSemanticMapper


from .dense_waypoint import get_dense_lane_waypoints, WorldDescription, Field2DGenerator
from .config import config


def display_sensor_streams(rgb_image=None, depth_image=None, rear_image=None):
    """
    Opens OpenCV windows to display sensor data for debugging.
    This version uses fixed-range normalization for a clearer depth view.
    """
    # Visualization constants for depth
    VIZ_DEPTH_MIN = 0.1  # Meters
    VIZ_DEPTH_MAX = 5.0  # Meters (clip here for better color variation)

    if rgb_image is not None and rgb_image.size > 0:
        cv2.imshow("RGB Camera (Front)", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    if depth_image is not None and depth_image.size > 0:
        depth_to_show = np.squeeze(depth_image.copy())

        # Clip the depth values to our desired visualization range
        depth_to_show = np.clip(depth_to_show, VIZ_DEPTH_MIN, VIZ_DEPTH_MAX)

        # Normalize based on the FIXED range, not the dynamic min/max of the image
        normalized_depth = (depth_to_show - VIZ_DEPTH_MIN) / (VIZ_DEPTH_MAX - VIZ_DEPTH_MIN)
        
        # Convert to 8-bit and apply a colormap
        normalized_depth_8u = (normalized_depth * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized_depth_8u, cv2.COLORMAP_JET)
        
        cv2.imshow("Depth Image (Front)", colored_depth)

    if rear_image is not None and rear_image.size > 0:
        cv2.imshow("RGB Camera (Rear)", cv2.cvtColor(rear_image, cv2.COLOR_RGB2BGR))

    cv2.waitKey(1)
def close_windows():
    """Closes all OpenCV windows."""
    cv2.destroyAllWindows()

def quaternion_from_euler(roll, pitch, yaw): # <--- ADD THIS HELPER FUNCTION
    """Converts euler roll, pitch, yaw to a Quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q

class MaizeNavigationEnv(gymnasium.Env, Node):
    metadata = {'render_modes': ['human']}

    def __init__(self, robot_name='tracked_robot'):
        gymnasium.Env.__init__(self)
        Node.__init__(self, 'maize_drl_environment')
        self.config = config
        self.robot_name = robot_name
        self.FINAL_crop_SIZE = 256 # Final size for both image and LiDAR BEV

        self.get_logger().info("Attempting to load waypoints for the environment...")
        self.master_waypoints = get_dense_lane_waypoints()[0]
        if not self.master_waypoints:
            self.get_logger().fatal("CRITICAL: Failed to load waypoints. Cannot proceed.")
            rclpy.shutdown()
            return

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32
        )
        # --- MODIFIED: Observation Space for RAW Data ---
        # Define raw image dimensions for the NEW front RGB-D camera
        FRONT_IMG_HEIGHT = 576
        FRONT_IMG_WIDTH = 1024
        # Dimensions for the original rear camera
        REAR_IMG_HEIGHT = 576 
        REAR_IMG_WIDTH = 1024
        self.MAX_LIDAR_POINTS = 10080
        STATE_VECTOR_SIZE = 4
        # Far clipping plane from URDF, used for observation space and depth processing
        self.camera_far_clip = 10.0
        # --- MAPPER CONFIGURATION ---
        # --- MAPPER CONFIGURATION ---
        # 0.015625 = 1/64 meters per pixel.
        # Initial size 512px (~8m x 8m). Expands dynamically.
        self.mapper = RealTimeSemanticMapper(resolution=0.015625, initial_map_size=512)
        obs_space_dict = {
            'image_raw': spaces.Box(low=0, high=255, shape=(FRONT_IMG_HEIGHT, FRONT_IMG_WIDTH, 3), dtype=np.uint8),
            'depth_raw': spaces.Box(low=0.0, high=self.camera_far_clip, shape=(FRONT_IMG_HEIGHT, FRONT_IMG_WIDTH, 1), dtype=np.float32),
            'lidar_raw': spaces.Box(low=-np.inf, high=np.inf, shape=(self.MAX_LIDAR_POINTS, 3), dtype=np.float32),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_VECTOR_SIZE,), dtype=np.float32),
            'gt_waypoints': spaces.Box(low=-np.inf, high=np.inf, shape=(4, 2), dtype=np.float32),
            'bev_semantic' : spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            'semantic_raw': spaces.Box(low=0, high=2, shape=(FRONT_IMG_HEIGHT, FRONT_IMG_WIDTH), dtype=np.uint8)
        }
        
        if not self.config.ignore_rear:
            self.get_logger().info("Rear camera is ENABLED in the environment.")
            # Use the separate dimensions for the rear camera
            obs_space_dict['image_rear_raw'] = spaces.Box(low=0, high=255, shape=(REAR_IMG_HEIGHT, REAR_IMG_WIDTH, 3), dtype=np.uint8)
        
        self.observation_space = spaces.Dict(obs_space_dict)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.reset_sim_client = self.create_client(Empty, '/reset_simulation')
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')
        ### MODIFIED SUBSCRIBERS for RGB-D Camera ###
        self.get_logger().info("Subscribing to RGB-D camera topics...")
        self.camera_sub = self.create_subscription(Image, '/tracked_robot/front_rgbd_camera/image_raw', self.camera_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/tracked_robot/front_rgbd_camera/depth/image_raw', self.depth_callback, 10)
        self.lidar_3d_sub = self.create_subscription(PointCloud2, '/points', self.lidar_3d_callback, 10)
        # --- MODIFICATION: Conditional Subscriber ---
        self.current_rear_image_raw = None # Initialize as None
        if not self.config.ignore_rear:
            self.rear_camera_sub = self.create_subscription(Image, '/tracked_robot/rear_camera/image_raw', self.rear_camera_callback, 10)
            self.current_rear_image_raw = np.zeros((REAR_IMG_HEIGHT, REAR_IMG_WIDTH, 3), dtype=np.uint8)
       
        self.bridge = CvBridge()
        
        self.get_logger().info("Connecting to Gazebo services...")
        while not self.set_state_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Service /set_entity_state not available, waiting...')
        while not self.reset_sim_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Service /reset_simulation not available, waiting...')
        self.get_logger().info("Successfully connected to Gazebo services.")

        # --- MODIFIED: Prepare for dynamic spawn point initialization ---
        self.spawn_points = {}
        self.spawn_point_keys = []
        self.spawn_points_initialized = False # Flag to run initialization only once
        
        # --- Initialize other state variables ---
        self.current_odom = None
        self.current_scan = np.full(360, 2.0, dtype=np.float32)
        # Initialize with expected raw camera size
        self.current_image_raw = np.zeros((FRONT_IMG_HEIGHT, FRONT_IMG_WIDTH, 3), dtype=np.uint8) 
        self.current_depth_raw = np.zeros((FRONT_IMG_HEIGHT, FRONT_IMG_WIDTH, 1), dtype=np.float32)
        # Initialize empty point cloud
        self.current_lidar_raw = np.zeros((0, 3), dtype=np.float32) 
        self.min_lidar_range = 0.14
        self.collision_threshold = 0.141   #era 0.16 disattivato per test
        self.too_far_lidar_threshold = 1.8
        self.waypoints, self.visited_waypoints = [], []
        self.num_waypoints_total = 0
        self.target_waypoint_index, self.previous_waypoint_index = None, None
        self.last_distance_to_target, self.REWARD_FACTOR_DISTANCE = 0.0, 15.0
        self.episode_done, self.last_action = False, np.array([0.0, 0.0], dtype=np.float32)
        self.waypoint_reach_threshold = 0.35
        self.turning_radius, self.turn_wp_step_distance = 0.6, 0.3
        self.original_target_after_turn_idx = None
        self.local_goal_waypoints = []
        self.distant_goal_world_coords = None
        self.boundary_crossing_counter = 0
        self.debug_counter = 0
        self.is_turning = False # Flag to detect when the robot is executing a U-turn

    def _initialize_spawn_points_dynamically(self):
        """
        Calculates spawn points based on the robot's actual initial odometry.
        This is run only once on the first reset.
        """
        try:
            self.get_logger().info("First reset: Dynamically initializing spawn points from initial odometry...")
            Z_OFFSET = 0.1
            
            # --- STEP 1: 'start_spawn' is the robot's current position after reset ---
            initial_pos = self.current_odom.pose.pose.position
            _roll, _pitch, initial_yaw = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
            start_spawn_pos = {
                "x": initial_pos.x, "y": initial_pos.y, "z": Z_OFFSET, "yaw": initial_yaw
            }
            self.spawn_points["start_spawn"] = start_spawn_pos

            # --- STEP 2: Find the 'start_of_lane' by finding the closest waypoint ---
            start_of_lane_wp, sol_wp_idx = self._find_closest_master_waypoint(start_spawn_pos)
            if start_of_lane_wp is None:
                raise ValueError("Could not find a closest waypoint to the initial robot position.")

            next_wp_for_sol = self.master_waypoints[sol_wp_idx + 1]
            start_lane_yaw = math.atan2(next_wp_for_sol['y'] - start_of_lane_wp['y'], next_wp_for_sol['x'] - start_of_lane_wp['x'])
            
            self.spawn_points["start_of_lane"] = {
                "x": start_of_lane_wp['x'], "y": start_of_lane_wp['y'], "z": Z_OFFSET, "yaw": start_lane_yaw
            }

            # --- STEP 3: Find the 'end_of_lane' ---
            first_lane_index = start_of_lane_wp.get('original_lane_index')
            if first_lane_index is None:
                raise ValueError("The identified 'start_of_lane' waypoint is missing a lane index.")

            end_of_lane_wp, eol_wp_idx = self._find_end_of_lane_master(first_lane_index)
            if end_of_lane_wp is None:
                raise ValueError(f"Could not find the end boundary for lane {first_lane_index}.")

            prev_wp_for_eol = self.master_waypoints[eol_wp_idx - 1]
            end_lane_yaw = math.atan2(end_of_lane_wp['y'] - prev_wp_for_eol['y'], end_of_lane_wp['x'] - prev_wp_for_eol['x'])
            
            self.spawn_points["end_of_lane"] = {
                "x": end_of_lane_wp['x'], "y": end_of_lane_wp['y'], "z": Z_OFFSET, "yaw": end_lane_yaw,
                "lane_to_skip": first_lane_index
            }
            
            self.spawn_point_keys = list(self.spawn_points.keys())
            self.spawn_points_initialized = True
            for name, data in self.spawn_points.items():
                self.get_logger().info(f"  - Initialized '{name}': x={data['x']:.2f}, y={data['y']:.2f}, yaw={data['yaw']:.2f}")

        except Exception as e:
            self.get_logger().fatal(f"CRITICAL: Failed to dynamically initialize spawn points: {e}")
            self.get_logger().fatal("Cannot proceed with training. Shutting down.")
            rclpy.shutdown()

    def odom_callback(self, msg):
        self.current_odom = msg

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max
        ranges[ranges < self.min_lidar_range] = msg.range_max
        self.current_scan = ranges

    ### MODIFIED CALLBACKS ###
    def camera_callback(self, msg):
        """Stores raw incoming RGB camera images."""
        try:
            self.current_image_raw = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f"Error in camera_callback: {e}")

    ### NEW: Callback for Depth Image ###
    def depth_callback(self, msg):
        """Stores raw incoming depth images."""
        try:
            # The encoding for depth images in Gazebo is typically 32FC1 (32-bit float, 1 channel)
            # 'passthrough' will keep the original data type and depth
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # The observation space expects a 3D array (H, W, 1)
            depth_image_3d = np.expand_dims(cv_image, axis=-1)

            # Replace any NaN (Not a Number) values with the far clipping distance
            self.current_depth_raw = np.nan_to_num(depth_image_3d, nan=self.camera_far_clip)
            
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")
        
    def rear_camera_callback(self, msg):
        """Stores raw incoming rear camera images."""
        try:
            self.current_rear_image_raw = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f"Error in rear_camera_callback: {e}")

    def lidar_3d_callback(self, msg):
        try:
            point_generator = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            points_list = []
            for p in point_generator:
                points_list.append((p[0], p[1], p[2]))

            # --- START OF PADDING LOGIC ---
            num_points = len(points_list)
            
            # Create a fixed-size array filled with zeros
            fixed_size_cloud = np.zeros((self.MAX_LIDAR_POINTS, 3), dtype=np.float32)
            
            if num_points > 0:
                # Convert the list to a NumPy array
                point_cloud_np = np.array(points_list, dtype=np.float32)
                
                # Truncate if we have more points than expected (unlikely but safe)
                num_to_copy = min(num_points, self.MAX_LIDAR_POINTS)
                
                # Copy the actual points into the fixed-size array
                fixed_size_cloud[:num_to_copy, :] = point_cloud_np[:num_to_copy, :]
            
            # Store the padded, fixed-size array
            self.current_lidar_raw = fixed_size_cloud
            # --- END OF PADDING LOGIC ---
            
        except Exception as e:
            self.get_logger().error(f"Error in lidar_3d_callback: {e}")

      
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.get_logger().info("Resetting environment...")
        self.previous_waypoint_index = None
        self.is_turning = False

        while rclpy.ok():
            # 1. Reset the simulation to its default state
            reset_future = self.reset_sim_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, reset_future, timeout_sec=5.0)
            if not reset_future.done() or reset_future.result() is None:
                self.get_logger().warn("Reset service call failed or timed out. Retrying...")
                time.sleep(1.0)
                continue

            # 2. Get initial odometry (for first-time spawn point calculation)
            self.current_odom = None
            start_time, timeout_seconds = time.time(), 10.0
            while time.time() - start_time < timeout_seconds:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.current_odom is not None: break
            if self.current_odom is None:
                self.get_logger().warn("Timed out waiting for initial odom after reset. Retrying.")
                continue
                
            # 3. Initialize spawn points if not already done
            if not self.spawn_points_initialized:
                self._initialize_spawn_points_dynamically()
                if not self.spawn_points_initialized:
                    self.get_logger().error("Shutting down due to failed spawn point initialization.")
                    return None, {}

            # --- START OF MODIFICATION: WEIGHTED SPAWN SELECTION ---
            # 4. Choose a spawn point with a higher probability for 'end_of_lane'
            
            # Define the desired probability for the 'end_of_lane' spawn
            end_of_lane_prob = 0.99
            
            # Get all spawn point keys except for 'end_of_lane'
            other_keys = [key for key in self.spawn_point_keys if key != 'end_of_lane']
            
            if not other_keys: # Safety check in case 'end_of_lane' is the only key
                num_other_keys = 0
                other_prob_total = 0.0
            else:
                num_other_keys = len(other_keys)
                # The remaining probability is distributed equally among the other keys
                other_prob_total = 1.0 - end_of_lane_prob

            # Create the list of spawn keys and their corresponding weights
            spawn_candidates = []
            spawn_weights = []

            if 'end_of_lane' in self.spawn_point_keys:
                spawn_candidates.append('end_of_lane')
                spawn_weights.append(end_of_lane_prob)
            
            if num_other_keys > 0:
                prob_per_other = other_prob_total / num_other_keys
                for key in other_keys:
                    spawn_candidates.append(key)
                    spawn_weights.append(prob_per_other)

            # Use random.choices() to select one spawn key based on the weights
            # The [0] is because random.choices returns a list (e.g., ['end_of_lane'])
            chosen_spawn_key = random.choices(spawn_candidates, weights=spawn_weights, k=1)[0]
            
            # --- END OF MODIFICATION ---

            spawn_loc = self.spawn_points[chosen_spawn_key]
            self.get_logger().info(f"Attempting to teleport robot to: '{chosen_spawn_key}' (Selected with weighted probability)")
            # --- START OF NOISE MODIFICATION ---
            # Add random noise to the spawn orientation (+-10 degrees)
            noise_degrees = 15.0
            noise_radians = math.radians(random.uniform(-noise_degrees, noise_degrees))
            noisy_yaw = spawn_loc['yaw'] + noise_radians
            # --- END OF NOISE MODIFICATION ---

            pose = Pose(position=Point(x=spawn_loc['x'], y=spawn_loc['y'], z=spawn_loc['z']),
                        orientation=quaternion_from_euler(0.0, 0.0, noisy_yaw))
            req = SetEntityState.Request()
            req.state.name = self.robot_name
            req.state.pose = pose
            req.state.reference_frame = "world"
            
            set_state_future = self.set_state_client.call_async(req)
            rclpy.spin_until_future_complete(self, set_state_future, timeout_sec=5.0)
            if not set_state_future.done() or not set_state_future.result().success:
                self.get_logger().warn("Failed to set robot state. Retrying reset...")
                continue
            
            # 5. Wait for the simulation to settle
            time.sleep(0.5)
            self.current_odom, self.current_scan = None, np.full(360, 2.0, dtype=np.float32)
            self.current_lidar_raw = np.zeros((0, 3), dtype=np.float32) # Reset raw LiDAR
            # Correctly get dimensions from the observation space definition
            FRONT_IMG_HEIGHT = self.observation_space['image_raw'].shape[0]
            FRONT_IMG_WIDTH = self.observation_space['image_raw'].shape[1]

            self.current_image_raw = np.zeros((FRONT_IMG_HEIGHT, FRONT_IMG_WIDTH, 3), dtype=np.uint8) 
            self.current_depth_raw = np.zeros((FRONT_IMG_HEIGHT, FRONT_IMG_WIDTH, 1), dtype=np.float32)

            self.current_lidar_raw = np.zeros((0, 3), dtype=np.float32)
            start_time, got_fresh_data = time.time(), False
            while time.time() - start_time < timeout_seconds:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.current_odom is not None and not np.all(self.current_scan == 2.0):
                    got_fresh_data = True
                    break
                time.sleep(0.2)
            if not got_fresh_data:
                self.get_logger().warn("Timed out waiting for sensor data after teleport. Retrying.")
                continue 
            
            # 6. Initialize waypoints and determine the next target
            self.boundary_crossing_counter = 0
            self.waypoints = [wp.copy() for wp in self.master_waypoints]
            self.visited_waypoints = [False] * len(self.waypoints)
            self.num_waypoints_visited_current_episode = 0
            self.num_waypoints_total = len(self.waypoints)
            
            self.target_waypoint_index = None

            ### THIS IS THE CORRECTED LOGIC BASED ON YOUR INSTRUCTION ###
            if 'lane_to_skip' in spawn_loc:
                # Set up the environment to trigger the U-turn logic on the first step
                lane_to_skip_idx = spawn_loc['lane_to_skip']
                self.get_logger().info(f"Staging lane {lane_to_skip_idx} to trigger U-turn on next step.")
                
                # Find the index of the very last waypoint in this lane
                end_of_lane_wp_idx = -1
                for i, wp in reversed(list(enumerate(self.waypoints))):
                    if wp.get('original_lane_index') == lane_to_skip_idx:
                        end_of_lane_wp_idx = i
                        break
                
                if end_of_lane_wp_idx != -1:
                    # Mark all waypoints in the lane as visited, EXCEPT the last one
                    for i, wp in enumerate(self.waypoints):
                        if wp.get('original_lane_index') == lane_to_skip_idx and i != end_of_lane_wp_idx:
                            if not self.visited_waypoints[i]:
                                self.visited_waypoints[i] = True
                                self.num_waypoints_visited_current_episode += 1
                    
                    # Set the target to be that single, unvisited, end-of-lane waypoint
                    self.target_waypoint_index = end_of_lane_wp_idx
                    # Set the previous waypoint to allow the U-turn logic to calculate direction
                    self.previous_waypoint_index = end_of_lane_wp_idx - 1
                    self.boundary_crossing_counter = 1
                else:
                    # Fallback if something goes wrong
                    self.target_waypoint_index = self._find_closest_unvisited_waypoint()

            else:
                # Standard logic for other spawns
                self.target_waypoint_index = self._find_closest_unvisited_waypoint()
            
            if self.target_waypoint_index is None:
                self.get_logger().warn(f"Could not find a valid target from spawn '{chosen_spawn_key}'. Retrying reset.")
                time.sleep(1.0)
                continue
            
            # 7. Finalize the reset process
            self._initialize_local_goals()
            initial_lane_idx = self.waypoints[self.target_waypoint_index].get('original_lane_index')
            self.distant_goal_world_coords = self._find_end_of_lane(initial_lane_idx) if initial_lane_idx is not None else None
            
            if self.previous_waypoint_index is None:
                 self.previous_waypoint_index = self.target_waypoint_index

            self.original_target_after_turn_idx = None

            target_wp = self.waypoints[self.target_waypoint_index]
            robot_pos = self.current_odom.pose.pose.position
            initial_x = self.current_odom.pose.pose.position.x
            initial_y = self.current_odom.pose.pose.position.y
            
            # Reset the map and center it on the spawn point
            self.mapper.reset_map(initial_x, initial_y)
                
            self.last_distance_to_target = math.sqrt((target_wp['x'] - robot_pos.x)**2 + (target_wp['y'] - robot_pos.y)**2)
            
            self.episode_done = False
            self.last_action = np.array([0.0, 0.0], dtype=np.float32)
            self.render()
            break
        
        observation = self._get_observation()
        info = self._get_info()
        self.get_logger().info(f"Reset complete. Initial target: #{self.target_waypoint_index}")
        return observation, info

        
    def render(self, mode='human'):
            # 1. Update Map and Get Segmentation Image (Existing Logic)
            seg_viz = None 
            if self.current_odom:
                rx = self.current_odom.pose.pose.position.x
                ry = self.current_odom.pose.pose.position.y
                _r, _p, ryaw = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
                
                if self.current_image_raw is not None and len(self.current_lidar_raw) > 0:
                    seg_viz, sem_raw = self.mapper.process_one_step(
                        self.current_image_raw, 
                        self.current_lidar_raw, 
                        rx, ry, ryaw
                    )
                    if seg_viz is not None:
                        cv2.imshow("Semantic Segmentation", seg_viz)

            # 2. Display Sensor Streams (Existing Logic)
            display_sensor_streams(
                rgb_image=self.current_image_raw, 
                depth_image=self.current_depth_raw,
                rear_image=self.current_rear_image_raw
            )

            # 3. --- NEW: Transfuser Input Debug Visualization with Trajectory ---
            if self.current_lidar_raw is not None and len(self.current_lidar_raw) > 0:
                # A. Get BEV features (H, W, 2)
                lidar_bev = lidar_to_histogram_features(self.current_lidar_raw)

                # B. Create Display Image (BGR)
                # Blue=Ground (Ch 0), Red=Obstacles (Ch 1)
                h, w, _ = lidar_bev.shape
                transfuser_display = np.zeros((h, w, 3), dtype=np.uint8)
                transfuser_display[:, :, 0] = lidar_bev[:, :, 0] 
                transfuser_display[:, :, 2] = lidar_bev[:, :, 1] 

                # C. Collect Points to Draw (Robot -> 4 WPs -> Distant Goal)
                points_to_draw = []
                
                # 1. Robot Position (0,0 in local frame)
                points_to_draw.append({'coords': (0.0, 0.0), 'type': 'robot'})

                # 2. Next 4 Waypoints
                relative_goals = self._get_relative_local_goals()
                for wp in relative_goals:
                    points_to_draw.append({'coords': (wp[0], wp[1]), 'type': 'waypoint'})
                
                # 3. Distant Goal
                distant_local = self._get_local_coords_from_world_point(self.distant_goal_world_coords)
                points_to_draw.append({'coords': distant_local, 'type': 'goal'})

                # D. Convert to Pixels (Logic strictly matches transfuser_util.py)
                x_meters = 2.0      # Forward range (+/-)
                y_max_meters = 2.0  # Side range (+/-)
                pixels_per_meter = 64
                feature_map_height = int(x_meters * 2 * pixels_per_meter)
                
                pixel_poly_list = [] # For drawing the line

                for pt in points_to_draw:
                    rx, ry = pt['coords']
                    
                    # Math from draw_target_point:
                    # Map Y (Left/Right)
                    col = int((-ry + y_max_meters) * pixels_per_meter)
                    # Map X (Forward/Backward) - shift origin and flip axis
                    row = int(feature_map_height - 1 - ((rx + x_meters) * pixels_per_meter))
                    
                    # Check bounds before drawing
                    if 0 <= col < w and 0 <= row < h:
                        point_pixel = (col, row)
                        
                        if pt['type'] == 'robot':
                            pixel_poly_list.append(point_pixel)
                        
                        elif pt['type'] == 'waypoint':
                            pixel_poly_list.append(point_pixel)
                            # Draw Orange Circle for Waypoints
                            cv2.circle(transfuser_display, point_pixel, radius=4, color=(0, 165, 255), thickness=-1)
                        
                        elif pt['type'] == 'goal':
                            # Draw Purple Circle for Distant Goal
                            cv2.circle(transfuser_display, point_pixel, radius=6, color=(255, 0, 255), thickness=-1)
                            # Add label
                            cv2.putText(transfuser_display, "G", (col+4, row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                # E. Draw Trajectory Line (Yellow)
                if len(pixel_poly_list) > 1:
                    pts = np.array(pixel_poly_list, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(transfuser_display, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

                # F. Scale and Show
                display_size = 512
                transfuser_display_large = cv2.resize(transfuser_display, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Transfuser Debug (Yel=Path, Pur=Goal)", transfuser_display_large)

            # 4. Global Map Visualization (Existing Logic)
            if self.current_odom:
                rx = self.current_odom.pose.pose.position.x
                ry = self.current_odom.pose.pose.position.y
                
                global_map = self.mapper.get_debug_image(rx, ry)
                disp_size = 600
                global_map_small = cv2.resize(global_map, (disp_size, disp_size), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Global Map", global_map_small)



            cv2.waitKey(1)

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

        self._dynamic_waypoint_lookahead()
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
        gt_waypoints_rel = self._get_relative_local_goals()
        distant_goal_rel = self._get_local_coords_from_world_point(self.distant_goal_world_coords)
        linear_vel = self.last_action[0]
        angular_vel = self.last_action[1]

        # Initialize containers for semantic segmentation
        sem_raw = np.zeros((576, 1024), dtype=np.uint8) # Default blank

        if self.current_odom:
            rx = self.current_odom.pose.pose.position.x
            ry = self.current_odom.pose.pose.position.y
            _r, _p, ryaw = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
        else:
            rx, ry, ryaw = 0.0, 0.0, 0.0

        # Only update if we have valid sensor data
        if self.current_image_raw is not None and len(self.current_lidar_raw) > 0:
            # --- UNPACK THE TUPLE HERE ---
            _ , sem_raw = self.mapper.process_one_step(
                self.current_image_raw, 
                self.current_lidar_raw, 
                rx, ry, ryaw
            )

        state_obs = np.concatenate([
            np.array(distant_goal_rel, dtype=np.float32),
            np.array([linear_vel, angular_vel], dtype=np.float32)
        ])

        
        obs_dict = {
            'image_raw': self.current_image_raw,
            'depth_raw': self.current_depth_raw,
            'lidar_raw': self.current_lidar_raw,
 
            'semantic_raw': sem_raw, # <--- ADD THIS
            'state': state_obs,
            'gt_waypoints': gt_waypoints_rel,
            'pose': np.array([rx, ry, ryaw], dtype=np.float32) # Essential for Post-Processing
        }
        
        
        if not self.config.ignore_rear and self.current_rear_image_raw is not None:
            obs_dict['image_rear_raw'] = self.current_rear_image_raw
        
        return obs_dict
        
    def _find_closest_unvisited_waypoint(self):
        """
        Finds the next unvisited waypoint. It prioritizes searching for the closest
        waypoint within the robot's current lane. If the lane is finished or the robot
        is not in a lane, it falls back to finding the globally closest unvisited waypoint.
        """
        if self.current_odom is None:
            return None

        robot_pos = self.current_odom.pose.pose.position
        current_lane_idx = None

        # --- Step 1: Determine the current lane from the last visited or current target waypoint ---
        if self.previous_waypoint_index is not None:
            current_lane_idx = self.waypoints[self.previous_waypoint_index].get('original_lane_index')
        elif self.target_waypoint_index is not None:
            # A less reliable but useful fallback for initialization
            current_lane_idx = self.waypoints[self.target_waypoint_index].get('original_lane_index')

        # --- Step 2: Prioritized search for the closest waypoint WITHIN the current lane ---
        if current_lane_idx is not None:
            closest_in_lane_dist_sq = float('inf')
            closest_in_lane_idx = None

            # Iterate through all waypoints to find the closest one *in this specific lane*
            for i, wp in enumerate(self.waypoints):
                if not self.visited_waypoints[i] and wp.get('original_lane_index') == current_lane_idx:
                    dist_sq = (wp['x'] - robot_pos.x)**2 + (wp['y'] - robot_pos.y)**2
                    if dist_sq < closest_in_lane_dist_sq:
                        closest_in_lane_dist_sq = dist_sq
                        closest_in_lane_idx = i
            
            # If a suitable waypoint was found in the current lane, we're done. Return it.
            if closest_in_lane_idx is not None:
                return closest_in_lane_idx

            # If we reach here, it means the current lane is complete.
            # We log this and proceed to the global fallback search to find the next lane.
            self.get_logger().info(f"Lane {current_lane_idx} complete. Falling back to global closest waypoint search.")

        # --- Step 3: Fallback to globally closest waypoint search ---
        # This code runs if we are not in a defined lane (e.g., at the start) or
        # if the prioritized search above found no unvisited waypoints in the current lane.
        closest_dist_sq = float('inf')
        candidate_indices = []
        
        for i, visited in enumerate(self.visited_waypoints):
            if not visited:
                wp = self.waypoints[i]
                dist_sq = (wp['x'] - robot_pos.x)**2 + (wp['y'] - robot_pos.y)**2
                
                # Standard logic for finding the closest point globally
                if dist_sq < closest_dist_sq - 1e-6:
                    closest_dist_sq = dist_sq
                    candidate_indices = [i]
                elif abs(dist_sq - closest_dist_sq) < 1e-6: # Handle ties
                    candidate_indices.append(i)
        
        # If candidates were found, tie-break by choosing the waypoint with the smallest index
        if candidate_indices:
            return min(candidate_indices)
            
        return None # Return None if no unvisited waypoints are left at all
    
    def _dynamic_waypoint_lookahead(self):
        """
        Checks if the robot is closer to a subsequent waypoint (index +/- 2) than the 
        current target. If so, it skips the intermediate waypoints and updates the target.
        This prevents 'locking' onto a waypoint the robot has already passed or drifted by.
        """
        if self.target_waypoint_index is None or self.current_odom is None:
            return

        robot_pos = self.current_odom.pose.pose.position
        
        # Calculate distance to currently locked target
        current_wp = self.waypoints[self.target_waypoint_index]
        current_dist_sq = (current_wp['x'] - robot_pos.x)**2 + (current_wp['y'] - robot_pos.y)**2
        
        current_lane = current_wp.get('original_lane_index')
        
        # Define search window: +/- 2 indices. 
        # This allows for incremental or decremental path flows.
        candidates = [-1, 1, -2, 2]
        
        best_shortcut_idx = None
        # We start with the current distance; we only switch if we find something strictly closer
        best_shortcut_dist_sq = current_dist_sq 

        for offset in candidates:
            check_idx = self.target_waypoint_index + offset
            
            # 1. Bounds check
            if 0 <= check_idx < len(self.waypoints):
                cand_wp = self.waypoints[check_idx]
                
                # 2. Visited check: Never jump back to a waypoint we've already marked visited
                if self.visited_waypoints[check_idx]:
                    continue
                
                # 3. Lane check: CRITICAL per requirements.
                # Only shortcut if the candidate is in the SAME lane.
                # This prevents jumping to a parallel lane if the index diff is small but spatial diff is large.
                cand_lane = cand_wp.get('original_lane_index')
                if cand_lane != current_lane:
                    continue

                # 4. Distance check
                cand_dist_sq = (cand_wp['x'] - robot_pos.x)**2 + (cand_wp['y'] - robot_pos.y)**2
                
                # Logic: If candidate is closer than current target
                # AND is within a reasonable absolute radius (e.g. 3.0m) to prevent 
                # weird geometry jumps (like loop-backs).
                if cand_dist_sq < best_shortcut_dist_sq and cand_dist_sq < (3.0**2):
                    best_shortcut_idx = check_idx
                    best_shortcut_dist_sq = cand_dist_sq

        # If we found a better target
        if best_shortcut_idx is not None:
            # Determine direction of skip (forward or backward in array)
            step_dir = 1 if best_shortcut_idx > self.target_waypoint_index else -1
            
            skipped_count = 0
            # Mark the OLD target and any intermediate waypoints as visited
            # We range from current to new (exclusive of new)
            for i in range(self.target_waypoint_index, best_shortcut_idx, step_dir):
                if not self.visited_waypoints[i]:
                    self.visited_waypoints[i] = True
                    self.num_waypoints_visited_current_episode += 1
                    skipped_count += 1
            
            self.get_logger().info(f"Dynamic Update: Skipped {skipped_count} WPs. Jumped #{self.target_waypoint_index} -> #{best_shortcut_idx} (Dist: {math.sqrt(best_shortcut_dist_sq):.2f}m)")

            # Update state
            self.previous_waypoint_index = self.target_waypoint_index # Track logic needs previous
            self.target_waypoint_index = best_shortcut_idx
            self.last_distance_to_target = math.sqrt(best_shortcut_dist_sq)
            
            # Immediately refresh local goals for the observation
            self._update_local_goals()
            
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
                self.get_logger().info(f"REWARD_FN: Reached waypoint #{wp_reached_idx}.  Visited: {self.num_waypoints_visited_current_episode}/{self.num_waypoints_total}")
                self.get_logger().info(f"  - Waypont lane index: {wp_just_reached.get('original_lane_index', 'N/A')}")
     
                
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
                            # log to check when is called
                            
                            self.target_waypoint_index = next_sequential_idx
                        else:
                            self.target_waypoint_index = self.original_target_after_turn_idx
                            self.original_target_after_turn_idx = None
                    else:
                        # We reached a normal (non-assist) waypoint
                        if self.is_turning:
                            self.get_logger().info("U-turn sequence completed.")
                            self.is_turning = False
                        potential_next_target_idx = self._find_closest_unvisited_waypoint()
                        if potential_next_target_idx is not None:
                            lane_reached = wp_just_reached.get('original_lane_index', -1)
                            lane_next = self.waypoints[potential_next_target_idx].get('original_lane_index', -2)
                            
                            # Check for a lane change, which triggers a U-turn.
                            # This happens when we are at the END of a lane.
                            if is_boundary_wp and lane_reached != lane_next and self.previous_waypoint_index is not None:
                                self.get_logger().info(f"END OF LANE {lane_reached}: Initiating turn.")
                                self.is_turning = True

                               
                                potential_wp = self.waypoints[potential_next_target_idx]
                                start_of_next_lane_idx = potential_next_target_idx
                                
                                if not potential_wp.get('is_lane_boundary', False):
                                    self.get_logger().warn(
                                        f"Closest waypoint #{potential_next_target_idx} in new lane {lane_next} is not a boundary. "
                                        f"Searching for the correct boundary entry point."
                                    )

                                    # Use the new helper to find the TRUE entry point
                                    corrected_idx = self._find_closest_boundary_of_lane(lane_next, robot_pos)
                                    
                                    if corrected_idx is not None:
                                        self.get_logger().info(f"Correction successful. New target for U-turn is #{corrected_idx}.")
                                        potential_next_target_idx = corrected_idx
                                    else:
                                        self.get_logger().error(
                                            f"CRITICAL: Could not find any unvisited boundary for lane {lane_next}. "
                                            f"Proceeding with potentially incorrect waypoint #{potential_next_target_idx}."
                                        )
                                # ################################################################
                                # ## END OF NEW VALIDATION AND CORRECTION LOGIC ##
                                # ###############################################################
                                # 1. RESET the counter. This signifies a turn has started.
                                self.boundary_crossing_counter = 0
                                
                                # 2. SET the distant goal to be the START of the next lane.
                                self.distant_goal_world_coords = self.waypoints[potential_next_target_idx]
                                self.get_logger().info(f"New distant goal is START of lane {lane_next}.")

                                self.original_target_after_turn_idx = potential_next_target_idx
                                num_turn_wps_added = self._generate_dubins_uturn_waypoints(self.previous_waypoint_index, wp_reached_idx, potential_next_target_idx)
                                # self._plot_waypoint_debug_state(
                                #     trigger_wp_idx=wp_reached_idx,
                                #     next_lane_target_idx=potential_next_target_idx
                                # )
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
        close_windows() # Call the helper function to destroy all OpenCV windows.
        
        if self.cmd_vel_pub: self.destroy_publisher(self.cmd_vel_pub)
        if self.odom_sub: self.destroy_subscription(self.odom_sub)
        if self.scan_sub: self.destroy_subscription(self.scan_sub)
        if self.camera_sub: self.destroy_subscription(self.camera_sub)
        if self.lidar_3d_sub: self.destroy_subscription(self.lidar_3d_sub)
        ### NEW: Destroy depth subscriber ###
        if hasattr(self, 'depth_sub') and self.depth_sub: self.destroy_subscription(self.depth_sub)
        if hasattr(self, 'rear_camera_sub') and self.rear_camera_sub: self.destroy_subscription(self.rear_camera_sub)
        if hasattr(self, 'mapper'):
            self.mapper.save_3d_map("final_session_map.ply")
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


    ### NEW/MODIFIED HELPER FUNCTIONS ###
        
    def _get_local_coords_from_world_point(self, world_point: dict | None) -> tuple[float, float]:
        """
        Computes the local-frame (x,y) of a world_point robustly.
        Accepts either:
        - dict-like: {'x':..., 'y':...}
        - object-like: has attributes .x and .y (e.g. geometry_msgs/Point)
        Returns (0.0, 0.0) if data is unavailable.
        This explicitly rotates the (dx,dy) by -robot_yaw: local = R(-yaw) * (dx,dy).
        """
        if self.current_odom is None or world_point is None:
            return (0.0, 0.0)

        # get robot pose yaw
        _roll, _pitch, robot_yaw = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)

        # extract x,y robustly (support dicts and objects)
        try:
            wx = world_point['x']
            wy = world_point['y']
        except Exception:
            # fallback to attribute access
            wx = getattr(world_point, 'x', None)
            wy = getattr(world_point, 'y', None)

        if wx is None or wy is None:
            # last fallback: might be a waypoint object with different keys
            try:
                wx = float(world_point.x)
                wy = float(world_point.y)
            except Exception:
                return (0.0, 0.0)

        dx = wx - self.current_odom.pose.pose.position.x
        dy = wy - self.current_odom.pose.pose.position.y

        # rotate by -yaw explicitly (world -> robot)
        cosy = math.cos(-robot_yaw)
        siny = math.sin(-robot_yaw)
        local_x = dx * cosy - dy * siny
        local_y = dx * siny + dy * cosy

        return (local_x, local_y)


    
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



    def _find_end_of_lane_master(self, lane_index: int) -> tuple[dict | None, int | None]:
        """
        Finds the last waypoint marked as a boundary for a specific lane from the master list.
        Returns the waypoint dictionary and its index.
        """
        end_wp = None
        end_idx = None
        for i, wp in enumerate(self.master_waypoints):
            if (wp.get('original_lane_index') == lane_index and wp.get('is_lane_boundary', False)):
                end_wp = wp # The last one found for this lane will be the end
                end_idx = i
        return end_wp.copy() if end_wp else None, end_idx

    def _find_closest_master_waypoint(self, position: dict) -> tuple[dict | None, int | None]:
        """
        Finds the closest waypoint from the master list to a given world position.
        Returns the waypoint dictionary and its index.
        """
        closest_dist_sq = float('inf')
        closest_wp = None
        closest_idx = None
        for i, wp in enumerate(self.master_waypoints):
            dist_sq = (wp['x'] - position['x'])**2 + (wp['y'] - position['y'])**2
            if dist_sq < closest_dist_sq:
                closest_dist_sq = dist_sq
                closest_wp = wp
                closest_idx = i
        return closest_wp, closest_idx


    def _calculate_dubins_path(self, prev_wp_idx, end_of_lane_wp_idx, start_of_next_lane_wp_idx):
        """
        Calculates a Dubins path for a U-turn.
        This version robustly finds the adjacent waypoint in the new lane to determine the correct
        entry orientation, making it independent of waypoint ordering.
        """
        try:
            # --- Ensure all required indices are valid ---
            if prev_wp_idx is None or end_of_lane_wp_idx is None or start_of_next_lane_wp_idx is None:
                self.get_logger().error("[DubinsCalc] Failed: One or more waypoint indices are None.")
                return []

            # --- Get the waypoint data ---
            wp_prev = self.waypoints[prev_wp_idx]
            wp_end_lane = self.waypoints[end_of_lane_wp_idx]
            wp_start_next_lane = self.waypoints[start_of_next_lane_wp_idx]

            # --- Define the start configuration (q0) ---
            x0, y0 = wp_end_lane['x'], wp_end_lane['y']
            dx_start, dy_start = wp_end_lane['x'] - wp_prev['x'], wp_end_lane['y'] - wp_prev['y']
            
            # Use current robot yaw if waypoints are identical, otherwise calculate from vector
            if abs(dx_start) < 1e-6 and abs(dy_start) < 1e-6:
                if self.current_odom: 
                    _, _, yaw_start = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
                else: 
                    self.get_logger().warn("[DubinsCalc] Could not determine start yaw, prev and end waypoints are identical.")
                    return []
            else: 
                yaw_start = math.atan2(dy_start, dx_start)
            q0 = (x0, y0, yaw_start)

            # --- Define the end configuration (q1) ---
            x1, y1 = wp_start_next_lane['x'], wp_start_next_lane['y']
            
            # ### START OF MODIFIED LOGIC ###
            # If the target is a boundary, we must find its neighbor *inside* the new lane
            # to determine the correct entry angle for the U-turn.
            is_turn_to_boundary = wp_start_next_lane.get('is_lane_boundary', False)
            #search the actual boundary if is not
            if not is_turn_to_boundary:
                actual_boundary_idx = self._find_closest_boundary_of_lane(wp_start_next_lane.get('original_lane_index', -1), wp_start_next_lane)
                if actual_boundary_idx is not None:
                    start_of_next_lane_wp_idx = actual_boundary_idx
                    wp_start_next_lane = self.waypoints[start_of_next_lane_wp_idx]
                    x1, y1 = wp_start_next_lane['x'], wp_start_next_lane['y']
                    is_turn_to_boundary = True
                    self.get_logger().info(f"[DubinsCalc] Corrected target to actual boundary waypoint #{start_of_next_lane_wp_idx}.")
                else:
                    self.get_logger().warn(f"[DubinsCalc] Could not find any boundary waypoint for lane {wp_start_next_lane.get('original_lane_index', -1)}. Proceeding with original target #{start_of_next_lane_wp_idx}.")
            if is_turn_to_boundary:
                self.get_logger().info(f"[DubinsCalc] Target #{start_of_next_lane_wp_idx} is a boundary. Searching for adjacent non-boundary waypoint.")
                
                adjacent_wp_idx = None
                
                # Candidate 1: Check index - 1
                idx_minus_1 = start_of_next_lane_wp_idx - 1
                if 0 <= idx_minus_1 < len(self.waypoints):
                    # If the waypoint at -1 is NOT a boundary, it's the one we want.
                    if not self.waypoints[idx_minus_1].get('is_lane_boundary', False):
                        adjacent_wp_idx = idx_minus_1
                        self.get_logger().info(f"[DubinsCalc] Adjacent waypoint is at index -1 (#{adjacent_wp_idx}).")

                # Candidate 2: Check index + 1 if the first candidate was also a boundary or was invalid
                if adjacent_wp_idx is None:
                    idx_plus_1 = start_of_next_lane_wp_idx + 1
                    if 0 <= idx_plus_1 < len(self.waypoints):
                         # If the waypoint at +1 is NOT a boundary, it's the one we want.
                        if not self.waypoints[idx_plus_1].get('is_lane_boundary', False):
                            adjacent_wp_idx = idx_plus_1
                            self.get_logger().info(f"[DubinsCalc] Adjacent waypoint is at index +1 (#{adjacent_wp_idx}).")

                # If we still haven't found a valid adjacent waypoint, we cannot calculate the turn.
                if adjacent_wp_idx is None:
                    self.get_logger().error(f"[DubinsCalc] CRITICAL: Could not find a valid non-boundary adjacent waypoint for target #{start_of_next_lane_wp_idx}. Aborting turn.")
                    return []
                    
                # Calculate yaw using the correct adjacent waypoint.
                # The direction is FROM the boundary waypoint (x1, y1) TO the waypoint inside the lane.
                wp_adjacent = self.waypoints[adjacent_wp_idx]
                dx_next, dy_next = wp_adjacent['x'] - x1, wp_adjacent['y'] - y1
                yaw_end = math.atan2(dy_next, dx_next)
            else:
                # Fallback for non-boundary targets: calculate yaw in the standard forward direction.
                self.get_logger().warn(f"[DubinsCalc] Target is not a boundary. Calculating standard forward yaw.")
                next_to_target_idx = start_of_next_lane_wp_idx + 1
                if next_to_target_idx >= len(self.waypoints):
                    self.get_logger().error(f"[DubinsCalc] Cannot determine end yaw, next index {next_to_target_idx} is out of bounds.")
                    return []
                
                wp_after_start_next = self.waypoints[next_to_target_idx]
                dx_next, dy_next = wp_after_start_next['x'] - x1, wp_after_start_next['y'] - y1
                yaw_end = math.atan2(dy_next, dx_next)
            # ### END OF MODIFIED LOGIC ###

            q1 = (x1, y1, yaw_end)

            # --- Generate the path ---
            path = dubins.shortest_path(q0, q1, self.turning_radius)
            configurations, _ = path.sample_many(self.turn_wp_step_distance)
            if not configurations or len(configurations) < 2:
                return []

            # --- Format the generated points into waypoint dictionaries ---
            new_turn_waypoints = []
            lane_id_of_turn = wp_end_lane.get('original_lane_index', -1)
            # We skip the very first and very last points as they are the start/end poses
            if len(configurations) > 2:
                for config in configurations[1:-1]:
                    new_turn_waypoints.append({
                        'x': float(config[0]),
                        'y': float(config[1]),
                        'original_lane_index': lane_id_of_turn,
                        'is_turn_assist_wp': True,
                        'is_planned_wp': True
                    })
            
            return new_turn_waypoints

        except Exception as e:
            self.get_logger().error(f"Error during Dubins path calculation: {e}")
            import traceback
            traceback.print_exc()
            return []
    def _generate_dubins_uturn_waypoints(self, prev_wp_idx, end_of_lane_wp_idx, start_of_next_actual_lane_wp_idx):
        """
        Calculates and INSERTS new U-turn waypoints into the environment's main waypoint list.
        This function modifies the state of the environment for the current episode.
        """
        # 1. Calculate the path using the new helper. This separates planning from execution.
        new_turn_waypoints = self._calculate_dubins_path(prev_wp_idx, end_of_lane_wp_idx, start_of_next_actual_lane_wp_idx)

        if not new_turn_waypoints:
            return 0

        # 2. If waypoints were generated, insert them into the master lists for this episode.
        insertion_point_idx = end_of_lane_wp_idx + 1
        for i, turn_wp in enumerate(new_turn_waypoints):
            # Remove the temporary 'is_planned_wp' flag before inserting
            turn_wp.pop('is_planned_wp', None)
            self.waypoints.insert(insertion_point_idx + i, turn_wp)
            self.visited_waypoints.insert(insertion_point_idx + i, False)

        # 3. Update the total waypoint count for the episode.
        self.num_waypoints_total += len(new_turn_waypoints)
        self.get_logger().info(f"Dubins: Inserted {len(new_turn_waypoints)} U-turn waypoints. New total: {self.num_waypoints_total}.")
        # debug: log a sample of generated turn waypoints (world + local)
        # for j, turn_wp in enumerate(new_turn_waypoints[:10]):  # limit to first 10 to avoid spamming
        #     try:
        #         lx, ly = self._get_local_coords_from_world_point(turn_wp)
        #     except Exception as e:
        #         lx, ly = None, None
        #     self.get_logger().info(f"[Dubins DEBUG] Generated turn WP #{j}: world=({turn_wp['x']:.3f},{turn_wp['y']:.3f}) local=({lx},{ly})")

        return len(new_turn_waypoints)

    def _find_previous_waypoint_in_lane(self, current_wp_idx: int) -> int | None:
        """Finds the index of the waypoint just before the current one in the same lane."""
        current_lane = self.waypoints[current_wp_idx].get('original_lane_index')
        if current_lane is None:
            return None
        # Search backwards from the waypoint before the current one
        for i in range(current_wp_idx - 1, -1, -1):
            if self.waypoints[i].get('original_lane_index') == current_lane:
                return i
        return None

    def _populate_local_goals_plan(self):
        """
        Intelligently fills the self.local_goal_waypoints list up to 4 waypoints.
        This function handles pre-planning U-turns when it detects a lane boundary.
        """
        while len(self.local_goal_waypoints) < 4:
            if not self.local_goal_waypoints:
                first_wp_idx = self._find_closest_unvisited_waypoint()
                if first_wp_idx is None:
                    self.get_logger().info("[LocalPlan] No unvisited waypoints available to initialize the plan.")
                    return 
                
                new_wp = self.waypoints[first_wp_idx].copy()
                new_wp['master_index'] = first_wp_idx
                self.local_goal_waypoints.append(new_wp)
                continue 

            last_real_wp_in_plan = None
            for wp in reversed(self.local_goal_waypoints):
                if not wp.get('is_planned_wp', False):
                    last_real_wp_in_plan = wp
                    break
            
            if last_real_wp_in_plan is None:
                self.get_logger().warn("[LocalPlan] Could not find a 'real' waypoint in plan to use as reference. Breaking.")
                return 

            current_plan_indices = {wp['master_index'] for wp in self.local_goal_waypoints if 'master_index' in wp}
            potential_next_wp_idx = self._find_next_waypoint_from_ref(last_real_wp_in_plan, current_plan_indices)
            
            if potential_next_wp_idx is None:
                self.get_logger().info("[LocalPlan] No more subsequent waypoints found. Plan is as full as it can be.")
                return 

            last_real_lane = last_real_wp_in_plan.get('original_lane_index')
            next_wp_data = self.waypoints[potential_next_wp_idx]
            next_lane = next_wp_data.get('original_lane_index')
            is_boundary = last_real_wp_in_plan.get('is_lane_boundary', False)

            if is_boundary and last_real_lane is not None and next_lane is not None and last_real_lane != next_lane:
                
                if not next_wp_data.get('is_lane_boundary', False):
                     ref_point = Point(x=last_real_wp_in_plan['x'], y=last_real_wp_in_plan['y'], z=0.0)
                     corrected_idx = self._find_closest_boundary_of_lane(next_lane, ref_point)
                     if corrected_idx is not None:
                         potential_next_wp_idx = corrected_idx
                     else:
                         self.get_logger().error(f"[LocalPlan] CRITICAL: Could not find boundary for lane {next_lane}.")
                
                # --- START OF CORRECTED LOGIC ---
                prev_wp_for_turn = None
                # First, try to find the previous waypoint within the current plan
                if len(self.local_goal_waypoints) > 1:
                    for i in range(len(self.local_goal_waypoints) - 2, -1, -1):
                        wp = self.local_goal_waypoints[i]
                        if not wp.get('is_planned_wp', False):
                            prev_wp_for_turn = wp
                            break
                
                # FALLBACK: If not found in the plan (e.g., at initialization), use the new helper
                if prev_wp_for_turn is None:
                    self.get_logger().info("[LocalPlan] Previous WP not in plan. Using fallback search.")
                    prev_wp_master_idx = self._find_previous_waypoint_in_lane(last_real_wp_in_plan['master_index'])
                    if prev_wp_master_idx is not None:
                        prev_wp_for_turn = self.waypoints[prev_wp_master_idx].copy()
                        prev_wp_for_turn['master_index'] = prev_wp_master_idx # Ensure it has the index
                
                if prev_wp_for_turn is not None:
                    self.get_logger().info(f"[LocalPlan] U-Turn detected. Planning turn from #{last_real_wp_in_plan['master_index']} to #{potential_next_wp_idx} using prev #{prev_wp_for_turn['master_index']}.")
                    planned_turn_wps = self._calculate_dubins_path(
                        prev_wp_for_turn['master_index'], last_real_wp_in_plan['master_index'], potential_next_wp_idx
                    )
                    self.local_goal_waypoints.extend(planned_turn_wps)
                else:
                    self.get_logger().warn(f"[LocalPlan] Could not determine previous waypoint to calculate U-Turn. Skipping Dubins path generation.")
                # --- END OF CORRECTED LOGIC ---

                if potential_next_wp_idx not in {wp.get('master_index') for wp in self.local_goal_waypoints}:
                    next_wp_after_turn = self.waypoints[potential_next_wp_idx].copy()
                    next_wp_after_turn['master_index'] = potential_next_wp_idx
                    self.local_goal_waypoints.append(next_wp_after_turn)
                
                continue

            next_wp = self.waypoints[potential_next_wp_idx].copy()
            next_wp['master_index'] = potential_next_wp_idx
            self.local_goal_waypoints.append(next_wp)

        final_plan_indices = [wp.get('master_index', 'P') for wp in self.local_goal_waypoints]
        

    def _initialize_local_goals(self):
        """
        Initializes the 4-waypoint plan at the start of an episode by clearing the
        old plan and calling the new intelligent populator.
        """
        self.local_goal_waypoints.clear()
        self._populate_local_goals_plan()

    def _update_local_goals(self):
        """
        Updates the 4-waypoint plan during an episode by removing the reached waypoint
        and then calling the intelligent populator to fill the plan back up to 4.
        """
        if self.local_goal_waypoints:
            self.local_goal_waypoints.pop(0)
        
        self._populate_local_goals_plan()

    def _get_relative_local_goals(self) -> np.ndarray:
        """
        Computes the coordinates of the next 4 local waypoints relative to the
        robot's current position and orientation using the generic helper function.
        """
        relative_coords = []
        # Take only the first 4 waypoints from the plan for the observation
        plan_for_obs = self.local_goal_waypoints[:4]

        for wp in plan_for_obs:
            local_coords = self._get_local_coords_from_world_point(wp)
            relative_coords.append(list(local_coords))

        # Pad the list with [0.0, 0.0] pairs if the plan has fewer than 4 waypoints
        while len(relative_coords) < 4:
            relative_coords.append([0.0, 0.0])
        
        
        # self.get_logger().debug(f"Relative local goals (4): {relative_coords}")
        return np.array(relative_coords, dtype=np.float32)


    def _find_closest_boundary_of_lane(self, lane_index: int, reference_pos: Point) -> int | None:
        """
        Searches a specific lane for the closest unvisited boundary waypoint
        relative to a given reference position (e.g., the robot's current position).

        Args:
            lane_index: The 'original_lane_index' of the lane to search within.
            reference_pos: The robot's current position message (or any Point).

        Returns:
            The integer index of the closest boundary waypoint, or None if none are found.
        """
        closest_boundary_idx = None
        min_dist_sq = float('inf')

        # Iterate through all waypoints to find candidates in the target lane
        for i, wp in enumerate(self.waypoints):
            # We are only interested in waypoints that meet ALL three criteria:
            # 1. They belong to the target lane.
            # 2. They are marked as a lane boundary.
            # 3. They have not been visited yet.
            if (wp.get('original_lane_index') == lane_index and
                wp.get('is_lane_boundary', False) and
                not self.visited_waypoints[i]):
                
                # Calculate distance from the robot to this candidate boundary waypoint
                dist_sq = (wp['x'] - reference_pos.x)**2 + (wp['y'] - reference_pos.y)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_boundary_idx = i
        self.target_waypoint_index = closest_boundary_idx
        return closest_boundary_idx


        ### NEW FUNCTION FOR DEBUGGING ###
    def _plot_waypoint_debug_state(self, trigger_wp_idx: int, next_lane_target_idx: int):
        """
        Generates and displays a plot of the current waypoint state for debugging.
        This function is intended to be called when a U-turn is about to be computed.
        It will pause the execution of the environment until the plot window is closed.

        Args:
            trigger_wp_idx: The index of the end-of-lane waypoint that triggered the turn.
            next_lane_target_idx: The index of the identified start of the next lane.
        """
        self.get_logger().info("--- Generating Waypoint Debug Plot ---")

        if self.current_odom is None:
            self.get_logger().warn("Cannot plot debug state: current_odom is None.")
            return

        # Prepare data containers
        all_pts_x, all_pts_y = [], []
        visited_pts_x, visited_pts_y = [], []
        boundary_pts_x, boundary_pts_y = [], []
        lane_pts_x, lane_pts_y = [], [] # For the lane being exited
        assist_pts_x, assist_pts_y = [], [] # For Dubins waypoints
        #PRINT LOCAL POSITIONS OF WAYPOIN
        trigger_wp = self.waypoints[trigger_wp_idx]
        current_lane_id = trigger_wp.get('original_lane_index')

        # Categorize all waypoints
        for i, wp in enumerate(self.waypoints):
            all_pts_x.append(wp['x'])
            all_pts_y.append(wp['y'])

            if self.visited_waypoints[i]:
                visited_pts_x.append(wp['x'])
                visited_pts_y.append(wp['y'])
            
            if wp.get('is_lane_boundary', False):
                boundary_pts_x.append(wp['x'])
                boundary_pts_y.append(wp['y'])

            if wp.get('original_lane_index') == current_lane_id:
                lane_pts_x.append(wp['x'])
                lane_pts_y.append(wp['y'])
            
            if wp.get('is_turn_assist_wp', False):
                assist_pts_x.append(wp['x'])
                assist_pts_y.append(wp['y'])

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot different categories with different styles
        ax.scatter(all_pts_x, all_pts_y, c='lightgray', s=20, label='All Waypoints', zorder=1)
        if lane_pts_x:
            ax.scatter(lane_pts_x, lane_pts_y, c='cyan', s=30, label=f'Exiting Lane ({current_lane_id})', zorder=2)
        if assist_pts_x:
            ax.scatter(assist_pts_x, assist_pts_y, c='purple', s=35, marker='P', label='Assist WPs', zorder=3)
        if visited_pts_x:
            ax.scatter(visited_pts_x, visited_pts_y, c='orange', s=40, label='Visited', zorder=4)
        if boundary_pts_x:
            ax.scatter(boundary_pts_x, boundary_pts_y, c='red', s=60, marker='x', label='Boundaries', zorder=5)

        # Highlight key waypoints
        next_target_wp = self.waypoints[next_lane_target_idx]
        ax.plot(trigger_wp['x'], trigger_wp['y'], 'm*', markersize=20, label=f'Turn Trigger WP #{trigger_wp_idx}', zorder=6)
        ax.plot(next_target_wp['x'], next_target_wp['y'], 'g*', markersize=20, label=f'Next Lane Target WP #{next_lane_target_idx}', zorder=6)

        # Plot robot position and orientation
        robot_pos = self.current_odom.pose.pose.position
        _roll, _pitch, robot_yaw = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
        ax.plot(robot_pos.x, robot_pos.y, 'bo', markersize=12, label='Robot Position', zorder=7)
        ax.arrow(robot_pos.x, robot_pos.y, 
                 0.5 * math.cos(robot_yaw), 0.5 * math.sin(robot_yaw),
                 head_width=0.1, head_length=0.15, fc='blue', ec='blue', zorder=7)

        # Add labels and title for clarity
        ax.set_title("Waypoint State at U-Turn Trigger")
        ax.set_xlabel("World X Coordinate")
        ax.set_ylabel("World Y Coordinate")
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box') # Very important for correct spatial representation
        
        self.get_logger().info("Displaying debug plot. Close the plot window to continue simulation...")
        plt.show() # This will pause execution until the window is closed