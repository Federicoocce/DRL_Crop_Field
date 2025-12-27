import os
import ujson
from skimage.transform import rotate
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from pathlib import Path
import cv2
import random
from copy import deepcopy
import io
from .config import config

from .utils import get_vehicle_to_virtual_lidar_transform, get_vehicle_to_lidar_transform, get_lidar_to_vehicle_transform, get_lidar_to_bevimage_transform


def get_depth(data):
    """
    Computes the normalized depth
    """
    data = np.transpose(data, (1,2,0))
    data = data.astype(np.float32)

    normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
    normalized /=  (256 * 256 * 256 - 1)
    # in_meters = 1000 * normalized
    #clip to 50 meters
    normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
    normalized = normalized * 20.0 # Rescale map to lie in [0,1]

    return normalized


def get_waypoints(labels, len_labels):
    assert(len(labels) == len_labels)
    num = len_labels
    waypoints = {}
    
    for result in labels[0]:
        car_id = result["id"]
        waypoints[car_id] = [[result['ego_matrix'], True]]
        for i in range(1, num):
            for to_match in labels[i]:
                if to_match["id"] == car_id:
                    waypoints[car_id].append([to_match["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints

# this is only for visualization, For training, we should use vehicle coordinate

def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    T = get_vehicle_to_virtual_lidar_transform()
    
    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix
            
    return waypoints

def align(lidar_0, degree=0):
    """
    Applies a 2D rotation to the LiDAR point cloud for data augmentation.
    This version is simplified for ROS and removes all CARLA-specific transforms.

    Args:
        lidar_0 (np.array): The input LiDAR point cloud (Nx3 or Nx4).
        degree (float): The rotation angle in degrees.

    Returns:
        np.array: The rotated LiDAR point cloud.
    """
    # Create a 2D rotation matrix from the augmentation angle.
    # In ROS (X-fwd, Y-left), a positive rotation is counter-clockwise.
    rad = np.deg2rad(degree)
    degree_matrix_2d = np.array([[np.cos(rad), -np.sin(rad)],
                                 [np.sin(rad),  np.cos(rad)]])

    # Apply the rotation to the X and Y coordinates of the point cloud.
    # The Z coordinate (lidar_0[:, 2]) remains unchanged.
    lidar_rotated_xy = (degree_matrix_2d @ lidar_0[:, :2].T).T
    
    # Re-assemble the point cloud with the rotated XY and original Z/Intensity
    rotated_lidar = np.hstack((lidar_rotated_xy, lidar_0[:, 2:]))
    
    return rotated_lidar

def lidar_to_histogram_features(point_cloud_np, crop=256):
    """
    Converts a (potentially rotated) LiDAR point cloud to a BEV histogram.
    The BEV is now centered on the robot, covering -16m to +16m forward.
    """
    if point_cloud_np.shape[0] == 0:
         return np.zeros((crop, crop, 2), dtype=np.uint8)

    # --- Parameters (Modified for Centered BEV) ---
    # ==================== START OF THE FIX ====================
    # Define the forward/backward range. The total range is 32 meters.
    x_meters = 2.0
    # ===================== END OF THE FIX =====================
    y_max_meters = 2.0  # Side range (+/-) remains the same
    pixels_per_meter = 64
    hist_max_per_pixel = 5
    z_threshold = 0.0

    # Define boundaries
    # ==================== START OF THE FIX ====================
    # Create bins from -16 to +16 meters for the forward axis
    x_bins = np.linspace(-x_meters, x_meters, int(x_meters * 2 * pixels_per_meter) + 1)
    # ===================== END OF THE FIX =====================
    y_bins = np.linspace(-y_max_meters, y_max_meters, int(y_max_meters * 2 * pixels_per_meter) + 1)

    # Split & Histogram
    below = point_cloud_np[point_cloud_np[..., 2] <= z_threshold]
    above = point_cloud_np[point_cloud_np[..., 2] > z_threshold]

    below_hist = np.histogram2d(below[..., 1], below[..., 0], bins=(y_bins, x_bins))[0]
    above_hist = np.histogram2d(above[..., 1], above[..., 0], bins=(y_bins, x_bins))[0]

    # Normalize & Clip
    below_hist[below_hist > hist_max_per_pixel] = hist_max_per_pixel
    above_hist[above_hist > hist_max_per_pixel] = hist_max_per_pixel
    
    below_features = (below_hist / hist_max_per_pixel * 255).astype(np.uint8)
    above_features = (above_hist / hist_max_per_pixel * 255).astype(np.uint8)

    # Orient correctly (Y-forward as rows, 0 at bottom)
    features = np.stack([np.flipud(below_features.T), np.flipud(above_features.T)], axis=-1)
        # The histogram2d and transpose process results in a mirrored left/right axis.
    # We must flip it horizontally to match the draw_target_point coordinate system.
    features = np.fliplr(features)

    # Ensure exact output size
    h, w, c = features.shape
    output = np.zeros((crop, crop, 2), dtype=np.uint8)
    # Center result if bins resulted in smaller image
    start_h = (crop - h) // 2
    start_w = (crop - w) // 2
    output[start_h:start_h+h, start_w:start_w+w, :] = features
    
    return output

# --- SHARED HELPER FUNCTION ---
def _project_and_clip_to_bev(local_point, crop=256, pixels_per_meter=64, x_meters_range=2.0, y_meters_range=2.0):
    """
    Projects a local coordinate (x_forward, y_left) to pixel coordinates.
    If the point is outside the BEV box, it clips it to the edge while maintaining direction.
    """
    rx, ry = local_point

    # 1. Calculate map dimensions based on range
    feature_map_height = int(x_meters_range * 2 * pixels_per_meter)
    feature_map_width = int(y_meters_range * 2 * pixels_per_meter)

    # 2. Calculate center offsets (assuming square crop)
    start_h = (crop - feature_map_height) // 2
    start_w = (crop - feature_map_width) // 2

    # 3. Calculate Raw Pixel Coordinates (Unclamped)
    # Map Y (Left/Right) -> Col. Positive Y is Left, which maps to Column 0.
    raw_col = start_w + int((-ry + y_meters_range) * pixels_per_meter)
    
    # Map X (Forward/Back) -> Row. Positive X is Forward, which maps to Row 0.
    raw_row = start_h + int(feature_map_height - 1 - ((rx + x_meters_range) * pixels_per_meter))

    # 4. Clipping Logic
    center_x = crop / 2.0
    center_y = crop / 2.0
    
    vec_x = raw_col - center_x
    vec_y = raw_row - center_y
    
    # Margin to keep the drawn shape fully inside the image
    margin = 5 
    half_size = (crop / 2.0) - margin
    
    final_col = raw_col
    final_row = raw_row
    is_out_of_bounds = False

    # Check if vector magnitude exceeds box half-size
    if abs(vec_x) > half_size or abs(vec_y) > half_size:
        is_out_of_bounds = True
        scale = float('inf')
        
        if vec_x != 0:
            scale = min(scale, abs(half_size / vec_x))
        if vec_y != 0:
            scale = min(scale, abs(half_size / vec_y))
            
        final_col = center_x + vec_x * scale
        final_row = center_y + vec_y * scale

    return int(final_col), int(final_row), is_out_of_bounds


# --- UPDATED DRAW TARGET POINT (For Network Input) ---
def draw_target_point(target_point_world, crop=256):
    """
    Draws the immediate target point on the input tensor.
    """
    image = np.zeros((crop, crop), dtype=np.uint8)

    col, row, is_clipped = _project_and_clip_to_bev(
        target_point_world, crop=crop, pixels_per_meter=64, x_meters_range=2.0, y_meters_range=2.0
    )

    radius = 5 if not is_clipped else 4
    if 0 <= col < crop and 0 <= row < crop:
        cv2.circle(image, (col, row), radius=radius, color=255, thickness=-1)
    
    image = image.reshape(1, crop, crop)
    return image.astype(np.float32) / 255.0


# --- UPDATED DISTANT GOAL DRAWING (For Semantic Label) ---
def draw_distant_goal_on_bev(bev_map, goal_local, crop=256, pixels_per_meter=64, x_meters_range=2.0, y_meters_range=2.0):
    """
    Draws the Distant Goal (Class 3) on the semantic BEV label.
    Uses exact same projection/clipping logic as draw_target_point.
    """
    if goal_local is None:
        return bev_map

    col, row, is_clipped = _project_and_clip_to_bev(
        goal_local, crop=crop, pixels_per_meter=pixels_per_meter, 
        x_meters_range=x_meters_range, y_meters_range=y_meters_range
    )

    # Visual Style for Label: Square Box
    box_size = 6 if not is_clipped else 4
    
    top_left = (col - box_size, row - box_size)
    bottom_right = (col + box_size, row + box_size)
    
    # Draw Class ID 3 (Goal)
    # Note: This modifies the map in-place
    cv2.rectangle(bev_map, top_left, bottom_right, color=(3), thickness=-1)
    
    return bev_map

def debug_visualize_batch(batch):
    """
    Displays the first sample of a batch for debugging in separate windows.
    - Shows the RGB camera input.
    - Shows the BEV input fed to the model, with a corrected orientation and
      an intuitive color mapping for human understanding:
      - RED:   LiDAR points ABOVE the split height (obstacles).
      - GREEN: The target point image.
      - BLUE:  LiDAR points BELOW the split height (ground).
    """
    # --- Select the first item from the batch for visualization ---
    rgb_tensor = batch['rgb'][0]
    lidar_bev_tensor = batch['lidar_bev'][0]          # 2 channels: below/above points
    target_point_image = batch['target_point_image'][0] # 1 channel: target point
    target_point_coords = batch['target_point'][0]      # [x, y] coordinates for text

    # --- 1. Process and Display RGB Image ---
    rgb_numpy = rgb_tensor.cpu().numpy()
    rgb_numpy = np.transpose(rgb_numpy, (1, 2, 0)) # C, H, W -> H, W, C
    rgb_numpy = (rgb_numpy * 255).astype(np.uint8)
    rgb_display = cv2.cvtColor(rgb_numpy, cv2.COLOR_RGB2BGR) # PyTorch is RGB, OpenCV is BGR

    # Add target point coordinates as text on the image
    coords = target_point_coords.cpu().numpy()
    text = f"Target WP: ({coords[0]:.2f}m, {coords[1]:.2f}m)"
    cv2.putText(rgb_display, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("RGB Camera Input", rgb_display)

    # --- 2. Process and Display the BEV Input with Corrected Orientation and Colors ---
    
    # Unpack the two LiDAR channels and the target point image.
    # The data from the preprocessor is already oriented with "forward" pointing "up".
    # No rotation is needed for visualization.
    lidar_below_viz = lidar_bev_tensor[0].cpu().numpy() # Channel 0 is 'below' points
    lidar_above_viz = lidar_bev_tensor[1].cpu().numpy() # Channel 1 is 'above' points
    target_img_viz = target_point_image[0].cpu().numpy()

    # Create an empty 3-channel image (BGR for OpenCV)
    h, w = lidar_above_viz.shape
    bev_display = np.zeros((h, w, 3), dtype=np.uint8)

    # Map each data channel to a unique BGR color channel.
    # This color scheme makes it easy to interpret the BEV.
    bev_display[:, :, 0] = (lidar_below_viz * 255).astype(np.uint8)  # Blue Channel: Ground points
    bev_display[:, :, 1] = (target_img_viz * 255).astype(np.uint8)   # Green Channel: Target waypoint
    bev_display[:, :, 2] = (lidar_above_viz * 255).astype(np.uint8)  # Red Channel: Obstacle points

    # Update the window title to be descriptive of the color mapping
    cv2.imshow("BEV Input (R=Obstacles, G=Target, B=Ground)", bev_display)
    
    # Update windows and allow for a small delay
    cv2.waitKey(50)


def debug_augmentation_visualize(original_batch, augmented_batch, original_degree, augmented_degree, original_wps, augmented_wps, original_sem=None, augmented_sem=None):
    """
    Displays a side-by-side comparison of an original and augmented data sample.
    Draws the full 4-waypoint trajectory graphically on the BEV.
    """
    # --- Helper function to prepare a single BEV image for display ---
    def create_bev_display_image(batch, degree, waypoints):
        lidar_bev_tensor = batch['lidar_bev'][0]
        target_point_image = batch['target_point_image'][0]
        
        lidar_below_viz = lidar_bev_tensor[0].cpu().numpy()
        lidar_above_viz = lidar_bev_tensor[1].cpu().numpy()
        target_img_viz = target_point_image[0].cpu().numpy()

        h, w = lidar_above_viz.shape
        bev_display = np.zeros((h, w, 3), dtype=np.uint8)
        
        bev_display[:, :, 0] = (lidar_below_viz * 255).astype(np.uint8)  # Blue: Ground
        bev_display[:, :, 1] = (target_img_viz * 255).astype(np.uint8)   # Green: Target
        bev_display[:, :, 2] = (lidar_above_viz * 255).astype(np.uint8)  # Red: Obstacles

        # ==================== START OF THE FIX ====================
        # --- Draw the waypoint trajectory graphically on the BEV ---
        
        # 1. Define BEV parameters (must match lidar_to_histogram_features)
        x_meters = 2.0
        y_max_meters = 2.0
        pixels_per_meter = 64
        crop = h

        # 2. Convert all waypoints from meters to pixel coordinates
        pixel_points = []
        for wp in waypoints:
            robot_x_forward, robot_y_left = wp
            
            feature_map_height = int(x_meters * 2 * pixels_per_meter)
            feature_map_width = int(y_max_meters * 2 * pixels_per_meter)
            
            pixel_col = int((-robot_y_left + y_max_meters) * pixels_per_meter)
            pixel_row = int(feature_map_height - 1 - ((robot_x_forward + x_meters) * pixels_per_meter))
            
            pixel_points.append((pixel_col, pixel_row))

        # 3. Draw the trajectory line connecting the waypoints
        # cv2.polylines expects a list of arrays, so we reshape.
        pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(bev_display, [pts], isClosed=False, color=(0, 255, 255), thickness=2) # Yellow line

        # 4. Draw a circle at each waypoint to make them distinct
        for point in pixel_points:
            cv2.circle(bev_display, point, radius=4, color=(0, 165, 255), thickness=-1) # Orange circles

        # Add text for the rotation degree
        rot_text = f"Rot: {degree:.2f} deg"
        cv2.putText(bev_display, rot_text, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # ===================== END OF THE FIX =====================

        return bev_display

    # --- Helper function to prepare a single RGB image for display (unchanged) ---
    def create_rgb_display_image(batch, degree):
        # ... (This helper function remains unchanged)
        rgb_tensor = batch['rgb'][0]
        target_point_coords = batch['target_point'][0]

        rgb_numpy = rgb_tensor.cpu().numpy()
        rgb_numpy = np.transpose(rgb_numpy, (1, 2, 0))
        rgb_numpy = (rgb_numpy * 255).astype(np.uint8)
        rgb_display = cv2.cvtColor(rgb_numpy, cv2.COLOR_RGB2BGR)

        coords = target_point_coords.cpu().numpy()
        text_tgt = f"Target: ({coords[0]:.2f}, {coords[1]:.2f})"
        text_rot = f"Rot: {degree:.2f} deg"
        cv2.putText(rgb_display, text_tgt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(rgb_display, text_rot, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return rgb_display
    
    def create_sem_display_image(sem_class_grid, degree):
        h, w = sem_class_grid.shape
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color Mapping
        # 0 (Unk)   -> Black
        # 1 (Drive) -> Gray
        # 2 (Obs)   -> White
        # 3 (Goal)  -> Magenta (Fixes the black square issue)
        
        viz[sem_class_grid == 1] = [100, 100, 100]
        viz[sem_class_grid == 2] = [255, 255, 255]
        viz[sem_class_grid == 3] = [255, 0, 255] # BGR: Magenta
        
        cv2.putText(viz, f"Rot: {degree:.2f}", (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return viz

    # --- Create and show the four images ---
    
    # Pass the full waypoint arrays (plural wps) to the BEV helper
    original_rgb_display = create_rgb_display_image(original_batch, original_degree)
    original_bev_display = create_bev_display_image(original_batch, original_degree, original_wps)
    cv2.imshow("Original RGB", original_rgb_display)
    cv2.imshow("Original BEV (Orange Dots = Waypoints)", original_bev_display)

    augmented_rgb_display = create_rgb_display_image(augmented_batch, augmented_degree)
    augmented_bev_display = create_bev_display_image(augmented_batch, augmented_degree, augmented_wps)
    cv2.imshow("Augmented RGB", augmented_rgb_display)
    cv2.imshow("Augmented BEV (Orange Dots = Waypoints)", augmented_bev_display)
    # --- Display Logic ---
    if original_sem is not None and augmented_sem is not None:
        orig_sem_viz = create_sem_display_image(original_sem, original_degree)
        aug_sem_viz = create_sem_display_image(augmented_sem, augmented_degree)
        
        cv2.imshow("Orig Semantic Label", orig_sem_viz)
        cv2.imshow("Aug Semantic Label", aug_sem_viz)

    cv2.waitKey(50)

def render_sensor_data(camera_image, lidar_bev, camera_image_rear=None, uncertainty=None):
    """
    Displays sensor data in pop-up windows.
    Args:
        camera_image (np.array): The front camera image (H, W, 3).
        lidar_bev (np.array): The processed LiDAR BEV (H, W, 2).
        camera_image_rear (np.array, optional): The rear camera image.
        uncertainty (float, optional): Aleatoric Uncertainty (Sigma) in meters.
    """
    try:
        # --- Front Camera Visualization ---
        if camera_image is not None:
            img_bgr = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
            img_display = cv2.resize(img_bgr, (400, 400), interpolation=cv2.INTER_AREA)
            
            # --- NEW: Draw Uncertainty Overlay ---
            if uncertainty is not None:
                # Convert to cm for easier reading
                unc_cm = uncertainty * 100.0 
                
                # Color logic: Green if < 20cm, Yellow < 40cm, Red > 40cm
                if unc_cm < 20.0:
                    text_color = (0, 255, 0) # Green
                elif unc_cm < 40.0:
                    text_color = (0, 255, 255) # Yellow
                else:
                    text_color = (0, 0, 255) # Red

                # Draw a background box for readability
                cv2.rectangle(img_display, (10, 10), (220, 50), (0, 0, 0), -1)
                
                # Text: "Unc: 8.25 cm"
                text = f"Unc: {unc_cm:.2f} cm"
                cv2.putText(img_display, text, (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            cv2.imshow("Front Camera View", img_display)

        # --- Rear Camera Visualization ---
        if camera_image_rear is not None:
            img_rear_bgr = cv2.cvtColor(camera_image_rear, cv2.COLOR_RGB2BGR)
            img_rear_display = cv2.resize(img_rear_bgr, (400, 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("Rear Camera View", img_rear_display)

        # --- BEV Visualization ---
        if lidar_bev is not None:
            if lidar_bev.ndim == 3 and lidar_bev.shape[2] == 2:
                bev_h, bev_w, _ = lidar_bev.shape
                bev_display = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
                bev_display[:, :, 0] = lidar_bev[:, :, 0] # Blue
                bev_display[:, :, 2] = lidar_bev[:, :, 1] # Red
                
                bev_display_resized = cv2.resize(bev_display, (400, 400), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("LiDAR BEV", bev_display_resized)

        cv2.waitKey(1)

    except Exception as e:
        print(f"Error in render_sensor_data: {e}")

def close_windows():
    cv2.destroyAllWindows()