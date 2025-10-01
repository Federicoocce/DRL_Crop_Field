# transfuser_utils.py

import numpy as np
from PIL import Image
import cv2
import math

# ==========================================
# NEW: Augmentation / Rotation Functions
# ==========================================


def rotate_point_cloud(point_cloud_np, angle_deg):
    """
    Rotates a point cloud around the Z-axis (yaw).
    Args:
        point_cloud_np (np.array): (N, 3) array of points (x, y, z).
        angle_deg (float): Rotation angle in degrees.
    Returns:
        np.array: Rotated (N, 3) point cloud.
    """
    if abs(angle_deg) < 1e-6 or point_cloud_np.shape[0] == 0:
        return point_cloud_np

    rad = np.deg2rad(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    # Standard 2D rotation matrix applied to X and Y
    # [ x']   [ cos -sin ] [ x ]
    # [ y'] = [ sin  cos ] [ y ]
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a,  cos_a]], dtype=np.float32)

    # Separate XY and Z
    xy_points = point_cloud_np[:, :2]
    z_points = point_cloud_np[:, 2:] # Keep shape (N, 1)

    # Apply rotation (transpose needed for matrix multiplication shapes)
    rotated_xy = (rotation_matrix @ xy_points.T).T

    # Recombine
    return np.concatenate([rotated_xy, z_points], axis=1)

def rotate_waypoints(waypoints_np, angle_deg):
    """
    Rotates relative waypoints around the robot (origin 0,0).
    Args:
        waypoints_np (np.array): (T, 2) array of waypoints (x, y).
        angle_deg (float): Rotation angle in degrees.
    Returns:
        np.array: Rotated (T, 2) waypoints.
    """
    if abs(angle_deg) < 1e-6:
        return waypoints_np

    rad = np.deg2rad(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a,  cos_a]], dtype=np.float32)

    # Apply rotation
    rotated_wp = (rotation_matrix @ waypoints_np.T).T
    return rotated_wp.astype(np.float32)

# ==========================================
# Existing Preprocessing Functions (Unchanged logic)
# ==========================================
### MODIFIED: Now accepts a crop_shift parameter ###
def scale_and_crop_image(image_np, scale=1, crop=256, crop_shift=0):
    """
    Processes the image by scaling and then applying a shifted center crop.
    This mimics the augmentation style from the TransFuser paper.
    """
    # 1. Scale (if necessary, though we use scale=1)
    if scale != 1:
        (h, w) = image_np.shape[:2]
        new_w, new_h = int(w // scale), int(h // scale)
        image = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        image = image_np

    # 2. Center Crop with a horizontal shift
    img_h, img_w, _ = image.shape
    
    # Calculate initial top-left corner for a center crop
    start_y = img_h // 2 - crop // 2
    start_x = img_w // 2 - crop // 2
    
    # Apply the horizontal shift
    start_x += int(crop_shift)

    # Ensure crop window is within image bounds
    start_x = np.clip(start_x, 0, img_w - crop)
    start_y = np.clip(start_y, 0, img_h - crop)
    
    end_x = start_x + crop
    end_y = start_y + crop
    
    # Perform the crop
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    # Final check to ensure size is exactly as expected
    if cropped_image.shape[0] != crop or cropped_image.shape[1] != crop:
        padded = np.zeros((crop, crop, 3), dtype=np.uint8)
        h_c, w_c, _ = cropped_image.shape
        padded[:h_c, :w_c, :] = cropped_image
        return padded

    return cropped_image
def lidar_to_histogram_features(point_cloud_np, crop=256):
    """
    Converts a (potentially rotated) LiDAR point cloud to a BEV histogram.
    """
    if point_cloud_np.shape[0] == 0:
         return np.zeros((crop, crop, 2), dtype=np.uint8)

    # --- Parameters (Same as before) ---
    x_max_meters = 2.0  # Forward range
    y_max_meters = 1.0  # Side range (+/-)
    pixels_per_meter = crop / (y_max_meters * 2) 
    hist_max_per_pixel = 5
    z_threshold = 0.0

    # Define boundaries
    x_bins = np.linspace(0, x_max_meters, int(x_max_meters * pixels_per_meter) + 1)
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

    # Ensure exact output size
    h, w, c = features.shape
    output = np.zeros((crop, crop, 2), dtype=np.uint8)
    # Center result if bins resulted in smaller image
    start_h = (crop - h) // 2
    start_w = (crop - w) // 2
    output[start_h:start_h+h, start_w:start_w+w, :] = features
    
    return output

# ... (render_sensor_data and close_windows remain unchanged) ...
def render_sensor_data(camera_image, lidar_bev):
    """Displays the camera image and the LiDAR BEV in pop-up windows."""
    try:
        if camera_image is not None:
            # Handle both raw (large) and processed (256x256) images
            img_bgr = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
            # Resize for display purposes only
            img_display = cv2.resize(img_bgr, (400, 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("Camera View", img_display)

        if lidar_bev is not None:
            # Only display if it's the processed BEV (H, W, 2)
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