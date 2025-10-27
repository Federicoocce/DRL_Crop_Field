# ==============================================================================
# --- FINAL, SELF-CONTAINED COORDINATE ALIGNMENT TEST SCRIPT ---
# This script has ZERO dependencies on your project files. It is guaranteed to run.
# ==============================================================================

import numpy as np
import cv2
import math

# --- The functions to test are now COPIED directly into this script ---
# --- This version uses the final, correct 32x32 meter view. ---

def lidar_to_histogram_features(point_cloud_np, crop=256):
    """
    Converts LiDAR to a BEV histogram.
    VIEW: Centered on robot, 32x32 meters.
    RESOLUTION: 8 pixels/meter.
    """
    if point_cloud_np.shape[0] == 0:
         return np.zeros((crop, crop, 2), dtype=np.uint8)

    x_meters = 16.0
    y_max_meters = 16.0
    pixels_per_meter = 8
    hist_max_per_pixel = 5
    z_threshold = 0.0

    x_bins = np.linspace(-x_meters, x_meters, int(x_meters * 2 * pixels_per_meter) + 1)
    y_bins = np.linspace(-y_max_meters, y_max_meters, int(y_max_meters * 2 * pixels_per_meter) + 1)

    below = point_cloud_np[point_cloud_np[..., 2] <= z_threshold]
    above = point_cloud_np[point_cloud_np[..., 2] > z_threshold]

    below_hist = np.histogram2d(below[..., 1], below[..., 0], bins=(y_bins, x_bins))[0]
    above_hist = np.histogram2d(above[..., 1], above[..., 0], bins=(y_bins, x_bins))[0]

    below_hist[below_hist > hist_max_per_pixel] = hist_max_per_pixel
    above_hist[above_hist > hist_max_per_pixel] = hist_max_per_pixel
    
    below_features = (below_hist / hist_max_per_pixel * 255).astype(np.uint8)
    above_features = (above_hist / hist_max_per_pixel * 255).astype(np.uint8)

    features = np.stack([np.flipud(below_features.T), np.flipud(above_features.T)], axis=-1)

    h, w, c = features.shape
    output = np.zeros((crop, crop, 2), dtype=np.uint8)
    start_h = (crop - h) // 2
    start_w = (crop - w) // 2
    output[start_h:start_h+h, start_w:start_w+w, :] = features
    
    return output

def draw_target_point(target_point_world, crop=256):
    """
    Draws a target point on a BEV canvas.
    MUST MATCH lidar_to_histogram_features parameters.
    VIEW: Centered on robot, 32x32 meters.
    RESOLUTION: 8 pixels/meter.
    """
    x_meters = 16.0
    y_max_meters = 16.0
    pixels_per_meter = 8

    image = np.zeros((crop, crop), dtype=np.uint8)
    robot_x_forward, robot_y_left = target_point_world
    
    feature_map_height = int(x_meters * 2 * pixels_per_meter)
    feature_map_width = int(y_max_meters * 2 * pixels_per_meter)

    pixel_col_feature = int((-robot_y_left + y_max_meters) * pixels_per_meter)
    
    shifted_x = robot_x_forward + x_meters
    pixel_row_feature = int(feature_map_height - 1 - (shifted_x * pixels_per_meter))
    
    start_h = (crop - feature_map_height) // 2
    start_w = (crop - feature_map_width) // 2
    
    final_pixel_row = start_h + pixel_row_feature
    final_pixel_col = start_w + pixel_col_feature
    
    point_pixel = (final_pixel_col, final_pixel_row)

    if 0 <= final_pixel_row < crop and 0 <= final_pixel_col < crop:
        cv2.circle(image, point_pixel, radius=5, color=255, thickness=-1)
    
    image = image.reshape(1, crop, crop)
    return image.astype(np.float32) / 255.0

# --- The actual test logic ---

def run_test():
    print("Running coordinate system alignment test...")

    # Create a FAKE LiDAR "wall" 5 meters in front of the robot.
    wall_x = 5.0
    y_coords = np.linspace(-3.0, 3.0, 100)
    x_coords = np.full_like(y_coords, wall_x)
    z_coords = np.zeros_like(y_coords) # At ground level
    fake_lidar_points = np.stack([x_coords, y_coords, z_coords], axis=-1)

    # Create a FAKE target point on that wall.
    fake_target_point = np.array([5.0, 2.0]) # 5m forward, 2m left

    # Process the data using the functions copied above
    lidar_bev = lidar_to_histogram_features(fake_lidar_points)
    target_bev = draw_target_point(fake_target_point)

    # Create a visual output for verification
    h, w, _ = lidar_bev.shape
    debug_image = np.zeros((h, w, 3), dtype=np.uint8)

    debug_image[:, :, 0] = lidar_bev[:, :, 0] # Blue channel = LiDAR ground points
    debug_image[:, :, 1] = (target_bev.squeeze() * 255).astype(np.uint8) # Green channel = Target point

    print("\nDisplaying debug image. Press any key to exit.")
    print("--- EXPECTATION: You should see a green dot appearing ON TOP of a blue line. ---")

    cv2.imshow("Coordinate Alignment Test", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Test finished.")

if __name__ == '__main__':
    run_test()