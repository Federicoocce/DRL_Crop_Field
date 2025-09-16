# transfuser_utils.py

import numpy as np
from PIL import Image
import cv2

def scale_and_crop_image(image_np, scale=1, crop=256):
    """
    Adapts the paper's scale_and_crop_image function to work with NumPy arrays
    from ROS messages instead of loading from a file.
    
    Args:
        image_np (np.array): Input image as a NumPy array (H, W, C).
    
    Returns:
        np.array: Cropped image as a NumPy array (crop, crop, C).
    """
    # Convert NumPy array to PIL Image to use the original logic
    image_pil = Image.fromarray(image_np)
    
    (width, height) = (int(image_pil.width // scale), int(image_pil.height // scale))
    im_resized = image_pil.resize((width, height))
    
    image = np.asarray(im_resized)
    
    # The original code's crop logic
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    
    cropped_image = image[start_x:start_x + crop, start_y:start_y + crop]
    
    # The original returned (C, H, W), but our Gym space is (H, W, C).
    # The cropped_image is already in the correct (H, W, C) format.
    return cropped_image

def lidar_to_histogram_features(point_cloud_np, crop=256):
    """
    Converts a LiDAR point cloud to a 2-channel Bird's-Eye-View (BEV) histogram.
    This is a direct adaptation of the paper's `lidar_to_histogram_features` and
    `splat_points` functions, configured for our ROS environment's coordinate system.
    
    Args:
        point_cloud_np (np.array): LiDAR point cloud as an Nx3 NumPy array (X, Y, Z).
                                   Assumes ROS standard: X is forward, Y is left, Z is up.
    
    Returns:
        np.array: A (crop, crop, 2) BEV map.
    """
    # --- Parameters from the paper's `splat_points` function ---
    # The paper's coordinate system seems to be Y-forward, X-sideways.
    # We adapt it for ROS: X-forward, Y-sideways.
    
    # Range of points to consider in the robot's frame.
    # The paper uses a 32m x 32m grid.
    x_max_meters = 2.0  # Forward range
    y_max_meters = 1.0  # Side range (+/-)
    
    # Resolution of the BEV grid
    pixels_per_meter = crop / (y_max_meters * 2) # e.g., 256 / 32 = 8 pixels/meter
    hist_max_per_pixel = 5
    
    # Height threshold for splitting points into "below" and "above" channels
    # The paper uses -2.0; we use a more general 0.0 for the ground plane.
    z_threshold = 0.0

    # Define the boundaries of our BEV grid in meters
    x_bins = np.linspace(0, x_max_meters, int(x_max_meters * pixels_per_meter) + 1)
    y_bins = np.linspace(-y_max_meters, y_max_meters, int(y_max_meters * 2 * pixels_per_meter) + 1)

    # Split points based on height
    below = point_cloud_np[point_cloud_np[..., 2] <= z_threshold]
    above = point_cloud_np[point_cloud_np[..., 2] > z_threshold]

    # Create histograms for each channel
    # Note: `histogram2d`'s first argument is x, second is y.
    # In ROS, our point cloud is (X-fwd, Y-left). We want the histogram's
    # Y-axis to represent forward motion and X-axis for side motion.
    below_hist = np.histogram2d(below[..., 1], below[..., 0], bins=(y_bins, x_bins))[0]
    above_hist = np.histogram2d(above[..., 1], above[..., 0], bins=(y_bins, x_bins))[0]

    # Normalize and clip the histograms
    below_hist[below_hist > hist_max_per_pixel] = hist_max_per_pixel
    above_hist[above_hist > hist_max_per_pixel] = hist_max_per_pixel
    
    below_features = (below_hist / hist_max_per_pixel * 255).astype(np.uint8)
    above_features = (above_hist / hist_max_per_pixel * 255).astype(np.uint8)

    # The histogram has X (sides) as rows and Y (front) as columns.
    # We want it as an image where Y (front) is rows. So we transpose.
    # We also flip the forward axis so that 0 is at the bottom of the image.
    features = np.stack([np.flipud(below_features.T), np.flipud(above_features.T)], axis=-1)

    # Ensure the output size is exactly as expected, padding if necessary
    # This can happen due to discretization in the histogram binning
    h, w, c = features.shape
    output = np.zeros((crop, crop, 2), dtype=np.uint8)
    output[0:h, 0:w, :] = features
    
    return output

### NEW VISUALIZATION FUNCTIONS ###

def render_sensor_data(camera_image, lidar_bev):
    """
    Displays the camera image and the LiDAR BEV in pop-up windows.
    This function contains all the OpenCV logic.
    """
    try:
        if camera_image is not None and lidar_bev is not None:
            # --- Camera Image Visualization ---
            img_bgr = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
            img_display = cv2.resize(img_bgr, (400, 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("Camera View", img_display)

            # --- BEV Visualization ---
            bev_h, bev_w, _ = lidar_bev.shape
            bev_display = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
            
            # Channel 0 (below points) -> Mapped to Blue channel
            bev_display[:, :, 0] = lidar_bev[:, :, 0]
            # Channel 1 (above points) -> Mapped to Red channel
            bev_display[:, :, 2] = lidar_bev[:, :, 1]
            
            bev_display_resized = cv2.resize(bev_display, (400, 400), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("LiDAR BEV", bev_display_resized)

            cv2.waitKey(1)
    except Exception as e:
        # Using print here as this is a utility file without a ROS logger
        print(f"Error in render_sensor_data: {e}")

def close_windows():
    """Closes all OpenCV windows."""
    cv2.destroyAllWindows()