#!/usr/bin/env python3
import pickle
import cv2
import numpy as np
import torch
import os
import sys
import random
import math

# --- CONFIGURATION ---
DATASET_PATH = "/home/fede/ros2_ws/drl_datasets/imitation_learning/180_auxiliary.pkl"
# DATASET_PATH = "path/to/your/dataset.pkl" 

# Re-implementation of Config for standalone running
class DebugConfig:
    def __init__(self):
        self.camera_fov = 130.0
        self.img_resolution = (160, 704) # H, W
        self.lidar_resolution_width = 256
        self.lidar_resolution_height = 256
        self.bev_resolution_width = 160
        self.bev_resolution_height = 160
        self.pixels_per_meter = 8.0
        self.lidar_pos = [1.3, 0.0, 2.5]
        
        # Augmentation settings
        self.augment = True
        self.inv_augment_prob = 0.0 # Force augmentation for debug
        self.aug_max_rotation = 20.0 # Degrees

        # Toggles matching your training config
        self.use_aux_depth = True
        self.use_aux_bev = True
        self.use_aux_semantic = True

# --- UTILITIES ---

def lidar_to_histogram_features(point_cloud_np, crop=256):
    """Standard BEV Histogram generation"""
    if point_cloud_np.shape[0] == 0:
         return np.zeros((crop, crop, 2), dtype=np.uint8)
    x_meters = 2.0
    y_max_meters = 2.0
    pixels_per_meter = 64
    hist_max_per_pixel = 5
    x_bins = np.linspace(-x_meters, x_meters, int(x_meters * 2 * pixels_per_meter) + 1)
    y_bins = np.linspace(-y_max_meters, y_max_meters, int(y_max_meters * 2 * pixels_per_meter) + 1)
    below = point_cloud_np[point_cloud_np[..., 2] <= -0.25] # approximate ground
    above = point_cloud_np[point_cloud_np[..., 2] > -0.25]
    below_hist = np.histogram2d(below[..., 1], below[..., 0], bins=(y_bins, x_bins))[0]
    above_hist = np.histogram2d(above[..., 1], above[..., 0], bins=(y_bins, x_bins))[0]
    below_hist[below_hist > hist_max_per_pixel] = hist_max_per_pixel
    above_hist[above_hist > hist_max_per_pixel] = hist_max_per_pixel
    below_features = (below_hist / hist_max_per_pixel * 255).astype(np.uint8)
    above_features = (above_hist / hist_max_per_pixel * 255).astype(np.uint8)
    features = np.stack([np.flipud(below_features.T), np.flipud(above_features.T)], axis=-1)
    features = np.fliplr(features)
    h, w, c = features.shape
    output = np.zeros((crop, crop, 2), dtype=np.uint8)
    start_h = (crop - h) // 2
    start_w = (crop - w) // 2
    output[start_h:start_h+h, start_w:start_w+w, :] = features
    return output

def draw_target_point(target_point_world, crop=256):
    x_meters = 2.0
    y_max_meters = 2.0
    pixels_per_meter = 64
    image = np.zeros((crop, crop), dtype=np.uint8)
    robot_x, robot_y = target_point_world
    feature_map_height = int(x_meters * 2 * pixels_per_meter)
    feature_map_width = int(y_max_meters * 2 * pixels_per_meter)
    pixel_col = int((-robot_y + y_max_meters) * pixels_per_meter)
    pixel_row = int(feature_map_height - 1 - ((robot_x + x_meters) * pixels_per_meter))
    start_h = (crop - feature_map_height) // 2
    start_w = (crop - feature_map_width) // 2
    final_row = start_h + pixel_row
    final_col = start_w + pixel_col
    if 0 <= final_row < crop and 0 <= final_col < crop:
        cv2.circle(image, (final_col, final_row), radius=5, color=255, thickness=-1)
    return image.reshape(1, crop, crop).astype(np.float32) / 255.0

def convert_bev_img_to_classes(bev_img_np):
    bev_gray = bev_img_np[:, :, 0] 
    classes = np.zeros_like(bev_gray, dtype=np.uint8)
    classes[(bev_gray > 50) & (bev_gray < 200)] = 1 # Drivable (Gray)
    classes[bev_gray >= 200] = 2 # Obstacle (White)
    # Viz mapping for debugging: 0=Black, 1=Green, 2=Red
    viz = np.zeros((*classes.shape, 3), dtype=np.uint8)
    viz[classes == 1] = [0, 255, 0]
    viz[classes == 2] = [0, 0, 255]
    return viz

def preprocess_front_camera_image(image_np, target_w, target_h, crop_shift=0):
    CAMERA_FULL_FOV = 130.0 
    SIDE_BUFFER_DEG = 20.0
    source_h, source_w, _ = image_np.shape
    effective_fov = CAMERA_FULL_FOV - 2 * SIDE_BUFFER_DEG
    crop_w = int((effective_fov / CAMERA_FULL_FOV) * source_w)
    target_aspect_ratio = target_w / target_h
    crop_h = int(crop_w / target_aspect_ratio)
    center_x = source_w / 2
    center_y = source_h / 2
    start_x = int(center_x - (crop_w / 2) + crop_shift)
    start_y = int(center_y - (crop_h / 2))
    panoramic_crop = image_np[start_y:start_y+crop_h, start_x:start_x+crop_w]
    resized = cv2.resize(panoramic_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_depth_image(image_np, target_w, target_h, crop_shift=0):
    # Identical to RGB but nearest neighbor and clipping
    if image_np.ndim == 3: image_np = np.squeeze(image_np, axis=-1)
    CAMERA_FULL_FOV = 130.0 
    SIDE_BUFFER_DEG = 20.0
    source_h, source_w = image_np.shape
    effective_fov = CAMERA_FULL_FOV - 2 * SIDE_BUFFER_DEG
    crop_w = int((effective_fov / CAMERA_FULL_FOV) * source_w)
    target_aspect_ratio = target_w / target_h
    crop_h = int(crop_w / target_aspect_ratio)
    center_x = source_w / 2
    center_y = source_h / 2
    start_x = int(center_x - (crop_w / 2) + crop_shift)
    start_y = int(center_y - (crop_h / 2))
    panoramic_crop = image_np[start_y:start_y+crop_h, start_x:start_x+crop_w]
    resized = cv2.resize(panoramic_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    MAX_DEPTH = 10.0
    normalized = np.clip(resized, 0.0, MAX_DEPTH) / MAX_DEPTH
    return normalized

def preprocess_semantic_image(image_np, target_w, target_h, crop_shift=0):
    # Identical to RGB but NEAREST NEIGHBOR for classes
    CAMERA_FULL_FOV = 130.0 
    SIDE_BUFFER_DEG = 20.0
    source_h, source_w = image_np.shape
    crop_w = int(((CAMERA_FULL_FOV - 2 * SIDE_BUFFER_DEG) / CAMERA_FULL_FOV) * source_w)
    target_aspect_ratio = target_w / target_h
    crop_h = int(crop_w / target_aspect_ratio)
    start_x = int((source_w / 2) - (crop_w / 2) + crop_shift)
    start_y = int((source_h / 2) - (crop_h / 2))
    panoramic_crop = image_np[start_y:start_y+crop_h, start_x:start_x+crop_w]
    resized = cv2.resize(panoramic_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized

# --- MAIN DEBUG LOOP ---

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset)} frames.")
    
    config = DebugConfig()
    
    for i, raw_obs in enumerate(dataset):
        print(f"\n--- Frame {i} ---")
        print("Keys available:", raw_obs.keys())
        
        # 1. Augmentation Logic
        degree = 0.0
        rad = 0.0
        if config.augment:
            degree = (random.random() * 2. - 1.) * config.aug_max_rotation
            rad = np.deg2rad(degree)
            print(f"Applied Rotation: {degree:.2f} degrees")

        # 2. Process RGB
        img_raw = raw_obs['image_raw']
        crop_shift = degree / config.camera_fov * img_raw.shape[1]
        img_proc = preprocess_front_camera_image(img_raw, config.img_resolution[1], config.img_resolution[0], crop_shift)
        
        # 3. Process LiDAR BEV (Input)
        pc_raw = raw_obs['lidar_raw']
        # Rotate PC
        rot_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pc_rot = pc_raw.copy()
        pc_rot[:, :2] = (rot_matrix @ pc_raw[:, :2].T).T
        lidar_hist = lidar_to_histogram_features(pc_rot)
        # Visualize Histogram: Red=Above, Blue=Below
        lidar_viz = np.zeros((256, 256, 3), dtype=np.uint8)
        lidar_viz[:, :, 0] = lidar_hist[:, :, 0] # Blue channel
        lidar_viz[:, :, 2] = lidar_hist[:, :, 1] # Red channel

        # 4. Process Depth
        depth_viz = np.zeros_like(img_proc)
        if 'depth_raw' in raw_obs:
            depth_raw = raw_obs['depth_raw']
            depth_proc = preprocess_depth_image(depth_raw, config.img_resolution[1], config.img_resolution[0], crop_shift)
            depth_viz = (depth_proc * 255).astype(np.uint8)
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
        
        # 5. Process BEV Semantic (Target)
        bev_sem_viz = np.zeros((160, 160, 3), dtype=np.uint8)
        if 'bev_semantic' in raw_obs:
            bev_raw = raw_obs['bev_semantic']
            h, w = bev_raw.shape[:2]
            # Rotate
            M = cv2.getRotationMatrix2D((w//2, h//2), degree, 1.0)
            bev_rot = cv2.warpAffine(bev_raw, M, (w, h), flags=cv2.INTER_NEAREST)
            # Convert to classes then back to RGB for viz
            bev_classes = convert_bev_img_to_classes(bev_rot) # (H, W, 3)
            bev_sem_viz = cv2.resize(bev_classes, (160, 160), interpolation=cv2.INTER_NEAREST)

        # 6. Process Front Semantic
        sem_viz = np.zeros_like(img_proc)
        if 'semantic_raw' in raw_obs:
            sem_raw = raw_obs['semantic_raw']
            sem_proc = preprocess_semantic_image(sem_raw, config.img_resolution[1], config.img_resolution[0], crop_shift)
            # Colorize: 0=Black, 1=Green, 2=Red
            sem_viz[sem_proc == 1] = [0, 255, 0]
            sem_viz[sem_proc == 2] = [0, 0, 255]

        # --- DISPLAY COMPOSITE ---
        
        # Row 1: RGB | Semantic | Depth
        row1 = np.hstack([img_proc, sem_viz, depth_viz])
        
        # Row 2: LiDAR Input | BEV Target
        # Resize BEVs to match height of row 1 somewhat for cleaner look
        target_h = 256
        lidar_viz_resized = cv2.resize(lidar_viz, (target_h, target_h))
        bev_sem_viz_resized = cv2.resize(bev_sem_viz, (target_h, target_h))
        
        # Create blank spacer to fill width
        spacer_width = row1.shape[1] - (target_h * 2)
        spacer = np.zeros((target_h, spacer_width, 3), dtype=np.uint8)
        
        row2 = np.hstack([lidar_viz_resized, bev_sem_viz_resized, spacer])
        
        # Combine
        # Resize row2 width to match row1 exactly if needed due to integer math
        row2 = cv2.resize(row2, (row1.shape[1], target_h))
        
        final = np.vstack([row1, row2])
        
        cv2.imshow("Dataset Debugger (Press 'n' for next, 'q' to quit)", final)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()