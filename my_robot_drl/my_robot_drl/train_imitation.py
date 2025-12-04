# train_imitation.py (MODIFIED VERSION)

import rclpy
import os
import sys
import torch
import math
import numpy as np
import random
from std_srvs.srv import Empty
import time
import wandb
from torch import optim
import cv2
import pickle
import argparse
from tqdm import tqdm

# --- Main Project Imports ---
from .drl_env import MaizeNavigationEnv
from .model import LidarCenterNet
from .config import GlobalConfig

# --- Utility Imports (Corrected) ---
# --- Utility Imports (Remove the incorrect 'align' import) ---
from .transfuser_util import (
    lidar_to_histogram_features,
    draw_target_point,
    debug_augmentation_visualize,
    draw_distant_goal_on_bev
    
)

# ===================================================================
# --- CONFIGURATION PARAMETERS ---
# ===================================================================
IL_EPOCHS = 100 # Max epochs for the initial training phase
IL_BATCH_SIZE = 32
IL_LEARNING_RATE = 1e-4
DATA_COLLECTION_FPS = 2.0
DATA_COLLECTION_FPS_TURNING = 10.0
EXPERT_KP_ANGULAR = 1.0
EXPERT_TARGET_LINEAR_VEL = 0.2
AGENT_KP_ANGULAR = 0.8
AGENT_TARGET_LINEAR_VEL = 0.2
TARGET_REWARD_THRESHOLD = 7800.0
MAX_GT_WAYPOINT_DEVIATION_X = 0.75 # meters
EARLY_STOPPING_PATIENCE = 5 # Epochs to wait for validation loss improvement
DAGGER_RETRAIN_EPOCHS = 5   # Fixed number of epochs for DAgger retraining

HOME_DIR = os.path.expanduser('~')
MODEL_SAVE_DIR = os.path.join(HOME_DIR, 'ros2_ws', 'drl_models', 'imitation_learning')
DATASET_SAVE_DIR = os.path.join(HOME_DIR, 'ros2_ws', 'drl_datasets', 'imitation_learning')
# Define separate paths for the final model and the best-performing model
FINAL_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transfuser_il_final_model.pth')
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transfuser_il_best_model.pth')
BEST_VAL_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transfuser_il_best_val_model.pth')
BEST_UNCERTAINTY_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transfuser_il_best_uncertainty_model.pth')

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transfuser_il_full_model.pth')
EXPERT_DATASET_PATH = os.path.join(DATASET_SAVE_DIR, '180_auxiliary_mixed.pkl')
TRAIN_DATASET_PATH = "/workspace/180_auxiliary_sin_cs_mixed_ss.pkl"
VAL_DATASET_PATH = "/workspace/180_auxiliary_straight.pkl"


# ===================================================================
# --- DATA PREPROCESSOR CLASS ---
# ===================================================================



# ===================================================================
# --- DATA PREPROCESSOR CLASS (CORRECTED) ---
# ===================================================================
def convert_bev_img_to_classes(bev_img_np):
    """
    Converts 3-channel BEV image to single channel class indices.
    Mapper outputs: [0,0,0] (Unk), [100,100,100] (Drive), [255,255,255] (Obs)
    """
    # Take one channel since it's grayscale
    bev_gray = bev_img_np[:, :, 0] 
    classes = np.zeros_like(bev_gray, dtype=np.int64)
    
    # Thresholds based on realtime_mapper.py values
    # Class 1 (Drivable): value 100
    classes[(bev_gray > 50) & (bev_gray < 200)] = 1
    # Class 2 (Obstacle): value 255
    classes[bev_gray >= 200] = 2
    
    return classes

def preprocess_semantic_image(image_np, target_w, target_h, crop_shift=0):
    """
    Transforms the raw semantic mask (H, W) using Nearest Neighbor interpolation.
    """
    CAMERA_FULL_FOV = 130.0 
    SIDE_BUFFER_DEG = 20.0

    source_h, source_w = image_np.shape # 2D array

    effective_fov = CAMERA_FULL_FOV - 2 * SIDE_BUFFER_DEG
    crop_w = int((effective_fov / CAMERA_FULL_FOV) * source_w)
    target_aspect_ratio = target_w / target_h
    crop_h = int(crop_w / target_aspect_ratio)

    center_x = source_w / 2
    center_y = source_h / 2
    start_x_centered = center_x - (crop_w / 2)
    start_y = int(center_y - (crop_h / 2))
    start_x = int(start_x_centered + crop_shift)
    
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    
    panoramic_crop = image_np[start_y:end_y, start_x:end_x]
    
    # Crucial: INTER_NEAREST prevents creating new class values (e.g. 1.5 between 1 and 2)
    resized_image = cv2.resize(panoramic_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Return as (H, W) - no transpose needed for 2D mask usually, 
    # unless your model expects (1, H, W), but usually CrossEntropy expects (B, H, W) targets.
    return resized_image

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
    
    def process_observation(self, raw_obs_dict):
            degree = 0.0
            rad = 0.0
            if self.config.augment and random.random() > self.config.inv_augment_prob:
                degree = (random.random() * 2. - 1.) * self.config.aug_max_rotation
                rad = np.deg2rad(degree)

            # 1. Process RGB (Front)
            img_raw = raw_obs_dict['image_raw']
            crop_shift = degree / self.config.camera_fov * img_raw.shape[1]
            target_h, target_w = self.config.img_resolution
            img_processed = preprocess_front_camera_image(img_raw, target_w, target_h, crop_shift)

            # 2. Process LiDAR and Waypoints (Augmentation)
            pc_raw = raw_obs_dict['lidar_raw']
            gt_waypoints = raw_obs_dict['gt_waypoints']
            local_command_point = raw_obs_dict['state'][0:2]

            rotation_matrix_2d = np.array([[np.cos(rad), -np.sin(rad)],
                                        [np.sin(rad),  np.cos(rad)]])

            pc_rotated = pc_raw.copy()
            pc_rotated[:, :2] = (rotation_matrix_2d @ pc_raw[:, :2].T).T
            
            lidar_bev_hwc = lidar_to_histogram_features(pc_rotated)
            lidar_bev_chw = np.transpose(lidar_bev_hwc, (2, 0, 1))
            
            gt_waypoints_rot = (rotation_matrix_2d @ gt_waypoints.T).T
            local_command_point_rot = (rotation_matrix_2d @ local_command_point.T).T
            
            # 3. Assemble Basic Data
            processed_data = {
                'rgb': torch.from_numpy(img_processed.copy()).float(),
                'lidar_bev': torch.from_numpy(lidar_bev_chw.copy()).float(),
                'target_point': torch.from_numpy(local_command_point_rot.copy()).float(),
                'ego_vel': torch.tensor([raw_obs_dict['state'][2]]).float(),
                'ego_waypoint': torch.from_numpy(gt_waypoints_rot.copy()).float(),
                'target_point_image': torch.from_numpy(draw_target_point(local_command_point_rot).copy()).float(),
            }



            # --- 4. Handle Required Arguments (Auxiliary & Placeholders) ---
            
            # LABEL (Required positional arg in forward): Dummy tensor for bounding boxes
            # We create a dummy tensor of shape (20, 7) as expected by the model's loss function
            processed_data['label'] = torch.zeros(20, 7)

            # DEPTH
            if self.config.use_aux_depth and 'depth_raw' in raw_obs_dict:
                depth_raw = raw_obs_dict['depth_raw']
                depth_processed = preprocess_depth_image(depth_raw, target_w, target_h, crop_shift)
                processed_data['depth'] = torch.from_numpy(depth_processed.copy()).float()
            else:
                processed_data['depth'] = torch.zeros(target_h, target_w).float()

            # A. Semantic BEV Label
            # We look for the 'bev_semantic' key which was added during Post-Processing
            if self.config.use_aux_bev:
                # 1. Base BEV (Drivable/Obstacle)
                if 'bev_semantic' in raw_obs_dict:
                    bev_source_img = raw_obs_dict['bev_semantic']
                    h, w = bev_source_img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, degree, 1.0) 
                    bev_rotated = cv2.warpAffine(bev_source_img, M, (w, h), flags=cv2.INTER_NEAREST)
                    bev_classes = convert_bev_img_to_classes(bev_rotated)
                else:
                    bev_classes = np.zeros((self.config.bev_resolution_height, self.config.bev_resolution_width), dtype=np.int64)

                # 2. Resize
                if bev_classes.shape[0] != self.config.bev_resolution_height:
                    bev_classes = cv2.resize(bev_classes.astype(np.uint8), 
                                        (self.config.bev_resolution_width, self.config.bev_resolution_height), 
                                        interpolation=cv2.INTER_NEAREST).astype(np.int64)

                # 3. Add Distant Goal (Class 3)
                if 'state' in raw_obs_dict:
                        # 'state' contains [goal_x, goal_y, ...] relative to robot
                        dist_goal_rel = raw_obs_dict['state'][0:2] 
                        
                        # Apply augmentation rotation
                        dist_goal_aug = (rotation_matrix_2d @ dist_goal_rel.T).T
                        
                        # --- CRITICAL FIX: Calculate PPM for the RESIZED map ---
                        # The map covers bev_x_meters (4.0m) in bev_resolution_height (160px)
                        # PPM = 160 / 4.0 = 40.0 pixels per meter
                        target_ppm = self.config.bev_resolution_height / self.config.bev_x_meters
                        
                        # Ensure map is writable uint8 for drawing
                        bev_classes_uint8 = bev_classes.astype(np.uint8)
                        
                        bev_classes_uint8 = draw_distant_goal_on_bev(
                            bev_classes_uint8, 
                            dist_goal_aug, 
                            crop=self.config.bev_resolution_height, # 160
                            pixels_per_meter=target_ppm,            # 40.0
                            x_meters_range=self.config.bev_x_meters / 2.0, # 2.0m
                            y_meters_range=self.config.bev_y_meters / 2.0  # 2.0m
                        )
                        
                        bev_classes = bev_classes_uint8.astype(np.int64)

                processed_data['bev'] = torch.from_numpy(bev_classes).long()
            else:
                # Placeholder BEV (Required to prevent TypeError)
                processed_data['bev'] = torch.zeros(self.config.bev_resolution_height, self.config.bev_resolution_width, dtype=torch.long)


            # FRONT SEMANTIC
            if self.config.use_aux_semantic and 'semantic_raw' in raw_obs_dict:
                sem_raw = raw_obs_dict['semantic_raw']
                sem_processed = preprocess_semantic_image(sem_raw, target_w, target_h, crop_shift)
                processed_data['semantic'] = torch.from_numpy(sem_processed).long()
            else:
                processed_data['semantic'] = torch.zeros(target_h, target_w, dtype=torch.long)

            return processed_data, degree
# ===================================================================
# --- DATA HANDLING FUNCTIONS ---
# ===================================================================
# ===================================================================
# --- MAP POST-PROCESSING ---
# ===================================================================

def augment_data_with_final_map(env, raw_dataset, config, logger):
    """
    Called AFTER the episode finishes. 
    The robot has stopped, but the env.mapper holds the complete map of the field.
    We iterate through the recorded trajectory and extract the "Ground Truth" BEV.
    """
    if not raw_dataset: return raw_dataset
    
    # --- ADDED LOGGING HERE ---
    logger.info(f"Generating Ground Truth BEV Labels for {len(raw_dataset)} frames using FOV: {config.lidar_fov_deg}°...")
    
    for sample in tqdm(raw_dataset, desc="Extracting Labels", leave=False):
        if 'pose' not in sample: continue
        
        rx, ry, ryaw = sample['pose']
        
        # Extract the BEV from the FULL map
        perfect_bev = env.mapper.get_local_bev(
            rx, ry, ryaw, 
            fov_deg=config.lidar_fov_deg
        )
        
        # Save as label
        sample['bev_semantic'] = perfect_bev
        
    return raw_dataset

def save_dataset(dataset, path, logger):
    """Saves the collected dataset to a file using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"Successfully saved dataset with {len(dataset)} points to: {path}")
    except Exception as e:
        logger.error(f"Error saving dataset to {path}: {e}")

def load_datasets(train_path, val_path, logger):
    """Loads the training and validation datasets from files."""
    try:
        with open(train_path, 'rb') as f:
            train_dataset = pickle.load(f)
        logger.info(f"Loaded training dataset with {len(train_dataset)} points from: {train_path}")

        with open(val_path, 'rb') as f:
            val_dataset = pickle.load(f)
        logger.info(f"Loaded validation dataset with {len(val_dataset)} points from: {val_path}")
        return train_dataset, val_dataset
    except Exception as e:
        logger.error(f"FATAL: Error loading datasets: {e}")
        sys.exit(1)

# ===================================================================
# --- NEW: STRATIFIED SUBSET CREATION ---
# ===================================================================
def create_representative_subset(dataset, num_samples=64):
    """
    Scans the dataset to create a fixed subset that balances
    difficult turns and easy straights.
    """
    print("Scanning validation dataset to create representative subset...")
    indices_with_curvature = []
    
    for i, sample in enumerate(dataset):
        # Calculate 'curvature' based on the angle of the last waypoint
        wps = sample['gt_waypoints'] # Shape (4, 2) usually
        last_wp = wps[-1]
        # Angle relative to robot (0,0)
        angle_rad = abs(math.atan2(last_wp[1], last_wp[0]))
        indices_with_curvature.append((i, angle_rad))
    
    # Sort by curvature: [Straightest .... Curviest]
    indices_with_curvature.sort(key=lambda x: x[1])
    
    # Select 50% Easy (Straights) and 50% Hard (Turns)
    half_k = num_samples // 2
    
    # Take from start of list (Low angle)
    easy_indices = [x[0] for x in indices_with_curvature[:half_k]]
    
    # Take from end of list (High angle)
    hard_indices = [x[0] for x in indices_with_curvature[-half_k:]]
    
    # Combine and shuffle so they appear mixed in batches
    final_indices = easy_indices + hard_indices
    random.shuffle(final_indices)
    
    print(f"Created subset: {len(easy_indices)} Straights + {len(hard_indices)} Turns.")
    return final_indices

# ===================================================================
# --- UPDATED GENERALIZATION TEST ---
# ===================================================================
def test_generalization_mc_dropout(model, val_dataset, config, device, num_samples=10, fixed_indices=None):
    """
    Runs MC Dropout on a specific list of indices (samples), not batches.
    """
    preprocessor = DataPreprocessor(config)
    model.train() # ENABLE DROPOUT
    
    total_uncertainty = 0.0
    num_tested = 0
    
    # If no indices provided, fallback to random (not recommended for stability)
    if fixed_indices is None:
        fixed_indices = random.sample(range(len(val_dataset)), min(64, len(val_dataset)))

    # Process in chunks to respect batch size logic (though here we build custom batches)
    # We iterate through our list of specific INDICES
    
    current_batch_indices = []
    
    # Helper to process a collected batch
    def process_mini_batch(indices):
        raw_samples = [val_dataset[idx] for idx in indices]
        processed_list = [preprocessor.process_observation(s)[0] for s in raw_samples]
        
        # Stack
        batch = {key: torch.stack([s[key] for s in processed_list]).to(device) for key in processed_list[0].keys()}
        
        inference_args = {
            'rgb': batch['rgb'], 'lidar_bev': batch['lidar_bev'],
            'target_point': batch['target_point'], 'target_point_image': batch['target_point_image'],
            'ego_vel': batch['ego_vel']
        }
        
        predictions = []
        for _ in range(num_samples):
            pred_wp, _ = model.forward_ego(**inference_args)
            predictions.append(pred_wp)
            
        predictions = torch.stack(predictions) # (N, B, 4, 2)
        std_dev = torch.std(predictions, dim=0) # (B, 4, 2)
        
        return torch.sum(std_dev).item(), std_dev.numel()

    # Loop through fixed indices and batch them up
    batch_uncertainty_sum = 0
    batch_count_sum = 0
    
    with torch.no_grad():
        for idx in fixed_indices:
            current_batch_indices.append(idx)
            
            # If batch is full, process it
            if len(current_batch_indices) >= IL_BATCH_SIZE:
                unc, count = process_mini_batch(current_batch_indices)
                batch_uncertainty_sum += unc
                batch_count_sum += count
                current_batch_indices = []
        
        # Process remaining
        if len(current_batch_indices) > 0:
            unc, count = process_mini_batch(current_batch_indices)
            batch_uncertainty_sum += unc
            batch_count_sum += count

    model.eval()
    
    if batch_count_sum == 0: return 0.0
    return batch_uncertainty_sum / batch_count_sum

def collect_expert_data(env, config, logger):
    """
    Phase 1: Navigate the field using an expert controller and record observations.
    """
    logger.info("="*50)
    logger.info(f"PHASE 1: Starting Expert Data Collection")
    logger.info(f"         Normal FPS: {DATA_COLLECTION_FPS}, Turning FPS: {DATA_COLLECTION_FPS_TURNING}")
    logger.info("="*50)

    collection_interval_normal = 1.0 / DATA_COLLECTION_FPS
    collection_interval_turning = 1.0 / DATA_COLLECTION_FPS_TURNING

    while True:
        obs, info = env.reset()
        

        current_time = time.time()

        dataset = [obs.copy()]
        
        last_collection_time = current_time
        done = False
        step = 0

        while not done:
            gt_waypoints = obs['gt_waypoints']
            angle_to_target = math.atan2(gt_waypoints[0][1], gt_waypoints[0][0])
            action = np.array([EXPERT_TARGET_LINEAR_VEL, EXPERT_KP_ANGULAR * angle_to_target], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Determine collection rate based on behavior
            current_interval = collection_interval_turning if env.is_turning else collection_interval_normal
            
            # --- Capture current time ---
            now = time.time()
            
            if now - last_collection_time >= current_interval:

                obs_to_save = obs.copy()

                
                dataset.append(obs_to_save)
                last_collection_time = now

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        if info.get("termination_reason") == "all_waypoints_visited":
            dataset = augment_data_with_final_map(env, dataset, config , logger)
            logger.info(f"SUCCESS: Expert completed the field in {step} steps.")
            logger.info(f"Final dataset size: {len(dataset)}")
            return dataset
        else:
            reason = info.get("termination_reason", "unknown")
            logger.error(f"FAILURE: Expert failed (Reason: {reason}). Restarting collection.")
            time.sleep(2)

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


def train_model(model, optimizer, config, train_dataset, val_dataset, logger, pause_client, unpause_client, env_node, run_count, global_epoch_counter, max_epochs, use_early_stopping=False):
    """
    Phase 2: Train the LidarCenterNet model offline with optional early stopping.
    """
    phase_name = "Initial Training" if use_early_stopping else f"DAgger Retraining (Run #{run_count})"
    logger.info("="*50)
    logger.info(f"PHASE 2: Starting {phase_name} for max {max_epochs} Epochs")
    logger.info(f"         Training Set: {len(train_dataset)} | Validation Set: {len(val_dataset)}")
    logger.info("="*50)
    rclpy.spin_until_future_complete(env_node, pause_client.call_async(Empty.Request()))

    preprocessor = DataPreprocessor(config)
    loss_weights = {loss_name: weight for loss_name, weight in zip(config.detailed_losses, config.detailed_losses_weights)}
    logger.info("Using the following loss weights for training:")
    for name, weight in loss_weights.items():
        if weight > 0:
            logger.info(f"  - {name}: {weight}")

    # Track best WP loss specifically for saving models
    # Trackers
    best_val_wp_loss = float('inf')
    best_mc_uncertainty = float('inf') # For uncertainty-based saving
    patience_counter = 0
    
    debug_save_dir = os.path.join(MODEL_SAVE_DIR, 'debug_viz')
    os.makedirs(debug_save_dir, exist_ok=True)
    # We create a subset of ~64 samples (or less if dataset is small)
    # This guarantees we test on Hard Turns and Easy Straights every epoch.
    subset_size = min(64, len(val_dataset))
    mc_test_indices = create_representative_subset(val_dataset, num_samples=subset_size)
    logger.info(f"Fixed MC Dropout Subset Size: {len(mc_test_indices)} (Stratified)")

    try:
        for epoch in range(max_epochs):
            # ==========================================
            # --- TRAINING PASS ---
            # ==========================================
            model.train()
            random.shuffle(train_dataset)
            
            # accumulators for logging average epoch losses
            epoch_train_total_loss = 0.0
            epoch_train_individual_losses = {k: 0.0 for k in loss_weights.keys() if loss_weights[k] > 0}
            num_train_batches = 0

            for i in range(0, len(train_dataset), IL_BATCH_SIZE):
                raw_batch = train_dataset[i:i + IL_BATCH_SIZE]
                if len(raw_batch) < IL_BATCH_SIZE: continue

                # --- START: Augmentation Debug Visualization ---
                if i % 4 == 0: 
                    first_sample_raw = raw_batch[0]
                    processed_augmented, augmented_degree = preprocessor.process_observation(first_sample_raw)
                    if abs(augmented_degree) > 19.0:
                        original_augment_state = preprocessor.config.augment
                        preprocessor.config.augment = False
                        processed_original, original_degree = preprocessor.process_observation(first_sample_raw)
                        preprocessor.config.augment = original_augment_state
                        
                        batch_original = {key: val.unsqueeze(0).to(model.device) for key, val in processed_original.items()}
                        batch_augmented = {key: val.unsqueeze(0).to(model.device) for key, val in processed_augmented.items()}
                        
                        original_wps = batch_original['ego_waypoint'][0].cpu().numpy()
                        augmented_wps = batch_augmented['ego_waypoint'][0].cpu().numpy()
                        
                        # --- MODIFIED: Extract Semantic BEV Labels (Class Indices) ---
                        original_sem_bev = batch_original['bev'][0].cpu().numpy()
                        augmented_sem_bev = batch_augmented['bev'][0].cpu().numpy()

                        debug_augmentation_visualize(
                            batch_original, batch_augmented, 
                            original_degree, augmented_degree, 
                            original_wps, augmented_wps,
                            original_sem_bev, augmented_sem_bev # <--- Pass these new arguments
                        )
                # --- END: Augmentation Debug Visualization ---

                processed_list = [preprocessor.process_observation(s)[0] for s in raw_batch] 
                batch = {key: torch.stack([s[key] for s in processed_list]).to(model.device) for key in processed_list[0].keys()}

                if 'timestamp' in batch:
                    del batch['timestamp'] 
                
                # Pass save path for model debug visualization
                batch['save_path'] = debug_save_dir 

                losses = model(**batch)
                
                total_loss = torch.tensor(0.0, device=model.device)
                
                # Aggregate losses
                for key, value in losses.items():
                    # Always track raw value if it's a known loss
                    if key in epoch_train_individual_losses:
                        epoch_train_individual_losses[key] += value.item()

                    # Apply weight for optimization
                    if key in loss_weights and loss_weights[key] > 0:
                        weighted_loss = loss_weights[key] * value
                        total_loss += weighted_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_train_total_loss += total_loss.item()
                num_train_batches += 1
            
            # Calculate Training Averages
            avg_train_total = epoch_train_total_loss / num_train_batches if num_train_batches > 0 else 0
            avg_train_individual = {k: v / num_train_batches for k, v in epoch_train_individual_losses.items() if num_train_batches > 0}

            # ==========================================
            # --- VALIDATION PASS ---
            # ==========================================
            model.eval()
            epoch_val_total_loss = 0.0
            epoch_val_individual_losses = {k: 0.0 for k in loss_weights.keys() if loss_weights[k] > 0}
            num_val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(val_dataset), IL_BATCH_SIZE):
                    raw_batch = val_dataset[i:i + IL_BATCH_SIZE]
                    if len(raw_batch) < IL_BATCH_SIZE: continue

                    processed_list = [preprocessor.process_observation(s)[0] for s in raw_batch] 
                    batch = {key: torch.stack([s[key] for s in processed_list]).to(model.device) for key in processed_list[0].keys()}
                    if 'timestamp' in batch:
                        del batch['timestamp'] 
                    # Note: We don't pass save_path here to avoid saving validation images
                    
                    losses = model(**batch)

                    total_loss = torch.tensor(0.0, device=model.device)
                    
                    for key, value in losses.items():
                        # Always track raw value
                        if key in epoch_val_individual_losses:
                            epoch_val_individual_losses[key] += value.item()

                        # Apply weight for total metric
                        if key in loss_weights and loss_weights[key] > 0:
                            weighted_loss = loss_weights[key] * value
                            total_loss += weighted_loss
                        
                    epoch_val_total_loss += total_loss.item()
                    num_val_batches += 1

            # Calculate Validation Averages
            avg_val_total = epoch_val_total_loss / num_val_batches if num_val_batches > 0 else 0
            avg_val_individual = {k: v / num_val_batches for k, v in epoch_val_individual_losses.items() if num_val_batches > 0}

            # ==========================================
            # --- LOGGING & SAVING ---
            # ==========================================

            current_global_epoch = global_epoch_counter + epoch + 1
            current_val_wp_loss = avg_val_individual.get('loss_wp', avg_val_total)

            # --- RUN MC DROPOUT TEST ---
            # Now passing the STRATIFIED list of indices
            current_uncertainty = test_generalization_mc_dropout(
                model, val_dataset, config, model.device, 
                num_samples=10, 
                fixed_indices=mc_test_indices # <--- Key Change
            )


            # 3. Construct Console Log String
            # Base info
            log_str = f"[Epoch {epoch + 1}] Train Total: {avg_train_total:.4f} | Val Total: {avg_val_total:.4f} | Val WP: {current_val_wp_loss:.4f} | MC Unc: {current_uncertainty:.4f}"
            
            # --- RESTORED LOGIC: Add specific auxiliary losses ---
            aux_keys = ['loss_bev', 'loss_depth', 'loss_semantic', 'loss_center_heatmap']
            for k in aux_keys:
                if k in avg_train_individual and avg_train_individual[k] > 1e-6:
                    log_str += f" | Tr {k.replace('loss_', '')}: {avg_train_individual[k]:.4f}"
                if k in avg_val_individual and avg_val_individual[k] > 1e-6:
                    log_str += f" | Val {k.replace('loss_', '')}: {avg_val_individual[k]:.4f}"

            logger.info(log_str)
            
            # 4. Construct WandB Dictionary
            wandb_log_data = {
                "loss_train/total_weighted": avg_train_total,
                "loss_val/total_weighted": avg_val_total,
                "loss_val/wp_loss_specific": current_val_wp_loss, # Explicit decision metric
                "generalization/mc_uncertainty": current_uncertainty
            }
            # Add individual train losses (Restored)
            for k, v in avg_train_individual.items():
                wandb_log_data[f"loss_train/{k}"] = v
            # Add individual val losses (Restored)
            for k, v in avg_val_individual.items():
                wandb_log_data[f"loss_val/{k}"] = v
                
            wandb.log(wandb_log_data, step=current_global_epoch)

            # ==========================================
            # --- DUAL SAVING LOGIC ---
            # ==========================================
            
            # 4. Dual Saving & Combined Patience Logic
            improved_this_epoch = False

            # Check A: Best WP Loss (Standard Accuracy)
            if current_val_wp_loss < best_val_wp_loss:
                best_val_wp_loss = current_val_wp_loss
                logger.info(f"    >>> New Best WP Loss: {best_val_wp_loss:.6f}. Saving to {os.path.basename(BEST_VAL_MODEL_SAVE_PATH)}")
                torch.save(model.state_dict(), BEST_VAL_MODEL_SAVE_PATH)
                improved_this_epoch = True

            # Check B: Best Uncertainty (Robustness)
            if current_uncertainty < best_mc_uncertainty:
                best_mc_uncertainty = current_uncertainty
                logger.info(f"    >>> New Best Uncertainty: {best_mc_uncertainty:.6f}. Saving to {os.path.basename(BEST_UNCERTAINTY_MODEL_SAVE_PATH)}")
                torch.save(model.state_dict(), BEST_UNCERTAINTY_MODEL_SAVE_PATH)
                improved_this_epoch = True

            # Early Stopping Check
            if use_early_stopping:
                if improved_this_epoch:
                    if patience_counter > 0:
                        logger.info(f"    Improvement detected. Resetting patience from {patience_counter} to 0.")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info(f"    No improvement in WP Loss OR Uncertainty. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered. Both metrics plateaued for {EARLY_STOPPING_PATIENCE} epochs.")
                    break
                
    finally:
        rclpy.spin_until_future_complete(env_node, unpause_client.call_async(Empty.Request()))

    return global_epoch_counter + epoch + 1

def evaluate_model(env, model, config, logger, device, global_epoch_counter):
    """
    Phase 3: Test the trained model and collect data from the run with controlled frequency.
    """
    logger.info("Starting evaluation run with controlled data collection frequency...")
    preprocessor = DataPreprocessor(config)
    raw_obs, info = env.reset()
    done = False
    total_reward = 0.0
    evaluation_data = [raw_obs.copy()] # Store the very first observation

    # --- START OF THE FIX ---
    # Add the same frequency control logic from the expert collector
    collection_interval_normal = 1.0 / DATA_COLLECTION_FPS
    collection_interval_turning = 1.0 / DATA_COLLECTION_FPS_TURNING
    last_collection_time = time.time()
    # --- END OF THE FIX ---

    model.eval()
    with torch.no_grad():
        while not done:
            if abs(raw_obs['gt_waypoints'][0][0]) > MAX_GT_WAYPOINT_DEVIATION_X:
                info["termination_reason"] = "deviated_from_path"
                break

            processed_obs, _ = preprocessor.process_observation(raw_obs)
            batch = {key: val.unsqueeze(0).to(device) for key, val in processed_obs.items()}

            inference_args = {
                'rgb': batch['rgb'],
                'lidar_bev': batch['lidar_bev'],
                'target_point': batch['target_point'],
                'target_point_image': batch['target_point_image'],
                'ego_vel': batch['ego_vel']
            }
            
            pred_wp, _ = model.forward_ego(**inference_args)

            predicted_first_wp = pred_wp[0, 0].cpu().numpy()
            angle_to_target = math.atan2(predicted_first_wp[1], predicted_first_wp[0])
            action = np.array([AGENT_TARGET_LINEAR_VEL, AGENT_KP_ANGULAR * angle_to_target], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_raw_obs, reward, terminated, truncated, info = env.step(action)
            raw_obs = next_raw_obs
            done = terminated or truncated
            total_reward += reward

            # --- START OF THE FIX ---
            # Only append data if enough time has passed, based on whether the robot is turning
            current_interval = collection_interval_turning if env.is_turning else collection_interval_normal
            if time.time() - last_collection_time >= current_interval:
                evaluation_data.append(raw_obs.copy())
                last_collection_time = time.time()
            # --- END OF THE FIX ---

    model.train()

    logger.info(f"Evaluation finished. Total Reward: {total_reward:.2f}")
    wandb.log({"total_reward": total_reward}, step=global_epoch_counter)

    successful_run = info.get("termination_reason") == "all_waypoints_visited"
    if successful_run:
        logger.info("SUCCESS! Model completed the course.")
    else:
        reason = info.get("termination_reason", "unknown")
        logger.warn(f"FAILURE. Model did not complete (Reason: {reason}).")
    
    # Generate the BEV labels now that we have the full map
    evaluation_data = augment_data_with_final_map(env, evaluation_data, config, logger)

    return successful_run, evaluation_data, total_reward

def preprocess_front_camera_image(image_np, target_w, target_h, crop_shift=0):
    """
    Transforms the raw image by taking a panoramic crop and then moving a
    sampling window within it to simulate camera panning without padding.
    """
    # These constants define the 'camera' properties for the crop.
    # The FOV of the raw sensor image (e.g., 120 degrees).
    CAMERA_FULL_FOV = 130.0 
    # The buffer we leave on each side to allow for panning (20 degrees).
    SIDE_BUFFER_DEG = 20.0

    source_h, source_w, _ = image_np.shape

    # 1. Define the dimensions of the final cropping window at the source image's scale.
    # The effective FOV of our final image is the total FOV minus the side buffers.
    effective_fov = CAMERA_FULL_FOV - 2 * SIDE_BUFFER_DEG
    
    # Calculate the width of this effective FOV window in pixels.
    crop_w = int((effective_fov / CAMERA_FULL_FOV) * source_w)
    
    # Calculate the height required to maintain the target aspect ratio.
    target_aspect_ratio = target_w / target_h
    crop_h = int(crop_w / target_aspect_ratio)

    # 2. Calculate the crop boundaries based on the center and the shift.
    center_x = source_w / 2
    center_y = source_h / 2
    
    # Calculate the top-left corner of a perfectly centered crop.
    start_x_centered = center_x - (crop_w / 2)
    start_y = int(center_y - (crop_h / 2))
    
    # Apply the horizontal shift (pan) calculated from the augmentation degree.
    start_x = int(start_x_centered + crop_shift)
    
    # Define the final crop window's boundaries.
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    
    # 3. Extract the final crop from the source image.
    # The 20-degree buffer on each side ensures that for the maximum possible
    # crop_shift, this window will not go out of the image bounds.
    panoramic_crop = image_np[start_y:end_y, start_x:end_x]
    
    # 4. Resize the final crop to the model's required input dimensions.
    resized_image = cv2.resize(panoramic_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 5. Return as C, H, W for PyTorch.
    return np.transpose(resized_image, (2, 0, 1))

def preprocess_depth_image(image_np, target_w, target_h, crop_shift=0):
    """
    Transforms the raw depth image with the EXACT same cropping and resizing
    as the RGB image to ensure perfect alignment. Normalizes depth to [0, 1].
    """
    if image_np.ndim == 3:
        image_np = np.squeeze(image_np, axis=-1)

    CAMERA_FULL_FOV = 130.0
    SIDE_BUFFER_DEG = 20.0
    source_h, source_w = image_np.shape
    
    effective_fov = CAMERA_FULL_FOV - 2 * SIDE_BUFFER_DEG
    crop_w = int((effective_fov / CAMERA_FULL_FOV) * source_w)
    target_aspect_ratio = target_w / target_h
    crop_h = int(crop_w / target_aspect_ratio)

    center_x = source_w / 2
    center_y = source_h / 2
    start_x_centered = center_x - (crop_w / 2)
    start_y = int(center_y - (crop_h / 2))
    start_x = int(start_x_centered + crop_shift)
    
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    
    panoramic_crop = image_np[start_y:end_y, start_x:end_x]
    resized_image = cv2.resize(panoramic_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Normalize depth to [0, 1] for the loss function. Clip at a max distance.
    MAX_DEPTH_METERS = 10.0 
    resized_image = np.clip(resized_image, 0.0, MAX_DEPTH_METERS)
    normalized_depth = resized_image / MAX_DEPTH_METERS
    
    return normalized_depth # Returns (H, W)


def validate_best_model(env, config, logger, device, num_episodes=5):
    """
    Loads BOTH the Best WP model and the Best Uncertainty model and benchmarks them.
    """
    logger.info("="*60)
    logger.info(f"PHASE: COMPARATIVE VALIDATION")
    logger.info("="*60)

    # List of models to evaluate
    models_to_eval = [
        ("Best WP Loss Model", BEST_VAL_MODEL_SAVE_PATH),
        ("Best Uncertainty Model", BEST_UNCERTAINTY_MODEL_SAVE_PATH)
    ]

    model = LidarCenterNet(config, device, backbone=config.backbone, use_velocity=config.use_velocity)
    preprocessor = DataPreprocessor(config)
    
    # Disable augmentation for validation
    original_augment_setting = config.augment
    config.augment = False

    for model_name, model_path in models_to_eval:
        logger.info(f"\n--- Evaluating: {model_name} ---")
        
        if not os.path.exists(model_path):
            logger.warn(f"Model file not found: {model_path}. Skipping.")
            continue

        try:
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            continue

        success_count = 0
        total_rewards = []

        for ep in range(num_episodes):
            raw_obs, info = env.reset()
            done = False
            episode_reward = 0.0
            
            # logger.info(f"  Episode {ep + 1}/{num_episodes}...")

            with torch.no_grad():
                while not done:
                    if abs(raw_obs['gt_waypoints'][0][0]) > MAX_GT_WAYPOINT_DEVIATION_X:
                        info["termination_reason"] = "deviated_from_path"
                        break

                    processed_obs, _ = preprocessor.process_observation(raw_obs)
                    batch = {key: val.unsqueeze(0).to(device) for key, val in processed_obs.items()}

                    inference_args = {k: batch[k] for k in ['rgb', 'lidar_bev', 'target_point', 'target_point_image', 'ego_vel']}
                    pred_wp, _ = model.forward_ego(**inference_args)

                    predicted_first_wp = pred_wp[0, 0].cpu().numpy()
                    angle_to_target = math.atan2(predicted_first_wp[1], predicted_first_wp[0])
                    action = np.array([AGENT_TARGET_LINEAR_VEL, AGENT_KP_ANGULAR * angle_to_target], dtype=np.float32)
                    action = np.clip(action, env.action_space.low, env.action_space.high)

                    raw_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
            
            is_success = info.get("termination_reason") == "all_waypoints_visited"
            if is_success: success_count += 1
            total_rewards.append(episode_reward)
            # logger.info(f"    Result: {'SUCCESS' if is_success else 'FAIL'} (Rew: {episode_reward:.1f})")

        # Report Stats for this model
        success_rate = (success_count / num_episodes) * 100.0
        avg_rew = np.mean(total_rewards)
        logger.info(f"RESULTS for {model_name}:")
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        logger.info(f"  Avg Reward:   {avg_rew:.2f}")

    # Restore config
    config.augment = original_augment_setting
    logger.info("="*60)

# ===================================================================
# --- MAIN EXECUTION ---
# ===================================================================

def main(args=None):
    # ==================== START OF THE FIX ====================
    # The launch file passes ROS 2 specific arguments.
    # We use parse_known_args() to separate our script's arguments
    # from the ones added by ROS 2.
    
    parser = argparse.ArgumentParser(description="Run Imitation Learning for AgriCobots")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'collect', 'validate'],
                        help="Set the script to 'train', 'collect' or 'validate' mode.")
    # This is the key change:
    cli_args, unknown = parser.parse_known_args()
    # ===================== END OF THE FIX =====================
    config = GlobalConfig()
    rclpy.init(args=args)

    # --- MODE 1: Data Collection ---
    # Note: we now use cli_args.mode instead of args.mode
    if cli_args.mode == 'collect':
        try:
            env = MaizeNavigationEnv()
            logger = env.get_logger()
            logger.info("Running in Data Collection Mode.")
            dataset = collect_expert_data(env, config, logger)
            # NOTE: After collection, you must manually split the expert_data.pkl file
            # into train_data.pkl and val_data.pkl before running training.
            save_dataset(dataset, EXPERT_DATASET_PATH, logger)
        except Exception as e:
            print(f"An error occurred during data collection: {e}")
        finally:
            if 'env' in locals() and env: env.close()
            rclpy.shutdown()
            print("Shutdown complete.")
            sys.exit(0)

    if cli_args.mode == 'validate':
        try:
            env = MaizeNavigationEnv()
            logger = env.get_logger()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            validate_best_model(env, config, logger, device)
        except Exception as e:
            print(f"Error during validation: {e}")
        finally:
            if 'env' in locals() and env: env.close()
            rclpy.shutdown()
            print("Validation complete.")
            sys.exit(0)

    # --- MODE 2: Training ---
    # ... (the rest of your main function remains exactly the same)
    wandb.init(project="agricobots", name="il_transfuser_full_model_final")
    
    try:
        env = MaizeNavigationEnv()
        logger = env.get_logger()
        logger.info("Maize Navigation Environment for Imitation Learning created.")
    except Exception as e:
        print(f"FATAL: Error creating MaizeNavigationEnv: {e}")
        rclpy.shutdown(); sys.exit(1)

    logger.info("Connecting to Gazebo physics services...")
    pause_client = env.create_client(Empty, "/pause_physics")
    unpause_client = env.create_client(Empty, "/unpause_physics")
    while not pause_client.wait_for_service(timeout_sec=2.0): logger.warn('/pause_physics service not available, waiting...')
    while not unpause_client.wait_for_service(timeout_sec=2.0): logger.warn('/unpause_physics service not available, waiting...')
    logger.info("Gazebo services connected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GlobalConfig()
    model = LidarCenterNet(config, device, backbone=config.backbone, use_velocity=config.use_velocity)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=IL_LEARNING_RATE)

    logger.info(f"LidarCenterNet model initialized on device: {device}")
    logger.info(f"Multitask mode: {config.multitask}")
    if not config.multitask:
        logger.info("Running in WAYPOINT-ONLY training mode.")

    # --- Main Training Loop ---
    train_dataset, val_dataset = load_datasets(TRAIN_DATASET_PATH, VAL_DATASET_PATH, logger)

    # 1. Initial training phase with early stopping
    global_epoch_counter = train_model(
        model, optimizer, config, train_dataset, val_dataset, logger,
        pause_client, unpause_client, env_node=env, run_count=0,
        global_epoch_counter=0, max_epochs=IL_EPOCHS, use_early_stopping=True
    )

    # Load the best model from the initial training phase before starting DAgger
    logger.info("\n" + "="*60)
    logger.info(f"Initial training complete. Loading best model from {BEST_VAL_MODEL_SAVE_PATH} for DAgger.")
    try:
        model.load_state_dict(torch.load(BEST_VAL_MODEL_SAVE_PATH))
        model.to(device) # Ensure the model is on the correct device after loading
        logger.info("Successfully loaded best validation model.")
    except FileNotFoundError:
        logger.error(f"Could not find best validation model at {BEST_VAL_MODEL_SAVE_PATH}. Continuing with the last epoch's model.")
    except Exception as e:
        logger.error(f"Error loading best validation model: {e}. Continuing with the last epoch's model.")
    logger.info("="*60 + "\n")

    
    # --- START OF THE FIX ---
    # Initialize variables for tracking the best model and evaluation stats
    best_average_reward_so_far = -float('inf')
    run_count = 0
    success_rate = 0.0
    NUMBER_OF_EVAL_RUNS = 5
    # --- END OF THE FIX ---

    # 2. DAgger loop: evaluate, aggregate data, and retrain for a fixed number of epochs
    while success_rate < 100.0:
        run_count += 1
        logger.info("\n" + "#"*60)
        logger.info(f"STARTING IMITATION LEARNING ATTEMPT #{run_count}")
        logger.info(f"Current Best Avg Reward: {best_average_reward_so_far:.2f} | Target Reward: {TARGET_REWARD_THRESHOLD}")
        logger.info("#"*60 + "\n")
        # ==============================================================================
        # --- BENCHMARK: BEST UNCERTAINTY MODEL (Pure Evaluation, No Data Collection) ---
        # ==============================================================================
        if os.path.exists(BEST_UNCERTAINTY_MODEL_SAVE_PATH):
            logger.info("\n>>> Benchmarking 'Best Uncertainty Model' (5 Episodes - No Data Collection)...")
            try:
                # Load weights
                model.load_state_dict(torch.load(BEST_UNCERTAINTY_MODEL_SAVE_PATH))
                model.to(device)
                
                unc_rewards = []
                unc_successes = 0
                
                for i in range(NUMBER_OF_EVAL_RUNS):
                    logger.info(f"    Uncertainty Eval Run {i+1}/{NUMBER_OF_EVAL_RUNS}...")
                    # Run evaluation, ignore the returned dataset (2nd arg)
                    succ, _, rew = evaluate_model(env, model, config, logger, device, global_epoch_counter)
                    unc_rewards.append(rew)
                    if succ: unc_successes += 1
                
                unc_avg_rew = np.mean(unc_rewards)
                unc_succ_rate = (unc_successes / NUMBER_OF_EVAL_RUNS) * 100.0
                
                logger.info(f"    [Uncertainty Model Results] Success: {unc_succ_rate:.1f}% | Avg Reward: {unc_avg_rew:.2f}")
                wandb.log({
                    "benchmark/uncertainty_success_rate": unc_succ_rate,
                    "benchmark/uncertainty_avg_reward": unc_avg_rew
                }, step=global_epoch_counter)
                
            except Exception as e:
                logger.error(f"Failed to benchmark Uncertainty Model: {e}")
        else:
            logger.warn("Best Uncertainty Model not found. Skipping benchmark.")

        # --- START OF THE FIX ---
        # Perform multiple evaluation runs and collect statistics
        evaluation_rewards = []
        all_new_data = []
        successful_runs_count = 0
        
        logger.info(f"Starting {NUMBER_OF_EVAL_RUNS} evaluation runs...")
        for i in range(NUMBER_OF_EVAL_RUNS):
            logger.info(f"  --- Evaluation Run [{i + 1}/{NUMBER_OF_EVAL_RUNS}] ---")
            successful_run, new_data, current_reward = evaluate_model(env, model, config, logger, device, global_epoch_counter=global_epoch_counter)
            
            evaluation_rewards.append(current_reward)
            if successful_run:
                successful_runs_count += 1
            else:
                # Only aggregate data from failed runs
                all_new_data.extend(new_data)
        
        # Calculate and log the statistics
        average_reward = np.mean(evaluation_rewards)
        success_rate = (successful_runs_count / NUMBER_OF_EVAL_RUNS) * 100.0
        
        logger.info("\n" + "-"*60)
        logger.info(f"Evaluation Complete. Avg Reward: {average_reward:.2f}, Success Rate: {success_rate:.1f}%")
        logger.info(f"Individual rewards: {[f'{r:.2f}' for r in evaluation_rewards]}")
        logger.info("-"*60 + "\n")

        wandb.log({
            "average_reward": average_reward, 
            "success_rate": success_rate,
            "best_average_reward": best_average_reward_so_far # Log the running best
        }, step=global_epoch_counter)

        # Check for a new best model and save it
        if average_reward > best_average_reward_so_far:
            best_average_reward_so_far = average_reward
            logger.info(f"*** New best average reward achieved: {best_average_reward_so_far:.2f}! Saving model. ***")
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            try:
                torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
                logger.info(f"    Successfully saved new best model to: {BEST_MODEL_SAVE_PATH}")
            except Exception as e:
                logger.error(f"    Could not save the best model. Error: {e}")

        # Check completion criteria before deciding to retrain
        if success_rate >= 100.0 and best_average_reward_so_far >= TARGET_REWARD_THRESHOLD:
            logger.info("Success criteria met! Ending DAgger loop.")
            break

        # Aggregate the data from all failed runs for retraining
        if all_new_data:
            logger.warn(f"Aggregating {len(all_new_data)} new data points from {NUMBER_OF_EVAL_RUNS - successful_runs_count} failed runs.")
            random.shuffle(all_new_data)
            split_index = int(0.8 * len(all_new_data))
            new_train_data = all_new_data[:split_index]
            new_val_data = all_new_data[split_index:]
            train_dataset.extend(new_train_data)
            val_dataset.extend(new_val_data)
            logger.info(f"  - Added {len(new_train_data)} points to training set (new size: {len(train_dataset)})")
            logger.info(f"  - Added {len(new_val_data)} points to validation set (new size: {len(val_dataset)})")
        else:
            logger.info("All evaluation runs were successful. No new data to aggregate.")
        # --- END OF THE FIX ---
        # Retrain using the exact same logic as the initial training phase:
        # Train for up to IL_EPOCHS, but stop early if validation loss plateaus.
        # The best model from this phase is saved to the same BEST_VAL_MODEL_SAVE_PATH, overwriting the previous best.
        logger.info(f"Starting DAgger retraining phase (Run #{run_count})...")
        global_epoch_counter = train_model(
            model, optimizer, config, train_dataset, val_dataset, logger,
            pause_client, unpause_client, env_node=env, run_count=run_count,
            global_epoch_counter=global_epoch_counter, max_epochs=IL_EPOCHS, 
            use_early_stopping=True, # Use early stopping
        )

        # After retraining, load the new best validation model for the next evaluation round.
        logger.info(f"DAgger retraining complete. Loading new best model from {BEST_VAL_MODEL_SAVE_PATH}...")
        try:
            model.load_state_dict(torch.load(BEST_VAL_MODEL_SAVE_PATH))
            model.to(device)
            logger.info("    Successfully loaded new best model for next evaluation.")
        except Exception as e:
            logger.error(f"    Error loading new best model: {e}. Continuing with the last epoch's model.")

    logger.info("\n" + "="*60)
    logger.info(f"SUCCESS CRITERIA MET: Agent achieved a 100% success rate and a best average reward of {best_average_reward_so_far:.2f}!")
    logger.info("TRAINING SUCCEEDED!")
    logger.info("="*60)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    try:
        # --- START OF THE FIX ---
        # Save the final model state for traceability
        torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
        logger.info(f"Successfully saved final trained model to: {FINAL_MODEL_SAVE_PATH}")
        
        # Create a W&B artifact of the BEST model
        logger.info(f"Creating W&B artifact from the best model saved at: {BEST_MODEL_SAVE_PATH}")
        artifact = wandb.Artifact(
            name='transfuser-best-model',
            type='model',
            description='The model that achieved the highest average reward during DAgger training.',
            metadata={"best_average_reward": best_average_reward_so_far, "final_success_rate": success_rate}
        )
        artifact.add_file(BEST_MODEL_SAVE_PATH)
        wandb.log_artifact(artifact)
        logger.info("Successfully saved best model to W&B!")
        # --- END OF THE FIX ---

    except Exception as e:
        logger.error(f"Could not save the final model or artifact. Error: {e}")

    logger.info("Closing environment and shutting down.")
    wandb.finish()
    env.close()
    rclpy.shutdown()
    logger.info("Shutdown complete.")


if __name__ == '__main__':
    main()