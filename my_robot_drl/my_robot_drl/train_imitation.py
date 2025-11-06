import rclpy
import os
import sys
import torch
import torch.nn as nn # <-- Import nn
import math
import numpy as np
import random
from std_srvs.srv import Empty
import time
import wandb
import pickle
import argparse

# --- Important: Make sure these imports point to your project's files ---
from .drl_env import MaizeNavigationEnv
from .custom_features_extractor import TransFuserFeaturesExtractor
from .transfuser_util import rotate_point_cloud, rotate_waypoints, scale_and_crop_image, lidar_to_histogram_features, render_sensor_data, close_windows
import random
from .config import config

# ===================================================================
# --- CONFIGURATION PARAMETERS ---
# ===================================================================

# --- Data Collection & Expert Controller ---
DATA_COLLECTION_FPS = 2.0
DATA_COLLECTION_FPS_TURNING = 15.0
EXPERT_KP_ANGULAR = 1.0
EXPERT_TARGET_LINEAR_VEL = 0.2

# --- Model Training (UPDATED) ---
IL_EPOCHS = 100 # Max epochs for the initial training phase
IL_BATCH_SIZE = 32
IL_LEARNING_RATE = 1e-4
VISUALIZE_TRAINING_SAMPLE = False
EARLY_STOPPING_PATIENCE = 5  # Epochs to wait for validation loss improvement
DAGGER_RETRAIN_EPOCHS = 5    # Fixed number of epochs for DAgger retraining

# --- Evaluation & Data Aggregation ---
AGENT_KP_ANGULAR = 0.8
AGENT_TARGET_LINEAR_VEL = 0.2
EVAL_DATA_COLLECTION_FPS = 1.0
EVAL_DATA_COLLECTION_FPS_TURNING = 5.0
TARGET_REWARD_THRESHOLD = 7800.0
MAX_GT_WAYPOINT_DEVIATION_X = 1.5

# --- File Paths (UPDATED) ---
HOME_DIR = os.path.expanduser('~')
MODEL_SAVE_DIR = os.path.join(HOME_DIR, 'ros2_ws', 'drl_models', 'imitation_learning')
DATASET_SAVE_DIR = "/dataset"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transfuser_il_model.pth')
# NOTE: After collecting data, you must manually split this file into train and val versions
EXPERT_DATASET_PATH = os.path.join(DATASET_SAVE_DIR, 'expert_data.pkl')
TRAIN_DATASET_PATH = os.path.join(DATASET_SAVE_DIR, 'train_data.pkl')
VAL_DATASET_PATH = os.path.join(DATASET_SAVE_DIR, 'val_data.pkl')


# --- Augmentation & Preprocessing ---
AUGMENT_ROTATION_DEG = 30.0
FINAL_PROCESS_SIZE = 256
CAMERA_H_FOV_DEG = np.rad2deg(2.2689)
CAMERA_H_FOV_DEG_REAR = 100.0
DEBUG_AUGMENTATION = False

# ===================================================================
# --- DATASET HELPER FUNCTIONS ---
# ===================================================================

def save_dataset(dataset, path, logger):
    """Saves the collected dataset to a file using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"Successfully saved dataset with {len(dataset)} points to: {path}")
        logger.info("IMPORTANT: You must now manually split this file into 'train_data.pkl' and 'val_data.pkl' before training.")
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
    except FileNotFoundError as e:
        logger.error(f"FATAL: Dataset file not found: {e}. Please run in '--mode collect' first, then manually split the data.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL: Error loading datasets: {e}")
        sys.exit(1)


# ===================================================================
# --- Data Preprocessor Class ---
# ===================================================================
class DataPreprocessor:
    """Handles augmentation and conversion from raw env obs to model inputs."""
    def __init__(self, crop_size=256, camera_fov_deg_front=130.0, camera_fov_deg_rear=100.0):
        self.crop_size = crop_size
        self.camera_fov_deg_front = camera_fov_deg_front
        self.camera_fov_deg_rear = camera_fov_deg_rear

    def process_observation(self, raw_obs_dict, apply_augmentation=False):
        """
        Takes one raw observation dict, optionally augments, and returns processed dict.
        """
        angle = 0.0
        if apply_augmentation:
            angle = random.uniform(-AUGMENT_ROTATION_DEG, AUGMENT_ROTATION_DEG)

        raw_img_width = raw_obs_dict['image_raw'].shape[1]
        crop_shift_front = (angle / self.camera_fov_deg_front) * raw_img_width if apply_augmentation else 0.0

        img_raw_front = raw_obs_dict['image_raw']
        pc_raw = raw_obs_dict['lidar_raw']
        gt_wp_raw = raw_obs_dict['gt_waypoints']
        state_raw = raw_obs_dict['state']

        img_processed_front = scale_and_crop_image(img_raw_front, scale=1, crop=self.crop_size, crop_shift=crop_shift_front)
        
        img_raw_rear = raw_obs_dict.get('image_rear_raw')
        img_processed_rear = None
        if not config.ignore_rear and img_raw_rear is not None:
             crop_shift_rear = (-angle / self.camera_fov_deg_rear) * raw_img_width if apply_augmentation else 0
             img_processed_rear = scale_and_crop_image(img_raw_rear, scale=1, crop=self.crop_size, crop_shift=crop_shift_rear)

        pc_rot = rotate_point_cloud(pc_raw, angle)
        bev_processed = lidar_to_histogram_features(pc_rot, crop=self.crop_size)
        gt_wp_rot = rotate_waypoints(gt_wp_raw, angle)
        
        state_rot = state_raw.copy()
        if apply_augmentation and abs(angle) > 1e-6:
            distant_goal_point_batch = state_raw[0:2].reshape(1, 2)
            rotated_goal_batch = rotate_waypoints(distant_goal_point_batch, angle)
            state_rot[0:2] = rotated_goal_batch.flatten()

        processed_data = {
            'image': img_processed_front,
            'lidar_bev': bev_processed,
            'state': state_rot,
            'gt_waypoints': gt_wp_rot
        }

        if img_processed_rear is not None:
            processed_data['image_rear'] = img_processed_rear
        
        return processed_data

# ===================================================================
# --- CORE IMITATION LEARNING PHASES ---
# ===================================================================

def collect_expert_data(env, logger):
    """
    Phase 1: Navigate using an expert controller and record observations.
    """
    logger.info("="*50)
    logger.info(f"PHASE 1: Starting Expert Data Collection")
    logger.info(f"         Normal FPS: {DATA_COLLECTION_FPS}, Turning FPS: {DATA_COLLECTION_FPS_TURNING}")
    logger.info("="*50)

    collection_interval_normal = 1.0 / DATA_COLLECTION_FPS
    collection_interval_turning = 1.0 / DATA_COLLECTION_FPS_TURNING

    while True:
        obs, info = env.reset()
        dataset = [obs.copy()]
        last_collection_time = time.time()
        done = False
        step = 0
        
        while not done:
            gt_waypoints = obs['gt_waypoints']
            angle_to_target = math.atan2(gt_waypoints[0][1], gt_waypoints[0][0])
            action = np.array([EXPERT_TARGET_LINEAR_VEL, EXPERT_KP_ANGULAR * angle_to_target], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            current_interval = collection_interval_turning if env.is_turning else collection_interval_normal
            if time.time() - last_collection_time >= current_interval:
                dataset.append(obs.copy())
                last_collection_time = time.time()

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
        if info.get("termination_reason") == "all_waypoints_visited":
            logger.info(f"SUCCESS: Expert completed the field in {step} steps.")
            logger.info(f"Final dataset size: {len(dataset)}")
            return dataset
        else:
            reason = info.get("termination_reason", "unknown")
            logger.error(f"FAILURE: Expert failed (Reason: {reason}). Restarting collection.")
            time.sleep(2)


def train_model(model, train_dataset, val_dataset, logger, pause_client, unpause_client, env_node, run_count, global_epoch_counter, max_epochs, use_early_stopping=False):
    """
    Phase 2: Train the model offline with a validation loop and optional early stopping.
    """
    device = next(model.parameters()).device
    phase_name = "Initial Training" if use_early_stopping else f"DAgger Retraining (Run #{run_count})"
    logger.info("="*50)
    logger.info(f"PHASE 2: Starting {phase_name} for max {max_epochs} Epochs")
    logger.info(f"         Training Set: {len(train_dataset)} | Validation Set: {len(val_dataset)}")
    logger.info("="*50)
    logger.info("Pausing simulation for training...")
    rclpy.spin_until_future_complete(env_node, pause_client.call_async(Empty.Request()))

    preprocessor = DataPreprocessor(
        crop_size=FINAL_PROCESS_SIZE,
        camera_fov_deg_front=CAMERA_H_FOV_DEG,
        camera_fov_deg_rear=CAMERA_H_FOV_DEG_REAR
    )

    # ==================== START OF THE FIX ====================
    # Define the loss function here for use in the validation loop.
    l1_loss_fn = nn.L1Loss()
    # ===================== END OF THE FIX =====================

    best_val_loss = float('inf')
    patience_counter = 0
    epochs_completed = 0

    try:
        for epoch in range(max_epochs):
            epochs_completed += 1
            # --- TRAINING PASS ---
            model.train()
            random.shuffle(train_dataset)
            epoch_train_loss = 0.0
            num_train_batches = 0

            for i in range(0, len(train_dataset), IL_BATCH_SIZE):
                raw_batch = train_dataset[i:i + IL_BATCH_SIZE]
                if len(raw_batch) < IL_BATCH_SIZE: continue

                processed_list = [preprocessor.process_observation(s, apply_augmentation=True) for s in raw_batch]
                batch_np = {key: np.array([s[key] for s in processed_list]) for key in processed_list[0].keys()}

                loss = model.train_imitation_learning(batch_np)
                epoch_train_loss += loss
                num_train_batches += 1

            avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0

            # --- VALIDATION PASS ---
            model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for i in range(0, len(val_dataset), IL_BATCH_SIZE):
                    raw_batch = val_dataset[i:i + IL_BATCH_SIZE]
                    if len(raw_batch) < IL_BATCH_SIZE: continue

                    processed_list = [preprocessor.process_observation(s, apply_augmentation=False) for s in raw_batch]
                    batch_np = {key: np.array([s[key] for s in processed_list]) for key in processed_list[0].keys()}
                    batch_torch = {key: torch.as_tensor(val).to(device) for key, val in batch_np.items()}
                    
                    image_list, lidar_list, target_point, velocity = model._get_transfuser_inputs(batch_torch)
                    pred_wp, _ = model.transfuser(image_list, lidar_list, target_point, velocity)
                    gt_wp = batch_torch['gt_waypoints']
                    
                    # ==================== START OF THE FIX ====================
                    # Use the locally defined loss function instead of calling the model's
                    val_loss = l1_loss_fn(pred_wp, gt_wp)
                    # ===================== END OF THE FIX =====================

                    epoch_val_loss += val_loss.item()
                    num_val_batches += 1

            avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
            current_global_epoch = global_epoch_counter + epoch + 1
            logger.info(f"  [Training] Epoch {epoch + 1}/{max_epochs} | Global Step: {current_global_epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=current_global_epoch)

            # --- EARLY STOPPING LOGIC ---
            if use_early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered after {patience_counter} epochs without validation loss improvement.")
                    break
    finally:
        logger.info("Training finished. Unpausing simulation...")
        rclpy.spin_until_future_complete(env_node, unpause_client.call_async(Empty.Request()))

    return global_epoch_counter + epochs_completed


def evaluate_model(env, model, logger, device, global_epoch_counter):
    """
    Phase 3: Test the trained model, log reward, and return data from the run.
    """
    logger.info("="*50)
    logger.info(f"PHASE 3: Starting Evaluation (after {global_epoch_counter} total epochs)")
    logger.info(f"         Data Collection FPS: Normal={EVAL_DATA_COLLECTION_FPS}, Turning={EVAL_DATA_COLLECTION_FPS_TURNING}")
    logger.info("="*50)
    
    collection_interval_normal = 1.0 / EVAL_DATA_COLLECTION_FPS
    collection_interval_turning = 1.0 / EVAL_DATA_COLLECTION_FPS_TURNING

    preprocessor = DataPreprocessor(
        crop_size=FINAL_PROCESS_SIZE, 
        camera_fov_deg_front=CAMERA_H_FOV_DEG,
        camera_fov_deg_rear=CAMERA_H_FOV_DEG_REAR
    )
    
    raw_obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    evaluation_data = [raw_obs.copy()]
    last_collection_time = time.time()
    
    while not done:
        if abs(raw_obs['gt_waypoints'][0][0]) > MAX_GT_WAYPOINT_DEVIATION_X:
            logger.warn(f"FAILURE: Deviated too far from the ground truth path.")
            info["termination_reason"] = "deviated_from_path"
            break

        with torch.no_grad():
            processed_obs = preprocessor.process_observation(raw_obs, apply_augmentation=False)
            batch_obs = {key: torch.as_tensor(np.expand_dims(val, axis=0)).to(device) for key, val in processed_obs.items()}
            
            image_list, lidar_list, target_point, velocity = model._get_transfuser_inputs(batch_obs)
            pred_wp_tensor, _ = model.transfuser(image_list, lidar_list, target_point, velocity)
        
        predicted_first_wp = pred_wp_tensor[0].cpu().numpy()[0]
        angle_to_target = math.atan2(predicted_first_wp[1], predicted_first_wp[0])
        action = np.array([AGENT_TARGET_LINEAR_VEL, AGENT_KP_ANGULAR * angle_to_target], dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        current_interval = collection_interval_turning if env.is_turning else collection_interval_normal
        if time.time() - last_collection_time >= current_interval:
            evaluation_data.append(raw_obs.copy())
            last_collection_time = time.time()
        
        next_raw_obs, reward, terminated, truncated, info = env.step(action)
        raw_obs = next_raw_obs
        done = terminated or truncated
        total_reward += reward
        step += 1

    logger.info(f"Evaluation finished. Total Reward: {total_reward:.2f}")
    wandb.log({"total_reward": total_reward}, step=global_epoch_counter)
    
    if info.get("termination_reason") == "all_waypoints_visited":
        logger.info(f"SUCCESS! Model completed the course in {step} steps.")
        return True, evaluation_data, total_reward
    else:
        reason = info.get("termination_reason", "unknown")
        logger.warn(f"FAILURE. Model did not complete the course (Reason: {reason}).")
        return False, evaluation_data, total_reward

# ===================================================================
# --- MAIN EXECUTION ---
# ===================================================================

def main(args=None):
    parser = argparse.ArgumentParser(description="Run Imitation Learning for AgriCobots")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'collect'],
                        help="Set the script to 'train' mode or 'collect' mode.")
    cli_args, _ = parser.parse_known_args()

    rclpy.init(args=args)

    # --- MODE 1: Data Collection Only ---
    if cli_args.mode == 'collect':
        try:
            env = MaizeNavigationEnv()
            logger = env.get_logger()
            logger.info("Running in Data Collection Mode.")
            dataset = collect_expert_data(env, logger)
            save_dataset(dataset, EXPERT_DATASET_PATH, logger)
        except Exception as e:
            print(f"An error occurred during data collection: {e}")
        finally:
            if 'env' in locals() and env: env.close()
            rclpy.shutdown()
            print("Shutdown complete.")
            sys.exit(0)

    # --- MODE 2: Full Training Workflow ---
    wandb.init(project="agricobots", name="il_transfuser_dagger")

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
    model = TransFuserFeaturesExtractor(env.observation_space, features_dim=72, lr=IL_LEARNING_RATE)
    model.to(device)
    logger.info(f"TransFuser model initialized on device: {device}")

    train_dataset, val_dataset = load_datasets(TRAIN_DATASET_PATH, VAL_DATASET_PATH, logger)

    # 1. Initial training phase with early stopping
    global_epoch_counter = train_model(
        model, train_dataset, val_dataset, logger, pause_client, unpause_client,
        env_node=env, run_count=0, global_epoch_counter=0,
        max_epochs=IL_EPOCHS, use_early_stopping=True
    )

    best_reward_so_far = -float('inf')
    run_count = 0
    course_completed = False

    # 2. DAgger loop: evaluate, aggregate data, and retrain for a fixed number of epochs
    while not course_completed or best_reward_so_far < TARGET_REWARD_THRESHOLD:
        run_count += 1
        logger.info("\n" + "#"*60)
        logger.info(f"STARTING DAGGER ATTEMPT #{run_count}")
        logger.info(f"Current Best Reward: {best_reward_so_far:.2f} | Target Reward: {TARGET_REWARD_THRESHOLD}")
        logger.info(f"Course Completed in Previous Run: {course_completed}")
        logger.info("#"*60 + "\n")

        successful_run, new_data, current_reward = evaluate_model(env, model, logger, device, global_epoch_counter=global_epoch_counter)
        course_completed = successful_run

        if current_reward > best_reward_so_far:
            best_reward_so_far = current_reward
            logger.info(f"New best reward achieved: {best_reward_so_far:.2f}")
            wandb.log({"best_reward": best_reward_so_far}, step=global_epoch_counter)

        # Check completion criteria before deciding to retrain
        if course_completed and best_reward_so_far >= TARGET_REWARD_THRESHOLD:
            break

        if not successful_run:
            logger.warn(f"Evaluation failed. Aggregating {len(new_data)} new data points to the training set.")
            train_dataset.extend(new_data)
            logger.info(f"New training set size: {len(train_dataset)}")

        # Retrain for a fixed number of epochs
        global_epoch_counter = train_model(
            model, train_dataset, val_dataset, logger, pause_client, unpause_client,
            env_node=env, run_count=run_count, global_epoch_counter=global_epoch_counter,
            max_epochs=DAGGER_RETRAIN_EPOCHS, use_early_stopping=False
        )

    logger.info("\n" + "="*60)
    logger.info(f"SUCCESS CRITERIA MET: Agent completed the course and achieved a reward of {best_reward_so_far:.2f}!")
    logger.info("TRAINING SUCCEEDED!")
    logger.info("="*60)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    try:
        # Save the underlying transfuser model's state dict, as is common
        torch.save(model.transfuser.state_dict(), MODEL_SAVE_PATH)
        logger.info(f"Successfully saved trained TransFuser model to: {MODEL_SAVE_PATH}")

        artifact = wandb.Artifact('transfuser-il-model-dagger', type='model')
        artifact.add_file(MODEL_SAVE_PATH)
        wandb.log_artifact(artifact)
        logger.info("Successfully saved model to W&B!")
    except Exception as e:
        logger.error(f"Could not save the final model. Error: {e}")

    logger.info("Closing environment and shutting down.")
    wandb.finish()
    env.close()
    rclpy.shutdown()
    logger.info("Shutdown complete.")


if __name__ == '__main__':
    main()
