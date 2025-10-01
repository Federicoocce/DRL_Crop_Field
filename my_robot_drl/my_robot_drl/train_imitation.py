import rclpy
import os
import sys
import torch
import math
import numpy as np
import random
from std_srvs.srv import Empty
import time
import matplotlib.pyplot as plt

# --- Important: Make sure these imports point to your project's files ---
from .drl_env import MaizeNavigationEnv
from .custom_features_extractor import TransFuserFeaturesExtractor
from .transfuser_util import rotate_point_cloud, rotate_waypoints, scale_and_crop_image, lidar_to_histogram_features, render_sensor_data, close_windows
import random
# ===================================================================
# --- CONFIGURATION PARAMETERS ---
# ===================================================================

# --- Data Collection & Expert Controller ---
DATA_COLLECTION_FPS = 10.0 # Target FPS for saving expert data
EXPERT_KP_ANGULAR = 1.0
EXPERT_TARGET_LINEAR_VEL = 0.2

# --- Model Training ---
IL_EPOCHS = 10
IL_BATCH_SIZE = 128  
IL_LEARNING_RATE = 1e-4
VISUALIZE_TRAINING_SAMPLE = False # <-- NEW: Control flag for visualization


# --- Evaluation Controller ---
EVALUATION_FPS = 2.0 # <--- NEW: Target FPS for agent's decision-making and data logging
AGENT_KP_ANGULAR = 0.8
AGENT_TARGET_LINEAR_VEL = 0.1

# --- File Paths ---
HOME_DIR = os.path.expanduser('~')
MODEL_SAVE_DIR = os.path.join(HOME_DIR, 'ros2_ws', 'drl_models', 'imitation_learning')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transfuser_il_model.pth')

# Add Augmentation Config
AUGMENT_ROTATION_DEG = 20.0 # +/- 20 degrees
FINAL_PROCESS_SIZE = 256

# ... (CONFIG PARAMETERS) ...
# Add a parameter for the camera's horizontal FOV in degrees.
# From your XACRO, the FOV is 2.2689 radians.
CAMERA_H_FOV_DEG = np.rad2deg(2.2689) # Approx 130 degrees
DEBUG_AUGMENTATION = False # <-- NEW: Control flag for debug prints
# ===================================================================
# --- Data Preprocessor Class (with Debug Prints) ---
# ===================================================================
class DataPreprocessor:
    """Handles augmentation and conversion from raw env obs to model inputs."""
    def __init__(self, crop_size=256, camera_fov_deg=130.0):
        self.crop_size = crop_size
        self.camera_fov_deg = camera_fov_deg

    def process_observation(self, raw_obs_dict, apply_augmentation=False):
        """
        Takes one raw observation dict, optionally augments, and returns processed dict.
        """
        angle = 0.0
        crop_shift = 0.0
        
        # --- NEW: Flag to trigger debug prints ---
        is_debugging_this_sample = apply_augmentation and DEBUG_AUGMENTATION

        if apply_augmentation:
            angle = random.uniform(-AUGMENT_ROTATION_DEG, AUGMENT_ROTATION_DEG)
            raw_img_width = raw_obs_dict['image_raw'].shape[1]
            crop_shift = (angle / self.camera_fov_deg) * raw_img_width

        # --- Get raw data (unchanged) ---
        img_raw = raw_obs_dict['image_raw']
        pc_raw = raw_obs_dict['lidar_raw']
        gt_wp_raw = raw_obs_dict['gt_waypoints']
        state_raw = raw_obs_dict['state']


        # --- Apply Transformations (unchanged) ---
        img_processed = scale_and_crop_image(img_raw, scale=1, crop=self.crop_size, crop_shift=crop_shift)
        pc_rot = rotate_point_cloud(pc_raw, angle)
        bev_processed = lidar_to_histogram_features(pc_rot, crop=self.crop_size)
        gt_wp_rot = rotate_waypoints(gt_wp_raw, angle)
        
        # --- Rotation logic for state (unchanged) ---
        state_rot = state_raw.copy()
        if apply_augmentation and abs(angle) > 1e-6:
            distant_goal_point_batch = state_raw[0:2].reshape(1, 2)
            rotated_goal_batch = rotate_waypoints(distant_goal_point_batch, angle)
            state_rot[0:2] = rotated_goal_batch.flatten()
        


        # --- Return dictionary (unchanged) ---
        return {
            'image': img_processed,
            'lidar_bev': bev_processed,
            'state': state_rot,
            'gt_waypoints': gt_wp_rot
        }

def collect_expert_data(env, logger):
    """
    Phase 1: Navigate the field using an expert controller and record
    observations at a fixed rate (e.g., 2 FPS) to avoid redundant data.
    """
    logger.info("="*50)
    logger.info(f"PHASE 1: Starting Expert Data Collection (at {DATA_COLLECTION_FPS} FPS)")
    logger.info("="*50)
    
    collection_interval = 1.0 / DATA_COLLECTION_FPS
    
    while True:
        obs, info = env.reset()
        dataset = []
        
        dataset.append(obs.copy())
        last_collection_time = time.time()
        
        done = False
        step = 0
        
        while not done:
            # Expert action selection (unchanged)
            gt_waypoints = obs['gt_waypoints']
            first_wp = gt_waypoints[0]
            angle_to_target = math.atan2(first_wp[1], first_wp[0])
            angular_vel = EXPERT_KP_ANGULAR * angle_to_target
            linear_vel = EXPERT_TARGET_LINEAR_VEL
            action = np.array([linear_vel, angular_vel], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            current_time = time.time()
            if current_time - last_collection_time >= collection_interval:
                dataset.append(obs.copy())
                last_collection_time = current_time 
                logger.info(f"  [Collection] Step {step}, Sampled data point. Dataset size: {len(dataset)}")

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            done = terminated or truncated
            step += 1
            
        if info.get("termination_reason") == "all_waypoints_visited":
            logger.info(f"SUCCESS: Expert completed the field in {step} steps.")
            logger.info(f"Final dataset size (sampled at {DATA_COLLECTION_FPS} FPS): {len(dataset)}")
            return dataset
        else:
            reason = info.get("termination_reason", "unknown")
            logger.error(f"FAILURE: Expert failed (Reason: {reason}). Restarting collection.")
            time.sleep(2)


def train_model(model, dataset, logger, pause_client, unpause_client, env_node, run_count):
    """
    Phase 2: Train the model offline and plot the loss.
    """
    # This function is unchanged
    logger.info("="*50)
    logger.info(f"PHASE 2: Starting Model Training for {IL_EPOCHS} Epochs (Run #{run_count})")
    logger.info(f"         Current dataset size: {len(dataset)}")
    logger.info("="*50)
    logger.info(f"Batch size: {IL_BATCH_SIZE}, Learning rate: {IL_LEARNING_RATE}")  
    logger.info("Pausing simulation for training...")
    pause_future = pause_client.call_async(Empty.Request())
    rclpy.spin_until_future_complete(env_node, pause_future, timeout_sec=5.0)
    loss_history = []
    preprocessor = DataPreprocessor(crop_size=FINAL_PROCESS_SIZE, camera_fov_deg=CAMERA_H_FOV_DEG)
    try:
        for epoch in range(IL_EPOCHS):
            random.shuffle(dataset)
            epoch_loss = 0.0
            num_batches = 0
            visualized_this_epoch = False
            # ### NEW: Debug flag for this epoch ###
            debug_printed_this_epoch = False

            for i in range(0, len(dataset), IL_BATCH_SIZE):
                raw_batch_samples = dataset[i:i + IL_BATCH_SIZE]
                if len(raw_batch_samples) < IL_BATCH_SIZE: continue

                processed_batch_list = []
                for raw_sample in raw_batch_samples:
                    # --- NEW: Logic to control when debug prints happen ---
                    # We'll print for the first augmented sample of the epoch.
                    should_debug_print = DEBUG_AUGMENTATION and not debug_printed_this_epoch
                    
                    proc_sample = preprocessor.process_observation(
                        raw_sample, 
                        apply_augmentation=True
                    )
                    
                    # If this was the sample we debugged, set the flag
                    if should_debug_print:
                        debug_printed_this_epoch = True

                    processed_batch_list.append(proc_sample)
                                # ### NEW: Visualization logic ###
                if VISUALIZE_TRAINING_SAMPLE and not visualized_this_epoch:
                    logger.info(f"  [Visualizing] Displaying first augmented sample of Epoch {epoch + 1}...")
                    # Get the first sample from the processed list
                    sample_to_show = processed_batch_list[0]
                    # Render it. The function handles cv2.waitKey(1) internally.
                    render_sensor_data(sample_to_show['image'], sample_to_show['lidar_bev'])
                    visualized_this_epoch = True
                # Collate into batch dictionary of numpy arrays
                training_batch = {
                    key: np.array([s[key] for s in processed_batch_list]) 
                    for key in processed_batch_list[0].keys()
                }
                loss = model.train_imitation_learning(training_batch)
                epoch_loss += loss
                num_batches += 1
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            loss_history.append(avg_loss)
            logger.info(f"  [Training] Epoch {epoch + 1}/{IL_EPOCHS} | Average Loss: {avg_loss:.6f}")
    finally:
        logger.info("Training finished. Unpausing simulation...")
        unpause_future = unpause_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(env_node, unpause_future, timeout_sec=5.0)
    logger.info("Generating and saving loss plot...")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, IL_EPOCHS + 1), loss_history, marker='o', linestyle='-')
        plt.title(f'Imitation Learning Loss Curve - Run #{run_count}')
        plt.xlabel('Epoch')
        plt.ylabel('Average MSE Loss')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        plot_save_path = os.path.join(MODEL_SAVE_DIR, f"il_loss_curve_run_{run_count}.png")
        plt.savefig(plot_save_path)
        plt.close()
        logger.info(f"Loss plot saved to: {plot_save_path}")
    except Exception as e:
        logger.error(f"Could not generate or save loss plot. Error: {e}")
    logger.info("Model training complete.")


# --- MODIFIED FUNCTION ---
def evaluate_model(env, model, logger, device):
    """
    Phase 3: Test the trained model with continuous control, where the agent
    "thinks" and logs data at a fixed FPS rate.
    """
    logger.info("="*50)
    logger.info(f"PHASE 3: Starting Evaluation ")
    logger.info("="*50)
    logger.info("Resetting environment for evaluation.")
    preprocessor = DataPreprocessor(crop_size=FINAL_PROCESS_SIZE, camera_fov_deg=CAMERA_H_FOV_DEG)
    
    raw_obs, info = env.reset() # Get RAW
    done = False
    step = 0
    evaluation_data = []
    
    # Initialize a default action (e.g., stay still)
    action = np.array([0.0, 0.0], dtype=np.float32)

    while not done:
        
        # 2. Run model inference to decide on the next action
        with torch.no_grad():
            processed_obs = preprocessor.process_observation(raw_obs, apply_augmentation=False)
            batch_obs = {
                key: torch.as_tensor(np.expand_dims(val, axis=0)).to(device) 
                for key, val in processed_obs.items()
            }
            image_list, lidar_list, target_point, velocity = model._get_transfuser_inputs(batch_obs)
            
            pred_wp_tensor, _ = model.transfuser(image_list, lidar_list, target_point, velocity)
        
        # 3. Calculate the action based on the model's prediction
        predicted_first_wp = pred_wp_tensor[0].cpu().numpy()[0]
        angle_to_target = math.atan2(predicted_first_wp[1], predicted_first_wp[0])
        angular_vel = AGENT_KP_ANGULAR * angle_to_target
        linear_vel = AGENT_TARGET_LINEAR_VEL
        action = np.array([linear_vel, angular_vel], dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        #debug waypoint prediction
        if step % 40 == 0:
            gt_waypoints = raw_obs['gt_waypoints']
            logger.info(f"  [Step {step}] Predicted WP #1: (x={predicted_first_wp[0]:.3f}, y={predicted_first_wp[1]:.3f}) | GT WP #1: (x={gt_waypoints[0,0]:.3f}, y={gt_waypoints[0,1]:.3f})")
            evaluation_data.append(raw_obs.copy())
        # --- Continuously step the environment with the most recent action ---
        # This loop runs as fast as possible, ensuring smooth physics.
        next_raw_obs, reward, terminated, truncated, info = env.step(action) # Get RAW
        raw_obs = next_raw_obs
        done = terminated or truncated
        step += 1

    # Final result logging
    if info.get("termination_reason") == "all_waypoints_visited":
        logger.info(f"SUCCESS! Model completed the course in {step} simulation steps.")
        return True, evaluation_data
    else:
        reason = info.get("termination_reason", "unknown")
        logger.warn(f"FAILURE. Model did not complete the course (Reason: {reason}).")
        return False, evaluation_data


def main(args=None):
    # This function is mostly unchanged, except for the call to evaluate_model
    rclpy.init(args=args)
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
    dataset = collect_expert_data(env, logger)
    successful_run = False
    run_count = 0
    while not successful_run:
        run_count += 1
        logger.info("\n" + "#"*60)
        logger.info(f"STARTING IMITATION LEARNING ATTEMPT #{run_count}")
        logger.info("#"*60 + "\n")
        train_model(model, dataset, logger, pause_client, unpause_client, env_node=env, run_count=run_count)
        
        # --- MODIFIED: Simplified call to the reworked evaluate_model ---
        successful_run, new_data = evaluate_model(env, model, logger, device)
        
        if not successful_run:
            logger.warn("Evaluation failed. Aggregating new data from the failed run.")
            logger.info(f"  - Current dataset size: {len(dataset)}")
            logger.info(f"  - New data points to add: {len(new_data)}")
            dataset.extend(new_data)
            logger.info(f"  - New total dataset size: {len(dataset)}")
            logger.info("The full cycle will be repeated with the augmented dataset.")
            time.sleep(3)
    logger.info("\n" + "="*60)
    logger.info("IMITATION LEARNING SUCCEEDED!")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    try:
        torch.save(model.transfuser.state_dict(), MODEL_SAVE_PATH)
        logger.info(f"Successfully saved trained TransFuser model to: {MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not save the final model. Error: {e}")
    logger.info("Closing environment and shutting down.")
    env.close()
    rclpy.shutdown()
    logger.info("Shutdown complete.")


if __name__ == '__main__':
    main()