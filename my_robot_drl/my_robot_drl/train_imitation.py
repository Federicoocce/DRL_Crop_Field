# train_imitation.py

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

# ===================================================================
# --- CONFIGURATION PARAMETERS ---
# ===================================================================

# --- Data Collection & Expert Controller ---
DATA_COLLECTION_FPS = 10.0 # <--- NEW: Target FPS for saving data
EXPERT_KP_ANGULAR = 1.0
EXPERT_TARGET_LINEAR_VEL = 0.2

# --- Model Training ---
IL_EPOCHS = 1
IL_BATCH_SIZE = 32
IL_LEARNING_RATE = 1e-4

# --- Evaluation Controller ---
AGENT_KP_ANGULAR = 1.0
AGENT_TARGET_LINEAR_VEL = 0.2

# --- File Paths ---
HOME_DIR = os.path.expanduser('~')
MODEL_SAVE_DIR = os.path.join(HOME_DIR, 'ros2_ws', 'drl_models', 'imitation_learning')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transfuser_il_model.pth')


# --- MODIFIED FUNCTION: Now collects data at a fixed rate ---
def collect_expert_data(env, logger):
    """
    Phase 1: Navigate the field using an expert controller and record
    observations at a fixed rate (e.g., 2 FPS) to avoid redundant data.
    """
    logger.info("="*50)
    logger.info(f"PHASE 1: Starting Expert Data Collection (at {DATA_COLLECTION_FPS} FPS)")
    logger.info("="*50)
    
    # --- NEW: Rate-limiting logic setup ---
    collection_interval = 1.0 / DATA_COLLECTION_FPS
    
    while True:
        obs, info = env.reset()
        dataset = []
        
        # --- NEW: Always save the very first observation of the trajectory ---
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
            
            # --- MODIFIED: Time-based data saving ---
            current_time = time.time()
            if current_time - last_collection_time >= collection_interval:
                dataset.append(obs.copy())
                last_collection_time = current_time # Reset the timer
                logger.info(f"  [Collection] Step {step}, Sampled data point. Dataset size: {len(dataset)}")

            # Step the environment (this runs as fast as possible for smooth control)
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
    # ... (This function is unchanged) ...
    logger.info("="*50)
    logger.info(f"PHASE 2: Starting Model Training for {IL_EPOCHS} Epochs (Run #{run_count})")
    logger.info(f"         Current dataset size: {len(dataset)}")
    logger.info("="*50)
    logger.info("Pausing simulation for training...")
    pause_future = pause_client.call_async(Empty.Request())
    rclpy.spin_until_future_complete(env_node, pause_future, timeout_sec=5.0)
    loss_history = []
    try:
        for epoch in range(IL_EPOCHS):
            random.shuffle(dataset)
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, len(dataset), IL_BATCH_SIZE):
                batch_samples = dataset[i:i + IL_BATCH_SIZE]
                if len(batch_samples) < IL_BATCH_SIZE: continue
                training_batch = {key: np.array([s[key] for s in batch_samples]) for key in batch_samples[0].keys()}
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


def evaluate_model(env, model, logger, device, pause_client, unpause_client, env_node):
    """
    Phase 3: Test the trained model with a synchronized "stop-think-act" loop.
    """
    # ... (This function is unchanged) ...
    logger.info("="*50)
    logger.info("PHASE 3: Starting Evaluation (with pause-on-inference)")
    logger.info("="*50)
    logger.info("Resetting environment for evaluation.")
    obs, info = env.reset()
    done = False
    step = 0
    evaluation_data = []
    while not done:
        evaluation_data.append(obs.copy())
        pause_future = pause_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(env_node, pause_future, timeout_sec=2.0)
        try:
            batch_obs = {key: torch.as_tensor(np.expand_dims(val, axis=0)).to(device) for key, val in obs.items()}
            image_list, lidar_list, target_point, velocity = model._get_transfuser_inputs(batch_obs)
            with torch.no_grad():
                pred_wp_tensor, _ = model.transfuser(image_list, lidar_list, target_point, velocity)
            if step % 30 == 0:
                all_gt_wp = obs['gt_waypoints']
                all_pred_wp = pred_wp_tensor[0].cpu().numpy()
                logger.info(f"--- Waypoint Debug @ Step {step} (Paused State) ---")
                for i in range(4):
                    gt_x, gt_y = all_gt_wp[i]; pred_x, pred_y = all_pred_wp[i]
                    logger.info(f"  WP #{i+1} | GT: (x={gt_x:6.2f}, y={gt_y:6.2f}) | Pred: (x={pred_x:6.2f}, y={pred_y:6.2f})")
                logger.info("-------------------------------------------------")
            predicted_first_wp = pred_wp_tensor[0].cpu().numpy()[0]
            angle_to_target = math.atan2(predicted_first_wp[1], predicted_first_wp[0])
            angular_vel = AGENT_KP_ANGULAR * angle_to_target
            linear_vel = AGENT_TARGET_LINEAR_VEL
            action = np.array([linear_vel, angular_vel], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
        finally:
            unpause_future = unpause_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(env_node, unpause_future, timeout_sec=2.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
        done = terminated or truncated
        step += 1
        if step % 50 == 0 and step > 0:
            logger.info(f"  [Evaluation] Step {step}...")
    if info.get("termination_reason") == "all_waypoints_visited":
        logger.info(f"SUCCESS! Model completed the course in {step} steps.")
        return True, evaluation_data
    else:
        reason = info.get("termination_reason", "unknown")
        logger.warn(f"FAILURE. Model did not complete the course (Reason: {reason}).")
        return False, evaluation_data


def main(args=None):
    # ... (This function is unchanged) ...
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
        successful_run, new_data = evaluate_model(env, model, logger, device, pause_client, unpause_client, env_node=env)
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