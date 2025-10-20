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
DATA_COLLECTION_FPS = 2.0 # Target FPS for saving expert data
DATA_COLLECTION_FPS_TURNING = 15.0 # HIGHER target FPS for turning maneuvers
EXPERT_KP_ANGULAR = 1.0
EXPERT_TARGET_LINEAR_VEL = 0.2

# --- Model Training ---
IL_EPOCHS = 10
IL_BATCH_SIZE = 32
IL_LEARNING_RATE = 1e-4
VISUALIZE_TRAINING_SAMPLE = False # <-- NEW: Control flag for visualization


# --- Evaluation & Data Aggregation ---
AGENT_KP_ANGULAR = 0.8
AGENT_TARGET_LINEAR_VEL = 0.2
EVAL_DATA_COLLECTION_FPS = 1.0 # Target FPS for saving data during a failed evaluation
EVAL_DATA_COLLECTION_FPS_TURNING = 5.0 # HIGHER target FPS for turning maneuvers during evaluation
TARGET_REWARD_THRESHOLD = 7800.0
MAX_GT_WAYPOINT_DEVIATION_X = 1.5 # <-- NEW: Stop eval if |gt_waypoint.x| exceeds this

# --- File Paths ---
HOME_DIR = os.path.expanduser('~')
MODEL_SAVE_DIR = os.path.join(HOME_DIR, 'ros2_ws', 'drl_models', 'imitation_learning')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transfuser_il_model.pth')

# Add Augmentation Config
AUGMENT_ROTATION_DEG = 30.0 # +/- 30 degrees
FINAL_PROCESS_SIZE = 256

# ... (CONFIG PARAMETERS) ...
# Add a parameter for the camera's horizontal FOV in degrees.
# From your XACRO, the FOV is 2.2689 radians.
CAMERA_H_FOV_DEG = np.rad2deg(2.2689) # Approx 130 degrees
CAMERA_H_FOV_DEG_REAR = 100.0 # <-- NEW: Rear camera FOV in degrees
DEBUG_AUGMENTATION = False # <-- NEW: Control flag for debug prints
# ===================================================================
# --- Data Preprocessor Class (with Debug Prints) ---
# ===================================================================
class DataPreprocessor:
    """Handles augmentation and conversion from raw env obs to model inputs."""
    # --- MODIFIED __init__ ---
    def __init__(self, crop_size=256, camera_fov_deg_front=130.0, camera_fov_deg_rear=100.0):
        self.crop_size = crop_size
        self.camera_fov_deg_front = camera_fov_deg_front
        self.camera_fov_deg_rear = camera_fov_deg_rear # <-- NEW

    # --- MODIFIED process_observation ---
    def process_observation(self, raw_obs_dict, apply_augmentation=False):
        """
        Takes one raw observation dict, optionally augments, and returns processed dict.
        """
        angle = 0.0
        crop_shift_front = 0.0
        crop_shift_rear = 0.0 # <-- NEW
        
        if apply_augmentation:
            angle = random.uniform(-AUGMENT_ROTATION_DEG, AUGMENT_ROTATION_DEG)
            raw_img_width = raw_obs_dict['image_raw'].shape[1]
            
            # Calculate shift for FRONT camera
            crop_shift_front = (angle / self.camera_fov_deg_front) * raw_img_width


        # --- Get raw data ---
        img_raw_front = raw_obs_dict['image_raw']
        
        pc_raw = raw_obs_dict['lidar_raw']
        gt_wp_raw = raw_obs_dict['gt_waypoints']
        state_raw = raw_obs_dict['state']

        # --- Apply Transformations ---
        img_processed_front = scale_and_crop_image(img_raw_front, scale=1, crop=self.crop_size, crop_shift=crop_shift_front)
        
                # --- MODIFICATION: Check config before processing ---
        img_raw_rear = raw_obs_dict.get('image_rear_raw') # Safely get the data
        img_processed_rear = None

        if not config.ignore_rear and img_raw_rear is not None:
             # Calculate shift for REAR camera
             crop_shift_rear = (-angle / self.camera_fov_deg_rear) * raw_img_width if apply_augmentation else 0
             img_processed_rear = scale_and_crop_image(img_raw_rear, scale=1, crop=self.crop_size, crop_shift=crop_shift_rear)

        pc_rot = rotate_point_cloud(pc_raw, angle)
        bev_processed = lidar_to_histogram_features(pc_rot, crop=self.crop_size)
        gt_wp_rot = rotate_waypoints(gt_wp_raw, angle)
        
        # ... (state rotation logic remains the same) ...
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

        # --- MODIFICATION: Conditionally add the key ---
        if img_processed_rear is not None:
            processed_data['image_rear'] = img_processed_rear
        
        return processed_data

def collect_expert_data(env, logger):
    """
    Phase 1: Navigate the field using an expert controller and record
    observations. Samples more frequently during U-turns to balance the dataset.
    """
    logger.info("="*50)
    logger.info(f"PHASE 1: Starting Expert Data Collection")
    logger.info(f"         Normal FPS: {DATA_COLLECTION_FPS}, Turning FPS: {DATA_COLLECTION_FPS_TURNING}")
    logger.info("="*50)

    # Calculate the two sampling intervals
    collection_interval_normal = 1.0 / DATA_COLLECTION_FPS
    collection_interval_turning = 1.0 / DATA_COLLECTION_FPS_TURNING

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
            
            # Check the robot's turning state to select the correct sampling rate
            is_currently_turning = env.is_turning
            current_collection_interval = collection_interval_turning if is_currently_turning else collection_interval_normal
            
            current_time = time.time()
            if current_time - last_collection_time >= current_collection_interval:
                dataset.append(obs.copy())
                last_collection_time = current_time 
                

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
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


def train_model(model, dataset, logger, pause_client, unpause_client, env_node, run_count, global_epoch_counter):
    """
    Phase 2: Train the model offline and log the loss.
    """
    logger.info("="*50)
    logger.info(f"PHASE 2: Starting Model Training for {IL_EPOCHS} Epochs (Run #{run_count})")
    logger.info(f"         Current dataset size: {len(dataset)}")
    logger.info("="*50)
    logger.info(f"Batch size: {IL_BATCH_SIZE}, Learning rate: {IL_LEARNING_RATE}")  
    logger.info("Pausing simulation for training...")
    pause_future = pause_client.call_async(Empty.Request())
    rclpy.spin_until_future_complete(env_node, pause_future, timeout_sec=5.0)
    
    preprocessor = DataPreprocessor(
        crop_size=FINAL_PROCESS_SIZE, 
        camera_fov_deg_front=CAMERA_H_FOV_DEG,
        camera_fov_deg_rear=CAMERA_H_FOV_DEG_REAR
    )
    try:
        for epoch in range(IL_EPOCHS):
            random.shuffle(dataset)
            epoch_loss = 0.0
            num_batches = 0
            visualized_this_epoch = False
            debug_printed_this_epoch = False

            for i in range(0, len(dataset), IL_BATCH_SIZE):
                raw_batch_samples = dataset[i:i + IL_BATCH_SIZE]
                if len(raw_batch_samples) < IL_BATCH_SIZE: continue

                processed_batch_list = []
                for raw_sample in raw_batch_samples:
                    should_debug_print = DEBUG_AUGMENTATION and not debug_printed_this_epoch
                    
                    proc_sample = preprocessor.process_observation(
                        raw_sample, 
                        apply_augmentation=True
                    )
                    
                    if should_debug_print:
                        debug_printed_this_epoch = True
                    processed_batch_list.append(proc_sample)

                if VISUALIZE_TRAINING_SAMPLE and not visualized_this_epoch:
                    logger.info(f"  [Visualizing] Displaying first augmented sample of Epoch {epoch + 1}...")
                    sample_to_show = processed_batch_list[0]
                    render_sensor_data(sample_to_show['image'], sample_to_show['lidar_bev'], sample_to_show.get('image_rear'))
                    visualized_this_epoch = True

                training_batch = {
                    key: np.array([s[key] for s in processed_batch_list]) 
                    for key in processed_batch_list[0].keys()
                }
                loss = model.train_imitation_learning(training_batch)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # Calculate the current global epoch step
            current_global_epoch = global_epoch_counter + epoch + 1
            logger.info(f"  [Training] Epoch {epoch + 1}/{IL_EPOCHS} (Global Step: {current_global_epoch}) | Average Loss: {avg_loss:.6f}")
            
            # Log the loss to wandb with the continuous global step
            wandb.log({"loss": avg_loss}, step=current_global_epoch)
            
    finally:
        logger.info("Training finished. Unpausing simulation...")
        unpause_future = unpause_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(env_node, unpause_future, timeout_sec=5.0)
        
    logger.info("Model training complete.")
    # Return the new global epoch count
    return global_epoch_counter + IL_EPOCHS


### START OF MODIFIED FUNCTION ###
def evaluate_model(env, model, logger, device, global_epoch_counter):
    """
    Phase 3: Test the trained model, log the total reward, and return it.
    Data is collected during evaluation runs to be added to the main dataset
    if the run fails, mimicking the expert collection process.
    """
    logger.info("="*50)
    logger.info(f"PHASE 3: Starting Evaluation (after {global_epoch_counter} total epochs)")
    logger.info(f"         Data Collection FPS: Normal={EVAL_DATA_COLLECTION_FPS}, Turning={EVAL_DATA_COLLECTION_FPS_TURNING}")
    logger.info(f"         Failure Condition: |GT Waypoint X| > {MAX_GT_WAYPOINT_DEVIATION_X}m")
    logger.info("="*50)
    
    collection_interval_normal = 1.0 / EVAL_DATA_COLLECTION_FPS
    collection_interval_turning = 1.0 / EVAL_DATA_COLLECTION_FPS_TURNING

    logger.info("Resetting environment for evaluation.")
    preprocessor = DataPreprocessor(
        crop_size=FINAL_PROCESS_SIZE, 
        camera_fov_deg_front=CAMERA_H_FOV_DEG,
        camera_fov_deg_rear=CAMERA_H_FOV_DEG_REAR
    )
    
    raw_obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    evaluation_data = []
    
    evaluation_data.append(raw_obs.copy())
    last_collection_time = time.time()
    
    action = np.array([0.0, 0.0], dtype=np.float32)

    while not done:
        # --- NEW: Check for deviation from the ground truth path ---
        gt_first_wp_x = raw_obs['gt_waypoints'][0][0]
        if abs(gt_first_wp_x) > MAX_GT_WAYPOINT_DEVIATION_X:
            logger.warn(f"FAILURE: Deviated too far from the ground truth path (x-error: {gt_first_wp_x:.2f}m > {MAX_GT_WAYPOINT_DEVIATION_X}m).")
            info["termination_reason"] = "deviated_from_path" # Set a custom reason
            break # Exit the evaluation loop

        # --- Model Prediction (unchanged) ---
        with torch.no_grad():
            processed_obs = preprocessor.process_observation(raw_obs, apply_augmentation=False)
            batch_obs = {
                key: torch.as_tensor(np.expand_dims(val, axis=0)).to(device) 
                for key, val in processed_obs.items()
            }
            image_list, lidar_list, target_point, velocity = model._get_transfuser_inputs(batch_obs)
            pred_wp_tensor, _ = model.transfuser(image_list, lidar_list, target_point, velocity)
        
        predicted_first_wp = pred_wp_tensor[0].cpu().numpy()[0]
        angle_to_target = math.atan2(predicted_first_wp[1], predicted_first_wp[0])
        angular_vel = AGENT_KP_ANGULAR * angle_to_target
        linear_vel = AGENT_TARGET_LINEAR_VEL
        action = np.array([linear_vel, angular_vel], dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        is_currently_turning = env.is_turning
        current_collection_interval = collection_interval_turning if is_currently_turning else collection_interval_normal
            
        current_time = time.time()
        if current_time - last_collection_time >= current_collection_interval:
            evaluation_data.append(raw_obs.copy())
            last_collection_time = current_time 
            log_prefix = "[Eval Collection - TURNING]" if is_currently_turning else "[Eval Collection]"
            logger.info(f"  {log_prefix} Step {step}, Sampled data point. New data size: {len(evaluation_data)}")
        
        next_raw_obs, reward, terminated, truncated, info = env.step(action)
        raw_obs = next_raw_obs
        done = terminated or truncated
        total_reward += reward
        step += 1

    logger.info(f"Evaluation finished. Total Reward: {total_reward:.2f}")
    wandb.log({"total_reward": total_reward}, step=global_epoch_counter)
    
    if info.get("termination_reason") == "all_waypoints_visited":
        logger.info(f"SUCCESS! Model completed the course in {step} simulation steps.")
        return True, evaluation_data, total_reward
    else:
        reason = info.get("termination_reason", "unknown")
        logger.warn(f"FAILURE. Model did not complete the course (Reason: {reason}).")
        return False, evaluation_data, total_reward
### END OF MODIFIED FUNCTION ###


def main(args=None):
    rclpy.init(args=args)
    
    # Initialize wandb
    wandb.init(project="agricobots", name="il_smart_goal")
    
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
    
    best_reward_so_far = -float('inf')
    run_count = 0
    global_epoch_counter = 0
    course_completed = False 
    
    # ### START OF CORRECTION ###
    # The condition is changed from 'and' to 'or' and checks for 'not course_completed'
    # This ensures the loop runs until BOTH conditions (course completion and reward threshold) are met.
    while not course_completed or best_reward_so_far < TARGET_REWARD_THRESHOLD:
    # ### END OF CORRECTION ###
        run_count += 1
        logger.info("\n" + "#"*60)
        logger.info(f"STARTING IMITATION LEARNING ATTEMPT #{run_count}")
        logger.info(f"Current Best Reward: {best_reward_so_far:.2f} | Target Reward: {TARGET_REWARD_THRESHOLD}")
        logger.info(f"Course Completed in Previous Run: {course_completed}")
        logger.info("#"*60 + "\n")
        
        global_epoch_counter = train_model(model, dataset, logger, pause_client, unpause_client, env_node=env, run_count=run_count, global_epoch_counter=global_epoch_counter)
        
        successful_run, new_data, current_reward = evaluate_model(env, model, logger, device, global_epoch_counter=global_epoch_counter)
        
        course_completed = successful_run
        
        if current_reward > best_reward_so_far:
            best_reward_so_far = current_reward
            logger.info(f"New best reward achieved: {best_reward_so_far:.2f}")
            wandb.log({"best_reward": best_reward_so_far}, step=global_epoch_counter)
        
        if not successful_run:
            logger.warn("Evaluation failed. Aggregating new data from the failed run.")
            logger.info(f"  - Current dataset size: {len(dataset)}")
            logger.info(f"  - New data points to add: {len(new_data)}")
            dataset.extend(new_data)
            logger.info(f"  - New total dataset size: {len(dataset)}")
        
        ### --- NEW DIAGNOSTIC LOGGING --- ###
        logger.info("-" * 50)
        logger.info("END OF LOOP CHECK:")
        logger.info(f"  - Course Completed? {'YES' if course_completed else 'NO'}")
        logger.info(f"  - Best Reward ({best_reward_so_far:.2f}) < Target ({TARGET_REWARD_THRESHOLD})? {'YES' if best_reward_so_far < TARGET_REWARD_THRESHOLD else 'NO'}")

        # This diagnostic now accurately reflects the loop continuation logic
        if not course_completed or best_reward_so_far < TARGET_REWARD_THRESHOLD:
            logger.info("--> RESULT: Loop will CONTINUE.")
            time.sleep(3)
        else:
            logger.info("--> RESULT: Loop will TERMINATE because all success criteria were met.")
        logger.info("-" * 50)
        ### --- END OF NEW DIAGNOSTIC LOGGING --- ###

            
    logger.info("\n" + "="*60)
    # ### START OF CORRECTION ###
    # Simplified the final message, as exiting the loop means both conditions were met.
    logger.info(f"SUCCESS CRITERIA MET: Agent completed the course and achieved a reward of {best_reward_so_far:.2f}!")
    # ### END OF CORRECTION ###
    logger.info("TRAINING SUCCEEDED!")
    logger.info("="*60)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    try:
        torch.save(model.transfuser.state_dict(), MODEL_SAVE_PATH)
        logger.info(f"Successfully saved trained TransFuser model to: {MODEL_SAVE_PATH}")
        
        logger.info("Saving model to Weights & Biases as an Artifact...")
        artifact = wandb.Artifact(
            name='transfuser-il-model', 
            type='model',
            description='A trained TransFuser model that met the success criteria.',
            metadata={
                "run_name": wandb.run.name, 
                "total_epochs": global_epoch_counter,
                "final_reward": best_reward_so_far,
                "course_completed": course_completed
            }
        )
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