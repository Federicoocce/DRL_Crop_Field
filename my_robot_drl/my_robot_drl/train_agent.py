# train_agent.py

import rclpy
import os
import sys
import torch
import math
import numpy as np
import gymnasium
from collections import deque
import random
from std_srvs.srv import Empty

from .drl_env import MaizeNavigationEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from .custom_sac import CustomSAC
from .custom_features_extractor import TransFuserFeaturesExtractor
from stable_baselines3.sac import MlpPolicy as SACPolicy

def main(args=None):
    rclpy.init(args=args)

    # ... (Mode selection and path definitions are unchanged) ...
    mode = 'train'
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['train', 'eval', 'il_only']:
            mode = sys.argv[1].lower()
        else:
            print(f"Warning: Unknown mode '{sys.argv[1]}'. Defaulting to 'train'.")
    home_dir = os.path.expanduser('~')
    log_path = os.path.join(home_dir, 'ros2_ws', 'drl_logs', 'sac_maize_nav_logs')
    model_save_path_dir = os.path.join(home_dir, 'ros2_ws', 'drl_models')


    if mode == 'train':
        # ... (This section is unchanged and remains correct) ...
        print("\n--- Starting in DRL TRAINING mode (SAC + IL) ---\n")
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_save_path_dir, exist_ok=True)
        try:
            train_raw_env = MaizeNavigationEnv()
            train_raw_env.get_logger().info("Training environment created.")
        except Exception as e:
            print(f"Error creating Training MaizeNavigationEnv: {e}")
            rclpy.shutdown()
            sys.exit(1)
        episode_length = 20000
        train_env = TimeLimit(train_raw_env, max_episode_steps=episode_length)
        train_env = Monitor(train_env)
        train_raw_env.get_logger().info("Training environment wrapped.")
        train_raw_env.get_logger().info("Periodic evaluation is DISABLED.")
        model = CustomSAC(
            policy=SACPolicy,
            env=train_env,
            env_node=train_raw_env,
            transfuser_train_freq=20, sac_train_freq=1,
            transfuser_gradient_steps=5, sac_gradient_steps=1,
            verbose=1, tensorboard_log=log_path,
            learning_rate=0.0003, batch_size=32,
            buffer_size=20000, learning_starts=19900,
            gamma=0.99, tau=0.005,
            train_freq=1, gradient_steps=1,
            policy_kwargs=dict(
                features_extractor_class=TransFuserFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=72),
                net_arch=[256, 256]
            )
        )
        train_raw_env.get_logger().info("CustomSAC model with alternating schedule defined.")
        total_training_steps = 200000
        train_raw_env.get_logger().info(f"Starting DRL training for {total_training_steps} timesteps...")
        try:
            model.learn(total_timesteps=total_training_steps, log_interval=10)
        except Exception as e:
            train_raw_env.get_logger().error(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
        finally:
            train_raw_env.get_logger().info("Training process finished or interrupted.")
            final_model_path = os.path.join(model_save_path_dir, 'sac_maize_nav_final')
            try:
                model.save(final_model_path)
                train_raw_env.get_logger().info(f"Final model saved to {final_model_path}.zip")
            except Exception as e:
                train_raw_env.get_logger().error(f"Error saving final model: {e}")
            train_raw_env.get_logger().info("Closing environment...")
            train_env.close()
            rclpy.shutdown()
            train_raw_env.get_logger().info("Shutdown complete.")

    # ===================================================================
    # --- IMITATION LEARNING ONLY MODE ---
    # ===================================================================
    elif mode == 'il_only':
        print("\n--- Starting in IMITATION LEARNING ONLY mode ---\n")
        os.makedirs(log_path, exist_ok=True)
        
        # 1. Environment and Service Setup
        try:
            il_raw_env = MaizeNavigationEnv()
            il_raw_env.get_logger().info("IL-Only environment created.")
        except Exception as e:
            print(f"Error creating IL-Only MaizeNavigationEnv: {e}")
            rclpy.shutdown()
            sys.exit(1)

        il_raw_env.get_logger().info("IL-Only: Initializing Gazebo service clients...")
        pause_client = il_raw_env.create_client(Empty, "/pause_physics")
        unpause_client = il_raw_env.create_client(Empty, "/unpause_physics")
        while not pause_client.wait_for_service(timeout_sec=2.0):
            il_raw_env.get_logger().warn('/pause_physics service not available, waiting...')
        while not unpause_client.wait_for_service(timeout_sec=2.0):
            il_raw_env.get_logger().warn('/unpause_physics service not available, waiting...')
        il_raw_env.get_logger().info("IL-Only: Gazebo service clients connected.")
        
        episode_length = 20000
        il_env = TimeLimit(il_raw_env, max_episode_steps=episode_length)
        il_env = Monitor(il_env)
        il_raw_env.get_logger().info("IL-Only environment wrapped.")

        # 2. Model Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features_extractor = TransFuserFeaturesExtractor(il_env.observation_space, features_dim=72, lr=1e-4)
        features_extractor.to(device)
        il_raw_env.get_logger().info(f"TransFuser model loaded on device: {device}")

        # --- START OF MODIFICATION ---
        # 3. PID Controller and Replay Buffer Configuration
        KP_ANGULAR = 1.0
        TARGET_LINEAR_VEL = 0.1
        
        IL_BUFFER_SIZE = 15000  # Increased buffer size
        IL_LEARNING_STARTS = 1000 # Number of random steps before training
        
        replay_buffer = deque(maxlen=IL_BUFFER_SIZE)
        batch_size = 32
        train_freq = 3
        
        il_raw_env.get_logger().info(f"IL-Only Params: Buffer Size={IL_BUFFER_SIZE}, Learning Starts={IL_LEARNING_STARTS}")
        # --- END OF MODIFICATION ---
        
        # 4. Main Execution Loop
        total_timesteps = 200000
        obs, info = il_env.reset()
        ep_reward = 0
        ep_len = 0
        
        il_raw_env.get_logger().info(f"Starting IL-Only run for {total_timesteps} timesteps...")

        for step in range(total_timesteps):
            
            # --- START OF MODIFICATION ---
            # A. Action Selection: Random exploration or PID control
            if step < IL_LEARNING_STARTS:
                # Take a random action to explore and fill the buffer
                action = il_env.action_space.sample()
                if step % 500 == 0: # Log progress during exploration
                    il_raw_env.get_logger().info(f"Random exploration phase: {step}/{IL_LEARNING_STARTS} steps...")
            else:
                if step == IL_LEARNING_STARTS: # Log when training starts
                     il_raw_env.get_logger().info(f"LEARNING STARTS: Buffer filled. Switching to PID actions and training.")
                # Use PID controller based on TransFuser prediction
                batch_obs = {key: torch.as_tensor(np.expand_dims(val, axis=0)).to(device) for key, val in obs.items()}
                image_list, lidar_list, target_point, velocity = features_extractor._get_transfuser_inputs(batch_obs)
                with torch.no_grad():
                    pred_wp, _ = features_extractor.transfuser(image_list, lidar_list, target_point, velocity)
                all_gt_wp = obs['gt_waypoints']
                all_pred_wp = pred_wp[0].cpu().numpy()
                first_wp = all_pred_wp[0] # Get the first one for the controller
                angle_to_target = math.atan2(first_wp[1], first_wp[0])
                angular_vel = KP_ANGULAR * angle_to_target
                linear_vel = TARGET_LINEAR_VEL
                action = np.array([linear_vel, angular_vel], dtype=np.float32)
                action = np.clip(action, il_env.action_space.low, il_env.action_space.high)
                #START OF CONTROLLER DEBUGGING BLOCK
                # Log detailed debugging info every 20 steps
                if step % 20 == 0: # Log periodically
                    distant_goal_local = obs['state'][:2]
                    

                    il_raw_env.get_logger().info("--- WP Prediction Debug ---")
                    il_raw_env.get_logger().info(f"  Distant Goal (Local):   x={distant_goal_local[0]:.2f}, y={distant_goal_local[1]:.2f}")
                    
                    # Loop and print all 4 waypoints for easy comparison
                    for i in range(4):
                        gt_wp = all_gt_wp[i]
                        il_raw_env.get_logger().info(f"  GT   WP #{i+1}: x={gt_wp[0]:.2f}, y={gt_wp[1]:.2f}")

                    il_raw_env.get_logger().info(f"  Resulting Angle (from WP #1): {angle_to_target:.1f} RADIANS")
                    il_raw_env.get_logger().info(f"  Final Action: [Lin: {action[0]:.2f}, Ang: {action[1]:.2f}]")
                    il_raw_env.get_logger().info("---------------------------")
                # # --- START OF MODIFIED DEBUGGING BLOCK ---
                # if step % 20 == 0: # Log periodically
                    
                #     distant_goal_local = obs['state'][:2]
                #     angle_to_target_deg = math.degrees(angle_to_target)

                #     il_raw_env.get_logger().info("--- WP Prediction Debug ---")
                #     il_raw_env.get_logger().info(f"  Distant Goal (Local):   x={distant_goal_local[0]:.2f}, y={distant_goal_local[1]:.2f}")
                    
                #     # Loop and print all 4 waypoints for easy comparison
                #     for i in range(4):
                #         gt_wp = all_gt_wp[i]
                #         pred_wp = all_pred_wp[i]
                #         il_raw_env.get_logger().info(f"  GT   WP #{i+1}: x={gt_wp[0]:.2f}, y={gt_wp[1]:.2f}")
                #         il_raw_env.get_logger().info(f"  Pred WP #{i+1}: x={pred_wp[0]:.2f}, y={pred_wp[1]:.2f}")

                #     il_raw_env.get_logger().info(f"  Resulting Angle (from WP #1): {angle_to_target_deg:.1f} degrees")
                #     il_raw_env.get_logger().info(f"  Final Action: [Lin: {action[0]:.2f}, Ang: {action[1]:.2f}]")
                #     il_raw_env.get_logger().info("---------------------------")
                # # --- END OF MODIFIED DEBUGGING BLOCK ---
            # --- END OF MODIFICATION ---

            # B. Step the environment
            next_obs, reward, terminated, truncated, info = il_env.step(action)
            ep_reward += reward
            ep_len += 1
            
            # C. Store experience in buffer (always)
            replay_buffer.append(obs.copy())

            # D. Train the model (only after learning_starts)
            if step >= IL_LEARNING_STARTS and len(replay_buffer) > batch_size and step % train_freq == 0:
                
                
                pause_future = pause_client.call_async(Empty.Request())
                rclpy.spin_until_future_complete(il_raw_env, pause_future, timeout_sec=2.0)

                try:
                    samples = random.sample(replay_buffer, batch_size)
                    training_batch = {key: np.array([s[key] for s in samples]) for key in samples[0].keys()}
                    loss = features_extractor.train_imitation_learning(training_batch)
                    
                    if step % (train_freq * 10) == 0:
                         il_raw_env.get_logger().info(f"Step {step}, IL Loss: {loss:.6f}")
                finally:
                    
                    unpause_future = unpause_client.call_async(Empty.Request())
                    rclpy.spin_until_future_complete(il_raw_env, unpause_future, timeout_sec=2.0)

            # E. Handle episode end
            obs = next_obs
            if terminated or truncated:
                il_raw_env.get_logger().info(f"Episode finished after {ep_len} steps. Reward: {ep_reward:.2f}")
                ep_reward = 0
                ep_len = 0
                obs, info = il_env.reset()
        
        il_raw_env.get_logger().info("IL-Only run finished.")
        il_env.close()
        rclpy.shutdown()

    elif mode == 'eval':
        # ... (This section is unchanged and remains correct) ...
        print("\n--- Starting in EVALUATION mode ---\n")
        model_to_load_path = os.path.join(model_save_path_dir, 'sac_maize_nav_final.zip')
        if not os.path.exists(model_to_load_path):
            print(f"ERROR: Model file not found at '{model_to_load_path}'.")
            rclpy.shutdown()
            sys.exit(1)
        try:
            eval_raw_env = MaizeNavigationEnv()
            eval_raw_env.get_logger().info("Evaluation environment created.")
        except Exception as e:
            print(f"Error creating Evaluation MaizeNavigationEnv: {e}")
            rclpy.shutdown()
            sys.exit(1)
        episode_length = 10000
        eval_env = TimeLimit(eval_raw_env, max_episode_steps=episode_length)
        eval_env = Monitor(eval_env)
        eval_raw_env.get_logger().info("Evaluation environment wrapped.")
        try:
            model = CustomSAC.load(
                model_to_load_path, env=eval_env, env_node=eval_raw_env,
                custom_objects={"policy_kwargs": {"features_extractor_class": TransFuserFeaturesExtractor, "features_extractor_kwargs": {"features_dim": 72}}}
            )
            eval_raw_env.get_logger().info(f"Successfully loaded model from {model_to_load_path}")
        except Exception as e:
            eval_raw_env.get_logger().error(f"Error loading the model: {e}")
            import traceback
            traceback.print_exc()
            eval_env.close()
            rclpy.shutdown()
            sys.exit(1)
        num_eval_episodes = 10
        episode_rewards, episode_lengths = [], []
        eval_raw_env.get_logger().info(f"Starting evaluation for {num_eval_episodes} episodes...")
        for i in range(num_eval_episodes):
            obs, info = eval_env.reset()
            terminated, truncated = False, False
            while not terminated and not truncated:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_info = info.get('episode')
            if ep_info:
                print(f"Episode {i + 1}/{num_eval_episodes} finished.")
                print(f"  Reward: {ep_info['r']:.2f}, Length: {ep_info['l']} steps")
                episode_rewards.append(ep_info['r'])
                episode_lengths.append(ep_info['l'])
            else:
                 eval_raw_env.get_logger().warn(f"Could not find episode stats for episode {i+1}.")
        if episode_rewards:
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            print("\n--- Evaluation Summary ---")
            print(f"Average Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Average Episode Length: {mean_length:.2f} steps")
            print("--------------------------\n")
        else:
            print("\n--- No episodes completed, cannot calculate summary statistics. ---\n")
        eval_raw_env.get_logger().info("Closing evaluation environment...")
        eval_env.close()
        rclpy.shutdown()
        eval_raw_env.get_logger().info("Shutdown complete.")

if __name__ == '__main__':
    main()