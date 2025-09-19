import rclpy
import os
from .drl_env import MaizeNavigationEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor # For EvalCallback
from gymnasium.wrappers import TimeLimit
import sys
import gymnasium
import numpy as np # For calculating stats in evaluation mode
from .custom_sac import CustomSAC
from .custom_features_extractor import TransFuserFeaturesExtractor
from stable_baselines3.sac import MlpPolicy as SACPolicy # Use the standard MlpPolicy
def main(args=None):
    rclpy.init(args=args)

    # --- Determine Mode (Train vs. Eval) from command-line arguments ---
    mode = 'train' # Default mode
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'eval':
            mode = 'eval'
        elif sys.argv[1].lower() != 'train':
            print(f"Warning: Unknown mode '{sys.argv[1]}'. Defaulting to 'train'.")

    # --- Common Paths ---
    home_dir = os.path.expanduser('~')
    log_path = os.path.join(home_dir, 'ros2_ws', 'drl_logs', 'sac_maize_nav_logs')
    model_save_path_dir = os.path.join(home_dir, 'ros2_ws', 'drl_models')
    best_model_save_path = os.path.join(model_save_path_dir, 'sac_maize_nav_best_model')


    # ===================================================================
    # --- TRAINING MODE ---
    # ===================================================================
    if mode == 'train':
        print("\n--- Starting in TRAINING mode ---\n")
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_save_path_dir, exist_ok=True)
        os.makedirs(best_model_save_path, exist_ok=True)

        # 1. Create the custom training environment
        try:
            train_raw_env = MaizeNavigationEnv()
            train_raw_env.get_logger().info("Training environment created.")
        except Exception as e:
            print(f"Error creating Training MaizeNavigationEnv: {e}")
            rclpy.shutdown()
            sys.exit(1)

        episode_length = 20000
        # --- IMPORTANT: Keep the raw env instance before wrapping it ---
        train_env = TimeLimit(train_raw_env, max_episode_steps=episode_length)
        train_env = Monitor(train_env) # Monitor wraps TimeLimit
        train_raw_env.get_logger().info("Training environment wrapped.")


        # --- Create a separate evaluation environment for the callback ---
        try:
            eval_raw_env_callback = MaizeNavigationEnv()
            eval_raw_env_callback.get_logger().info("Callback evaluation environment created.")
        except Exception as e:
            print(f"Error creating Callback Evaluation MaizeNavigationEnv: {e}")
            rclpy.shutdown()
            sys.exit(1)

        eval_env_callback = TimeLimit(eval_raw_env_callback, max_episode_steps=episode_length)
        eval_env_callback = Monitor(eval_env_callback) # Monitor wraps TimeLimit
        eval_raw_env_callback.get_logger().info("Callback evaluation environment wrapped.")

 

        # 2. Define Callbacks
        eval_callback = EvalCallback(
            eval_env_callback,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=20000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1
        )
        train_raw_env.get_logger().info("EvalCallback configured.")

        # 3. Define the DRL model
        # --- MODIFIED MODEL CREATION ---
        model = CustomSAC(
            policy=SACPolicy,
            env=train_env,
            env_node=train_raw_env,  # <-- PASS THE NODE INSTANCE HERE
            verbose=1,
            tensorboard_log=log_path,
            learning_rate=0.0003,
            batch_size=32,
            buffer_size=2000,
            learning_starts=2000,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(
                features_extractor_class=TransFuserFeaturesExtractor,
                features_extractor_kwargs=dict(
                    features_dim=72  # 64 from z + 8 from waypoints
                ),
                net_arch=[256, 256]
            )
            # You can remove aux_loss_weight as it's not a standard SAC parameter
        )
        train_raw_env.get_logger().info("SAC model defined.")

        # ... (The rest of the file remains the same) ...
        # 4. Train the model
        total_training_steps = 200000
        train_raw_env.get_logger().info(f"Starting DRL training for {total_training_steps} timesteps...")
        try:
            model.learn(
                total_timesteps=total_training_steps,
                log_interval=10,
                callback=eval_callback
            )
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

            # 6. Clean up
            train_raw_env.get_logger().info("Closing environments...")
            train_env.close()
            eval_env_callback.close()
            rclpy.shutdown()
            train_raw_env.get_logger().info("Shutdown complete.")

    # ... (Evaluation mode code remains the same) ...
    elif mode == 'eval':
        # NOTE: The evaluation mode does not need pausing, so it will still work.
        # However, to load the model properly, we must tell SB3 about our custom SAC class.
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
            # When loading, we pass the env_node to the constructor via custom_objects
            model = CustomSAC.load(
                model_to_load_path, 
                env=eval_env,
                env_node=eval_raw_env, # <-- PASS NODE HERE TOO
                custom_objects={
                    "policy_kwargs": {
                        "features_extractor_class": TransFuserFeaturesExtractor,
                        "features_extractor_kwargs": {"features_dim": 72}
                    }
                }
            )
            eval_raw_env.get_logger().info(f"Successfully loaded model from {model_to_load_path}")
        except Exception as e:
            eval_raw_env.get_logger().error(f"Error loading the model: {e}")
            import traceback
            traceback.print_exc()
            eval_env.close()
            rclpy.shutdown()
            sys.exit(1)

        # ... (rest of eval code) ...
        num_eval_episodes = 10
        episode_rewards = []
        episode_lengths = []

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
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
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