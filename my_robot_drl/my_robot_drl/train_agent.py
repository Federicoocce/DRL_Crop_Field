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

        # Check spaces
        if not isinstance(train_env.observation_space, gymnasium.spaces.Box):
            train_raw_env.get_logger().error(f"Observation space is not a Box: {type(train_env.observation_space)}")
            train_raw_env.close()
            eval_raw_env_callback.close()
            rclpy.shutdown()
            return

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
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=log_path,
            learning_rate=0.0003,
            batch_size=256,
            buffer_size=1000000,
            learning_starts=5000,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
        )
        train_raw_env.get_logger().info("SAC model defined.")

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

    # ===================================================================
    # --- EVALUATION MODE ---
    # ===================================================================
    elif mode == 'eval':
        print("\n--- Starting in EVALUATION mode ---\n")

        # --- THIS IS THE ONLY LINE THAT CHANGED ---
        model_to_load_path = os.path.join(model_save_path_dir, 'sac_maize_nav_final.zip')

        if not os.path.exists(model_to_load_path):
            print(f"ERROR: Model file not found at '{model_to_load_path}'.")
            print("Please run in 'train' mode first to generate the final model.")
            rclpy.shutdown()
            sys.exit(1)

        # 1. Create the evaluation environment
        try:
            eval_raw_env = MaizeNavigationEnv()
            eval_raw_env.get_logger().info("Evaluation environment created.")
        except Exception as e:
            print(f"Error creating Evaluation MaizeNavigationEnv: {e}")
            rclpy.shutdown()
            sys.exit(1)

        episode_length = 10000
        eval_env = TimeLimit(eval_raw_env, max_episode_steps=episode_length)
        eval_env = Monitor(eval_env) # Monitor wraps TimeLimit
        eval_raw_env.get_logger().info("Evaluation environment wrapped.")
        # 2. Load the pre-trained model
        try:
            model = SAC.load(model_to_load_path, env=eval_env)
            eval_raw_env.get_logger().info(f"Successfully loaded model from {model_to_load_path}")
        except Exception as e:
            eval_raw_env.get_logger().error(f"Error loading the model: {e}")
            eval_env.close()
            rclpy.shutdown()
            sys.exit(1)

        # 3. Run evaluation episodes
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
            
            # The Monitor wrapper logs the episode statistics in the 'info' dictionary
            # when the episode is done
            ep_info = info.get('episode')
            if ep_info:
                print(f"Episode {i + 1}/{num_eval_episodes} finished.")
                print(f"  Reward: {ep_info['r']:.2f}, Length: {ep_info['l']} steps")
                episode_rewards.append(ep_info['r'])
                episode_lengths.append(ep_info['l'])
            else:
                 eval_raw_env.get_logger().warn(f"Could not find episode stats for episode {i+1}.")


        # 4. Print summary statistics
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

        # 5. Clean up
        eval_raw_env.get_logger().info("Closing evaluation environment...")
        eval_env.close()
        rclpy.shutdown()
        eval_raw_env.get_logger().info("Shutdown complete.")

if __name__ == '__main__':
    main()