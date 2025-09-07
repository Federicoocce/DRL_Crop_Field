import rclpy
import os
from .drl_env import MaizeNavigationEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor # For EvalCallback
from gymnasium.wrappers import TimeLimit
import sys
import gymnasium

def main(args=None):
    rclpy.init(args=args)

    # --- Paths ---
    home_dir = os.path.expanduser('~')
    log_path = os.path.join(home_dir, 'ros2_ws', 'drl_logs', 'sac_maize_nav_logs') # Specific log for this run
    model_save_path_dir = os.path.join(home_dir, 'ros2_ws', 'drl_models')
    best_model_save_path = os.path.join(model_save_path_dir, 'sac_maize_nav_best_model') # For EvalCallback

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_save_path_dir, exist_ok=True)
    os.makedirs(best_model_save_path, exist_ok=True) # EvalCallback saves in a subdirectory

    # 1. Create the custom training environment
    try:
        train_raw_env = MaizeNavigationEnv()
        train_raw_env.get_logger().info("Training environment created.")
    except Exception as e:
        print(f"Error creating Training MaizeNavigationEnv: {e}")
        rclpy.shutdown()
        sys.exit(1)

    episode_length = 5000
    # Important: Wrap with Monitor before TimeLimit for EvalCallback to get proper episode stats
    train_env = Monitor(train_raw_env) # Monitor wrapper for SB3 logging
    train_env = TimeLimit(train_env, max_episode_steps=episode_length)
    train_raw_env.get_logger().info("Training environment wrapped.")


    # --- Create a separate evaluation environment ---
    # It's good practice to evaluate on an env instance not used for training updates
    try:
        eval_raw_env = MaizeNavigationEnv() # Create a new instance
        eval_raw_env.get_logger().info("Evaluation environment created.")
    except Exception as e:
        print(f"Error creating Evaluation MaizeNavigationEnv: {e}")
        rclpy.shutdown()
        sys.exit(1)

    eval_env = Monitor(eval_raw_env) # Wrap with Monitor for EvalCallback
    eval_env = TimeLimit(eval_env, max_episode_steps=episode_length)
    eval_raw_env.get_logger().info("Evaluation environment wrapped.")

    # Check spaces (using train_env as it's the one passed to the model directly)
    if not isinstance(train_env.observation_space, gymnasium.spaces.Box):
        train_raw_env.get_logger().error(f"Observation space is not a Box: {type(train_env.observation_space)}")
        train_raw_env.close()
        eval_raw_env.close()
        rclpy.shutdown()
        return
    # ... (similar check for action_space if needed)

    # 2. Define Callbacks
    # Stop training if a reward threshold is achieved (optional)
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)

    eval_callback = EvalCallback(
        eval_env, # The environment to use for evaluation
        best_model_save_path=best_model_save_path, # Path to save the best model
        log_path=log_path, # Path to save evaluation logs
        eval_freq=20000, # Evaluate every 10,000 steps
        n_eval_episodes=5, # Number of episodes to run for evaluation
        deterministic=True, # Use deterministic actions for evaluation
        render=False, # We are in headless mode
        # callback_on_new_best=callback_on_best # Optional: trigger another callback when a new best is found
        verbose=1
    )
    train_raw_env.get_logger().info("EvalCallback configured.")

    # 3. Define the DRL model
    model = SAC(
        "MlpPolicy",
        train_env, # Pass the training env
        verbose=1,
        tensorboard_log=log_path, # Main TensorBoard log path
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
        # Pass the eval_callback to the learn method
        model.learn(
            total_timesteps=total_training_steps,
            log_interval=10, # Log training stats every 10 episodes
            callback=eval_callback # Add the evaluation callback here
        )
    except Exception as e:
        train_raw_env.get_logger().error(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        train_raw_env.get_logger().info("Training process finished or interrupted.")

        # 5. Save the final model (the best model is already saved by EvalCallback)
        final_model_path = os.path.join(model_save_path_dir, 'sac_maize_nav_final')
        try:
            model.save(final_model_path)
            train_raw_env.get_logger().info(f"Final model saved to {final_model_path}.zip")
        except Exception as e:
            train_raw_env.get_logger().error(f"Error saving final model: {e}")

        # 6. Clean up
        train_raw_env.get_logger().info("Closing environments...")
        train_env.close() # This should close train_raw_env due to Monitor
        eval_env.close()  # This should close eval_raw_env due to Monitor
        rclpy.shutdown()
        train_raw_env.get_logger().info("Shutdown complete.")

if __name__ == '__main__':
    main()