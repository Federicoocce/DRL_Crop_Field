import rclpy
import os
from .drl_env import MaizeNavigationEnv # Assuming drl_env.py is in the same directory
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit
import sys
import gymnasium # Import gymnasium explicitly

def main(args=None):
    rclpy.init(args=args)

    # 1. Create the custom environment
    try:
        raw_env = MaizeNavigationEnv() # This is your ROS 2 Node
    except Exception as e:
        print(f"Error creating MaizeNavigationEnv: {e}")
        rclpy.shutdown()
        sys.exit(1)

    episode_length = 3000
    env = TimeLimit(raw_env, max_episode_steps=episode_length) # env is the wrapped environment for SB3

    # 2. Define the DRL model and logging paths
    log_path = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'drl_logs')
    os.makedirs(log_path, exist_ok=True)
    
    model_save_path_dir = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'drl_models')
    os.makedirs(model_save_path_dir, exist_ok=True)

    # Check if the environment's observation and action spaces are valid for SAC's MlpPolicy
    # Use raw_env for checks if env might not expose them directly, or use env if it does
    if not isinstance(env.observation_space, gymnasium.spaces.Box):
        raw_env.get_logger().error(f"Observation space is not a Box: {type(env.observation_space)}")
        raw_env.close() # Close the raw_env
        rclpy.shutdown()
        return
    if not isinstance(env.action_space, gymnasium.spaces.Box):
        raw_env.get_logger().error(f"Action space is not a Box: {type(env.action_space)}")
        raw_env.close() # Close the raw_env
        rclpy.shutdown()
        return

    model = SAC(
        "MlpPolicy",
        env, # Pass the wrapped env to SB3
        verbose=1,
        tensorboard_log=log_path,
        learning_rate=0.0003, 
        batch_size=256,
        buffer_size=1000000, 
        learning_starts=10000,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
    )
    
    # 3. Train the model
    total_training_steps = 1000000
    # --- USE raw_env for get_logger ---
    raw_env.get_logger().info(f"Starting DRL training for {total_training_steps} timesteps...")
    try:
        model.learn(total_timesteps=total_training_steps, log_interval=10)
    except Exception as e:
        # --- USE raw_env for get_logger ---
        raw_env.get_logger().error(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- USE raw_env for get_logger ---
        raw_env.get_logger().info("Training process finished or interrupted.")

        # 4. Save the trained model
        model_path = os.path.join(model_save_path_dir, 'sac_maize_nav')
        try:
            model.save(model_path)
            # --- USE raw_env for get_logger ---
            raw_env.get_logger().info(f"Model saved to {model_path}.zip")
        except Exception as e:
            # --- USE raw_env for get_logger ---
            raw_env.get_logger().error(f"Error saving model: {e}")

        # 5. Clean up
        # Closing raw_env should be sufficient as it's the actual node.
        # The TimeLimit wrapper doesn't typically have its own resources to close
        # beyond what the underlying environment handles.
        raw_env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()