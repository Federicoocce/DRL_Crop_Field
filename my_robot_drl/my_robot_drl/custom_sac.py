# custom_sac.py

from stable_baselines3 import SAC
import torch
import torch.nn.functional as F
import numpy as np

# --- NEW IMPORTS ---
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

class CustomSAC(SAC):

    # --- MODIFIED __init__ ---
    def __init__(self, policy, env, env_node: Node, **kwargs):
        """
        Custom SAC constructor.
        
        :param env_node: The instance of the MaizeNavigationEnv, which is a ROS Node.
                         This is required to create ROS service clients.
        """
        super().__init__(policy, env, **kwargs)
        
        # Store the environment node to use its ROS capabilities
        self.env_node = env_node
        self.env_node.get_logger().info("CustomSAC is initializing Gazebo service clients...")

        # Create clients for pausing and unpausing the simulation
        self.pause_client = self.env_node.create_client(Empty, "/pause_physics")
        self.unpause_client = self.env_node.create_client(Empty, "/unpause_physics")

        # Wait for the services to become available to prevent errors on startup
        while not self.pause_client.wait_for_service(timeout_sec=2.0):
            self.env_node.get_logger().warn('/pause_physics service not available, waiting...')
        while not self.unpause_client.wait_for_service(timeout_sec=2.0):
            self.env_node.get_logger().warn('/unpause_physics service not available, waiting...')
        
        self.env_node.get_logger().info("Gazebo service clients connected successfully.")

    # --- MODIFIED train method ---
    def train(self, gradient_steps: int, batch_size: int) -> None:
        
        # --- 0. PAUSE THE SIMULATION ---
        self.env_node.get_logger().info("--- Pausing simulation for training step... ---")
        pause_future = self.pause_client.call_async(Empty.Request())
        # Wait for the service call to complete
        rclpy.spin_until_future_complete(self.env_node, pause_future, timeout_sec=2.0)

        try:
            # --- 1. SEPARATE UPDATE STEP FOR THE FEATURE EXTRACTOR ---
            
            # The actor holds the canonical instance of the feature extractor
            features_extractor = self.actor.features_extractor
            
            # Ensure the feature extractor and its optimizer are in training mode
            features_extractor.train()

            waypoint_losses = []
            for _ in range(gradient_steps):
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                
                # Perform a forward pass through the feature extractor to get predictions.
                _ = features_extractor(replay_data.observations)
                
                # Retrieve the stored waypoint predictions
                pred_waypoints = features_extractor.last_pred_wp
                gt_waypoints = replay_data.observations['gt_waypoints']

                # Calculate the waypoint loss
                waypoint_loss = F.l1_loss(pred_waypoints, gt_waypoints)
                
                # Optimize the feature extractor
                features_extractor.optimizer.zero_grad()
                waypoint_loss.backward()
                features_extractor.optimizer.step()
                
                waypoint_losses.append(waypoint_loss.item())

            avg_wp_loss = np.mean(waypoint_losses) if waypoint_losses else 0
            print(f"--- Feature Extractor Waypoint Loss (avg over {gradient_steps} steps): {avg_wp_loss:.6f} ---", flush=True)
            self.logger.record("train/waypoint_loss", avg_wp_loss)


            # --- 2. STANDARD SB3 SAC TRAINING STEP ---
            
            print("\n--- Starting standard SAC training step... ---", flush=True)
            self.policy.set_training_mode(True)
            super().train(gradient_steps, batch_size)

        finally:
            # --- 3. UNPAUSE THE SIMULATION ---
            # This 'finally' block ensures the simulation is ALWAYS unpaused,
            # even if an error occurs during training.
            self.env_node.get_logger().info("--- Training step complete. Unpausing simulation... ---")
            unpause_future = self.unpause_client.call_async(Empty.Request())
            # Wait for the service call to complete
            rclpy.spin_until_future_complete(self.env_node, unpause_future, timeout_sec=2.0)