# custom_sac.py

from stable_baselines3 import SAC
import torch
import torch.nn.functional as F
import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
import time

class CustomSAC(SAC):

    # ... (__init__ is unchanged) ...
    def __init__(self, 
                 policy, 
                 env, 
                 env_node: Node, 
                 transfuser_train_freq: int = 20,
                 transfuser_gradient_steps: int = 5,
                 sac_train_freq: int = 1,
                 sac_gradient_steps: int = 1,
                 **kwargs):
        
        super().__init__(policy, env, **kwargs)
        
        self.env_node = env_node
        self.env_node.get_logger().info("CustomSAC is initializing Gazebo service clients...")

        self.pause_client = self.env_node.create_client(Empty, "/pause_physics")
        self.unpause_client = self.env_node.create_client(Empty, "/unpause_physics")
        while not self.pause_client.wait_for_service(timeout_sec=2.0):
            self.env_node.get_logger().warn('/pause_physics service not available, waiting...')
        while not self.unpause_client.wait_for_service(timeout_sec=2.0):
            self.env_node.get_logger().warn('/unpause_physics service not available, waiting...')
        self.env_node.get_logger().info("Gazebo service clients connected successfully.")

        self.transfuser_train_freq = transfuser_train_freq
        self.transfuser_gradient_steps = transfuser_gradient_steps
        self.sac_train_freq = sac_train_freq
        self.sac_gradient_steps = sac_gradient_steps
        self._train_step_counter = 0 
        self.env_node.get_logger().info(
            f"Schedule: TransFuser (freq={transfuser_train_freq}, steps={transfuser_gradient_steps}), "
            f"SAC (freq={sac_train_freq}, steps={sac_gradient_steps})."
        )


    def train(self, gradient_steps: int, batch_size: int) -> None:
        self._train_step_counter += 1

 
        pause_future = self.pause_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.env_node, pause_future, timeout_sec=5.0)

        try:
            # --- TRAIN TRANSFUSER (IMITATION LEARNING) ---
            if self.num_timesteps <= self.learning_starts or self._train_step_counter % self.transfuser_train_freq == 0:
                print(f"\n--- Training Step #{self._train_step_counter}: Updating TransFuser Feature Extractor ---", flush=True)
                features_extractor = self.policy.actor.features_extractor # Correct way to access it
                features_extractor.train()

                waypoint_losses = []
                for i in range(self.transfuser_gradient_steps): 
                    replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                    
                    # --- NEW CLEAN METHOD CALL ---
                    # This call encapsulates the forward pass, loss calculation, and backward pass.
                    # The computation graph is created and destroyed entirely within this call.
                    loss_item = features_extractor.train_imitation_learning(replay_data.observations)
                    
                    waypoint_losses.append(loss_item)
                    print(f"    > TransFuser Grad Step {i+1}/{self.transfuser_gradient_steps}, Loss: {loss_item:.6f}", flush=True)

                avg_wp_loss = np.mean(waypoint_losses) if waypoint_losses else 0
                print(f"--- Feature Extractor Waypoint Loss (avg over {self.transfuser_gradient_steps} steps): {avg_wp_loss:.6f} ---", flush=True)
                self.logger.record("train/waypoint_loss", avg_wp_loss)

            # --- TRAIN SAC (REINFORCEMENT LEARNING) ---
            if self._train_step_counter % self.sac_train_freq == 0:
                if self.num_timesteps > self.learning_starts:
      
                    self.policy.set_training_mode(True)
                    super().train(self.sac_gradient_steps, batch_size)

        finally:
            unpause_future = self.unpause_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.env_node, unpause_future, timeout_sec=5.0)
            time.sleep(0.1)

  
