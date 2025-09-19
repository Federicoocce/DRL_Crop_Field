# custom_sac.py

from stable_baselines3 import SAC
import torch
import torch.nn.functional as F
import numpy as np

class CustomSAC(SAC):

    def train(self, gradient_steps: int, batch_size: int) -> None:
        
        # --- 1. SEPARATE UPDATE STEP FOR THE FEATURE EXTRACTOR ---
        
        # The actor holds the canonical instance of the feature extractor
        features_extractor = self.actor.features_extractor
        
        # Ensure the feature extractor and its optimizer are in training mode
        features_extractor.train()

        waypoint_losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Perform a forward pass through the feature extractor to get predictions.
            # This attaches gradients to the TransFuser model.
            _ = features_extractor(replay_data.observations)
            
            # Retrieve the stored waypoint predictions (still attached to the graph)
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
        
        # Now, call the original SAC's train method. It will handle the actor/critic
        # updates using the features provided by our (now separately trained) extractor.
        print("\n--- Starting standard SAC training step... ---", flush=True)
        # We need to switch the policy back to training mode for the SAC part
        self.policy.set_training_mode(True)
        super().train(gradient_steps, batch_size)