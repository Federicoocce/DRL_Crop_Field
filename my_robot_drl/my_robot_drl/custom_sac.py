# custom_sac.py

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union


import torch
import torch.nn.functional as F
import numpy as np

class CustomSAC(SAC):
    def __init__(self, *args, aux_loss_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight

    def train(self, gradient_steps: int, batch_size: int) -> None:
        print(f"\n--- CUSTOM SAC TRAIN START: {gradient_steps} gradient steps, batch size {batch_size} ---", flush=True)
        
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update learning rate
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, ent_losses, waypoint_losses = [], [], [], []

        for gradient_step in range(gradient_steps):
            print(f"\n--- SAC GRADIENT STEP {gradient_step + 1}/{gradient_steps} ---", flush=True)
            
            # Sample replay buffer
            print("Sampling replay buffer...", flush=True)
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            print("Replay buffer sampled successfully", flush=True)

            # 1. --- CRITIC UPDATE ---
            print("Starting critic update...", flush=True)
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - self.ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            print("Target Q values computed", flush=True)

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            print("Critic loss computed, starting backward pass...", flush=True)

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            print("Critic update completed", flush=True)

            # 2. --- ACTOR UPDATE ---
            print("Starting actor update...", flush=True)
            
            # Compute actor loss
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            qf_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(qf_pi, dim=1, keepdim=True)

            actor_loss = (self.ent_coef * log_prob.reshape(-1, 1) - min_qf_pi).mean()

            print("Base actor loss computed", flush=True)

            # 3. --- WAYPOINT AUXILIARY LOSS ---
            print("Computing waypoint loss...", flush=True)
            
            # Get the predicted waypoints from the features extractor
            pred_waypoints = self.policy.features_extractor.last_pred_wp
            
            if pred_waypoints is not None:
                # Get the ground truth waypoints from the observations
                gt_waypoints = replay_data.observations['gt_waypoints']  # Shape: (batch_size, 4, 2)
                
                # Compute L1 loss between predicted and ground truth waypoints
                waypoint_loss = F.l1_loss(pred_waypoints, gt_waypoints)
                waypoint_losses.append(waypoint_loss.item())
                
                # Add auxiliary loss to actor loss
                total_actor_loss = actor_loss + self.aux_loss_weight * waypoint_loss
                print(f"Waypoint loss: {waypoint_loss.item():.6f}", flush=True)
            else:
                total_actor_loss = actor_loss
                waypoint_losses.append(0.0)
                print("No waypoint predictions available", flush=True)

            actor_losses.append(total_actor_loss.item())

            print("Actor loss computed, starting backward pass...", flush=True)

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor.optimizer.step()

            print("Actor update completed", flush=True)

            # 4. --- ENTROPY COEFFICIENT UPDATE ---
            print("Starting entropy coefficient update...", flush=True)
            
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't differentiate it
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_losses.append(ent_coef_loss.item())

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            print("Entropy coefficient update completed", flush=True)

            # 5. --- TARGET NETWORK UPDATE ---
            print("Starting target network update...", flush=True)
            
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            print("Target network update completed", flush=True)

        self._n_updates += gradient_steps

        print("Logging results...", flush=True)

        # --- Log the mean of all collected losses ---
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/waypoint_loss", np.mean(waypoint_losses))
        if len(ent_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_losses))
        self.logger.record("train/ent_coef", self.log_ent_coef.exp().item())

        print("--- CUSTOM SAC TRAIN COMPLETED ---", flush=True)