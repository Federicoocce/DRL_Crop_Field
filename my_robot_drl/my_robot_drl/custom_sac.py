# custom_sac.py

import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class CustomSAC(SAC):
    def __init__(self, *args, aux_loss_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Initialize lists to store losses for logging
        actor_losses, critic_losses, ent_losses, waypoint_losses = [], [], [], []

        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # --- Calculate all losses ---
            sac_losses = self._get_sac_losses(replay_data)
            
            # Append SAC losses to their lists
            actor_losses.append(sac_losses['actor_loss'].item())
            critic_losses.append(sac_losses['critic_loss'].item())
            ent_losses.append(sac_losses['ent_loss'].item())
            
            # --- Calculate our custom auxiliary loss ---
            # The features_extractor has been run during the SAC loss calculation,
            # so `last_pred_wp` is populated and ready.
            pred_waypoints = self.policy.features_extractor.last_pred_wp
            gt_waypoints = replay_data.observations['gt_waypoints']
            waypoint_loss = F.l1_loss(pred_waypoints, gt_waypoints)
            
            # Append the waypoint loss to its list
            waypoint_losses.append(waypoint_loss.item())
            
            # --- Combine all losses for backpropagation ---
            total_loss = sac_losses['actor_loss'] + sac_losses['critic_loss'] + sac_losses['ent_loss']
            total_loss += self.aux_loss_weight * waypoint_loss

            # --- Optimization ---
            self.policy.optimizer.zero_grad()
            total_loss.backward()
            self.policy.optimizer.step()

            # --- Update Entropy Coefficient ---
            if self.ent_coef_optimizer is not None:
                with torch.no_grad():
                     _, log_prob = self.actor.action_log_prob(replay_data.observations)
                ent_loss_alpha = -torch.mean(self.log_ent_coef * (log_prob + self.target_entropy).detach())
                self.ent_coef_optimizer.zero_grad()
                ent_loss_alpha.backward()
                self.ent_coef_optimizer.step()

            # --- Update Target Networks ---
            if self._n_updates % self.target_update_interval == 0:
                self.update_target_networks()
            
            self._n_updates += 1

        # --- Log the mean of all collected losses ---
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ent_loss", np.mean(ent_losses))
        self.logger.record("train/waypoint_loss", np.mean(waypoint_losses))
        if self.ent_coef_optimizer is not None:
             self.logger.record("train/ent_coef", self.log_ent_coef.exp().item())

    def _get_sac_losses(self, replay_data: ReplayBufferSamples) -> dict:
        """
        Calculates the standard actor, critic, and entropy losses for SAC.
        This forward pass through the policy also populates the 
        `self.policy.features_extractor.last_pred_wp` attribute.
        """
        # --- Critic Loss ---
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            target_q_values_list = self.critic_target(replay_data.next_observations, next_actions)
            target_q_values, _ = torch.min(torch.cat(target_q_values_list, dim=1), dim=1, keepdim=True)
            target_q_values = target_q_values - self.alpha * next_log_prob
            target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q_values
        
        current_q_values_list = self.critic(replay_data.observations, replay_data.actions)
        critic_loss = sum(F.mse_loss(current_q, target_q) for current_q in current_q_values_list)

        # --- Actor and Entropy Loss ---
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        q_values_pi_list = self.critic(replay_data.observations, actions_pi)
        q_values_pi, _ = torch.min(torch.cat(q_values_pi_list, dim=1), dim=1, keepdim=True)
        
        actor_loss = (self.alpha * log_prob - q_values_pi).mean()
        ent_loss = -self.alpha.detach() * log_prob.mean()

        return {'actor_loss': actor_loss, 'critic_loss': critic_loss, 'ent_loss': ent_loss}

    def update_target_networks(self) -> None:
        """Helper function to update the critic target networks."""
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)