# custom_features_extractor.py

import torch
from torch import optim
from gymnasium import spaces
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .model import TransFuser
from .config import GlobalConfig

class TransFuserFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 72, lr: float = 1e-4):
        super().__init__(observation_space, features_dim=features_dim)
        self.config = GlobalConfig()
        self.transfuser = TransFuser(self.config, 'cuda') 
        # REMOVED: self.last_pred_wp = None. We will not use this anti-pattern.
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        print("TransFuserFeaturesExtractor initialized with its own Adam optimizer.", flush=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_transfuser_inputs(self, observations: dict) -> tuple:
        """Helper to extract and format inputs for the TransFuser model."""
        all_obs_tensors = {}
        current_device = self.device
        
        for key, obs in observations.items():
            all_obs_tensors[key] = torch.as_tensor(obs, device=current_device)

        image_obs = all_obs_tensors['image'].float() / 255.0
        lidar_obs = all_obs_tensors['lidar_bev'].float() / 255.0
        state_obs = all_obs_tensors['state']

        image_list, lidar_list = [image_obs], [lidar_obs]
        
        target_point = state_obs[:, 0:2]
        if state_obs.dim() > 1:
             velocity = state_obs[:, 2].unsqueeze(1)
        else:
             velocity = state_obs[2].unsqueeze(0).unsqueeze(0)
             
        return image_list, lidar_list, target_point, velocity

    def forward(self, observations: dict) -> torch.Tensor:
        """
        This is the standard forward method used by the SAC actor and critic.
        It MUST return detached features to prevent SAC from training the backbone.
        """
        image_list, lidar_list, target_point, velocity = self._get_transfuser_inputs(observations)
        
        # We need the forward pass to get the features, but we don't want to
        # train the backbone with the SAC loss, so we use no_grad here.
        with torch.no_grad():
            pred_wp, z = self.transfuser(image_list, lidar_list, target_point, velocity)
        
        # The .detach() calls are technically redundant due to no_grad, but it's
        # good practice to be explicit. This is the feature vector for the policy.
        final_features = torch.cat([z.detach(), pred_wp.flatten(start_dim=1).detach()], dim=1)
        
        return final_features

    def train_imitation_learning(self, observations: dict) -> torch.Tensor:
        """
        A dedicated method for the imitation learning update step.
        This keeps the computation graph local to this function call.
        """
        # Get ground truth waypoints
        gt_waypoints = torch.as_tensor(observations['gt_waypoints'], device=self.device)

        # Get model inputs
        image_list, lidar_list, target_point, velocity = self._get_transfuser_inputs(observations)

        # --- CRITICAL ---
        # Perform a forward pass WITH gradient tracking
        pred_wp, _ = self.transfuser(image_list, lidar_list, target_point, velocity)
        
        
        # Calculate the loss
        waypoint_loss = F.l1_loss(pred_wp, gt_waypoints)
        
        # Perform the optimization
        self.optimizer.zero_grad()
        waypoint_loss.backward()
        self.optimizer.step()
        
        # Return the loss value for logging
        return waypoint_loss.item()