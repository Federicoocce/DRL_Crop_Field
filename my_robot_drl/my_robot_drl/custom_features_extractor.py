# custom_features_extractor.py

import torch
from torch import optim
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from .model import TransFuser
from .config import GlobalConfig

class TransFuserFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 72, lr: float = 1e-4):
        super().__init__(observation_space, features_dim=features_dim)
        self.config = GlobalConfig()
        self.transfuser = TransFuser(self.config, 'cuda') 
        self.last_pred_wp = None
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        print("TransFuserFeaturesExtractor initialized with its own Adam optimizer.", flush=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, observations: dict) -> torch.Tensor:
        all_obs_tensors = {}
        current_device = self.device
        
        for key, obs in observations.items():
            # SB3 handles adding the batch dimension, so obs is usually a tensor
            # We just need to ensure it's on the right device.
            all_obs_tensors[key] = torch.as_tensor(obs, device=current_device)

        image_obs = all_obs_tensors['image'].float() / 255.0
        lidar_obs = all_obs_tensors['lidar_bev'].float() / 255.0
        state_obs = all_obs_tensors['state']

        # --- FIX: REMOVED INCORRECT PERMUTATION ---
        # Stable Baselines 3 automatically converts image observations from (N, H, W, C)
        # to the PyTorch standard (N, C, H, W) before they reach the feature extractor.
        # The permute calls were therefore incorrect and corrupted the tensor shape.
        # image_obs = image_obs.permute(0, 3, 1, 2) # This line was removed.
        # lidar_obs = lidar_obs.permute(0, 3, 1, 2) # This line was removed.
        
        image_list, lidar_list = [image_obs], [lidar_obs]
        
        target_point = state_obs[:, 0:2]
        # Handle batch vs non-batch case for velocity
        if state_obs.dim() > 1:
             velocity = state_obs[:, 2].unsqueeze(1)
        else:
             velocity = state_obs[2].unsqueeze(0).unsqueeze(0)



        
        pred_wp, z = self.transfuser(image_list, lidar_list, target_point, velocity)

        self.last_pred_wp = pred_wp
        
        final_features = torch.cat([z.detach(), pred_wp.flatten(start_dim=1).detach()], dim=1)
  
        
        return final_features