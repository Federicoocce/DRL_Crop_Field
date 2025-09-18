# custom_features_extractor.py

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .model import TransFuser
from .config import GlobalConfig

class TransFuserFeaturesExtractor(BaseFeaturesExtractor):
    """
    - Runs the full TransFuser model to get both waypoints and the hidden state.
    - Outputs a concatenated feature vector of [hidden_state_z, predicted_waypoints].
    - Stores the predicted waypoints for the custom training loop to access for loss calculation.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 72):
        # features_dim = 64 (for hidden state z) + 4_waypoints * 2_coords = 72
        super().__init__(observation_space, features_dim=features_dim)
        self.config = GlobalConfig()
        self.transfuser = TransFuser(self.config, 'cuda')
        self.last_pred_wp = None

    def forward(self, observations: dict) -> torch.Tensor:
        image_obs = observations['image'].permute(0, 3, 1, 2)
        lidar_obs = observations['lidar_bev'].permute(0, 3, 1, 2)
        state_obs = observations['state']
        
        image_list, lidar_list = [image_obs], [lidar_obs]
        
        # State: [distant_goal_x, y, linear_vel, angular_vel]
        target_point = state_obs[:, 0:2]
        velocity = state_obs[:, 2].unsqueeze(1)

        # 1. Get predicted waypoints and hidden state `z` from the model
        pred_wp, z = self.transfuser(image_list, lidar_list, target_point, velocity)

        # Store the prediction for the training loop's loss calculation
        self.last_pred_wp = pred_wp
        
        # 2. Create the final feature vector for the SAC Actor/Critic
        final_features = torch.cat([z, pred_wp.flatten(start_dim=1)], dim=1)
        
        return final_features