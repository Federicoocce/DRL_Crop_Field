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
        """
        Helper to extract and format inputs for the TransFuser model.
        This version correctly processes each camera view individually and prepares
        a list of tensors for the model.
        """
        all_obs_tensors = {}
        current_device = self.device
        
        for key, obs in observations.items():
            all_obs_tensors[key] = torch.as_tensor(obs, device=current_device)

        # --- Reusable function to process any image tensor ---
        def process_image_tensor(img_tensor):
            # Permute from (B, H, W, C) to (B, C, H, W)
            if img_tensor.shape[-1] == 3 and img_tensor.ndim == 4:
                img_tensor = img_tensor.permute(0, 3, 1, 2)
            
            # Normalize to 0-1 range
            return img_tensor.float() / 255.0

        # --- Process each camera view ---
        image_obs_front = process_image_tensor(all_obs_tensors['image'])
        
        # Start building the list of camera views for the model
        image_list = [image_obs_front] 

        # Check for and process the rear camera
        if 'image_rear' in all_obs_tensors:
            image_obs_rear = process_image_tensor(all_obs_tensors['image_rear'])
            image_list.append(image_obs_rear)
        
        # --- Process LiDAR (unchanged, but cleaned up) ---
        lidar_obs = all_obs_tensors['lidar_bev']
        if lidar_obs.shape[-1] == 2 and lidar_obs.ndim == 4:
            lidar_obs = lidar_obs.permute(0, 3, 1, 2)
        # NOTE: The original Lidar processing in the paper does not divide by 255.
        # It expects histogram features. Let's assume your lidar_to_histogram_features
        # already outputs values in a reasonable range. If not, you might remove the division.
        lidar_obs = lidar_obs.float() # Removed / 255.0 for LiDAR
        lidar_list = [lidar_obs]
        
        # --- State processing (unchanged) ---
        state_obs = all_obs_tensors['state']
        target_point = state_obs[:, 0:2]
        velocity = state_obs[:, 2].unsqueeze(1) if state_obs.dim() > 1 else state_obs[2].unsqueeze(0).unsqueeze(0)

        # The image_list now correctly contains a list of processed tensors
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
        
        
        # --- START OF MODIFICATION: Weighted L1 Loss ---
        # Define a weight for the y-axis (lateral) error. 
        # A value > 1.0 puts more emphasis on it.
        y_weight = 20.0

        # Separate the x and y components of the waypoints. Shape is (batch, num_waypoints, 2)
        pred_x = pred_wp[:, :, 0]
        pred_y = pred_wp[:, :, 1]
        gt_x = gt_waypoints[:, :, 0]
        gt_y = gt_waypoints[:, :, 1]

        # Calculate L1 loss for each component separately
        loss_x = F.l1_loss(pred_x, gt_x)
        loss_y = F.l1_loss(pred_y, gt_y)

        # Combine the losses with the desired weight on the y-axis
        waypoint_loss = loss_x + (y_weight * loss_y)
        # --- END OF MODIFICATION ---

        # Perform the optimization
        self.optimizer.zero_grad()
        waypoint_loss.backward()
        self.optimizer.step()
        
        # Return the total loss value for logging
        return waypoint_loss.item()