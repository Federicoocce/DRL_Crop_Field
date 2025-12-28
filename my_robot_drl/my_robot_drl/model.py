from collections import deque
import torch.nn.functional as F
import cv2

from .utils import *
from .transfuser import TransfuserBackbone, SegDecoder, DepthDecoder
# from geometric_fusion import GeometricFusionBackbone
# from late_fusion import LateFusionBackbone
# from latentTF import latentTFBackbone
from copy import deepcopy
# from point_pillar import PointPillarNet


from PIL import Image, ImageFont, ImageDraw
from torchvision import models

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn



class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class LidarCenterNet(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        in_channels: input channels
    """

    def __init__(self, config, device, backbone, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        self.use_target_point_image = config.use_target_point_image
        self.gru_concat_target_point = config.gru_concat_target_point
        self.use_point_pillars = config.use_point_pillars

        if(self.use_point_pillars == True):
            self.point_pillar_net = PointPillarNet(config.num_input, config.num_features,
                                                   min_x = config.min_x, max_x = config.max_x,
                                                   min_y = config.min_y, max_y = config.max_y,
                                                   pixels_per_meter = int(config.pixels_per_meter),
                                                  )

        self.backbone = backbone


        if(backbone == 'transFuser'):
            self._model = TransfuserBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif(backbone == 'late_fusion'):
            self._model = LateFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif(backbone == 'geometric_fusion'):
            self._model = GeometricFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'latentTF'):
            self._model = latentTFBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        else:
            raise("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

        # --- MODIFIED: Conditional Decoder Initialization ---
        if config.multitask:
            # Only initialize Front Semantic Decoder if enabled
            if config.use_aux_semantic:
                self.seg_decoder = SegDecoder(self.config, self.config.perception_output_features).to(self.device)
            else:
                self.seg_decoder = None
                
            # Only initialize Depth Decoder if enabled
            if config.use_aux_depth:
                self.depth_decoder = DepthDecoder(self.config, self.config.perception_output_features).to(self.device)
            else:
                self.depth_decoder = None
        
        # BEV Head (Usually initialized regardless, but loss depends on config)
        channel = config.channel
        # --- MODIFIED: Output 4 classes instead of 3 ---
        self.pred_bev = nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channel, 4, kernel_size=(1, 1), stride=1, padding=0, bias=True) # 4 classes: Unk, Drive, Obs, Goal
        ).to(self.device)

        # prediction heads

        self.i = 0

        # waypoints prediction
        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)

        self.decoder = nn.GRUCell(input_size=4 if self.gru_concat_target_point else 2, # 2 represents x,y coordinate
                                  hidden_size=self.config.gru_hidden_size).to(self.device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # If predicting uncertainty, we need 2 outputs for (dx, dy) AND 2 for (log_var_x, log_var_y)
        # Original was 3 (dx, dy, unused/brake?). We adapt to 2 + 2 = 4.
        out_dim = 4 if self.config.predict_uncertainty else 3 
        self.output = nn.Linear(self.config.gru_hidden_size, out_dim).to(self.device)
        # pid controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

    def forward_gru(self, z, target_point):
        z = self.join(z)
    
        output_wp = list()
        output_log_var = list() # Store uncertainty predictions
        
        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

        target_point = target_point.clone()
        target_point[:, 1] *= -1
        
        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            if self.gru_concat_target_point:
                x_in = torch.cat([x, target_point], dim=1)
            else:
                x_in = x
            
            z = self.decoder(x_in, z)
            dx_full = self.output(z) # Get full output
            
            # Extract coordinates
            dx = dx_full[:, :2]
            x = dx + x
            output_wp.append(x[:,:2])
            
            # --- MODIFICATION START ---
            if self.config.predict_uncertainty:
                # Extract log variance (last 2 dimensions)
                log_var = dx_full[:, 2:4]
                output_log_var.append(log_var)
            # --- MODIFICATION END ---
            
        pred_wp = torch.stack(output_wp, dim=1)
        
        # Stack uncertainty if enabled
        pred_log_var = None
        if self.config.predict_uncertainty:
            pred_log_var = torch.stack(output_log_var, dim=1)

        # pred the waypoints in the vehicle coordinate...
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - self.config.lidar_pos[0]
            
        pred_brake = None
        steer = None
        throttle = None
        brake = None

        return pred_wp, pred_brake, steer, throttle, brake, pred_log_var
    
    def force_dropout_active(self):
        """
        Sets the model to eval mode (fixing BatchNorm), 
        but forces Dropout layers to train mode for MC Inference.
        """
        self.eval() # Default: Set everything to eval (BN stats fixed)
        
        # Iterate over all modules and enable Dropout specifically
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
    def control_pid(self, waypoints, velocity, is_stuck):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        # when training we transform the waypoints to lidar coordinate, so we need to change is back when control
        waypoints[:, 0] += self.config.lidar_pos[0]

        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0

        if is_stuck:
            desired_speed = np.array(self.config.default_speed) # default speed of 14.4 km/h

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if (speed < 0.01):
            angle = 0.0  # When we don't move we don't want the angle error to accumulate in the integral
        if brake:
            angle = 0.0
        
        steer = self.turn_controller.step(angle)

        steer = np.clip(steer, -1.0, 1.0) #Valid steering values are in [-1,1]

        return steer, throttle, brake
    
    def forward_ego(self, rgb, lidar_bev, target_point, target_point_image, ego_vel, bev_points=None, cam_points=None, save_path=None, expert_waypoints=None,
                    stuck_detector=0, forced_move=False, num_points=None, rgb_back=None, debug=False):
        
        if(self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3)) #For consitency this is also done in voxelization

        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        if (self.backbone == 'transFuser'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'late_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'geometric_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel, bev_points, cam_points)
        elif (self.backbone == 'latentTF'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        else:
            raise ("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

        pred_wp, _, _, _, _, pred_log_var = self.forward_gru(fused_features, target_point)

        # We are no longer predicting bounding boxes, so return an empty list
        rotated_bboxes = []
        self.i += 1
        if debug and self.i % 2 == 0 and not (save_path is None):
            pred_bev = self.pred_bev(features[0])
            pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width), mode='bilinear', align_corners=True)
            pred_semantic = self.seg_decoder(image_features_grid)
            pred_depth = self.depth_decoder(image_features_grid)

            self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                            pred_wp, pred_bev, pred_semantic, pred_depth,  self.device,
                            gt_bboxes=None, expert_waypoints=expert_waypoints, stuck_detector=stuck_detector, forced_move=forced_move)


        return pred_wp, pred_log_var

    def forward(self, rgb, lidar_bev, ego_waypoint, target_point, target_point_image, ego_vel, bev, depth, semantic, num_points=None, save_path=None, bev_points=None, cam_points=None):
        loss = {}
        preds = None
        pred_bev = None
        pred_depth = None
        pred_semantic = None    

        if(self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3)) #For consitency this is also done in voxelization


        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        if (self.backbone == 'transFuser'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'late_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'geometric_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel, bev_points, cam_points)
        elif (self.backbone == 'latentTF'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        else:
            raise ("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")


        # Unpack result
        pred_wp, _, _, _, _, pred_log_var = self.forward_gru(fused_features, target_point)
        if self.config.predict_uncertainty and pred_log_var is not None:
            # ALEATORIC UNCERTAINTY LOSS (Gaussian NLL)
            squared_error = (pred_wp - ego_waypoint) ** 2
            log_var = torch.clamp(pred_log_var, min=self.config.uncertainty_min_log_var, max=self.config.uncertainty_max_log_var)
            precision = torch.exp(-log_var)
            loss_wp_elementwise = 0.5 * precision * squared_error + 0.5 * log_var
            loss_wp = torch.mean(loss_wp_elementwise)
            
            loss.update({"loss_wp": loss_wp})
            
            # --- ADDED: Log helpful metrics for humans ---
            loss.update({"mean_uncertainty": torch.mean(log_var)}) 
            # Calculate pure L1 error (actual distance in meters) for logging only
            loss.update({"metric_l1": torch.mean(torch.abs(pred_wp - ego_waypoint))})
            
        else:
            # ... standard L1 loss ...
            loss_wp = torch.mean(torch.abs(pred_wp - ego_waypoint))
            loss.update({"loss_wp": loss_wp})
            loss.update({"metric_l1": loss_wp}) # In this case, loss IS the metric
            # Now, ONLY calculate auxiliary losses if their weight is not zero.
            # This makes the code robust to disabling tasks via the config.
                # --- MODIFIED: Auxiliary Losses ---

         # 2. BEV Semantic Loss
        if self.config.multitask and self.config.use_aux_bev:
            pred_bev = self.pred_bev(features[0])
            pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width), mode='bilinear', align_corners=True)
            
            # --- MODIFIED: Added weight for the 4th class (Goal) ---
            # Weights: [Unk, Drive, Obs, Goal]
            # We give Goal a high weight because it is a very small object (sparse).
            weight = torch.tensor([1.0, 1.0, 5.0, 5.0], dtype=torch.float32, device=pred_bev.device)
            
            loss_bev = F.cross_entropy(pred_bev, bev, weight=weight).mean()
            loss.update({"loss_bev": self.config.ls_bev * loss_bev})

        # 3. Depth Loss
        if self.config.multitask and self.config.use_aux_depth and self.depth_decoder is not None:
            pred_depth = self.depth_decoder(image_features_grid)
            loss_depth = self.config.ls_depth * F.l1_loss(pred_depth, depth).mean()
            loss.update({"loss_depth": loss_depth})

        # 4. Front Semantic Loss
        if self.config.multitask and self.config.use_aux_semantic and self.seg_decoder is not None:
            pred_semantic = self.seg_decoder(image_features_grid)
            loss_semantic = self.config.ls_seg * F.cross_entropy(pred_semantic, semantic).mean()
            loss.update({"loss_semantic": loss_semantic})
        # # Check if BEV loss is active
        # if self.config.detailed_losses_weights[self.config.detailed_losses.index('loss_bev')] > 0:
        #     pred_bev = self.pred_bev(features[0])
        #     pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width), mode='bilinear', align_corners=True)
        #     weight = torch.from_numpy(np.array([1., 1., 3.])).to(dtype=torch.float32, device=pred_bev.device)
        #     loss_bev = F.cross_entropy(pred_bev, bev, weight=weight).mean()
        #     loss.update({"loss_bev": loss_bev})

        # # Check if any detection loss is active
        # detection_loss_weights = [
        #     self.config.detailed_losses_weights[self.config.detailed_losses.index('loss_center_heatmap')],
        #     self.config.detailed_losses_weights[self.config.detailed_losses.index('loss_wh')],
        #     self.config.detailed_losses_weights[self.config.detailed_losses.index('loss_offset')]
        # ]
        # if any(w > 0 for w in detection_loss_weights):
        #     preds = self.head([features[0]])
        #     gt_labels = torch.zeros_like(label[:, :, 0])
        #     gt_bboxes_ignore = label.sum(dim=-1) == 0.
        #     loss_bbox = self.head.loss(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6],
        #                             [label], gt_labels=[gt_labels], gt_bboxes_ignore=[gt_bboxes_ignore], img_metas=None)
        #     loss.update(loss_bbox)
        
        # # --- END OF THE FIX ---

        # # This part was already correct
        # if self.config.multitask:
        #     #pred_semantic = self.seg_decoder(image_features_grid)
        #     pred_depth = self.depth_decoder(image_features_grid)
        #     #loss_semantic = self.config.ls_seg * F.cross_entropy(pred_semantic, semantic).mean()
        #     loss_depth = self.config.ls_depth * F.l1_loss(pred_depth, depth).mean()
        #     loss.update({
        #         "loss_depth": loss_depth
        #         #"loss_semantic": loss_semantic
        #     })
        self.i += 1
        if ((self.config.debug == True) and (self.i % self.config.train_debug_save_freq == 0) and (save_path != None)):
            with torch.no_grad():
                # HANDLE BBOXES: Check if preds exists
                if preds is not None:
                    results = self.head.get_bboxes(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6])
                    bboxes, _ = results[0]
                    bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]
                else:
                    # If detection was skipped, create an empty bbox tensor
                    bboxes = torch.zeros((0, 8)).to(self.device)


                self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                                   pred_wp, 
                                   pred_bev,        # May be None
                                   pred_semantic,   # May be None
                                   pred_depth,      # May be None
                                   bboxes, self.device,
                                    expert_waypoints=ego_waypoint, stuck_detector=0, forced_move=False,
                                   gt_depth=depth)
        return loss


    # Converts the coordinate system to x front y right, vehicle center at the origin.
    # Units are converted from pixels to meters
    def get_bbox_local_metric(self, bbox):
        x, y, w, h, yaw, speed, brake, confidence = bbox

        w = w / self.config.bounding_box_divisor / self.config.pixels_per_meter # We multiplied by 2 when collecting the data, and multiplied by 8 when loading the labels.
        h = h / self.config.bounding_box_divisor / self.config.pixels_per_meter # We multiplied by 2 when collecting the data, and multiplied by 8 when loading the labels.

        T = get_lidar_to_bevimage_transform()
        T_inv = np.linalg.inv(T)

        center = np.array([x,y,1.0])

        center_old_coordinate_sys = T_inv @ center

        center_old_coordinate_sys = center_old_coordinate_sys + np.array(self.config.lidar_pos)

        #Convert to standard CARLA right hand coordinate system
        center_old_coordinate_sys[1] =  -center_old_coordinate_sys[1]

        bbox = np.array([[-h, -w, 1],
                         [-h,  w, 1],
                         [ h,  w, 1],
                         [ h, -w, 1],
                         [ 0,  0, 1],
                         [ 0, h * speed * 0.5, 1]])

        R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0,                      0, 1]])

        for point_index in range(bbox.shape[0]):
            bbox[point_index] = R @ bbox[point_index]
            bbox[point_index] = bbox[point_index] + np.array([center_old_coordinate_sys[0], center_old_coordinate_sys[1],0])

        return bbox, brake, confidence

    # this is different
    def get_rotated_bbox(self, bbox):
        x, y, w, h, yaw, speed, brake =  bbox

        bbox = np.array([[h,   w, 1],
                         [h,  -w, 1],
                         [-h, -w, 1],
                         [-h,  w, 1],
                         [0, 0, 1],
                         [-h * speed * 0.5, 0, 1]])
        bbox[:, :2] /= self.config.bounding_box_divisor
        bbox[:, :2] = bbox[:, [1, 0]]

        c, s = np.cos(yaw), np.sin(yaw)
        # use y x because coordinate is changed
        r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

        bbox = r1_to_world @ bbox.T
        bbox = bbox.T

        return bbox, brake

    def draw_bboxes(self, bboxes, image, color=(255, 255, 255), brake_color=(0, 0, 255)):
        idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
        for bbox, brake in bboxes:
            bbox = bbox.astype(np.int32)[:, :2]
            for s, e in idx:
                if brake >= self.config.draw_brake_threshhold:
                    color = brake_color
                else:
                    color = color
                # brake is true while still have high velocity
                cv2.line(image, tuple(bbox[s]), tuple(bbox[e]), color=color, thickness=1)
        return image


    def draw_waypoints(self, label, waypoints, image, color=(255, 255, 255)):
            """
            Draws waypoints on the image.
            Assumes waypoints are in Ego-Frame (Meters): [x_forward, y_left].
            Matches projection logic of transfuser_util.py and draw_target_point.
            """
            # Handle Tensor vs Numpy
            if isinstance(waypoints, torch.Tensor):
                waypoints = waypoints.detach().cpu().numpy()
            
            # Config dimensions
            crop_size = image.shape[0]
            half_crop = crop_size / 2.0
            ppm = self.config.pixels_per_meter

            # Iterate over batch (usually batch size is 1 for visualization)
            for i in range(len(waypoints)):
                # Points: (Seq_Len, 2)
                points_m = waypoints[i] 
                
                pixel_coords = []
                
                for point in points_m:
                    x_m, y_m = point[0], point[1]
                    
                    # Projection Logic:
                    # X (Forward) -> Row (Up/Negative)
                    # Y (Left)    -> Col (Left/Negative relative to center)
                    # Formula: Center - (Coordinate * PPM)
                    
                    col = int(half_crop - (y_m * ppm))
                    row = int(half_crop - (x_m * ppm))
                    
                    pixel_coords.append([col, row])
                
                pixel_coords = np.array(pixel_coords, dtype=np.int32)

                # 1. Draw Lines (Trajectory Path)
                if len(pixel_coords) > 1:
                    # Reshape for polylines: (-1, 1, 2)
                    pts = pixel_coords.reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=2)

                # 2. Draw Points (Dots)
                for p in pixel_coords:
                    # Bounds check
                    if 0 <= p[0] < crop_size and 0 <= p[1] < crop_size:
                        cv2.circle(image, tuple(p), radius=4, color=color, thickness=-1)
                        
            return image


    def draw_target_point(self, target_point, image, color=(0, 255, 0)):
            """
            Draws the target point on the image using the correct BEV projection logic.
            Matches transfuser_util.py coordinate system.
            """
            # Ensure target_point is numpy
            if isinstance(target_point, torch.Tensor):
                target_point = target_point.cpu().numpy()

            # Config parameters
            crop_size = image.shape[0]  # Assuming square (e.g., 256)
            ppm = self.config.pixels_per_meter # e.g., 64
            
            # Assume the map is centered (range +/- 2.0m for 256px @ 64ppm)
            # Calculate max range in meters based on image size
            # x_range_meters = (crop_size / ppm) / 2.0 
            
            # Dimensions 
            half_crop = crop_size / 2.0

            # Projection Logic (Matches transfuser_util.py)
            # Y (Left/Right) -> Column. Positive Y is Left.
            # Image Col 0 is Max Left. Center is 0. Max Col is Max Right.
            # col = center + (-y * ppm)  <-- This puts positive Y (Left) to the left (smaller col index)?
            # Wait, usually image origin (0,0) is top-left.
            # If we want standard visualization:
            # Col: Center + (-y * ppm) => If y is positive (left), col decreases (moves left). Correct.
            col = int(half_crop + (-target_point[1] * ppm))
            
            # X (Forward) -> Row. Positive X is Forward.
            # Image Row 0 is Top.
            # Row: Center - (x * ppm) => If x is positive (forward), row decreases (moves up). Correct.
            # Adjusting offset to ensure (0,0) robot is at the center
            # Note: transfuser_util uses specific logic: feature_map_height - 1 - ((x + range) * ppm)
            # which simplifies to: (range * ppm) - (x * ppm) = center - (x * ppm)
            row = int(half_crop - (target_point[0] * ppm))

            # Draw if within bounds
            if 0 <= col < crop_size and 0 <= row < crop_size:
                # Use radius 5 to be visible
                cv2.circle(image, (col, row), radius=5, color=color, thickness=-1)
            
            return image
    
    def visualize_model_io(self, save_path, step, config, rgb, lidar_bev, target_point,
                        pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, device,
                        gt_bboxes=None, expert_waypoints=None, stuck_detector=0, forced_move=False,
                        gt_depth=None):
        
        i = 0 # Visualize first element of batch

        # --- 1. RGB (Front Camera) ---
        # Permute (C, H, W) -> (H, W, C) and convert to BGR
        rgb_image = rgb[i].permute(1, 2, 0).detach().cpu().numpy() 
        rgb_image = rgb_image[:, :, [2, 1, 0]] 
        rgb_image = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
        
        # --- 2. Semantic Prediction (Front) ---
        if pred_semantic is not None:
            # pred_semantic: (B, Num_Classes, H, W)
            # Resize to match RGB for visualization if needed, or keep original
            sem_logits = pred_semantic[i].detach().cpu()
            sem_idx = sem_logits.argmax(dim=0).numpy().astype(np.uint8)
            
            # Semantic Color Map (BGR)
            # 0: Unk (Black), 1: Drive (Gray), 2: Obs (Red)
            # Adjust these to match your dataset classes
            sem_colors = np.array([
                [0, 0, 0],       # 0: Unknown / Background
                [128, 128, 128], # 1: Drivable (Gray)
                [0, 0, 255],     # 2: Obstacle (Red)
                [0, 255, 0],     # 3: Other?
                [255, 0, 0]      # 4: Other?
            ], dtype=np.uint8)
            
            # Handle class indices larger than color map
            sem_idx[sem_idx >= len(sem_colors)] = 0
            
            sem_viz = sem_colors[sem_idx]
            
            # Resize to match RGB height if different
            if sem_viz.shape[:2] != rgb_image.shape[:2]:
                sem_viz = cv2.resize(sem_viz, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            cv2.putText(sem_viz, "Pred Semantic", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            sem_viz = np.zeros_like(rgb_image)

        # --- 3. Depth (Combined GT & Pred) ---
        depth_stack = None
        
        # Helper to process depth
        def process_depth(d_tensor, label):
            d_img = d_tensor.detach().cpu().numpy()
            if d_img.ndim == 3: d_img = d_img[0] # Handle (1, H, W)
            d_img = (np.clip(d_img, 0, 1) * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_img, cv2.COLORMAP_MAGMA)
            cv2.putText(d_color, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return d_color

        if pred_depth is not None:
            depth_stack = process_depth(pred_depth[i], "Pred Depth")

        if gt_depth is not None:
            gt_d_viz = process_depth(gt_depth[i], "GT Depth")
            if depth_stack is not None:
                depth_stack = np.concatenate((gt_d_viz, depth_stack), axis=0) # Vertical stack
            else:
                depth_stack = gt_d_viz
        
        # If no depth, make placeholder
        if depth_stack is None:
            depth_stack = np.zeros_like(rgb_image)
        else:
            # Resize depth stack width to match RGB width
            d_h, d_w = depth_stack.shape[:2]
            scale = rgb_image.shape[1] / d_w
            depth_stack = cv2.resize(depth_stack, (rgb_image.shape[1], int(d_h * scale)))

        # --- 4. BEV Input (Ground Truth) ---
        # lidar_bev tensor shape: (C, H, W). 
        # Typically: Ch0=Below(Ground), Ch1=Above(Obs), and if concat=True, Ch2=Target
        lidar_data = lidar_bev[i].detach().cpu().numpy()
        bev_h, bev_w = lidar_data.shape[1], lidar_data.shape[2]
        bev_input_viz = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

        # Map channels to colors (BGR)
        # Blue: Ground (Ch0)
        bev_input_viz[:, :, 0] = (lidar_data[0] * 255).astype(np.uint8)
        # Red: Obstacle (Ch1)
        bev_input_viz[:, :, 2] = (lidar_data[1] * 255).astype(np.uint8)
        
        # Green: Target (Ch2 if exists)
        if lidar_data.shape[0] > 2:
             bev_input_viz[:, :, 1] = (lidar_data[2] * 255).astype(np.uint8)

        # Draw Waypoints on BEV Input
        label_dummy = torch.zeros((1, 1, 7)).to(device) # Dummy label for helper
        if expert_waypoints is not None:
            bev_input_viz = self.draw_waypoints(label_dummy[0], expert_waypoints[i:i+1], bev_input_viz, color=(0, 255, 255)) # Yellow
        
        bev_input_viz = self.draw_waypoints(label_dummy[0], deepcopy(pred_wp[i:i + 1]), bev_input_viz, color=(255, 0, 0)) # Blue
        cv2.putText(bev_input_viz, "BEV Input (GT)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # --- 5. BEV Prediction (4 Classes) ---
        if pred_bev is not None:
            # pred_bev: (B, 4, H, W) -> Argmax -> (H, W)
            bev_cls = pred_bev[i].detach().cpu().numpy().argmax(axis=0).astype(np.uint8)
            
            # Color Map for 4 Classes (BGR)
            # 0: Unk (Black), 1: Drive (Gray), 2: Obs (Red), 3: Goal (Magenta)
            bev_colors = np.array([
                [0, 0, 0],       # 0: Unknown
                [100, 100, 100], # 1: Drivable
                [0, 0, 255],     # 2: Obstacle
                [255, 0, 255]    # 3: Goal
            ], dtype=np.uint8)
            
            bev_pred_viz = bev_colors[bev_cls]
            
            # Draw Waypoints overlay on Prediction
            if expert_waypoints is not None:
                bev_pred_viz = self.draw_waypoints(label_dummy[0], expert_waypoints[i:i+1], bev_pred_viz, color=(0, 255, 255))
            bev_pred_viz = self.draw_waypoints(label_dummy[0], deepcopy(pred_wp[i:i + 1]), bev_pred_viz, color=(255, 0, 0))
            
            # Draw Target Point overlay
            bev_pred_viz = self.draw_target_point(target_point[i], bev_pred_viz, color=(0, 255, 0))
            
            cv2.putText(bev_pred_viz, "BEV Pred", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            bev_pred_viz = np.zeros_like(bev_input_viz)

        # --- 6. Final Layout Assembly ---
        # Resize BEVs to 256x256 if they aren't already
        target_bev_size = (256, 256)
        if bev_input_viz.shape[:2] != target_bev_size:
            bev_input_viz = cv2.resize(bev_input_viz, target_bev_size, interpolation=cv2.INTER_NEAREST)
        if bev_pred_viz.shape[:2] != target_bev_size:
            bev_pred_viz = cv2.resize(bev_pred_viz, target_bev_size, interpolation=cv2.INTER_NEAREST)

        # Column 1: RGB + Semantic
        col1 = np.concatenate((rgb_image, sem_viz), axis=0)
        
        # Column 2: BEV Input + BEV Pred
        col2 = np.concatenate((bev_input_viz, bev_pred_viz), axis=0)
        # Resize Col2 to match Col1 height logic if needed, but usually we just stack nicely
        # Let's ensure Col2 width matches Col1 for cleaner concatenation? 
        # Actually, separate columns is better.
        
        # Resize Col2 to have same width as Col1
        scale = col1.shape[1] / col2.shape[1]
        col2 = cv2.resize(col2, (col1.shape[1], int(col2.shape[0] * scale)))

        # Column 3: Depth Stack
        # Resize to match Col1 width
        scale = col1.shape[1] / depth_stack.shape[1]
        col3 = cv2.resize(depth_stack, (col1.shape[1], int(depth_stack.shape[0] * scale)))

        # Final Grid: [Col1, Col2, Col3] side by side
        # Pad heights to match the tallest column
        max_h = max(col1.shape[0], col2.shape[0], col3.shape[0])
        
        def pad_img(img, target_h):
            if img.shape[0] < target_h:
                pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                return np.concatenate((img, pad), axis=0)
            return img

        final_image = np.concatenate((pad_img(col1, max_h), pad_img(col2, max_h), pad_img(col3, max_h)), axis=1)

        # Save
        cv2.imwrite(str(save_path + ("/%d.png" % (step // 2))), final_image)
    # def visualize_model_io(self, save_path, step, config, rgb, lidar_bev, target_point,
    #                     pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, device,
    #                     gt_bboxes=None, expert_waypoints=None, stuck_detector=0, forced_move=False):
    #     font = ImageFont.load_default()
    #     i = 0 # We only visualize the first image if there is a batch of them.
    #     if config.multitask:
    #         classes_list = config.classes_list
    #         converter = np.array(classes_list)

    #         depth_image = pred_depth[i].detach().cpu().numpy()

    #         indices = np.argmax(pred_semantic.detach().cpu().numpy(), axis=1)
    #         semantic_image = converter[indices[i, ...], ...].astype('uint8')

    #         ds_image = np.stack((depth_image, depth_image, depth_image), axis=2)
    #         ds_image = (ds_image * 255).astype(np.uint8)
    #         ds_image = np.concatenate((ds_image, semantic_image), axis=0)
    #         ds_image = cv2.resize(ds_image, (640, 256))
    #         ds_image = np.concatenate([ds_image, np.zeros_like(ds_image[:50])], axis=0)

    #     images = np.concatenate(list(lidar_bev.detach().cpu().numpy()[i][:2]), axis=1)
    #     images = (images * 255).astype(np.uint8)
    #     images = np.stack([images, images, images], axis=-1)
    #     images = np.concatenate([images, np.zeros_like(images[:50])], axis=0)

    #     # draw bbox GT
    #     if (not (gt_bboxes is None)):
    #         rotated_bboxes_gt = []
    #         for bbox in gt_bboxes.detach().cpu().numpy()[i]:
    #             bbox = self.get_rotated_bbox(bbox)
    #             rotated_bboxes_gt.append(bbox)
    #         images = self.draw_bboxes(rotated_bboxes_gt, images, color=(0, 255, 0), brake_color=(0, 255, 128))

    #     rotated_bboxes = []
    #     for bbox in bboxes.detach().cpu().numpy():
    #         bbox = self.get_rotated_bbox(bbox[:7])
    #         rotated_bboxes.append(bbox)
    #     images = self.draw_bboxes(rotated_bboxes, images, color=(255, 0, 0), brake_color=(0, 255, 255))

    #     label = torch.zeros((1, 1, 7)).to(device)
    #     label[:, -1, 0] = 128.
    #     label[:, -1, 1] = 256.

    #     if not expert_waypoints is None:
    #         images = self.draw_waypoints(label[0], expert_waypoints[i:i+1], images, color=(0, 0, 255))

    #     images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, 2:]), images, color=(255, 255, 255)) # Auxliary waypoints in white
    #     images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, :2]), images, color=(255, 0, 0))     # First two, relevant waypoints in blue

    #     # draw target points
    #     images = self.draw_target_point(target_point[i].detach().cpu().numpy(), images)

    #     # stuck text
    #     images = Image.fromarray(images)
    #     draw = ImageDraw.Draw(images)
    #     draw.text((10, 0), "stuck detector:   %04d" % (stuck_detector), font=font)
    #     draw.text((10, 30), "forced move:      %s" % (" True" if forced_move else "False"), font=font,
    #               fill=(255, 0, 0, 255) if forced_move else (255, 255, 255, 255))
    #     images = np.array(images)

    #     bev = pred_bev[i].detach().cpu().numpy().argmax(axis=0) / 2.
    #     bev = np.stack([bev, bev, bev], axis=2) * 255.
    #     bev_image = bev.astype(np.uint8)
    #     bev_image = cv2.resize(bev_image, (256, 256))
    #     bev_image = np.concatenate([bev_image, np.zeros_like(bev_image[:50])], axis=0)

    #     if not expert_waypoints is None:
    #         bev_image = self.draw_waypoints(label[0], expert_waypoints[i:i+1], bev_image, color=(0, 0, 255))

    #     bev_image = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, 2:]), bev_image, color=(255, 255, 255))
    #     bev_image = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, :2]), bev_image, color=(255, 0, 0))

    #     bev_image = self.draw_target_point(target_point[i].detach().cpu().numpy(), bev_image)

    #     if (not (expert_waypoints is None)):
    #         aim = expert_waypoints[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
    #         expert_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))

    #         aim = pred_wp[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
    #         ego_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))
    #         angle_error = normalize_angle_degree(expert_angle - ego_angle)

    #         bev_image = Image.fromarray(bev_image)
    #         draw = ImageDraw.Draw(bev_image)
    #         draw.text((0, 0), "Angle error:        %.2f°" % (angle_error), font=font)

    #     bev_image = np.array(bev_image)

    #     rgb_image = rgb[i].permute(1, 2, 0).detach().cpu().numpy()[:, :, [2, 1, 0]]
    #     rgb_image = cv2.resize(rgb_image, (1280 + 128, 320 + 32))
    #     assert (config.multitask)
    #     images = np.concatenate((bev_image, images, ds_image), axis=1)

    #     images = np.concatenate((rgb_image, images), axis=0)

    #     cv2.imwrite(str(save_path + ("/%d.png" % (step // 2))), images)