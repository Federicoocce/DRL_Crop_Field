import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
import os
import random
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class RealTimeSemanticMapper:
    def __init__(self, device='cuda', resolution=0.02, initial_map_size=512):
        """
        resolution: 0.02 = 2cm voxels (High Density)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing RealTimeMapper on {self.device} (Strict Filtering | No Sky)...")

        # --- AI Model (Segformer B1) ---
        model_name = "nvidia/segformer-b1-finetuned-ade-512-512" 
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device).half().eval()
        
        # Load Labels for Debugging
        self.id2label = self.model.config.id2label
        self.res = resolution 
        # Track current size dynamically
        self.map_h = initial_map_size
        self.map_w = initial_map_size
        
        # Origin (0,0) of the grid in World Coordinates
        # Initially center the robot (0,0 world) in the middle of the grid
        self.origin_x = -(self.map_w * self.res) / 2.0
        self.origin_y = -(self.map_h * self.res) / 2.0
        
        # 2D Grid [H, W, 3]
        self.grid = torch.zeros((self.map_h, self.map_w, 3), device=self.device, dtype=torch.float32)
        # Higher Hit log-odds to fill map faster
        self.L_HIT = 1.2 
        self.L_MISS = -0.1

        # --- 3D Voxel Grid ---
        self.voxel_grid = {} 
        self.collect_3d = True

        # --- Caching ---
        self.cached_seg_viz = None
        self.frame_count = 0
        
        # --- Calibration ---
        self.img_w, self.img_h = 1024, 576
        self.inference_size = (512, 288)
        self.fx, self.fy = 241.5, 241.5 
        self.cx, self.cy = 512.0, 288.0
        
        self.lidar_to_cam = torch.tensor([
            [0, -1, 0, 0], 
            [0, 0, -1, 0], 
            [1, 0, 0, 0], 
            [0, 0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        self.lidar_x_offset = 0.20

        # --- STRICT FARM MAPPING ---
        # Any ID NOT in these lists will be mapped to 0 (Unknown) and IGNORED by the mapper.
        
        # 1 = DRIVABLE
        self.drivable_ids = [
            9,   # Grass
            13,  # Earth
            14,  # Ground
            29,  # Field
            46,  # Sand
            91,  # Dirt
            6,   # Road
            52,  # Path
            94   # Step/Stair (often misclassified dirt bumps)
        ]

        # 2 = OBSTACLE
        self.obstacle_ids = [
            17,  # Plant
            4,   # Tree
            66,  # Flower
            72,  # Palm
            126, # Pole (often corn stalks)
            12,  # Person
            10   # Fence
        ]
        
        # Debug Colors
        np.random.seed(42)
        self.class_colors = np.random.randint(0, 255, (200, 3), dtype=np.uint8)

    def reset_map(self, start_x, start_y):
        """Resets grid to initial size and centers on new start position."""
        initial_size = 512
        self.map_h = initial_size
        self.map_w = initial_size
        
        self.grid = torch.zeros((self.map_h, self.map_w, 3), device=self.device, dtype=torch.float32)
        
        # Center grid on robot start
        self.origin_x = start_x - (self.map_w * self.res) / 2.0
        self.origin_y = start_y - (self.map_h * self.res) / 2.0
        
        self.voxel_grid = {} 
        self.frame_count = 0
        self.cached_seg_viz = None

    def _expand_map(self, min_u, max_u, min_v, max_v):
        """Dynamically expands the grid tensor if points fall out of bounds."""
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0
        expansion_buffer = 128 # Prevent resizing every frame

        # Check Horizontal
        if min_u < 0: pad_left = abs(min_u) + expansion_buffer
        if max_u >= self.map_w: pad_right = (max_u - self.map_w) + 1 + expansion_buffer
            
        # Check Vertical
        if min_v < 0: pad_top = abs(min_v) + expansion_buffer
        if max_v >= self.map_h: pad_bottom = (max_v - self.map_h) + 1 + expansion_buffer

        if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0: return

        new_w = self.map_w + pad_left + pad_right
        new_h = self.map_h + pad_top + pad_bottom
        
        new_grid = torch.zeros((new_h, new_w, 3), device=self.device, dtype=torch.float32)
        
        # Copy old data into center
        new_grid[pad_top : pad_top + self.map_h, pad_left : pad_left + self.map_w, :] = self.grid
        
        self.grid = new_grid
        self.map_w = new_w
        self.map_h = new_h
        
        # Adjust origin if we padded left/top
        self.origin_x -= (pad_left * self.res)
        self.origin_y -= (pad_top * self.res)

    def process_one_step(self, img_np, lidar_np, pose_x, pose_y, pose_yaw):
            self.frame_count += 1
            
            # [Initialize cache if None ...]
            if self.cached_seg_viz is None:
                self.cached_seg_viz = img_np.copy()
                # Initialize cached mask as zeros
                self.cached_sem_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

            # [Skip frames optimization ...] 
            if self.frame_count % 2 != 0:
                return self.cached_seg_viz, self.cached_sem_mask # <--- CHANGED: Return tuple

            if len(lidar_np) < 10: return self.cached_seg_viz, self.cached_sem_mask

            # --- 1. Prepare Data ---
            img_small = cv2.resize(img_np, self.inference_size)
            inputs = self.processor(images=img_small, return_tensors="pt").to(self.device)
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}
            
            lidar_t = torch.from_numpy(lidar_np).to(self.device).float()
            
            # [Range Filter ...]
            dists = torch.norm(lidar_t, dim=1)
            valid_pts = (dists > 0.15) & (dists < 3.5)
            lidar_t = lidar_t[valid_pts]
            if lidar_t.shape[0] == 0: return self.cached_seg_viz, self.cached_sem_mask

            # --- 2. Inference ---
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = F.interpolate(outputs.logits.float(), size=(self.img_h, self.img_w), 
                                    mode="bilinear", align_corners=False)
                seg_ids = logits.argmax(dim=1)[0]

            # --- 3. Prepare Semantic Mask (NEW) ---
            # Map raw Segformer IDs (0-150) to our Farm Classes (0:Unk, 1:Drive, 2:Obs)
            sem_labels = seg_ids.clamp(0, 199)
            lookup = torch.zeros(200, device=self.device, dtype=torch.uint8)
            lookup[self.drivable_ids] = 1 
            lookup[self.obstacle_ids] = 2 
            
            # Create the dense mask (H, W)
            class_mask = lookup[sem_labels] 
            
            # Cache this mask for return
            self.cached_sem_mask = class_mask.cpu().numpy().astype(np.uint8)

            # --- 4. Debug Viz (CPU) ---
            seg_cpu = seg_ids.cpu().numpy().astype(np.uint8)
            color_mask = self.class_colors[seg_cpu]
            self.cached_seg_viz = cv2.addWeighted(img_np, 0.6, color_mask, 0.4, 0)

            # --- 5. Projection (Lidar to Camera) ---
            ones = torch.ones((lidar_t.shape[0], 1), device=self.device)
            pts_hom = torch.cat([lidar_t, ones], dim=1)
            pts_cam = (self.lidar_to_cam @ pts_hom.T).T
            
            u = ((self.fx * pts_cam[:, 0]) / (pts_cam[:, 2] + 1e-5)) + self.cx
            v = ((self.fy * pts_cam[:, 1]) / (pts_cam[:, 2] + 1e-5)) + self.cy
            valid_uv = (pts_cam[:, 2] > 0.1) & (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
            
            u, v = u[valid_uv].long(), v[valid_uv].long()
            pts_local = lidar_t[valid_uv]
            
            # Use the computed class_mask to get classes for lidar points
            point_classes = class_mask[v, u].long()

            valid_map_points = point_classes > 0 
            pts_local = pts_local[valid_map_points]
            point_classes = point_classes[valid_map_points]
            
            if pts_local.shape[0] == 0: return self.cached_seg_viz, self.cached_sem_mask

            # --- 6. Update Dynamic 2D Map ---
            # ... (Keep existing logic for dynamic map expansion and voxel grid) ...
            # (This part remains identical to your previous file)
            pts_base_x = pts_local[:, 0] + self.lidar_x_offset
            pts_base_y = pts_local[:, 1]
            
            cos_y, sin_y = np.cos(pose_yaw), np.sin(pose_yaw)
            x_global = pts_base_x * cos_y - pts_base_y * sin_y + pose_x
            y_global = pts_base_x * sin_y + pts_base_y * cos_y + pose_y

            grid_u = ((x_global - self.origin_x) / self.res).long()
            grid_v = ((y_global - self.origin_y) / self.res).long()
            
            if grid_u.numel() > 0:
                min_u, max_u = grid_u.min().item(), grid_u.max().item()
                min_v, max_v = grid_v.min().item(), grid_v.max().item()
                
                if min_u < 0 or max_u >= self.map_w or min_v < 0 or max_v >= self.map_h:
                    self._expand_map(int(min_u), int(max_u), int(min_v), int(max_v))
                    grid_u = ((x_global - self.origin_x) / self.res).long()
                    grid_v = ((y_global - self.origin_y) / self.res).long()

            in_bounds = (grid_u >= 0) & (grid_u < self.map_w) & (grid_v >= 0) & (grid_v < self.map_h)
            gu, gv, gcls = grid_u[in_bounds], grid_v[in_bounds], point_classes[in_bounds]

            if gu.numel() > 0:
                flat_idx = gv * self.map_w + gu
                one_hot = F.one_hot(gcls, num_classes=3).float()
                updates = one_hot * self.L_HIT
                updates[one_hot == 0] = self.L_MISS
                self.grid.view(-1, 3).scatter_add_(0, flat_idx.unsqueeze(1).repeat(1, 3), updates)

            # --- 7. Update 3D Voxel Grid ---
            if self.collect_3d:
                gx, gy = x_global.cpu().numpy(), y_global.cpu().numpy()
                gz = pts_local[:, 2].cpu().numpy()
                cls = point_classes.cpu().numpy()
                ix = np.floor(gx / self.res).astype(int)
                iy = np.floor(gy / self.res).astype(int)
                iz = np.floor(gz / self.res).astype(int)
                for i in range(len(ix)):
                    key = (ix[i], iy[i], iz[i])
                    c = cls[i]
                    if key not in self.voxel_grid: self.voxel_grid[key] = [0, 0, 0] 
                    self.voxel_grid[key][c] += 1
                
            return self.cached_seg_viz, self.cached_sem_mask # <--- Return both

    def get_local_bev(self, pose_x, pose_y, pose_yaw, view_size=256, physical_range=4.0):
            """
            Extracts a local BEV patch from the dynamic global map.
            Handles dynamic map size, origin, and flips Y-axis to match model.
            """
            # 1. Convert Map to Color
            map_probs = self.grid.argmax(dim=2).byte().cpu().numpy()
            bev_img = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
            bev_img[map_probs == 1] = [100, 100, 100]
            bev_img[map_probs == 2] = [255, 255, 255]

            # 2. Scale Logic
            pixels_needed_in_source = physical_range / self.res
            scale = view_size / pixels_needed_in_source

            # 3. Center Calculation (using current dynamic origin)
            center_u = (pose_x - self.origin_x) / self.res
            center_v = (pose_y - self.origin_y) / self.res
            
            # 4. Rotation Matrix
            # Rotates so Robot Forward aligns with Image Up
            angle_deg = np.degrees(pose_yaw) + 90 
            M = cv2.getRotationMatrix2D((center_u, center_v), angle_deg, scale)
            
            # 5. Translation to output center
            M[0, 2] += (view_size / 2) - center_u
            M[1, 2] += (view_size / 2) - center_v
            
            # 6. Warp
            local_bev = cv2.warpAffine(
                bev_img, 
                M, 
                (view_size, view_size), 
                flags=cv2.INTER_NEAREST, 
                borderValue=(0,0,0)
            )

            # 7. Flip horizontally (Local Y Flip)
            # Matches np.fliplr from transfuser_util.py to align lateral coordinates
            local_bev = cv2.flip(local_bev, 1)

            return local_bev
    def get_debug_image(self, pose_x, pose_y):
        map_indices = self.grid.argmax(dim=2).byte().cpu().numpy()
        color_map = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        color_map[map_indices == 1] = [100, 100, 100]
        color_map[map_indices == 2] = [0, 200, 0] 
        
        ru = int((pose_x - self.origin_x) / self.res)
        rv = int((pose_y - self.origin_y) / self.res)
        
        if 0 <= ru < self.map_w and 0 <= rv < self.map_h:
            cv2.circle(color_map, (ru, rv), 5, (0, 0, 255), -1) 
        
        return cv2.rotate(color_map, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def save_3d_map(self, filename="final_session_map.ply"):
        if not self.voxel_grid:
            print("No 3D points collected.")
            return
        
        full_path = os.path.abspath(filename)
        print(f"Filtering and Building 3D Map from {len(self.voxel_grid)} voxels...")
        
        final_pts = []
        final_colors = []
        
        for key, votes in self.voxel_grid.items():
            total_votes = sum(votes)
            if total_votes < 3: continue 
            
            winner_class = np.argmax(votes)
            
            # Skip if the winner is Class 0 (just in case)
            if winner_class == 0: continue

            # Use center of voxel
            px = key[0] * self.res + (self.res/2)
            py = key[1] * self.res + (self.res/2)
            pz = key[2] * self.res + (self.res/2)
            
            final_pts.append([px, py, pz])
            
            if winner_class == 1:
                final_colors.append([0.6, 0.4, 0.2]) # Soil Brown
            elif winner_class == 2:
                final_colors.append([0.0, 1.0, 0.0]) # Crop Green
                
        if len(final_pts) == 0:
            print("Map empty after voting.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(final_pts, dtype=np.float32))
        pcd.colors = o3d.utility.Vector3dVector(np.array(final_colors, dtype=np.float32))

        print("Running Radius Outlier Removal...")
        cl, ind = pcd.remove_radius_outlier(nb_points=6, radius=0.03)
        pcd_clean = pcd.select_by_index(ind)
        
        o3d.io.write_point_cloud(full_path, pcd_clean)
        print(f"Saved Cleaned 3D Map to: {full_path}")