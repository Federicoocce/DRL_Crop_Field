import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
import os
import random
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class RealTimeSemanticMapper:
    def __init__(self, device='cuda', resolution=0.02, map_size_px=2000):
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

        # --- Map Config ---
        self.res = resolution 
        self.map_size = map_size_px
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        # 2D Grid [H, W, 3]
        self.grid = torch.zeros((self.map_size, self.map_size, 3), device=self.device, dtype=torch.float32)
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
        self.grid.zero_()
        self.origin_x = start_x
        self.origin_y = start_y
        self.voxel_grid = {} 
        self.frame_count = 0
        self.cached_seg_viz = None

    def process_one_step(self, img_np, lidar_np, pose_x, pose_y, pose_yaw):
        self.frame_count += 1
        
        if self.cached_seg_viz is None:
            self.cached_seg_viz = img_np.copy()

        if self.frame_count % 2 != 0:
            return self.cached_seg_viz

        if len(lidar_np) < 10: return self.cached_seg_viz

        # --- 1. Prepare Data ---
        img_small = cv2.resize(img_np, self.inference_size)
        inputs = self.processor(images=img_small, return_tensors="pt").to(self.device)
        inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}
        
        lidar_t = torch.from_numpy(lidar_np).to(self.device).float()
        
        # Range Filter
        dists = torch.norm(lidar_t, dim=1)
        valid_pts = (dists > 0.15) & (dists < 3.5)
        lidar_t = lidar_t[valid_pts]
        if lidar_t.shape[0] == 0: return self.cached_seg_viz

        # --- 2. Inference ---
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = F.interpolate(outputs.logits.float(), size=(self.img_h, self.img_w), 
                                  mode="bilinear", align_corners=False)
            seg_ids = logits.argmax(dim=1)[0] # [H, W]

        # --- 3. DEBUG VISUALIZATION (Shows ALL classes, even Sky) ---
        seg_cpu = seg_ids.cpu().numpy().astype(np.uint8)
        unique_classes = np.unique(seg_cpu)
        
        # Colored overlay
        color_mask = self.class_colors[seg_cpu]
        viz_img = cv2.addWeighted(img_np, 0.6, color_mask, 0.4, 0)
        
        # Draw Labels
        for cls_id in unique_classes:
            if cls_id == 0: continue # Skip background
            
            mask = (seg_cpu == cls_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_cnt) > 400: 
                    M = cv2.moments(largest_cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        label_name = self.id2label.get(cls_id, "Unknown")
                        text = f"{label_name} [{cls_id}]"
                        
                        cv2.putText(viz_img, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(viz_img, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        self.cached_seg_viz = viz_img

        # --- 4. Projection & Logic ---
        sem_labels = seg_ids.clamp(0, 199)
        lookup = torch.zeros(200, device=self.device, dtype=torch.uint8)
        lookup[self.drivable_ids] = 1 
        lookup[self.obstacle_ids] = 2 
        # Everything else (Sky, Building, etc) stays 0
        
        class_mask = lookup[sem_labels] # [H, W]
        
        # Project Lidar
        ones = torch.ones((lidar_t.shape[0], 1), device=self.device)
        pts_hom = torch.cat([lidar_t, ones], dim=1)
        pts_cam = (self.lidar_to_cam @ pts_hom.T).T
        
        u = ((self.fx * pts_cam[:, 0]) / (pts_cam[:, 2] + 1e-5)) + self.cx
        v = ((self.fy * pts_cam[:, 1]) / (pts_cam[:, 2] + 1e-5)) + self.cy
        valid_uv = (pts_cam[:, 2] > 0.1) & (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        
        u, v = u[valid_uv].long(), v[valid_uv].long()
        pts_local = lidar_t[valid_uv]
        point_classes = class_mask[v, u].long()

        # --- STRICT FILTERING START ---
        # Ignore anything that is Class 0 (Unknown/Sky/Others)
        # We ONLY want Drive(1) or Obstacle(2)
        valid_map_points = point_classes > 0 
        
        # Apply mask
        pts_local = pts_local[valid_map_points]
        point_classes = point_classes[valid_map_points]
        
        if pts_local.shape[0] == 0: return self.cached_seg_viz
        # --- STRICT FILTERING END ---

        # --- 5. Update 2D Map ---
        pts_base_x = pts_local[:, 0] + self.lidar_x_offset
        pts_base_y = pts_local[:, 1]
        
        cos_y, sin_y = np.cos(pose_yaw), np.sin(pose_yaw)
        x_global = pts_base_x * cos_y - pts_base_y * sin_y + pose_x
        y_global = pts_base_x * sin_y + pts_base_y * cos_y + pose_y

        grid_u = ((x_global - self.origin_x) / self.res) + (self.map_size / 2)
        grid_v = ((y_global - self.origin_y) / self.res) + (self.map_size / 2)
        
        in_bounds = (grid_u >= 0) & (grid_u < self.map_size) & (grid_v >= 0) & (grid_v < self.map_size)
        gu, gv, gcls = grid_u[in_bounds].long(), grid_v[in_bounds].long(), point_classes[in_bounds]

        flat_idx = gu * self.map_size + gv
        one_hot = F.one_hot(gcls, num_classes=3).float()
        updates = one_hot * self.L_HIT
        updates[one_hot == 0] = self.L_MISS
        self.grid.view(-1, 3).scatter_add_(0, flat_idx.unsqueeze(1).repeat(1, 3), updates)

        # --- 6. Update 3D Voxel Grid ---
        if self.collect_3d:
            gx = x_global.cpu().numpy()
            gy = y_global.cpu().numpy()
            gz = pts_local[:, 2].cpu().numpy()
            cls = point_classes.cpu().numpy()

            ix = np.floor(gx / self.res).astype(int)
            iy = np.floor(gy / self.res).astype(int)
            iz = np.floor(gz / self.res).astype(int)
            
            for i in range(len(ix)):
                key = (ix[i], iy[i], iz[i])
                c = cls[i] # c is guaranteed to be 1 or 2 now
                
                if key not in self.voxel_grid:
                    self.voxel_grid[key] = [0, 0, 0] 
                
                self.voxel_grid[key][c] += 1
            
        return self.cached_seg_viz

    def get_local_bev(self, pose_x, pose_y, pose_yaw, view_size=256):
        map_probs = self.grid.argmax(dim=2).byte().cpu().numpy()
        bev_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        bev_img[map_probs == 1] = [100, 100, 100]
        bev_img[map_probs == 2] = [255, 255, 255]

        center_u = int(((pose_x - self.origin_x) / self.res) + (self.map_size / 2)) 
        center_v = int(((pose_y - self.origin_y) / self.res) + (self.map_size / 2)) 
        angle_deg = np.degrees(pose_yaw) + 90 
        
        M = cv2.getRotationMatrix2D((center_v, center_u), angle_deg, 1.0)
        M[0, 2] += (view_size / 2) - center_v
        M[1, 2] += (view_size / 2) - center_u
        return cv2.warpAffine(bev_img, M, (view_size, view_size), borderValue=(0,0,0))

    def get_debug_image(self, pose_x, pose_y):
        map_indices = self.grid.argmax(dim=2).byte().cpu().numpy()
        color_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        color_map[map_indices == 1] = [100, 100, 100]
        color_map[map_indices == 2] = [0, 200, 0] 
        
        ru = int(((pose_x - self.origin_x) / self.res) + (self.map_size / 2)) 
        rv = int(((pose_y - self.origin_y) / self.res) + (self.map_size / 2)) 
        
        if 0 <= ru < self.map_size and 0 <= rv < self.map_size:
            cv2.circle(color_map, (rv, ru), 5, (0, 0, 255), -1) 
        
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
        cl, ind = pcd.remove_radius_outlier(nb_points=6, radius=0.1)
        pcd_clean = pcd.select_by_index(ind)
        
        o3d.io.write_point_cloud(full_path, pcd_clean)
        print(f"Saved Cleaned 3D Map to: {full_path}")