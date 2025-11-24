import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class RealTimeSemanticMapper:
    def __init__(self, device='cuda', resolution=0.02, map_size_px=2000):
        """
        resolution: 0.02 = 2cm voxels (High Density)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing RealTimeMapper on {self.device} (Farm Logic | Dense Grid)...")

        # --- AI Model ---
        # We use the robust NVIDIA model, but interpret the classes as Farm objects
        model_name = "nvidia/segformer-b1-finetuned-ade-512-512" 
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device).half().eval()

        # --- Map Config ---
        self.res = resolution 
        self.map_size = map_size_px
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        # 2D Grid [H, W, 3]
        self.grid = torch.zeros((self.map_size, self.map_size, 3), device=self.device, dtype=torch.float32)
        # Higher Hit log-odds to fill map faster (denser feel)
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

        # --- FARM SEMANTIC MAPPING (ADE20k IDs) ---
        # 1 = DRIVABLE (Soil, Grass)
        self.drivable_ids = [
            9,   # Grass
            13,  # Earth
            14,  # Ground
            29,  # Field
            46,  # Sand
            91,  # Dirt
            6,   # Road
            52,  # Path
            94   # Step/Stair (often misclassified dirt)
        ]

        # 2 = OBSTACLE (Crops, Trees, Plants)
        self.obstacle_ids = [
            17,  # Plant
            4,   # Tree
            66,  # Flower
            72,  # Palm
            126, # Pole (often corn stalks)
            5,   # Ceiling (sometimes sky/tall crops artifact)
            12   # Person (Scarecrows?)
        ]

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

        # Run every 2 frames for speed
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
            seg_ids = logits.argmax(dim=1)[0]

        # --- 3. Visualization ---
        sem_labels = seg_ids.clamp(0, 199)
        lookup = torch.zeros(200, device=self.device, dtype=torch.uint8)
        lookup[self.drivable_ids] = 1 
        lookup[self.obstacle_ids] = 2 
        
        class_mask = lookup[sem_labels] # [H, W]
        
        mask_cpu = class_mask.cpu().numpy().astype(np.uint8)
        color_viz = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        # BGR Colors for OpenCV
        color_viz[mask_cpu == 1] = [100, 100, 100] # Gray (Soil/Grass)
        color_viz[mask_cpu == 2] = [0, 255, 0]     # Green (Crops)
        color_viz[mask_cpu == 0] = [0, 0, 255]     # Red (Unknown)
        
        self.cached_seg_viz = cv2.addWeighted(img_np, 0.6, color_viz, 0.4, 0)

        # --- 4. Projection ---
        ones = torch.ones((lidar_t.shape[0], 1), device=self.device)
        pts_hom = torch.cat([lidar_t, ones], dim=1)
        pts_cam = (self.lidar_to_cam @ pts_hom.T).T
        
        u = ((self.fx * pts_cam[:, 0]) / (pts_cam[:, 2] + 1e-5)) + self.cx
        v = ((self.fy * pts_cam[:, 1]) / (pts_cam[:, 2] + 1e-5)) + self.cy
        valid_uv = (pts_cam[:, 2] > 0.1) & (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        
        u, v = u[valid_uv].long(), v[valid_uv].long()
        pts_local = lidar_t[valid_uv]
        point_classes = class_mask[v, u].long()

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

        # --- 6. Update 3D Voxel Grid (CPU) ---
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
                c = cls[i]
                if key not in self.voxel_grid:
                    self.voxel_grid[key] = [0, 0, 0] 
                if c < 3:
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
        
        # --- 1. VOTING FILTER ---
        for key, votes in self.voxel_grid.items():
            total_votes = sum(votes)
            # Threshold: Voxel must be seen at least 4 times to be saved
            if total_votes < 4: continue 
            
            winner_class = np.argmax(votes)
            
            # Use center of voxel
            px = key[0] * self.res + (self.res/2)
            py = key[1] * self.res + (self.res/2)
            pz = key[2] * self.res + (self.res/2)
            
            final_pts.append([px, py, pz])
            
            if winner_class == 1:
                final_colors.append([0.6, 0.4, 0.2]) # Soil Brown
            elif winner_class == 2:
                final_colors.append([0.0, 1.0, 0.0]) # Crop Green
            else:
                final_colors.append([0.3, 0.3, 0.3]) # Unk Dark Gray
                
        if len(final_pts) == 0:
            print("Map empty after voting.")
            return

        # Create Open3D Object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(final_pts, dtype=np.float32))
        pcd.colors = o3d.utility.Vector3dVector(np.array(final_colors, dtype=np.float32))

        # --- 2. RADIUS OUTLIER REMOVAL (Clean Scattered Points) ---
        print("Running Radius Outlier Removal...")
        # Every point must have at least 5 neighbors within a 6cm radius
        # Since resolution is 4cm, neighbors should be adjacent voxels
        cl, ind = pcd.remove_radius_outlier(nb_points=5, radius=0.06)
        pcd_clean = pcd.select_by_index(ind)
        
        o3d.io.write_point_cloud(full_path, pcd_clean)
        print(f"Saved Cleaned 3D Map to: {full_path}")