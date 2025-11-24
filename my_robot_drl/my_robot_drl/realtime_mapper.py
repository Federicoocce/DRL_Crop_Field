import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class RealTimeSemanticMapper:
    def __init__(self, device='cuda', resolution=0.05, map_size_px=1000):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing RealTimeMapper on {self.device} (Model: Segformer-B0)...")

        # --- 1. Use Faster B0 Model ---
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512" 
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device).eval()

        # --- Map Config ---
        self.res = resolution
        self.map_size = map_size_px
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        # Grid: [H, W, 3] 
        self.grid = torch.zeros((self.map_size, self.map_size, 3), device=self.device, dtype=torch.float32)
        self.L_HIT = 0.8
        self.L_MISS = -0.1

        # --- 3D Accumulation ---
        self.collect_3d = True
        self.global_pcd_points = []
        self.global_pcd_colors = []
        self.frame_count = 0
        
        # --- Calibration ---
        self.img_w, self.img_h = 1024, 576
        self.inference_size = (512, 288)
        
        # Recalculated Focal Length for 130deg FOV
        self.fx, self.fy = 241.5, 241.5 
        self.cx, self.cy = 512.0, 288.0
        
        self.lidar_to_cam = torch.tensor([
            [0, -1, 0, 0], 
            [0, 0, -1, 0], 
            [1, 0, 0, 0], 
            [0, 0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        self.drivable_ids = [13, 9, 52, 91, 14, 29] 
        self.obstacle_ids = [4, 17, 66, 72] 
        self.lidar_x_offset = 0.20

    def reset_map(self, start_x, start_y):
        self.grid.zero_()
        self.origin_x = start_x
        self.origin_y = start_y
        self.global_pcd_points = []
        self.global_pcd_colors = []
        self.frame_count = 0

    def process_one_step(self, img_np, lidar_np, pose_x, pose_y, pose_yaw):
        self.frame_count += 1
        
        # Process every 2nd frame to let the B0 model breathe, change to 1 if you have a 3090/4090
        if self.frame_count % 2 != 0:
            return None

        if len(lidar_np) < 10: return None

        # --- 1. Prepare Data ---
        img_small = cv2.resize(img_np, self.inference_size)
        inputs = self.processor(images=img_small, return_tensors="pt").to(self.device)
        
        lidar_t = torch.from_numpy(lidar_np).to(self.device).float()
        
        # Filter Range
        dists = torch.norm(lidar_t, dim=1)
        valid_pts = (dists > 0.15) & (dists < 3.0) 
        lidar_t = lidar_t[valid_pts]
        if lidar_t.shape[0] == 0: return None

        # --- 2. Inference ---
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = F.interpolate(outputs.logits, size=(self.img_h, self.img_w), 
                                  mode="bilinear", align_corners=False)
            seg_ids = logits.argmax(dim=1)[0]

        # --- 3. CREATE SEGMENTATION VISUALIZATION ---
        sem_labels = seg_ids.clamp(0, 199)
        lookup = torch.zeros(200, device=self.device, dtype=torch.uint8)
        lookup[self.drivable_ids] = 1 
        lookup[self.obstacle_ids] = 2 
        
        class_mask = lookup[sem_labels] # [H, W]
        
        # CPU Visualization
        mask_cpu = class_mask.cpu().numpy().astype(np.uint8)
        color_viz = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        color_viz[mask_cpu == 1] = [128, 128, 128] # Gray Road
        color_viz[mask_cpu == 2] = [0, 255, 0]     # Bright Green Crops
        
        seg_viz = cv2.addWeighted(img_np, 0.7, color_viz, 0.3, 0)

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

        # Debug Print to verify we are actually hitting things
        # num_obs = (point_classes == 2).sum().item()
        # num_drv = (point_classes == 1).sum().item()
        # if num_obs > 0 or num_drv > 0:
        #    print(f"Mapping: {num_obs} Obstacle pts, {num_drv} Drive pts")

        # --- 5. Update Map ---
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

        # --- 6. Accumulate 3D Points ---
        if self.collect_3d and self.frame_count % 5 == 0:
            p_x, p_y = pts_base_x.cpu().numpy(), pts_base_y.cpu().numpy()
            p_z = pts_local[:, 2].cpu().numpy()
            cls_cpu = point_classes.cpu().numpy()
            
            pts_gl = np.zeros((len(p_x), 3), dtype=np.float32)
            pts_gl[:, 0] = p_x * cos_y - p_y * sin_y + pose_x
            pts_gl[:, 1] = p_x * sin_y + p_y * cos_y + pose_y
            pts_gl[:, 2] = p_z
            
            # Vivid Colors for 3D Map
            cols = np.zeros_like(pts_gl)
            cols[cls_cpu == 1] = [0.8, 0.8, 0.8] # Bright White/Gray
            cols[cls_cpu == 2] = [0.0, 1.0, 0.0] # Bright Green
            cols[cls_cpu == 0] = [0.0, 0.0, 1.0] # Blue for Unknown (Debug)

            self.global_pcd_points.append(pts_gl)
            self.global_pcd_colors.append(cols)
            
        return seg_viz

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
        """ Returns Full Global Map with Robot Marker """
        map_indices = self.grid.argmax(dim=2).byte().cpu().numpy()
        
        color_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        color_map[map_indices == 1] = [100, 100, 100]
        color_map[map_indices == 2] = [0, 200, 0] 
        
        ru = int(((pose_x - self.origin_x) / self.res) + (self.map_size / 2)) 
        rv = int(((pose_y - self.origin_y) / self.res) + (self.map_size / 2)) 
        
        if 0 <= ru < self.map_size and 0 <= rv < self.map_size:
            # Note: ru is Row (y), rv is Col (x) for array indexing
            # cv2.circle expects (x, y) -> (rv, ru)
            cv2.circle(color_map, (rv, ru), 5, (0, 0, 255), -1) 
        
        return cv2.rotate(color_map, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def save_3d_map(self, filename="final_session_map.ply"):
        if not self.global_pcd_points:
            print("No 3D points collected.")
            return
        
        full_path = os.path.abspath(filename)
        print(f"Building 3D Point Cloud ({len(self.global_pcd_points)} frames)...")
        pts = np.vstack(self.global_pcd_points)
        cols = np.vstack(self.global_pcd_colors)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        
        # Remove downsampling to ensure points aren't disappearing
        # pcd = pcd.voxel_down_sample(voxel_size=self.res)
        
        o3d.io.write_point_cloud(full_path, pcd)
        print(f"Saved 3D Map to: {full_path}")