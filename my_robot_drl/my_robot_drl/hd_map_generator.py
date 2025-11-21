import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pickle
import copy
from tqdm import tqdm

# --- FIX: UPDATED IMPORTS FOR NEW KISS-ICP ---
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import KISSConfig
# ---------------------------------------------

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# 1. FILE PATHS
INPUT_PKL_PATH = os.path.expanduser('~/ros2_ws/drl_datasets/imitation_learning/360_straight_depth.pkl')
OUTPUT_PKL_PATH = os.path.expanduser('~/ros2_ws/drl_datasets/imitation_learning/real_world_processed.pkl')

# 2. CAMERA INTRINSICS (From URDF: 1024x576, ~130 deg FOV)
IMG_WIDTH = 1024
IMG_HEIGHT = 576
FX = 236.5 
FY = 236.5 
CX = 512.0
CY = 288.0

# 3. EXTRINSICS: LIDAR TO CAMERA
# Rotates LiDAR (X-Fwd) to Camera Optical (Z-Fwd)
LIDAR_TO_CAM = np.array([
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [1,  0,  0,  0],
    [0,  0,  0,  1]
], dtype=np.float32)

# 4. MAPPING CONFIG
MAP_SIZE_PIXELS = 256
MAP_RES_METERS = 0.125 
MAP_SIZE_METERS = MAP_SIZE_PIXELS * MAP_RES_METERS 

# 5. SEGMENTATION METHOD
SEG_METHOD = "segformer" 

# ==============================================================================
# --- PROCESSOR CLASS ---
# ==============================================================================

class DatasetProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Processor on {self.device}...")
        

        # ------------------------------
                # We must manually set the voxel_size to a float to avoid the TypeError
        config = KISSConfig()
        config.mapping.voxel_size = 0.5 
        self.kiss_icp = KissICP(config=config)
        if SEG_METHOD == "segformer":
            print("Loading SegFormer...")
            self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-20k-semantic")
            self.seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-20k-semantic")
            self.seg_model.to(self.device)
            self.seg_model.eval()
            # ADE20K IDs: 13=earth, 29=field, 91=dirt track, 94=land
            self.drivable_ids = [13, 29, 91, 94] 

    def get_semantic_mask(self, image_rgb):
        if SEG_METHOD == "excess_green":
            img = image_rgb.astype(float) / 255.0
            R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
            exg = 2*G - R - B
            return exg < 0.05 
        else:
            inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.seg_model(**inputs)
            logits = F.interpolate(outputs.logits, size=(image_rgb.shape[0], image_rgb.shape[1]), mode="bilinear", align_corners=False)
            pred_seg = logits.argmax(dim=1)[0].cpu().numpy()
            return np.isin(pred_seg, self.drivable_ids)

    def paint_point_cloud(self, points_3d, image_rgb):
        mask = self.get_semantic_mask(image_rgb)
        num_pts = points_3d.shape[0]
        pts_hom = np.hstack((points_3d, np.ones((num_pts, 1))))
        pts_cam = (LIDAR_TO_CAM @ pts_hom.T).T
        depth = pts_cam[:, 2]
        
        u = ((FX * pts_cam[:, 0]) / (depth + 1e-6)) + CX
        v = ((FY * pts_cam[:, 1]) / (depth + 1e-6)) + CY
        u = u.astype(int)
        v = v.astype(int)
        
        valid_indices = (depth > 0.1) & (u >= 0) & (u < IMG_WIDTH) & (v >= 0) & (v < IMG_HEIGHT)
        
        labeled_points = np.zeros((num_pts, 4))
        labeled_points[:, :3] = points_3d
        
        u_valid = u[valid_indices]
        v_valid = v[valid_indices]
        
        is_drivable = mask[v_valid, u_valid]
        labeled_points[valid_indices, 3] = is_drivable.astype(float)
        
        return labeled_points

    def rasterize_bev(self, points_global, current_pose_inv):
        pts_hom = np.hstack((points_global[:, :3], np.ones((points_global.shape[0], 1))))
        pts_local = (current_pose_inv @ pts_hom.T).T
        
        # Height Filter
        mask_z = (pts_local[:, 2] > -1.0) & (pts_local[:, 2] < 2.0)
        pts_local = pts_local[mask_z]
        labels = points_global[mask_z, 3]
        
        if len(pts_local) == 0: return np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), dtype=np.uint8)

        half_dim_m = MAP_SIZE_METERS / 2
        row = MAP_SIZE_PIXELS - 1 - ((pts_local[:, 0] + half_dim_m) / MAP_RES_METERS).astype(int)
        col = (MAP_SIZE_PIXELS / 2) - (pts_local[:, 1] / MAP_RES_METERS).astype(int)
        
        valid_pixels = (row >= 0) & (row < MAP_SIZE_PIXELS) & (col >= 0) & (col < MAP_SIZE_PIXELS)
        row = row[valid_pixels]
        col = col[valid_pixels]
        labels = labels[valid_pixels]
        
        bev_map = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), dtype=np.uint8)
        
        # OpenCV is BGR
        # Channel 0 (Blue) = Drivable
        drivable = labels > 0.5
        bev_map[row[drivable], col[drivable], 0] = 255 
        
        # Channel 2 (Red) = Obstacle
        obstacle = labels <= 0.5
        bev_map[row[obstacle], col[obstacle], 2] = 255 
        
        kernel = np.ones((3,3), np.uint8)
        bev_map = cv2.morphologyEx(bev_map, cv2.MORPH_CLOSE, kernel)
        return bev_map

# ==============================================================================
# --- VISUALIZATION FUNCTION ---
# ==============================================================================

def visualize_dataset(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"Error: File {pkl_path} not found.")
        return

    print(f"Loading dataset from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset)} frames.")
    print("Controls: [SPACE] Next Frame, [Q] Quit")

    for i, frame in enumerate(dataset):
        rgb_img = frame['image_raw'] 
        bev_map = frame['bev_semantic_label']
        
        bev_display = cv2.resize(bev_map, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        h, w, _ = rgb_img.shape
        scale = 512 / h
        new_w = int(w * scale)
        rgb_display = cv2.resize(rgb_img, (new_w, 512))

        cv2.putText(bev_display, "Red: Obstacle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(bev_display, "Blue: Drivable", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(bev_display, f"Frame: {i}", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        combined = np.hstack((rgb_display, bev_display))

        cv2.imshow("TransFuser Data Verification", combined)
        
        key = cv2.waitKey(0) 
        if key == ord('q'):
            break
        
    cv2.destroyAllWindows()

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

def process_data():
    if not os.path.exists(INPUT_PKL_PATH):
        print(f"Error: Input file not found at {INPUT_PKL_PATH}")
        return

    print(f"Loading input dataset from {INPUT_PKL_PATH}...")
    with open(INPUT_PKL_PATH, 'rb') as f:
        raw_dataset = pickle.load(f)
    
    proc = DatasetProcessor()
    processed_frames = [] 
    
    print("\n--- PHASE 1: SLAM & Segmentation ---")
    for i, frame in enumerate(tqdm(raw_dataset)):
        img = frame['image_raw']
        lidar = frame['lidar_raw'][:, :3].astype(np.float64)
        
        proc.kiss_icp.register_frame(lidar)
        pose = proc.kiss_icp.poses[-1]
        
        labeled_local = proc.paint_point_cloud(lidar.astype(np.float32), img)
        
        pts_hom = np.hstack((labeled_local[:, :3], np.ones((labeled_local.shape[0], 1))))
        pts_global = (pose @ pts_hom.T).T
        pts_global_labeled = np.hstack((pts_global[:, :3], labeled_local[:, 3:4]))
        
        processed_frames.append({'pose': pose, 'points_global': pts_global_labeled})

    print("\n--- PHASE 2: Map Generation & Packaging ---")
    final_dataset = []
    WINDOW_SIZE = 15 
    
    for i, original_frame in enumerate(tqdm(raw_dataset)):
        curr_pose = processed_frames[i]['pose']
        curr_pose_inv = np.linalg.inv(curr_pose)
        
        start = max(0, i - WINDOW_SIZE)
        end = min(len(processed_frames), i + WINDOW_SIZE)
        
        clouds_to_merge = [processed_frames[j]['points_global'] for j in range(start, end)]
        full_cloud = np.vstack(clouds_to_merge)
        
        bev_map = proc.rasterize_bev(full_cloud, curr_pose_inv)
        
        new_sample = copy.deepcopy(original_frame)
        new_sample['bev_semantic_label'] = bev_map
        final_dataset.append(new_sample)
        
    print(f"\nSaving processed dataset to {OUTPUT_PKL_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
    with open(OUTPUT_PKL_PATH, 'wb') as f:
        pickle.dump(final_dataset, f)
    print("Processing Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real World Data Processor")
    # Changed argument to --process (boolean flag) and --view (boolean flag) 
    # to match your command: python3 hd_map_generator.py --process
    parser.add_argument('--process', action='store_true', help="Generate the HD Map dataset")
    parser.add_argument('--view', action='store_true', help="Visualize the generated dataset")
    args = parser.parse_args()

    if args.process:
        process_data()
    elif args.view:
        visualize_dataset(OUTPUT_PKL_PATH)
    else:
        print("Please specify a mode: --process or --view")