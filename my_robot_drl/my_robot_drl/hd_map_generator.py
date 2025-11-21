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

# --- TRANSFORMERS IMPORTS ---
try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    print("Error: Transformers library not installed. Run: pip install transformers")
    sys.exit(1)

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
INPUT_PKL_PATH = os.path.expanduser('~/ros2_ws/drl_datasets/imitation_learning/360_straight_depth.pkl')
OUTPUT_PKL_PATH = os.path.expanduser('~/ros2_ws/drl_datasets/imitation_learning/real_world_processed.pkl')

IMG_WIDTH, IMG_HEIGHT = 1024, 576
FX, FY, CX, CY = 236.5, 236.5, 512.0, 288.0

LIDAR_TO_CAM = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)

# --- ZOOM SETTINGS ---
MAP_SIZE_PIXELS = 256
# Resolution: 2.5 cm per pixel -> 6.4m wide view
MAP_RES_METERS = 0.025 
MAP_SIZE_METERS = MAP_SIZE_PIXELS * MAP_RES_METERS 

# Simulation Step Time
DT = 0.1 

# Filter settings
MIN_LIDAR_DIST = 0.2   
MAX_LIDAR_DIST = 80.0  

# ==============================================================================
# --- ODOMETRY CLASS ---
# ==============================================================================

class DeadReckoningTracker:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
    def update(self, linear_vel, angular_vel, dt):
        self.theta += angular_vel * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        self.x += linear_vel * np.cos(self.theta) * dt
        self.y += linear_vel * np.sin(self.theta) * dt
        return self.get_pose_matrix()
    
    def get_pose_matrix(self):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        pose = np.eye(4)
        pose[0, 0] = c
        pose[0, 1] = -s
        pose[0, 3] = self.x
        pose[1, 0] = s
        pose[1, 1] = c
        pose[1, 3] = self.y
        return pose

# ==============================================================================
# --- PROCESSOR CLASS ---
# ==============================================================================

class DatasetProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Processor on {self.device}...")
        
        self.tracker = DeadReckoningTracker()
        
        # --- FIXED MODEL NAME ---
        model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
        print(f"Loading High-Accuracy Model: {model_name}...")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.seg_model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.seg_model.to(self.device).eval()
        except Exception as e:
            print(f"Model Load Error: {e}")
            sys.exit(1)
            
        # --- DYNAMIC CLASS MAPPING ---
        self.id2label = self.seg_model.config.id2label
        self.drivable_ids = []
        
        # Keywords for drivable terrain
        drivable_keywords = ['earth', 'ground', 'soil', 'dirt', 'path', 'road', 'field', 'grass', 'sand', 'land']
        
        print("\n--- Class Mapping ---")
        for i, label in self.id2label.items():
            if any(k in label.lower() for k in drivable_keywords):
                self.drivable_ids.append(i)
        
        print(f"Mapped {len(self.drivable_ids)} classes to DRIVABLE (Earth, Soil, etc.)")
        print("All other classes (Plants, Sky, Obstacles) are OBSTACLES.")
        
        # Generate random colors for debug visualization (150 classes)
        np.random.seed(42)
        self.color_palette = np.random.randint(0, 255, (200, 3), dtype=np.uint8)

    def get_semantic_raw(self, image_rgb):
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
        logits = F.interpolate(outputs.logits, size=(image_rgb.shape[0], image_rgb.shape[1]), mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy()

    def paint_point_cloud(self, points_3d, image_rgb):
        if len(points_3d) == 0: return np.zeros((0, 4))
        
        raw_ids = self.get_semantic_raw(image_rgb)
        
        pts_hom = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        pts_cam = (LIDAR_TO_CAM @ pts_hom.T).T
        depth = pts_cam[:, 2]
        
        u = ((FX * pts_cam[:, 0]) / (depth + 1e-6)) + CX
        v = ((FY * pts_cam[:, 1]) / (depth + 1e-6)) + CY
        
        valid = (depth > 0.1) & (u >= 0) & (u < IMG_WIDTH) & (v >= 0) & (v < IMG_HEIGHT)
        u, v = u.astype(int), v.astype(int)
        
        labeled = np.zeros((len(points_3d), 4))
        labeled[:, :3] = points_3d
        
        if valid.any():
            point_classes = raw_ids[v[valid], u[valid]]
            is_drivable = np.isin(point_classes, self.drivable_ids)
            labeled[valid, 3] = 1.0 - is_drivable.astype(float)
            
        return labeled

    def rasterize_bev(self, points_global, current_pose_inv):
        if len(points_global) == 0: return np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), dtype=np.uint8)
        
        pts_hom = np.hstack((points_global[:, :3], np.ones((len(points_global), 1))))
        pts_local = (current_pose_inv @ pts_hom.T).T
        
        mask = (pts_local[:, 2] > -1.0) & (pts_local[:, 2] < 3.0)
        pts_local, labels = pts_local[mask], points_global[mask, 3]
        if len(pts_local) == 0: return np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), dtype=np.uint8)

        half_dim = MAP_SIZE_METERS / 2
        
        row_float = (MAP_SIZE_PIXELS - 1) - ((pts_local[:, 0] + half_dim) / MAP_RES_METERS)
        col_float = (MAP_SIZE_PIXELS / 2) - (pts_local[:, 1] / MAP_RES_METERS)
        
        row = row_float.astype(int)
        col = col_float.astype(int)
        
        valid = (row >= 0) & (row < MAP_SIZE_PIXELS) & (col >= 0) & (col < MAP_SIZE_PIXELS)
        row, col, labels = row[valid], col[valid], labels[valid]
        
        bev = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), dtype=np.uint8)
        
        if len(row) > 0:
            mask_obs = labels > 0.5
            mask_drive = labels <= 0.5
            bev[row[mask_obs], col[mask_obs], 0] = 255   
            bev[row[mask_drive], col[mask_drive], 2] = 255 
            
        return cv2.morphologyEx(bev, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

def process_data():
    if not os.path.exists(INPUT_PKL_PATH):
        print(f"Error: Input not found at {INPUT_PKL_PATH}")
        return

    print(f"Loading data from {INPUT_PKL_PATH}...")
    with open(INPUT_PKL_PATH, 'rb') as f:
        raw_dataset = pickle.load(f)
    
    proc = DatasetProcessor()
    processed_frames = [] 
    
    print("\n--- PHASE 1: TRACKING & SEGMENTATION ---")
    
    for i, frame in enumerate(tqdm(raw_dataset)):
        img = frame['image_raw']
        lidar_raw = frame['lidar_raw'][:, :3]
        
        try:
            linear_v = float(frame['state'][2])
            angular_w = float(frame['state'][3])
        except:
            linear_v, angular_w = 0.0, 0.0

        # 1. UPDATE POSE
        pose = proc.tracker.update(linear_v, angular_w, DT)

        # 2. PAINT
        dists = np.linalg.norm(lidar_raw, axis=1)
        mask_valid = (dists > MIN_LIDAR_DIST) & (dists < MAX_LIDAR_DIST)
        lidar_clean = lidar_raw[mask_valid]
        
        labeled_local = proc.paint_point_cloud(lidar_clean.astype(np.float32), img)
        
        pts_hom = np.hstack((labeled_local[:, :3], np.ones((len(labeled_local), 1))))
        pts_global = (pose @ pts_hom.T).T
        pts_global_labeled = np.hstack((pts_global[:, :3], labeled_local[:, 3:4]))
        
        processed_frames.append({'pose': pose, 'points_global': pts_global_labeled})

        # --- DEBUG VISUALIZATION (Every 50 Frames) ---
        if i % 50 == 0:
            raw_ids = proc.get_semantic_raw(img)
            unique_ids = np.unique(raw_ids)
            color_mask = proc.color_palette[raw_ids]
            
            blended = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            
            print(f"\n[Frame {i}] Found Classes:")
            for uid in unique_ids:
                label_name = proc.id2label[uid]
                status = "DRIVABLE" if uid in proc.drivable_ids else "OBSTACLE"
                print(f"  - {label_name} (ID {uid}) -> {status}")
            
            cv2.imshow("B2 Segmentation (All Classes)", blended_bgr)
            cv2.waitKey(0) 

    print("\n--- PHASE 2: Map Generation ---")
    final_dataset = []
    WINDOW_SIZE = 15 
    
    for i, original_frame in enumerate(tqdm(raw_dataset)):
        if i >= len(processed_frames): break
        curr_pose_inv = np.linalg.inv(processed_frames[i]['pose'])
        
        start, end = max(0, i - WINDOW_SIZE), min(len(processed_frames), i + WINDOW_SIZE)
        clouds = [processed_frames[j]['points_global'] for j in range(start, end)]
        full_cloud = np.vstack(clouds) if clouds else np.zeros((0, 4))
        
        bev_map = proc.rasterize_bev(full_cloud, curr_pose_inv)
        
        if i % 50 == 0:
            cv2.imshow("BEV Map (Blue=Obstacle, Red=Drivable)", bev_map)
            cv2.waitKey(0) 
        
        new_sample = copy.deepcopy(original_frame)
        new_sample['bev_semantic_label'] = bev_map
        final_dataset.append(new_sample)
        
    cv2.destroyAllWindows()
    
    os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
    with open(OUTPUT_PKL_PATH, 'wb') as f:
        pickle.dump(final_dataset, f)
    print(f"Saved to {OUTPUT_PKL_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', action='store_true')
    args = parser.parse_args()
    if args.process: process_data()
    else: print("Use --process")