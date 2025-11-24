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

# --- LIBRARIES ---
try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    print("Error: Transformers not installed. Run: pip install transformers")
    sys.exit(1)

try:
    import open3d as o3d
    print("Libraries loaded successfully.")
except ImportError:
    print("Error: Open3D not found. Run: pip install open3d")
    sys.exit(1)

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
INPUT_PKL_PATH = os.path.expanduser('~/ros2_ws/drl_datasets/imitation_learning/180_straight_depth_timestamp.pkl')
OUTPUT_PKL_PATH = os.path.expanduser('~/ros2_ws/drl_datasets/imitation_learning/real_world_processed.pkl')

# Camera Intrinsics
IMG_WIDTH, IMG_HEIGHT = 1024, 576
FX, FY, CX, CY = 236.5, 236.5, 512.0, 288.0

LIDAR_TO_CAM = np.array([
    [0, -1, 0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1]
], dtype=np.float32)

# Map Config
GRID_SIZE = 512       
GRID_RES = 0.05       

# ==============================================================================
# --- MAP CLASS ---
# ==============================================================================
class BayesianSemanticMap:
    def __init__(self, device='cuda'):
        self.device = device
        self.num_classes = 3
        self.grid = torch.zeros((GRID_SIZE, GRID_SIZE, self.num_classes), device=device, dtype=torch.float32)
        
        self.PROB_HIT = 0.65 
        self.L_HIT = np.log(self.PROB_HIT / (1 - self.PROB_HIT))
        self.L_MISS = np.log((1 - self.PROB_HIT) / self.PROB_HIT) * 0.1 

    def update(self, points_global, labels, pose_inv):
        if len(points_global) == 0: return
        
        # Convert World XYZ -> Grid UV
        # We center the map (256, 256) at World (0,0)
        u = ((points_global[:, 0] / GRID_RES) + (GRID_SIZE / 2)).astype(int)
        v = ((points_global[:, 1] / GRID_RES) + (GRID_SIZE / 2)).astype(int)
        
        valid = (u >= 0) & (u < GRID_SIZE) & (v >= 0) & (v < GRID_SIZE)
        u, v, lbs = u[valid], v[valid], labels[valid]
        if len(u) == 0: return

        u_t = torch.from_numpy(u).to(self.device).long()
        v_t = torch.from_numpy(v).to(self.device).long()
        lbl_t = torch.from_numpy(lbs).to(self.device).long()

        flat_indices = u_t * GRID_SIZE + v_t
        
        updates = torch.full((len(lbl_t), self.num_classes), self.L_MISS, device=self.device)
        updates.scatter_(1, lbl_t.unsqueeze(1), self.L_HIT)

        for c in range(self.num_classes):
            self.grid[:, :, c].view(-1).scatter_add_(0, flat_indices, updates[:, c])

    def get_bev(self, current_pose_matrix):
        map_probs = self.grid.float()
        map_indices = torch.argmax(map_probs, dim=2).cpu().numpy().astype(np.uint8)
        bev_global = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        bev_global[map_indices == 1] = [100, 100, 100] 
        bev_global[map_indices == 2] = [255, 255, 255] 

        rx, ry = current_pose_matrix[0, 3], current_pose_matrix[1, 3]
        ru = int((rx / GRID_RES) + (GRID_SIZE / 2))
        rv = int((ry / GRID_RES) + (GRID_SIZE / 2))

        VIEW_SIZE = 256
        half = VIEW_SIZE // 2
        padded = cv2.copyMakeBorder(bev_global, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
        start_row, start_col = rv, ru 
        local_crop = padded[start_row:start_row+VIEW_SIZE, start_col:start_col+VIEW_SIZE]
        
        yaw = np.arctan2(current_pose_matrix[1, 0], current_pose_matrix[0, 0])
        degree = np.degrees(yaw)
        center = (VIEW_SIZE // 2, VIEW_SIZE // 2)
        
        # Rotate so Robot X is UP
        M = cv2.getRotationMatrix2D(center, degree + 90, 1.0)
        return cv2.warpAffine(local_crop, M, (VIEW_SIZE, VIEW_SIZE))

# ==============================================================================
# --- PROCESSOR CLASS ---
# ==============================================================================
class DatasetProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading Semantic Model...")
        model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device).eval()
        
        self.map = BayesianSemanticMap(self.device)
        self.drivable_ids = [13, 9, 52, 91, 14, 29] 
        self.obstacle_ids = [4, 17, 66, 72] 
        
        self.prev_pcd = None
        self.icp_threshold = 2.0 # Very permissive search

    def get_semantic_labels(self, img_rgb, points_lidar):
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.seg_model(**inputs).logits
        
        mask = F.interpolate(logits, size=img_rgb.shape[:2], mode="bilinear", align_corners=False)
        seg_ids = mask.argmax(dim=1)[0].cpu().numpy()

        pts_hom = np.hstack((points_lidar, np.ones((len(points_lidar), 1))))
        pts_cam = (LIDAR_TO_CAM @ pts_hom.T).T
        depth = pts_cam[:, 2]
        u = ((FX * pts_cam[:, 0]) / (depth + 1e-5)) + CX
        v = ((FY * pts_cam[:, 1]) / (depth + 1e-5)) + CY

        valid = (depth > 0.1) & (u >= 0) & (u < IMG_WIDTH) & (v >= 0) & (v < IMG_HEIGHT)
        u_valid, v_valid = u[valid].astype(int), v[valid].astype(int)
        p_valid = points_lidar[valid]
        
        if len(p_valid) == 0: return np.array([]), np.array([])

        pixel_classes = seg_ids[v_valid, u_valid]
        labels = np.zeros_like(pixel_classes)
        labels[np.isin(pixel_classes, self.drivable_ids)] = 1
        labels[np.isin(pixel_classes, self.obstacle_ids)] = 2
        return p_valid, labels

    def get_odometry_guess(self, v, w, dt):
        # Standard Differential Drive Kinematics
        dx = v * np.cos(w * dt / 2.0) * dt 
        dy = v * np.sin(w * dt / 2.0) * dt
        dtheta = w * dt
        
        guess = np.eye(4)
        guess[0, 0] = np.cos(dtheta)
        guess[0, 1] = -np.sin(dtheta)
        guess[1, 0] = np.sin(dtheta)
        guess[1, 1] = np.cos(dtheta)
        guess[0, 3] = dx
        guess[1, 3] = dy
        return guess

    def run(self):
        if not os.path.exists(INPUT_PKL_PATH):
            print(f"Error: Data not found at {INPUT_PKL_PATH}")
            return

        print(f"Loading dataset...")
        with open(INPUT_PKL_PATH, 'rb') as f:
            raw_dataset = pickle.load(f)

        processed_dataset = []
        global_pose = np.eye(4)
        prev_timestamp = None
        
        print(f"Processing {len(raw_dataset)} Frames...")
        for i, frame in enumerate(tqdm(raw_dataset)):
            img = frame['image_raw']
            lidar_raw = frame['lidar_raw']
            
            # --- DATA CLEANING ---
            lidar_clean = lidar_raw[~np.isnan(lidar_raw).any(axis=1)]
            lidar_clean = lidar_clean[~np.isinf(lidar_clean).any(axis=1)]
            norms = np.linalg.norm(lidar_clean, axis=1)
            # Keep points 0.1m to 6.0m (increased range to help ICP see further)
            valid_mask = (norms > 0.1) & (norms < 6.0) 
            lidar_clean = lidar_clean[valid_mask]

            # --- STATE & TIME ---
            try:
                v = float(frame['state'][2])
                w = float(frame['state'][3])
            except:
                v, w = 0.0, 0.0

            current_timestamp = frame.get('timestamp', None)
            if current_timestamp is None or prev_timestamp is None:
                dt = 0.1 # Default guess
            else:
                dt = current_timestamp - prev_timestamp
                if dt <= 0: dt = 0.01
                if dt > 1.0: dt = 0.1 # Cap huge jumps
            prev_timestamp = current_timestamp
            
            # --- 1. DEAD RECKONING GUESS (The "Trust" source) ---
            odom_guess = self.get_odometry_guess(v, w, dt)
            
            # --- 2. ICP CORRECTION (The "Verify" source) ---
            relative_trans = odom_guess # Default to Odom
            
            if len(lidar_clean) > 20:
                curr_pcd = o3d.geometry.PointCloud()
                curr_pcd.points = o3d.utility.Vector3dVector(lidar_clean)
                
                try:
                    curr_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
                    
                    if self.prev_pcd is not None:
                        try:
                            # Run Point-to-Plane ICP
                            reg_p2p = o3d.pipelines.registration.registration_icp(
                                curr_pcd, self.prev_pcd, self.icp_threshold, odom_guess,
                                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                            )
                            
                            # --- THE FIX: CORRIDOR LOGIC ---
                            # Extract translation distance from ICP result
                            icp_trans = reg_p2p.transformation
                            dx_icp = np.linalg.norm(icp_trans[:2, 3])
                            dx_odom = np.linalg.norm(odom_guess[:2, 3])
                            
                            # Condition 1: Fitness must be high
                            # Condition 2: If Odom says we moved (v > 0.1) but ICP says we didn't (dx < 0.05), 
                            #              it's the corridor problem -> REJECT ICP
                            
                            is_moving = (abs(v) > 0.1)
                            icp_says_stopped = (dx_icp < 0.05)
                            
                            if is_moving and icp_says_stopped:
                                # REJECT ICP, force Dead Reckoning
                                relative_trans = odom_guess
                            elif reg_p2p.fitness > 0.2:
                                # Accept ICP
                                relative_trans = icp_trans
                            else:
                                # Low fitness, fallback to Odom
                                relative_trans = odom_guess
                                
                        except Exception:
                            pass 
                    
                    self.prev_pcd = curr_pcd
                except Exception:
                    pass
            
            # --- 3. UPDATE GLOBAL POSE ---
            global_pose = global_pose @ relative_trans

            # --- 4. PAINT MAP ---
            if len(lidar_clean) > 0:
                pts_labeled, labels = self.get_semantic_labels(img, lidar_clean)
                if len(pts_labeled) > 0:
                    pts_hom = np.hstack((pts_labeled, np.ones((len(pts_labeled), 1))))
                    pts_global = (global_pose @ pts_hom.T).T[:, :3]
                    self.map.update(pts_global, labels, np.linalg.inv(global_pose))

            bev_img = self.map.get_bev(global_pose)
            
            # --- SAVE ---
            new_frame = frame.copy()
            new_frame['bev_semantic'] = bev_img 
            new_frame['pose_slam'] = global_pose
            processed_dataset.append(new_frame)
            
            if i % 50 == 0:
                cv2.imshow("Semantic Map Generation", bev_img)
                cv2.waitKey(1)

        with open(OUTPUT_PKL_PATH, 'wb') as f:
            pickle.dump(processed_dataset, f)
        print(f"Saved processed data to {OUTPUT_PKL_PATH}")
        cv2.destroyAllWindows()

def view_results(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    print(f"Loading dataset from {path}...")
    with open(path, 'rb') as f:
        dataset = pickle.load(f)

    idx = 0
    paused = False
    print("Controls: [SPACE] Pause, [N] Next, [B] Back, [Q] Quit")

    while True:
        frame = dataset[idx]
        img_rgb = frame['image_raw']
        bev_map = frame['bev_semantic']
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape
        bev_h, bev_w, _ = bev_map.shape
        scale = h / bev_h
        bev_resized = cv2.resize(bev_map, (int(bev_w * scale), h), interpolation=cv2.INTER_NEAREST)

        # Overlay Info
        cv2.putText(bev_resized, "BEV Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        v = float(frame['state'][2])
        cv2.putText(bev_resized, f"Vel: {v:.2f} m/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        combined = np.hstack((img_bgr, np.zeros((h, 10, 3), dtype=np.uint8), bev_resized))
        ts = frame.get('timestamp', 0)
        cv2.putText(combined, f"Frame {idx} | TS: {ts:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Map Verification", combined)

        key = cv2.waitKey(0 if paused else 30) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('n'): 
            idx = min(idx + 1, len(dataset) - 1)
            paused = True
        elif key == ord('b'): 
            idx = max(idx - 1, 0)
            paused = True

        if not paused:
            idx += 1
            if idx >= len(dataset): idx = 0

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', action='store_true', help="Run the map generation process")
    parser.add_argument('--view', action='store_true', help="Visualize the generated output")
    args = parser.parse_args()

    if args.process:
        proc = DatasetProcessor()
        proc.run()
    elif args.view:
        view_results(OUTPUT_PKL_PATH)
    else:
        print("Usage: python3 hd_map_generator.py --process OR --view")