import open3d as o3d
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm

# Path to the PROCESSED file (contains the 'pose_slam' key)
DATASET_PATH = os.path.expanduser('~/ros2_ws/drl_datasets/imitation_learning/real_world_processed.pkl')

def visualize_3d():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: File not found at {DATASET_PATH}")
        return

    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Reconstructing 3D Map from {len(dataset)} frames...")

    all_points = []
    all_colors = []
    trajectory_points = []

    # We skip frames to keep the visualizer lightweight (e.g., every 2nd frame)
    SKIP_FRAME = 2 

    for i, frame in enumerate(tqdm(dataset)):
        if i % SKIP_FRAME != 0: continue

        # 1. Get Data
        lidar_local = frame['lidar_raw']
        pose = frame['pose_slam']
        
        # Store Trajectory (Translation component)
        trajectory_points.append(pose[:3, 3])

        # 2. Clean Data (Remove padding and near-field noise)
        norms = np.linalg.norm(lidar_local, axis=1)
        valid = (norms > 0.2) & (norms < 6.0)
        clean_local = lidar_local[valid]
        
        if len(clean_local) == 0: continue

        # 3. Transform to Global Frame
        # R @ points + t
        # Note: lidar_local is (N, 3), we need (3, N) for matmul, then transpose back
        points_global = (pose[:3, :3] @ clean_local.T).T + pose[:3, 3]

        # 4. Colorize (Heuristic)
        # Ground (Low Z) = Brown, Crops (High Z) = Green
        # Adjust Z threshold based on your robot's lidar height. 
        # Assuming Lidar is at ~0.2m - 0.5m off ground.
        # Points below -0.1 relative to lidar might be ground? 
        # Let's use Local Z for coloring before transform
        
        local_z = clean_local[:, 2]
        colors = np.zeros_like(points_global)
        
        # Height coloring heuristic
        # Z > 0.1 -> Green (Crop)
        # Z <= 0.1 -> Brown (Ground)
        
        mask_crop = local_z > 0.05 
        
        # Green [0, 1, 0]
        colors[mask_crop] = [0.0, 0.8, 0.0] 
        # Brown/Gray [0.4, 0.4, 0.3]
        colors[~mask_crop] = [0.4, 0.4, 0.3]

        all_points.append(points_global)
        all_colors.append(colors)

    # 5. Build Point Cloud
    print("Building Open3D Geometry...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    # Downsample to make it responsive
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # 6. Build Trajectory Line
    if len(trajectory_points) > 1:
        traj_lines = o3d.geometry.LineSet()
        traj_points = np.array(trajectory_points)
        traj_lines.points = o3d.utility.Vector3dVector(traj_points)
        
        # Connect point i to i+1
        lines = [[i, i+1] for i in range(len(traj_points)-1)]
        traj_lines.lines = o3d.utility.Vector2iVector(lines)
        traj_lines.paint_uniform_color([1, 0, 0]) # Red Line path
    else:
        traj_lines = o3d.geometry.LineSet()

    # 7. Coordinate Frame (Start Point)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    print("\n=========================================")
    print("OPENING 3D VIEWER")
    print("-----------------------------------------")
    print(" RED LINE   = Robot Path")
    print(" GREEN DOTS = Crops")
    print(" BROWN DOTS = Ground")
    print("-----------------------------------------")
    print(" Controls:")
    print("   [Mouse Left]   Rotate")
    print("   [Mouse Wheel]  Zoom")
    print("   [Mouse Right]  Pan")
    print("   [+/-]          Increase/Decrease Point Size")
    print("=========================================")

    o3d.visualization.draw_geometries([pcd, traj_lines, axis])

if __name__ == "__main__":
    visualize_3d()