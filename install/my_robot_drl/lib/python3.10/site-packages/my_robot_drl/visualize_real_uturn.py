# visualize_real_uturn.py
#
# A script to load waypoints directly from dense_waypoint.py, visualize their
# layout, and test the Dubins U-turn logic for ALTERNATING turns.
#
# This script:
# 1. Calls get_dense_lane_waypoints() to load real waypoint data.
# 2. Plots an overview of all loaded waypoints with their indices.
# 3. Implements and visualizes two alternating U-turns (e.g., 0->1 and 1->2).
# 4. Includes all waypoints in the background of U-turn plots for context.
#
# Dependencies:
# pip install dubins matplotlib numpy

import dubins
import matplotlib.pyplot as plt
import math
import numpy as np

# Directly import the function from your provided file
from .dense_waypoint import get_dense_lane_waypoints

def plot_all_waypoints(waypoints):
    """
    Creates a scatter plot of all waypoints and annotates them with their index.
    """
    if not waypoints:
        print("Cannot plot waypoints: The list is empty.")
        return
        
    print("\n--- Plotting All Loaded Waypoints (Overview) ---")
    fig, ax = plt.subplots(figsize=(10, 12))
    
    lane_indices = [wp['original_lane_index'] for wp in waypoints]
    num_lanes = len(set(lane_indices))
    colors = plt.cm.jet(np.linspace(0, 1, num_lanes))
    
    for i, wp in enumerate(waypoints):
        lane_idx = wp['original_lane_index']
        ax.scatter(wp['x'], wp['y'], color=colors[lane_idx], s=50)
        ax.text(wp['x'] + 0.05, wp['y'], str(i), fontsize=9)
        
    ax.set_title('Overview of All Waypoints from dense_waypoint.py')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.grid(True)
    ax.axis('equal')
    plt.show()

def plot_specific_uturn(all_waypoints, from_lane_idx, to_lane_idx):
    """
    Identifies waypoints for an ALTERNATING U-turn, generates the Dubins path,
    and plots the result with all other waypoints in the background for context.
    """
    print(f"\n--- Implementing U-Turn from Lane {from_lane_idx} to Lane {to_lane_idx} ---")
    
    # --- Parameters matching drl_env.py ---
    TURNING_RADIUS = 0.7
    WAYPOINT_STEP = 0.2

    # 1. Identify relevant waypoints with alternating logic
    from_lane_wps = [wp for wp in all_waypoints if wp['original_lane_index'] == from_lane_idx]
    to_lane_wps = [wp for wp in all_waypoints if wp['original_lane_index'] == to_lane_idx]

    if len(from_lane_wps) < 2 or len(to_lane_wps) < 2:
        print(f"Error: Not enough waypoints in lane {from_lane_idx} or {to_lane_idx} to generate a turn.")
        return

    # ALTERNATING LOGIC:
    # An even 'from' lane (0, 2...) turns at the FAR end.
    # An odd 'from' lane (1, 3...) turns at the NEAR end.
    # This assumes a standard "snake" or "boustrophedon" path.
    if from_lane_idx % 2 == 0:
        print(f"  (Even Lane {from_lane_idx}: Assuming turn at far end of the field)")
        # 'from' lane ends at the last points
        wp_prev = from_lane_wps[-2]
        wp_end_lane = from_lane_wps[-1]
        # 'to' lane starts at its last points (coming back)
        wp_start_next_lane = to_lane_wps[-1]
        wp_after_start_next_lane = to_lane_wps[-2]
    else: # Odd 'from' lane
        print(f"  (Odd Lane {from_lane_idx}: Assuming turn at near end of the field)")
        # 'from' lane ends at the first points
        wp_prev = from_lane_wps[1]
        wp_end_lane = from_lane_wps[0]
        # 'to' lane starts at its first points (going away)
        wp_start_next_lane = to_lane_wps[0]
        wp_after_start_next_lane = to_lane_wps[1]
        
    # 2. Define Start Pose (q0) for the Dubins path
    x0, y0 = wp_end_lane['x'], wp_end_lane['y']
    yaw_start = math.atan2(wp_end_lane['y'] - wp_prev['y'], 
                           wp_end_lane['x'] - wp_prev['x'])
    q0 = (x0, y0, yaw_start)

    # 3. Define End Pose (q1) for the Dubins path
    x1, y1 = wp_start_next_lane['x'], wp_start_next_lane['y']
    yaw_end = math.atan2(wp_after_start_next_lane['y'] - wp_start_next_lane['y'],
                         wp_after_start_next_lane['x'] - wp_start_next_lane['x'])
    q1 = (x1, y1, yaw_end)

    print(f"  Generating Dubins path with:")
    print(f"    Start q0: (x={q0[0]:.2f}, y={q0[1]:.2f}, yaw={math.degrees(q0[2]):.1f} deg)")
    print(f"    End   q1: (x={q1[0]:.2f}, y={q1[1]:.2f}, yaw={math.degrees(q1[2]):.1f} deg)")
    
    # 4. Generate and sample the Dubins path
    path = dubins.shortest_path(q0, q1, TURNING_RADIUS)
    configurations, _ = path.sample_many(WAYPOINT_STEP)
    
    if not configurations:
        print("Dubins path generation failed.")
        return

    # --- Plotting ---
    path_points = np.array(configurations)
    fig, ax = plt.subplots(figsize=(10, 8))

    # PLOT 1: All waypoints in the background for context
    all_wp_array = np.array([[wp['x'], wp['y']] for wp in all_waypoints])
    ax.scatter(all_wp_array[:, 0], all_wp_array[:, 1], c='gray', alpha=0.4, label='All Field Waypoints', zorder=1)

    context_wps = {
        'Prev WP': wp_prev,
        'End of Lane WP (q0)': wp_end_lane,
        'Start of Next Lane WP (q1)': wp_start_next_lane,
        'After Start WP': wp_after_start_next_lane
    }

    # PLOT 2: Key waypoints for this specific turn
    for name, point in context_wps.items():
        ax.plot(point['x'], point['y'], 'ko', markersize=10, label=name, zorder=5)
        ax.text(point['x'] + 0.05, point['y'] + 0.05, name, fontsize=12)

    # PLOT 3: The generated Dubins path and its waypoints
    ax.plot(path_points[:, 0], path_points[:, 1], 'c-', label='Full Dubins Path', linewidth=2, zorder=3)
    if len(configurations) > 2:
        inserted_wps = path_points[1:-1, :]
        ax.plot(inserted_wps[:, 0], inserted_wps[:, 1], 'b.', markersize=8, label='Generated Turn Waypoints', zorder=4)

    # PLOT 4: Orientation arrows for start and end poses
    arrow_len = 0.4
    ax.arrow(q0[0], q0[1], arrow_len * math.cos(q0[2]), arrow_len * math.sin(q0[2]),
             head_width=0.08, head_length=0.1, fc='g', ec='g', label='Start Pose (q0)', zorder=6)
    ax.arrow(q1[0], q1[1], arrow_len * math.cos(q1[2]), arrow_len * math.sin(q1[2]),
             head_width=0.08, head_length=0.1, fc='r', ec='r', label='End Pose (q1)', zorder=6)
    
    ax.set_title(f'Dubins U-Turn Visualization (Lane {from_lane_idx} to {to_lane_idx})')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    plt.show()

def main():
    # 1. Load the master list of waypoints using your function
    master_waypoints = get_dense_lane_waypoints()
    
    if not master_waypoints:
        print("\nExecution finished: No waypoints were loaded. Cannot create visualizations.")
        return
        
    # 2. Plot all waypoints with their indices for a general overview
    plot_all_waypoints(master_waypoints)
    
    # 3. Get the set of available lane indices from the data
    available_lanes = sorted(list({wp['original_lane_index'] for wp in master_waypoints}))
    if len(available_lanes) < 2:
        print("\nNot enough lanes to demonstrate a U-turn.")
        return

    # 4. Demonstrate the FIRST U-turn (e.g., from lane 0 to 1 at the far end)
    plot_specific_uturn(master_waypoints, from_lane_idx=available_lanes[0], to_lane_idx=available_lanes[1])
    
    # 5. Demonstrate the SECOND, ALTERNATING U-turn (e.g., from lane 1 to 2 at the near end)
    if len(available_lanes) > 2:
        plot_specific_uturn(master_waypoints, from_lane_idx=available_lanes[1], to_lane_idx=available_lanes[2])
    else:
        print("\nSkipping second U-turn plot: Only 2 lanes found in the data.")


if __name__ == '__main__':
    main()