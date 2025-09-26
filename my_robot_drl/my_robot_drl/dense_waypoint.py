#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np
from collections import defaultdict
import dubins
import matplotlib.pyplot as plt
import math

# Dependencies for this script:
# pip install dubins matplotlib numpy virtual-maize-field

# It's good practice to import from a library if available, but defaultdict is also fine.
# from typing import DefaultDict

# These classes are needed for waypoint generation
from virtual_maize_field.world_generator.world_description import WorldDescription
from virtual_maize_field.world_generator.field_2d_generator import Field2DGenerator

def get_dense_lane_waypoints() -> tuple[list[dict], list]:
    """
    Generates a list of waypoints and returns them along with the plant rows.
    If rows in a pair are of unequal length, it continues generating waypoints
    for the longer row by pairing its remaining plants with the last plant
    of the shorter row.

    This function will now also trigger a plot to visualize its output.

    Returns:
        tuple[list[dict], list]: A tuple containing the list of waypoints
                                 and the list of plant rows (each row is a list of np.array).
    """
    print(f"\n--- Loading Dense Lane Waypoints & Plant Rows from Last Generated World ---")

    ros_home = Path.home() / ".ros"
    receipt_path = ros_home / "virtual_maize_field" / "last_used_world.json"

    if not receipt_path.is_file():
        print(f"\n[ERROR] World description file not found at: {receipt_path}")
        print("Please generate a world first...")
        return [], []

    print(f"  Found world description: {receipt_path}")

    try:
        wd = WorldDescription(load_from_file=str(receipt_path))
        fgen = Field2DGenerator(wd)

        def do_nothing_plot(self, *args, **kwargs):
            pass

        Field2DGenerator.plot_field = do_nothing_plot

        dummy_cache_folder = Path.home() / ".ros" / "my_robot_drl_temp_gen"
        dummy_cache_folder.mkdir(parents=True, exist_ok=True)

        _ = fgen.generate(cache_dir=dummy_cache_folder)

    except Exception as e:
        import traceback
        print(f"  [ERROR] Failed to initialize or run world/field generator: {e}")
        traceback.print_exc()
        return [], []

    if not hasattr(fgen, 'rows') or len(fgen.rows) < 2:
        print("  Field generator did not produce enough rows (at least 2 required).")
        return [], []

    # --- Step 1: Create all waypoints with a default boundary flag ---
    all_waypoints = []
    num_lanes = len(fgen.rows) - 1
    print(f"  Found {len(fgen.rows)} rows, creating waypoints for {num_lanes} lanes.")

    for i in range(num_lanes): # i is the original_lane_index
        row_a = fgen.rows[i]
        row_b = fgen.rows[i+1]

        len_a, len_b = len(row_a), len(row_b)

        if len_a == 0 or len_b == 0:
            print(f"  Skipping lane {i} as one or both rows are empty.")
            continue

        num_wp_in_lane = max(len_a, len_b)

        for j in range(num_wp_in_lane):
            plant_a = row_a[j] if j < len_a else row_a[-1]
            plant_b = row_b[j] if j < len_b else row_b[-1]

            midpoint = (plant_a + plant_b) / 2.0

            all_waypoints.append({
                'x': float(midpoint[0]),
                'y': float(midpoint[1]),
                'original_lane_index': i,
                'is_lane_boundary': False # Default to False
            })

    print(f"  Generated {len(all_waypoints)} raw waypoints.")

    # --- Step 2: After creation, iterate and flag the boundaries for each lane ---
    if not all_waypoints:
        return [], fgen.rows

    print("  Now flagging lane boundaries...")

    lanes_map = defaultdict(list)
    for wp in all_waypoints:
        lanes_map[wp['original_lane_index']].append(wp)

    for lane_index, waypoints_in_lane in lanes_map.items():
        if len(waypoints_in_lane) > 0:
            waypoints_in_lane[0]['is_lane_boundary'] = True
            waypoints_in_lane[-1]['is_lane_boundary'] = True

    print("  Boundary flagging complete.")

    # --- NEW: Call the plotting function from inside this function ---
    # This will visualize the generated waypoints and plant rows immediately.
    # print("\n  >>> Triggering visualization from within get_dense_lane_waypoints...")
    # plot_all_waypoints(all_waypoints, fgen.rows)

    return all_waypoints, fgen.rows

def plot_all_waypoints(waypoints, plant_rows):
    """
    Creates a plot of all waypoints and plant rows to visualize the lane structure.
    - Lanes are plotted with unique colors and connected by lines.
    - Boundary waypoints are highlighted with a distinct marker.
    - All waypoints are annotated with their overall index.
    """
    if not waypoints:
        print("Cannot plot waypoints: The list is empty.")
        return

    print("\n--- Plotting All Loaded Waypoints & Plant Rows (Overview) ---")
    fig, ax = plt.subplots(figsize=(12, 12))

    # --- Plot Plant Rows (as background) ---
    if plant_rows:
        all_plants_x = []
        all_plants_y = []
        for row in plant_rows:
            for plant in row:
                all_plants_x.append(plant[0])
                all_plants_y.append(plant[1])
        ax.scatter(all_plants_x, all_plants_y, c='green', s=15, alpha=0.6, label='Plants', marker='.', zorder=1)

    # --- Plot Waypoints Lane by Lane for a Clear Structure ---

    # 1. Group all waypoints by lane index and also collect boundary waypoints
    lanes_map = defaultdict(list)
    boundary_wps = []
    for wp in waypoints:
        lanes_map[wp['original_lane_index']].append(wp)
        if wp['is_lane_boundary']:
            boundary_wps.append(wp)

    # 2. Set up a color map for the lanes
    all_lane_indices = sorted(lanes_map.keys())
    num_lanes = len(all_lane_indices)
    # Using 'viridis' colormap which is more accessible than 'jet'
    colors = plt.cm.viridis(np.linspace(0, 1, num_lanes))
    lane_color_map = {idx: color for idx, color in zip(all_lane_indices, colors)}

    # 3. Plot each lane's waypoints connected by a line
    for lane_index, wps_in_lane in sorted(lanes_map.items()):
        if not wps_in_lane:
            continue

        x_coords = [wp['x'] for wp in wps_in_lane]
        y_coords = [wp['y'] for wp in wps_in_lane]

        ax.plot(x_coords, y_coords,
                marker='o',          # Add a small circle for each waypoint
                markersize=5,
                linestyle='-',       # Connect them with a line
                color=lane_color_map[lane_index],
                label=f'Lane {lane_index}' if num_lanes < 15 else None, # Avoid crowded legend
                zorder=3)

    # 4. Plot boundary waypoints on top with a distinct marker to highlight them
    if boundary_wps:
        bound_x = [wp['x'] for wp in boundary_wps]
        bound_y = [wp['y'] for wp in boundary_wps]
        ax.scatter(bound_x, bound_y,
                   edgecolor='black',
                   facecolor='yellow',
                   s=150,
                   marker='*',
                   zorder=5,
                   label='End-of-Row Waypoints')

    # 5. Annotate all waypoints with their original index for debugging
    for i, wp in enumerate(waypoints):
        ax.text(wp['x'] + 0.05, wp['y'], str(i), fontsize=8, zorder=4)

    ax.set_title('Waypoint Structure Overview')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    plt.show()

def plot_specific_uturn(all_waypoints, plant_rows, from_lane_idx, to_lane_idx):
    """
    Identifies waypoints for a U-turn, generates the Dubins path, and plots the
    result with all other waypoints and plant rows in the background for context.
    """
    print(f"\n--- Implementing U-Turn from Lane {from_lane_idx} to Lane {to_lane_idx} ---")

    TURNING_RADIUS = 0.7
    WAYPOINT_STEP = 0.2

    # 1. Identify relevant waypoints with alternating logic
    from_lane_wps = [wp for wp in all_waypoints if wp['original_lane_index'] == from_lane_idx]
    to_lane_wps = [wp for wp in all_waypoints if wp['original_lane_index'] == to_lane_idx]

    if len(from_lane_wps) < 2 or len(to_lane_wps) < 2:
        print(f"Error: Not enough waypoints in lane {from_lane_idx} or {to_lane_idx} to generate a turn.")
        return

    # ALTERNATING LOGIC for boustrophedon path
    if from_lane_idx % 2 == 0:
        print(f"  (Even Lane {from_lane_idx}: Assuming turn at far end of the field)")
        wp_prev, wp_end_lane = from_lane_wps[-2], from_lane_wps[-1]
        wp_start_next_lane, wp_after_start_next_lane = to_lane_wps[-1], to_lane_wps[-2]
    else: # Odd 'from' lane
        print(f"  (Odd Lane {from_lane_idx}: Assuming turn at near end of the field)")
        wp_prev, wp_end_lane = from_lane_wps[1], from_lane_wps[0]
        wp_start_next_lane, wp_after_start_next_lane = to_lane_wps[0], to_lane_wps[1]

    # 2. Define Start Pose (q0)
    x0, y0 = wp_end_lane['x'], wp_end_lane['y']
    yaw_start = math.atan2(wp_end_lane['y'] - wp_prev['y'], wp_end_lane['x'] - wp_prev['x'])
    q0 = (x0, y0, yaw_start)

    # 3. Define End Pose (q1)
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

    # PLOT 0: Plant rows in the far background
    if plant_rows:
        for row in plant_rows:
            row_arr = np.array(row)
            if row_arr.size > 0:
                ax.scatter(row_arr[:, 0], row_arr[:, 1], c='lightgreen', s=10, zorder=0)

    # PLOT 1: All waypoints in the background, with end-of-row points highlighted
    boundary_wps = [wp for wp in all_waypoints if wp['is_lane_boundary']]
    regular_wps = [wp for wp in all_waypoints if not wp['is_lane_boundary']]

    if regular_wps:
        reg_arr = np.array([[wp['x'], wp['y']] for wp in regular_wps])
        ax.scatter(reg_arr[:, 0], reg_arr[:, 1], c='gray', alpha=0.4, label='All Field Waypoints', zorder=1)

    if boundary_wps:
        bound_arr = np.array([[wp['x'], wp['y']] for wp in boundary_wps])
        ax.scatter(bound_arr[:, 0], bound_arr[:, 1], c='orange', marker='x', alpha=0.7, label='End-of-Row (Context)', zorder=2)

    # PLOT 2: Key waypoints for this specific turn
    context_wps = {'Prev WP': wp_prev, 'End of Lane WP (q0)': wp_end_lane,
                   'Start of Next Lane WP (q1)': wp_start_next_lane, 'After Start WP': wp_after_start_next_lane}
    for name, point in context_wps.items():
        ax.plot(point['x'], point['y'], 'ko', markersize=10, label=name, zorder=5)
        ax.text(point['x'] + 0.05, point['y'] + 0.05, name, fontsize=12)

    # PLOT 3: The generated Dubins path
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
    # 1. Load the master list of waypoints and plant rows.
    #    The plot for all waypoints will now be generated automatically by this function call.
    master_waypoints, plant_rows = get_dense_lane_waypoints()

    if not master_waypoints:
        print("\nExecution finished: No waypoints were loaded. Cannot create visualizations.")
        return

    # 2. The general overview plot is now called inside get_dense_lane_waypoints.
    #    The call `plot_all_waypoints(master_waypoints, plant_rows)` is no longer needed here.

    # 3. Get the set of available lane indices from the data
    available_lanes = sorted(list({wp['original_lane_index'] for wp in master_waypoints}))
    if len(available_lanes) < 2:
        print("\nNot enough lanes to demonstrate a U-turn.")
        return

    # # 4. Demonstrate the FIRST U-turn (e.g., from lane 0 to 1 at the far end)
    # plot_specific_uturn(master_waypoints, plant_rows, from_lane_idx=available_lanes[0], to_lane_idx=available_lanes[1])

    # # 5. Demonstrate the SECOND, ALTERNATING U-turn (e.g., from lane 1 to 2 at the near end)
    # if len(available_lanes) > 2:
    #     plot_specific_uturn(master_waypoints, plant_rows, from_lane_idx=available_lanes[1], to_lane_idx=available_lanes[2])
    # else:
    #     print("\nSkipping second U-turn plot: Only 2 lanes found in the data.")

if __name__ == '__main__':
    main()