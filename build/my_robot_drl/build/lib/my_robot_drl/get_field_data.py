#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np

from virtual_maize_field.world_generator.world_description import WorldDescription
from virtual_maize_field.world_generator.field_2d_generator import Field2DGenerator

def get_precise_row_waypoints() -> list[dict]:
    """
    Generates waypoints by loading the description of the world that was
    most recently generated for the simulator.
    Waypoints include start, middle, and end points for each lane in a serpentine pattern.
    """
    print(f"\n--- Loading Waypoints from Last Generated World ---")
    
    # Define the path to the "receipt" file.
    ros_home = Path.home() / ".ros"
    receipt_path = ros_home / "virtual_maize_field" / "last_used_world.json"
    
    if not receipt_path.is_file():
        print(f"\n[ERROR] World description file not found at: {receipt_path}")
        print("Please generate a world first by running:")
        print("  ros2 run virtual_maize_field generate_world <your_config_name>")
        print("or by running the simulation launch file that does this for you.\n")
        return []
        
    print(f"  Found world description: {receipt_path}")

    try:
        # Create a WorldDescription instance by loading directly from the file.
        # The seed will be loaded from the file, ensuring the RNG is identical.
        wd = WorldDescription(load_from_file=str(receipt_path))
        
        # Now proceed with generation, which will be identical to the simulator's.
        fgen = Field2DGenerator(wd)
        
        dummy_cache_folder = Path.home() / ".ros" / "my_robot_drl_temp_gen"
        dummy_cache_folder.mkdir(parents=True, exist_ok=True)
        _ = fgen.generate(cache_dir=dummy_cache_folder)
        
    except Exception as e:
        import traceback
        print(f"  [ERROR] Failed to initialize or run world/field generator from description file: {e}")
        traceback.print_exc()
        return []

    if not hasattr(fgen, 'rows') or len(fgen.rows) < 2:
        print("  Field generator did not produce enough rows (at least 2 required) to create lanes.")
        return []

    serpentine_waypoints = []
    num_lanes = len(fgen.rows) - 1
    print(f"  Found {len(fgen.rows)} rows, creating waypoints for {num_lanes} lanes.")

    for i in range(num_lanes):
        lane_num = i + 1
        row_a, row_b = fgen.rows[i], fgen.rows[i+1]
        
        if len(row_a) == 0 or len(row_b) == 0:
            print(f"  Skipping lane {lane_num} due to empty row(s).")
            continue
            
        # Physical start, middle, and end of the lane segments
        # Ensure rows have enough points for indexing
        # Midpoint uses integer division, safe for len=1 (idx 0)
        idx_mid_a = len(row_a) // 2
        idx_mid_b = len(row_b) // 2

        phys_midpoint_start = (row_a[0] + row_b[0]) / 2.0
        phys_midpoint_lane = (row_a[idx_mid_a] + row_b[idx_mid_b]) / 2.0
        phys_midpoint_end = (row_a[-1] + row_b[-1]) / 2.0
        
        wp_A_data = {'x': float(phys_midpoint_start[0]), 'y': float(phys_midpoint_start[1])}
        wp_M_data = {'x': float(phys_midpoint_lane[0]), 'y': float(phys_midpoint_lane[1])}
        wp_B_data = {'x': float(phys_midpoint_end[0]), 'y': float(phys_midpoint_end[1])}

        # Serpentine traversal:
        # Even lanes (0, 2, ... corresponding to lane_num 1, 3, ...): travel A -> M -> B
        # Odd lanes (1, 3, ... corresponding to lane_num 2, 4, ...): travel B -> M -> A
        if i % 2 == 0: # Forward traversal relative to physical row definition
            serpentine_waypoints.append({**wp_A_data, 'type': f'lane_{lane_num}_start'})
            serpentine_waypoints.append({**wp_M_data, 'type': f'lane_{lane_num}_mid'})
            serpentine_waypoints.append({**wp_B_data, 'type': f'lane_{lane_num}_end'})
        else: # Backward traversal relative to physical row definition
            serpentine_waypoints.append({**wp_B_data, 'type': f'lane_{lane_num}_start'})
            serpentine_waypoints.append({**wp_M_data, 'type': f'lane_{lane_num}_mid'})
            serpentine_waypoints.append({**wp_A_data, 'type': f'lane_{lane_num}_end'})
            
    print(f"  Generated {len(serpentine_waypoints)} waypoints (start, mid, end per lane).")
    return serpentine_waypoints


# Main function is now just for testing and not used by the visualizer
def main():
    precise_row_waypoints = get_precise_row_waypoints()
    print("\n--- Summary of Precise Lane Waypoints from Last Generated World ---")
    if precise_row_waypoints:
        for i, wp in enumerate(precise_row_waypoints):
            print(f"    {i+1}. Type: {wp['type']}, X: {wp['x']:.3f}, Y: {wp['y']:.3f}")
    else:
        print("  No precise lane waypoints were generated.")

if __name__ == '__main__':
    main()