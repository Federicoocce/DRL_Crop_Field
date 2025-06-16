#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np

from virtual_maize_field.world_generator.world_description import WorldDescription
from virtual_maize_field.world_generator.field_2d_generator import Field2DGenerator

def get_dense_lane_waypoints() -> list[dict]:
    """
    Generates a list of waypoints centered within each lane.
    Each waypoint contains 'x', 'y', and 'original_lane_index'.
    Waypoints within a lane are generated in their natural order from the field generator.
    """
    print(f"\n--- Loading Dense Lane Waypoints (Simplified) from Last Generated World ---")
    
    ros_home = Path.home() / ".ros"
    receipt_path = ros_home / "virtual_maize_field" / "last_used_world.json"
    
    if not receipt_path.is_file():
        print(f"\n[ERROR] World description file not found at: {receipt_path}")
        print("Please generate a world first...")
        return []
        
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
        return []

    if not hasattr(fgen, 'rows') or len(fgen.rows) < 2:
        print("  Field generator did not produce enough rows (at least 2 required).")
        return []

    all_waypoints = []
    num_lanes = len(fgen.rows) - 1
    print(f"  Found {len(fgen.rows)} rows, creating waypoints for {num_lanes} lanes.")

    for i in range(num_lanes): # i is the original_lane_index
        row_a = fgen.rows[i]
        row_b = fgen.rows[i+1]
        
        num_wp_in_lane = min(len(row_a), len(row_b))
        
        if num_wp_in_lane == 0:
            print(f"  Skipping lane {i+1} as one or both rows are empty.")
            continue
            
        for j in range(num_wp_in_lane):
            plant_a = row_a[j]
            plant_b = row_b[j]
            midpoint = (plant_a + plant_b) / 2.0
            all_waypoints.append({
                'x': float(midpoint[0]), 
                'y': float(midpoint[1]),
                'original_lane_index': i 
                # 'is_turn_assist_wp' will be added in the env if needed,
                # or assumed False for master_waypoints
            })
            
    print(f"  Generated {len(all_waypoints)} dense waypoints (simplified structure).")
    return all_waypoints

def main():
    # ... (main function for testing dense_waypoint.py can remain) ...
    dense_waypoints = get_dense_lane_waypoints()
    print("\n--- Summary of Dense Lane Waypoints (Simplified) ---")
    if dense_waypoints:
        print(f"  Total waypoints: {len(dense_waypoints)}")
        for i, wp in enumerate(dense_waypoints[:5]):
            print(f"    {i+1}. X: {wp['x']:.3f}, Y: {wp['y']:.3f}, LaneIdx: {wp['original_lane_index']}")
        if len(dense_waypoints) > 10:
            print("    ...")
        for i, wp in enumerate(dense_waypoints[-5:]):
            idx = len(dense_waypoints) - 5 + i
            print(f"    {idx+1}. X: {wp['x']:.3f}, Y: {wp['y']:.3f}, LaneIdx: {wp['original_lane_index']}")
    else:
        print("  No dense lane waypoints were generated.")

if __name__ == '__main__':
    main()