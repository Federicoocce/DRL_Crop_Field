#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
from shapely.geometry import LineString

try:
    from virtual_maize_field.world_generator.field_2d_generator import Field2DGenerator
    from virtual_maize_field.world_generator.world_description import WorldDescription
except ImportError:
    print("\n[ERROR] Could not import from 'virtual_maize_field'. Is the environment sourced?\n")
    # Return a dummy dict to prevent crashing importers if the environment isn't sourced
    def get_spawn_points() -> Dict[str, Dict[str, float]]:
        print("[WARN] Returning a single default spawn point at origin.")
        return {"default": {"x": 0.0, "y": 0.0, "z": 0.1, "yaw": 0.0}}
    
else:
    def calculate_yaw(p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculates the yaw angle in radians from point p1 to p2."""
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    def get_spawn_points() -> Dict[str, Dict[str, float]]:
        """
        Calculates spawn points based on the last generated world.
        Returns a dictionary of spawn locations.
        """
        # 1. Load the last world description
        ros_home = Path.home() / ".ros"
        receipt_path = ros_home / "virtual_maize_field" / "last_used_world.json"

        if not receipt_path.is_file():
            print(f"[ERROR] World description file not found at: {receipt_path}")
            print("[WARN] Returning a single default spawn point at origin.")
            return {"default": {"x": 0.0, "y": 0.0, "z": 0.1, "yaw": 0.0}}

        try:
            wd = WorldDescription(load_from_file=str(receipt_path))
            fgen = Field2DGenerator(wd)
            # Run the generator to get row data
            fgen.gather_available_models()
            fgen.chain_segments()
            fgen.center_plants()

            if not hasattr(fgen, "rows") or len(fgen.rows) < 2:
                raise ValueError("Field generator did not produce enough rows.")

        except Exception as e:
            print(f"  [ERROR] Failed to run world generator for spawn points: {e}")
            print("[WARN] Returning a single default spawn point at origin.")
            return {"default": {"x": 0.0, "y": 0.0, "z": 0.1, "yaw": 0.0}}

        spawn_data = {}
        Z_OFFSET = 0.1  # Spawn slightly above ground

        # --- Point 1: Default Spawn Point (near the 'START' marker) ---
        line = LineString([fgen.rows[0][0], fgen.rows[-1][0]])
        offset_start = line.parallel_offset(1.0, "right", join_style=2)
        default_pos = offset_start.centroid
        default_yaw = calculate_yaw(fgen.rows[0][0], fgen.rows[0][1])
        spawn_data["default"] = {"x": default_pos.x, "y": default_pos.y, "z": Z_OFFSET, "yaw": default_yaw}

        # --- Point 2: Start of the first lane ---
        row0_start, row1_start = fgen.rows[0][0], fgen.rows[1][0]
        lane_start_pos = (row0_start + row1_start) / 2.0
        lane_second_pos = (fgen.rows[0][1] + fgen.rows[1][1]) / 2.0
        lane_start_yaw = calculate_yaw(lane_start_pos, lane_second_pos)
        spawn_data["lane_start"] = {"x": lane_start_pos[0], "y": lane_start_pos[1], "z": Z_OFFSET, "yaw": lane_start_yaw}

        # --- Point 3: End of the first lane ---
        row0_end, row1_end = fgen.rows[0][-1], fgen.rows[1][-1]
        lane_end_pos = (row0_end + row1_end) / 2.0
        lane_penultimate_pos = (fgen.rows[0][-2] + fgen.rows[1][-2]) / 2.0
        lane_end_yaw = calculate_yaw(lane_penultimate_pos, lane_end_pos)
        spawn_data["lane_end"] = {"x": lane_end_pos[0], "y": lane_end_pos[1], "z": Z_OFFSET, "yaw": lane_end_yaw}
        
        print("--- Successfully calculated 3 spawn points ---")
        for name, data in spawn_data.items():
            print(f"  - {name}: x={data['x']:.2f}, y={data['y']:.2f}, yaw={data['yaw']:.2f}")

        return spawn_data

if __name__ == "__main__":
    # For testing the script directly
    get_spawn_points()