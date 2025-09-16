# test_dubins_uturn.py
#
# A standalone script to test and visualize the Dubins path generation for a U-turn,
# mimicking the logic from the MaizeNavigationEnv.
#
# Dependencies:
# pip install dubins matplotlib numpy

import dubins
import matplotlib.pyplot as plt
import math
import numpy as np

def plot_dubins_uturn(q0, q1, turning_radius, step_size, wp_points):
    """
    Generates and plots a Dubins path for a given U-turn scenario.

    Args:
        q0 (tuple): Start pose (x, y, yaw_radians).
        q1 (tuple): End pose (x, y, yaw_radians).
        turning_radius (float): The turning radius for the vehicle.
        step_size (float): The distance between sampled points on the path.
        wp_points (dict): A dictionary of original waypoints for plotting context.
    """
    print(f"Generating Dubins path with:")
    print(f"  Start q0: (x={q0[0]:.2f}, y={q0[1]:.2f}, yaw={math.degrees(q0[2]):.1f} deg)")
    print(f"  End   q1: (x={q1[0]:.2f}, y={q1[1]:.2f}, yaw={math.degrees(q1[2]):.1f} deg)")
    print(f"  Turning Radius: {turning_radius:.2f} m")
    print("-" * 30)

    # 1. Generate the shortest Dubins path
    path = dubins.shortest_path(q0, q1, turning_radius)
    
    # 2. Sample points along the path
    # The second return value is the distance array, which we ignore here.
    configurations, _ = path.sample_many(step_size)

    if not configurations:
        print("Dubins path generation failed or returned no points.")
        return

    print(f"Path type: {path.path_type()}")
    print(f"Generated {len(configurations)} points (including start and end).")
    # In the DRL env, we would insert configurations[1:-1]
    print(f"This would result in {max(0, len(configurations) - 2)} new waypoints being inserted.")
    
    # --- Plotting ---
    path_points = np.array(configurations)

    
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot original context waypoints
    for name, point in wp_points.items():
        ax.plot(point['x'], point['y'], 'ko', markersize=10, label=name, zorder=5)
        ax.text(point['x'] + 0.05, point['y'] + 0.05, name, fontsize=12)

    # Plot the full, smooth Dubins path
    ax.plot(path_points[:, 0], path_points[:, 1], 'c-', label='Full Dubins Path', linewidth=2, zorder=3)

    # Plot the sampled waypoints that would be added
    if len(configurations) > 2:
        inserted_wps = path_points[1:-1, :]
        ax.plot(inserted_wps[:, 0], inserted_wps[:, 1], 'b.', markersize=8, label='Generated Turn Waypoints', zorder=4)

    # Plot start and end poses with orientation arrows
    arrow_len = 0.3
    # Start pose q0
    ax.arrow(q0[0], q0[1], arrow_len * math.cos(q0[2]), arrow_len * math.sin(q0[2]),
             head_width=0.08, head_length=0.1, fc='g', ec='g', label='Start Pose (q0)', zorder=6)
    # End pose q1
    ax.arrow(q1[0], q1[1], arrow_len * math.cos(q1[2]), arrow_len * math.sin(q1[2]),
             head_width=0.08, head_length=0.1, fc='r', ec='r', label='End Pose (q1)', zorder=6)
    
    ax.set_title(f'Dubins U-Turn Visualization (Type: {path.path_type()})')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.legend()
    ax.axis('equal')  # CRITICAL for correct aspect ratio of turns
    plt.show()


def main(args=None):
    # --- Parameters matching drl_env.py ---
    TURNING_RADIUS = 0.7
    WAYPOINT_STEP = 0.2
    
    # ... (the rest of the logic from the if __name__ == "__main__": block)

    wp_prev = {'x': 0.0, 'y': 1.0}
    wp_end_lane = {'x': 0.0, 'y': 2.0}
    wp_start_next_lane = {'x': 0.8, 'y': 2.0}

    x0, y0 = wp_end_lane['x'], wp_end_lane['y']
    yaw_start = math.atan2(wp_end_lane['y'] - wp_prev['y'], wp_end_lane['x'] - wp_prev['x'])
    q0 = (x0, y0, yaw_start)

    x1, y1 = wp_start_next_lane['x'], wp_start_next_lane['y']
    yaw_end = yaw_start + math.pi
    yaw_end = (yaw_end + math.pi) % (2 * math.pi) - math.pi
    q1 = (x1, y1, yaw_end)

    context_wps = {
        'Prev WP': wp_prev,
        'End of Lane WP': wp_end_lane,
        'Start of Next Lane WP': wp_start_next_lane
    }

    plot_dubins_uturn(q0, q1, TURNING_RADIUS, WAYPOINT_STEP, context_wps)


if __name__ == '__main__':
    main()