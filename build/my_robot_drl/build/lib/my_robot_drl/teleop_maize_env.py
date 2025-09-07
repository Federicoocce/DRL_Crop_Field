#!/usr/bin/env python3

import rclpy
import time
import sys
import termios
import tty
import select # For non-blocking key read
import math
import numpy as np

from geometry_msgs.msg import Twist
# Assuming your MaizeNavigationEnv is in a package that can be imported.
# If your package is 'my_robot_drl_pkg' and drl_env.py is in 'my_robot_drl_pkg/my_robot_drl_pkg/',
# then the import would be:
# from my_robot_drl_pkg.drl_env import MaizeNavigationEnv
# If teleop_maize_env.py is in the same directory as drl_env.py and run directly,
# and that directory is in PYTHONPATH:
from .drl_env import MaizeNavigationEnv # Adjust if your DRL environment is in a package

# Key bindings
KEY_BINDINGS = {
    'w': (1.0, 0.0),  # Move forward
    's': (-1.0, 0.0), # Move backward
    'a': (0.0, 1.0),  # Turn left
    'd': (0.0, -1.0), # Turn right
    'x': (0.0, 0.0),  # Stop / Zero velocity
    'r': 'reset',     # Reset environment
    'q': 'quit'       # Quit
}
# Speed settings
SPEED_LINEAR = 0.25  # m/s
SPEED_ANGULAR = 0.6 # rad/s

def get_key_non_blocking():
    """Reads a single key press without blocking. Returns empty string if no key."""
    if select.select([sys.stdin], [], [], 0.05)[0]: # Timeout 0.05s
        return sys.stdin.read(1)
    return ''

def print_controls():
    print("\n" + "="*30)
    print("Teleoperation Controls:")
    print("  'w': Move Forward")
    print("  's': Move Backward")
    print("  'a': Turn Left")
    print("  'd': Turn Right")
    print("  'x': Stop Robot")
    print("  'r': Reset Environment")
    print("  'q': Quit Teleoperation")
    print("="*30 + "\n")

def print_status(env: MaizeNavigationEnv, current_twist: Twist, reward: float, info: dict):
    """Prints the current status of the robot and environment."""
    sys.stdout.write("\r") # Carriage return to overwrite previous line

    status_str = ""
    if env.current_odom:
        pos = env.current_odom.pose.pose.position
        _, _, yaw = env.euler_from_quaternion(env.current_odom.pose.pose.orientation)
        
    else:
        status_str += "Odom: N/A"

    target_idx = info.get('current_target_wp_idx', -1)
    if target_idx != -1 and env.waypoints and target_idx < len(env.waypoints):
        target_wp = env.waypoints[target_idx]
        status_str += f" | TargetWP:#{target_idx}(X:{target_wp['x']:.2f},Y:{target_wp['y']:.2f})"
        if target_wp.get('is_turn_assist_wp', False):
            status_str += "(ASSIST)"
        status_str += f" Dist:{info.get('distance_to_target', -1):.2f}m"
    else:
        status_str += " | TargetWP: None"

    status_str += f" | Visited:{info.get('waypoints_visited',0)}/{info.get('waypoints_total',0)}"
    # status_str += f" | Reward:{reward:.2f}" # Reward can be noisy for teleop display
    min_scan = info.get('collision_sensor_min_range', -1.0)
    status_str += f" | MinScan:{min_scan:.2f}m"
    
    # Pad with spaces to clear previous longer lines
    status_str = status_str.ljust(120) 
    sys.stdout.write(status_str)
    sys.stdout.flush()

def run_teleop_loop(env: MaizeNavigationEnv, old_terminal_settings):
    """Main loop for teleoperation."""
    tty.setraw(sys.stdin.fileno()) # Set terminal to raw mode
    print_controls()

    observation, info = env.reset() # Initial reset
    current_twist = Twist()
    linear_vel_target = 0.0
    angular_vel_target = 0.0
    
    is_running = True
    while is_running and rclpy.ok():
        key = get_key_non_blocking()

        if key: # Process key press
            if key in KEY_BINDINGS:
                command = KEY_BINDINGS[key]
                if command == 'reset':
                    env.get_logger().info("Resetting environment by key press...")
                    observation, info = env.reset()
                    current_twist = Twist() # Stop robot on reset
                    linear_vel_target, angular_vel_target = 0.0, 0.0
                    env.cmd_vel_pub.publish(current_twist)
                    env.get_logger().info("Environment reset.")
                    print_controls() # Reprint controls after reset messages
                    continue
                elif command == 'quit':
                    env.get_logger().info("Quitting teleoperation...")
                    is_running = False
                    break
                else: # Movement command
                    linear_vel_target = command[0] * SPEED_LINEAR
                    angular_vel_target = command[1] * SPEED_ANGULAR
            elif key == '\x03': # CTRL+C
                env.get_logger().info("CTRL+C pressed, quitting teleoperation...")
                is_running = False
                break
        
        current_twist.linear.x = linear_vel_target
        current_twist.angular.z = angular_vel_target
        
        # This is important for _calculate_reward if it uses self.last_action
        env.last_action = np.array([current_twist.linear.x, current_twist.angular.z], dtype=np.float32)

        env.cmd_vel_pub.publish(current_twist)
        rclpy.spin_once(env, timeout_sec=0.01) # Process odom/scan, etc. for the env Node

        # Call _calculate_reward to trigger waypoint progression logic (including U-turns)
        # The reward value itself isn't used by an agent here, but the side effects are what we want.
        reward = env._calculate_reward() 
        observation = env._get_observation() # Get updated observation
        info = env._get_info()               # Get updated info

        print_status(env, current_twist, reward, info)

        if env.episode_done:
            reason = info.get('termination_reason', 'unknown')
            print(f"\nEpisode finished! Reason: {reason}")
            print("Press 'r' to reset, or 'q' to quit.")
            
            # Wait for 'r' or 'q'
            while rclpy.ok():
                key_after_done = get_key_non_blocking()
                if key_after_done == 'r':
                    env.get_logger().info("Resetting environment after episode end...")
                    observation, info = env.reset()
                    current_twist = Twist()
                    linear_vel_target, angular_vel_target = 0.0, 0.0
                    env.cmd_vel_pub.publish(current_twist)
                    env.get_logger().info("Environment reset.")
                    print_controls()
                    break 
                elif key_after_done == 'q' or key_after_done == '\x03':
                    env.get_logger().info("Quitting after episode end...")
                    is_running = False
                    break
                time.sleep(0.1) # Don't spin too fast waiting for key
            
            if not is_running:
                break
        
        time.sleep(0.05) # Loop rate control (approx 20Hz with other delays)

    # Restore terminal settings and stop robot
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)
    print("\nRestored terminal settings.")
    stop_twist = Twist()
    env.cmd_vel_pub.publish(stop_twist)
    env.get_logger().info("Robot stopped.")


def main(args=None):
    rclpy.init(args=args)
    
    # Store original terminal settings
    old_terminal_settings = termios.tcgetattr(sys.stdin)
    
    env_instance = None
    try:
        # Important: Ensure last_used_world.json exists as per dense_waypoint.py
        # You might need to run your world generator first.
        env_instance = MaizeNavigationEnv()
        env_instance.get_logger().info("MaizeNavigationEnv created for teleoperation.")
        
        run_teleop_loop(env_instance, old_terminal_settings)

    except Exception as e:
        # Restore terminal settings in case of crash
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)
        print(f"\nException in teleop main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # This will be called after loop exits or if an exception occurs in `try`
        # old_terminal_settings should be restored by run_teleop_loop or by except block
        if env_instance:
            env_instance.get_logger().info("Closing environment.")
            env_instance.close() # This calls destroy_node for the env
        
        rclpy.shutdown()
        print("Teleoperation finished and RCLPY shutdown.")

if __name__ == '__main__':
    main()