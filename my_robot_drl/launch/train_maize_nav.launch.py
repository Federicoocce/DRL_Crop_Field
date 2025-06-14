import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    # Define package paths
    maize_field_pkg = get_package_share_directory('virtual_maize_field')
    maize_robot_bringup_pkg = get_package_share_directory('maize_robot_bringup')

    # --- 1. Generate the World ---
    # This command runs first, creating the world and the spawner launch file.
    world_config_name = 'fre22_task_navigation_mini'
    generate_world_cmd = ExecuteProcess(
        cmd=['ros2', 'run', 'virtual_maize_field', 'generate_world', world_config_name],
        output='screen'
    )

    # --- Define all other actions that will be launched later ---
    
    # Action to launch Gazebo
    gazebo_launch_file = os.path.join(maize_field_pkg, 'launch', 'simulation.launch.py')
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file)
    )
    
    # Action to launch your robot spawner (which also includes the robot_state_publisher)
    robot_bringup_launch_file = os.path.join(maize_robot_bringup_pkg, 'launch', 'spawn_tracked_robot.launch.py')
    start_and_spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_bringup_launch_file)
    )

    # Action to start the DRL Training Script
    train_agent_node = Node(
        package='my_robot_drl',
        executable='train_agent',
        name='drl_trainer',
        output='screen'
    )

    # --- Orchestration Logic ---
    return LaunchDescription([
        # Step 1: Run world generation.
        generate_world_cmd,
        
        # Step 2: When world generation finishes, start the next sequence of events.
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=generate_world_cmd,
                on_exit=[
                    # Launch Gazebo immediately after generation is done.
                    gazebo_sim,
                    
                    # Step 3: Use a timed delay after starting Gazebo before spawning the robot.
                    # This gives Gazebo time to initialize its services. 5 seconds is a safe delay.
                    TimerAction(
                        period=5.0,
                        actions=[start_and_spawn_robot]
                    ),
                    
                    # Step 4: Use a longer delay before starting the training agent.
                    # This ensures the robot is fully spawned and its plugins (like /odom) are active.
                    TimerAction(
                        period=10.0,
                        actions=[train_agent_node]
                    )
                ]
            )
        ),
    ])