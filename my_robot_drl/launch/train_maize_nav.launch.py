import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression # Added PythonExpression
from launch.conditions import IfCondition, UnlessCondition # Added conditions

def generate_launch_description():
    # Define package paths
    maize_field_pkg = get_package_share_directory('virtual_maize_field')
    maize_robot_bringup_pkg = get_package_share_directory('maize_robot_bringup')
    my_robot_drl_pkg = get_package_share_directory('my_robot_drl')

    # --- Declare Launch Argument for Headless Mode ---
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='true', # Default to headless for training
        description='Run Gazebo in headless mode (no GUI). Set to "false" for GUI.'
    )

    # --- 1. Generate the World ---
    world_config_name = 'fre22_task_navigation_mini'
    generate_world_cmd = ExecuteProcess(
        cmd=['ros2', 'run', 'virtual_maize_field', 'generate_world', world_config_name],
        output='screen'
    )

    # --- Define actions to be launched later ---
    world_file_path = os.path.join(os.path.expanduser('~'), '.ros', 'virtual_maize_field', 'generated.world')

    # Action to launch Gazebo Server (headless)
    gazebo_server_cmd = ExecuteProcess(
        cmd=['gzserver',
             '--verbose',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             world_file_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless')) # Only run if headless is true
    )

    # Action to launch Gazebo with GUI (not headless)
    # This uses your existing simulation.launch.py, assuming it starts both server and client
    # and can accept a world argument.
    # You might need to adjust simulation.launch.py to accept a 'world' argument.
    gazebo_gui_launch_file = os.path.join(maize_field_pkg, 'launch', 'simulation.launch.py')
    gazebo_gui_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_gui_launch_file),
        launch_arguments={'world': world_file_path}.items(), # Pass world file as argument
        condition=UnlessCondition(LaunchConfiguration('headless')) # Only run if headless is false
    )
    # If your simulation.launch.py doesn't take a 'world' arg, you might need separate
    # gzserver and gzclient ExecuteProcess actions for the GUI mode, similar to headless but adding gzclient.


    # Action to launch robot spawner
    robot_bringup_launch_file = os.path.join(maize_robot_bringup_pkg, 'launch', 'spawn_tracked_robot.launch.py')
    start_and_spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_bringup_launch_file),
        # launch_arguments={'use_sim_time': 'true'}.items() # Pass if needed
    )

    # Action to start DRL Training Script
    train_agent_node = Node(
        package='my_robot_drl',
        executable='train_agent',
        name='drl_trainer',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # --- Orchestration Logic ---
    return LaunchDescription([
        headless_arg, # Declare the argument first

        # Step 1: Run world generation.
        generate_world_cmd,
        
        # Step 2: When world generation finishes, start Gazebo (either server or full sim).
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=generate_world_cmd,
                on_exit=[
                    LogInfo(msg="World generation complete. Starting Gazebo..."),
                    # Conditionally launch Gazebo server or full simulation
                    gazebo_server_cmd, # This will only execute if headless is true
                    gazebo_gui_sim,    # This will only execute if headless is false
                ]
            )
        ),

        # Step 3: After Gazebo is assumed to be ready (using a delay), spawn the robot.
        TimerAction(
            period=8.0, # Adjusted delay, might need tuning based on Gazebo start time
            actions=[
                LogInfo(msg="Gazebo likely up. Spawning robot..."),
                start_and_spawn_robot
            ]
        ),
        
        # Step 4: After robot is assumed to be spawned, start training.
        TimerAction(
            period=16.0, # Adjusted delay
            actions=[
                LogInfo(msg="Robot likely spawned. Starting DRL training..."),
                train_agent_node
            ]
        )
    ])