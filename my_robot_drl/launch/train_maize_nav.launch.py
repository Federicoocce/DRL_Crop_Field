import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition

def generate_launch_description():
    # Define package paths
    maize_field_pkg = get_package_share_directory('virtual_maize_field')
    maize_robot_bringup_pkg = get_package_share_directory('maize_robot_bringup')
    my_robot_drl_pkg = get_package_share_directory('my_robot_drl')
    gazebo_ros_pkg = get_package_share_directory('gazebo_ros')

    # --- Declare Launch Argument for Headless Mode ---
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='true',
        description='Run Gazebo in headless mode (no GUI). Set to "false" for GUI.'
    )

    # --- Declare Launch Argument for Primary Train/Eval/IL Mode ---
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='train',
        description="Primary mode: 'train' (SAC), 'eval' (SAC), or 'il' (Imitation Learning)"
    )

    # --- NEW: Declare Launch Argument specifically for Imitation Learning Mode ---
    il_mode_arg = DeclareLaunchArgument(
        'il_mode',
        default_value='train',
        description="Mode for the imitation learning script: 'train' or 'collect'"
    )

    # --- 1. Generate the World ---
    world_config_name = 'my_world'
    generate_world_cmd = ExecuteProcess(
        cmd=['ros2', 'run', 'virtual_maize_field', 'generate_world', world_config_name],
        output='screen'
    )

    # --- Define actions to be launched later ---
    world_file_path = os.path.join(os.path.expanduser('~'), '.ros', 'virtual_maize_field', 'generated.world')

    # --- Gazebo Launch (Robust Method) ---
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file_path,
            'headless': LaunchConfiguration('headless'),
            'gui': PythonExpression(["not ", LaunchConfiguration('headless')]),
            'pause': 'false',
            'verbose': 'true',
        }.items()
    )

    # Action to launch robot spawner
    robot_bringup_launch_file = os.path.join(maize_robot_bringup_pkg, 'launch', 'spawn_tracked_robot.launch.py')
    start_and_spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_bringup_launch_file),
    )

    # Node for the ORIGINAL DRL training script (SAC)
    train_agent_node = Node(
        package='my_robot_drl',
        executable='train_agent',
        name='drl_trainer_sac',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[LaunchConfiguration('mode')],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('mode'), "' != 'il'"])
        )
    )

    # MODIFIED Node for the IMITATION LEARNING script
    # It now accepts the 'il_mode' launch argument and passes it to the script.
    train_imitation_node = Node(
        package='my_robot_drl',
        executable='train_imitation',
        name='drl_trainer_il',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['--mode', LaunchConfiguration('il_mode')], # This line passes the argument
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('mode'), "' == 'il'"])
        )
    )

    # --- Orchestration Logic ---
    return LaunchDescription([
        headless_arg,
        mode_arg,
        il_mode_arg, # Add the new argument to the launch description

        generate_world_cmd,

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=generate_world_cmd,
                on_exit=[
                    LogInfo(msg="World generation complete. Starting Gazebo..."),
                    gazebo_launch,
                ]
            )
        ),

        TimerAction(
            period=8.0,
            actions=[
                LogInfo(msg="Gazebo likely up. Spawning robot..."),
                start_and_spawn_robot
            ]
        ),

        TimerAction(
            period=16.0,
            actions=[
                LogInfo(msg="Robot likely spawned. Starting selected training script..."),
                train_agent_node,
                train_imitation_node
            ]
        )
    ])
