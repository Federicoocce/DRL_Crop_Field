import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from virtual_maize_field import get_spawner_launch_file


def generate_launch_description():
    """
    This launch file spawns a custom tracked robot into the maize field simulation.
    It assumes that the simulation has already been started in a separate terminal.
    """

    # --- Part 1: PUBLISH THE ROBOT'S DESCRIPTION ---
    
    # Find our custom robot description package
    pkg_my_tracked_robot = get_package_share_directory('my_tracked_robot_description')
    
    # Get the path to our robot_state_publisher launch file
    robot_state_publisher_launch_path = os.path.join(
        pkg_my_tracked_robot, 'launch', 'robot_state_publisher.launch.py'
    )

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Include the robot_state_publisher launch file. This will publish the
    # custom robot's URDF to the /robot_description topic.
    start_robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_state_publisher_launch_path),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )


    # --- Part 2: SPAWN THE ROBOT INTO THE WORLD ---

    # Use the function from the maize field package to get the generated spawner.
    # This spawner knows the correct starting coordinates.
    # The robot name here must match the <robot name="..."> in your URDF.
    robot_spawner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([get_spawner_launch_file()]),
        launch_arguments={"robot_name": "tracked_robot"}.items(),
    )

    return LaunchDescription([
        declare_use_sim_time_arg,
        start_robot_state_publisher,
        robot_spawner_launch,
    ])