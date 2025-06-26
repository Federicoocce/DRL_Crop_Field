# File: my_tracked_robot_description/launch/robot_state_publisher.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
# This import is required for the fix
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    pkg_path = os.path.join(get_package_share_directory('my_tracked_robot_description'))
    xacro_file = os.path.join(pkg_path, 'urdf', 'tracked_robot.urdf.xacro')

    robot_description_config = Command(['xacro ', xacro_file])

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            # THIS IS THE FIX: Explicitly mark the parameter as a string
            'robot_description': ParameterValue(robot_description_config, value_type=str)
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        robot_state_publisher_node,
    ])