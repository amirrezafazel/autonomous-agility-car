# Main launch file, use to start all the nodes.
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()

    # Sensing Launch (sensors)
    sensing_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('sensing'), 'launch', 'all_sensors.launch.py')
        )
    )
    ld.add_action(sensing_launch)

    # Localisation Launch
    localisation_node = Node(
        package='localisation',
        executable='new_ekf_localisation_3d',
        name='ekf_localisation_node',
        output='screen',
    )
    ld.add_action(localisation_node)

    return ld
