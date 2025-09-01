# Launch all the sensors connected to the robot.
# It automatically checks which sensors are connected and launches them.
import subprocess
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# USB Vendor:Product IDs for each device
ZED_USB_ID = "2b03:f780"
IMU_USB_ID = "0483:5740"
LIDAR_USB_ID = "10c4:ea60"
GNSS_USB_ID = "1735:0002"

def usb_device_connected(vendor_product_id):
    # Check if a USB device with given Vendor:Product ID is connected.
    try:
        result = subprocess.run(
            ["lsusb"], capture_output=True, text=True, check=True
        )
        return vendor_product_id.lower() in result.stdout.lower()
    except subprocess.CalledProcessError:
        return False


def generate_launch_description():
    ld = LaunchDescription()

    # ZED 2 Camera Launch
    if usb_device_connected(ZED_USB_ID):
        zed_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('zed_wrapper'), 'launch', 'zed_camera.launch.py')
            ),
            launch_arguments={'camera_model': 'zed2'}.items()
        )
        ld.add_action(zed_launch)
        print("ZED 2 camera detected - launching.")
    else:
        print("ZED 2 camera not detected - skipping.")

    # LORD Microstrain 3DM-GX5-25 IMU Launch
    if usb_device_connected(IMU_USB_ID):
        imu_params_file = os.path.join(get_package_share_directory('microstrain_inertial_driver'), 'config', 'params.yml')
        imu_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('microstrain_inertial_driver'), 'launch', 'microstrain_launch.py')
            ),
            launch_arguments={'params_file': imu_params_file}.items()
        )
        ld.add_action(imu_launch)
        print("Microstrain IMU detected - launching.")
    else:
        print("Microstrain IMU not detected - skipping.")

    # RPLidar S1 Launch
    if usb_device_connected(LIDAR_USB_ID):
        lidar_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('rplidar_ros'), 'launch', 'rplidar_s1_launch.py')
            )
        )
        ld.add_action(lidar_launch)
        print("RPLidar S1 detected - launching.")
    else:
        print("RPLidar S1 not detected - skipping.")

    # Hemisphere GNSS Launch
    if usb_device_connected(GNSS_USB_ID):
        gnss_node = Node(
            package='hemisphere_gnss',
            executable='gnss_data',
            name='hemisphere_gnss_node',
            output='screen',
            parameters=[{'gnss_port': '/dev/ttyACM1'}] if usb_device_connected(IMU_USB_ID) else [{'gnss_port': '/dev/ttyACM0'}]
        )
        ld.add_action(gnss_node)
        print("Hemisphere GNSS detected - launching.")
    else:
        print("Hemisphere GNSS not detected - skipping.")

    return ld
