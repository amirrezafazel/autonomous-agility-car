# Autonomous RC Car - ROS2

This project involves converting a Traxxas RC car into an autonomous vehicle using ROS2. The car is equipped with a ZED2i stereo camera, an RPLiDAR S1, a Microstrain 3DM-GX5-25 IMU, a Hemisphere GNSS base and rover, and an NVIDIA Jetson TX2i for processing. More details about the project can be found [here](assets/report.pdf).

<!-- ![Project Screenshot](assets/robots.png) -->

## 🏁 Getting Started

This section will guide you through the set up process for this project.

### **Prerequisites**

-   [Ubuntu 22.04 (Jammy Jellyfish)](https://releases.ubuntu.com/jammy/)
-   [ROS 2 Humble Hawksbill](https://docs.ros.org/en/humble/index.html)
-   [ZED SDK v4.2](https://www.stereolabs.com/en-gb/developers/release/4.2)
-   [CUDA](https://developer.nvidia.com/cuda-downloads) dependency


### **Installation**

1.  Create your workspace (or use an existing one).

    ```sh
    mkdir -p ~/ros2_ws/src/
    cd ~/ros2_ws/src/
    ```

2.  Clone the repository into your workspace.
    ```sh
    git clone https://github.com/amirrezafazel/autonomous-agility-car.git
    ```

3.  Clone the following repositories:

    - [zed-ros2-wrapper](github.com/stereolabs/zed-ros2-wrapper/tree/humble-v4.2.x) from Stereolabs
    - [zed-ros2-examples](https://github.com/stereolabs/zed-ros2-examples/tree/humble-v4.2.x) from Stereolabs
    - [microstrain_inertial](https://github.com/LORD-MicroStrain/microstrain_inertial/tree/ros2) from LORD-MicroStrain
    - [rplidar_ros](https://github.com/Slamtec/rplidar_ros/tree/ros2) from Slamtec

    ```sh
    git clone -b humble-v4.2.x --single-branch https://github.com/stereolabs/zed-ros2-examples.git
    git clone -b humble-v4.2.x --single-branch https://github.com/stereolabs/zed-ros2-wrapper.git
    git clone --recursive -b ros2 https://github.com/LORD-MicroStrain/microstrain_inertial.git
    git clone -b ros2 https://github.com/Slamtec/rplidar_ros.git
    ```

4.  Add the params.yml to microstrain_inertial_driver package.

    ```sh
    cp ./autonomous-agility-car/assets/params.yml ./microstrain_inertial/microstrain_inertial_driver/config/
    ```

5.  Update the dependencies and build the packages.

    ```sh
    cd ..
    sudo apt update
    rosdep update
    rosdep install --from-paths src --ignore-src -r -y # install dependencies
    colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc) # build the workspace
    ```

6.  Source the workspace to use its ROS packages and commands.

    ```sh
    source ./install/setup.bash
    ```

    To automatically source the installation in every new bash session (Optional):
    
    ```sh
    echo source $(pwd)/install/setup.bash >> ~/.bashrc
    ```

7.  Create udev rules for rplidar.

    ```sh
    source ./src/rplidar_ros/scripts/create_udev_rules.sh
    cd ../..
    ```

## 🏃 Usage

To run all the ros2 nodes use the following command:

```sh
ros2 launch bringup bringup.launch.py 
```

To only run the nodes related to sensing (data published from sensors): 

```sh
ros2 launch sensing all_sensors.launch.py
```

To check the ros2 topic available use `ros2 topic list` and to view the data published on a specific topic use `ros2 topic echo <topic_name>`

## 🛠️ Troubleshooting tips

If you encounter problems with building/running the following packages please have a look at their github pages:

- [zed-ros2-wrapper](github.com/stereolabs/zed-ros2-wrapper/tree/humble-v4.2.x)
- [zed-ros2-examples](https://github.com/stereolabs/zed-ros2-examples/tree/humble-v4.2.x)
- [microstrain_inertial](https://github.com/LORD-MicroStrain/microstrain_inertial/tree/ros2)
- [rplidar_ros](https://github.com/Slamtec/rplidar_ros/tree/ros2)

To check if the Zed SDK is installed correctly run the following diagnostic:
```sh
/usr/local/zed/tools/ZED_Diagnostic
```

If there is an issue related to the port the devices/sensors are connected to try changing/adding parameters in the `sensing/launch/all_sensors.launch.py`.

## Special Thanks

I would like to thank Prof. Saber Fallah and University of Surrey for giving me this opportunity to work on this project, as well as Danny Fallah for his contributions to this project. 