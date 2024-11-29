#!/bin/bash
# apt-get update

pwd

rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -y

source /opt/ros/noetic/setup.bash

catkin build

mv devel/lib/yunet_detector.cpython-38-x86_64-linux-gnu.so devel/lib/python3/dist-packages/

source devel/setup.bash

# rosrun openface2_ros openface2_ros_single _image_topic:=/naoqi_driver_node/camera/front/image_raw
# roslaunch grace_ros run_grace.launch 
chmod +x src/grace_ros/src/server_compare.py


# rosparam delete /use_sim_time

roslaunch grace_ros params_grace.launch

# Keep the container running
exec "$@"
