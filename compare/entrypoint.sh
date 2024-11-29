#!/bin/bash
# apt-get update

pwd

source /opt/ros/noetic/setup.bash

catkin build

source devel/setup.bash

# rosparam delete /use_sim_time
# rosrun openface2_ros openface2_ros_single _image_topic:=/naoqi_driver_node/camera/front/image_raw

chmod +x src/compare/src/run.py
chmod +x src/compare/src/optimizer.py
chmod +x src/compare/src/create_db.py
chmod +x src/compare/src/evaluate_corr.py

# rosrun compare run.py

# rosrun compare evaluate_corr.py 

rosrun compare create_db.py
# Keep the container running
exec "$@"
