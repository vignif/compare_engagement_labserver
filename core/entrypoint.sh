#!/bin/bash

# Source the ROS setup script
source /opt/ros/noetic/setup.bash

# Start roscore in the background
roscore &
ROSCORE_PID=$!

# Wait for roscore to fully start
sleep 5

# Set the use_sim_time parameter to true
# rosparam set /use_sim_time true

# Wait for the roscore process to complete
wait $ROSCORE_PID
