#!/bin/bash

# Source the ROS setup script
source /opt/ros/noetic/setup.bash

# Function to handle termination signals
terminate() {
    echo "Terminating all background processes..."
    kill $(jobs -p)
    exit 1
}

# Trap termination signals and call the terminate function
trap terminate SIGINT SIGTERM

echo "Waiting 60 seconds before starting"
sleep 60

# Start Python scripts in the background
# python3 /app/sim_clock.py &
# PID_SIM_CLOCK=$!

python3 /app/play_once.py &
PID_PLAY_ONCE=$!


# Wait for any of the background processes to exit
wait -n $PID_PLAY_ONCE 
# $PID_SIM_CLOCK

# If the script reaches here, it means one of the processes has terminated
terminate
