#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Check if the bag file is provided as an argument
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <bag_file> [topic1 topic2 ...]"
    exit 1
fi

# Extract the bag file from the first argument
BAG_FILE=$1
shift

# Check if specific topics are provided
if [ "$#" -gt 0 ]; then
    TOPICS=$@
else
    # Default topics if none are provided
    TOPICS="/naoqi_driver_node/camera/front/image_raw /naoqi_driver_node/camera/front/camera_info /naoqi_driver_node/camera/depth/image_raw /naoqi_driver_node/camera/depth/camera_info /tf"
fi

# Function to print progress
print_progress() {
    local TOTAL_DURATION=$1
    local START_TIME=$2
    while true; do
        CURRENT_TIME=$(rostopic echo -n 1 /clock | grep 'secs' | awk '{print $2}')
        if [ -z "$CURRENT_TIME" ]; then
            break
        fi
        ELAPSED_TIME=$(echo "$CURRENT_TIME - $START_TIME" | bc)
        PROGRESS=$(echo "$ELAPSED_TIME / $TOTAL_DURATION * 100" | bc -l)
        echo "Progress: $(printf "%.2f" $PROGRESS)%"
        sleep 1
    done
}

# Print the name of the bag file being played
echo "Playing rosbag: $BAG_FILE"

# Get the total duration of the bag file
TOTAL_DURATION=$(rosbag info -y -k duration "$BAG_FILE")
START_TIME=$(date +%s.%N)

# Start the progress monitor in the background
print_progress $TOTAL_DURATION $START_TIME &

# Play the bag file with selected topics and --clock option
rosbag play "$BAG_FILE" --clock -e $TOPICS

# Kill the progress monitor
kill $!

echo "Finished playing rosbag: $BAG_FILE"
