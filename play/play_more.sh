#!/bin/bash

# Ensure the script exits if any command fails
set -e

# List of rosbag files to play
ROSBAG_FILES=(
    "bags/user1_2017-03-03.bag"
    "bags/user108_2017-03-15.bag"
    "bags/user14_2017-06-14.bag"
    "bags/user16_2017-01-23.bag"
    "bags/user184_2017-02-03.bag"
    "bags/user191_2017-03-16.bag"
    "bags/user201_2017-02-03.bag"
    "bags/user218_2017-02-09.bag"
    "bags/user23_2017-01-20.bag"
    "bags/user230_2017-01-25.bag"
    "bags/user279_2017-04-26.bag"
    "bags/user28_2017-01-31.bag"
    "bags/user315_2017-02-10.bag"
    "bags/user36_2017-03-13.bag"
    "bags/user4_2017-02-17.bag"
    "bags/user555_2017-04-14.bag"
    "bags/user60_2017-02-20.bag"
    "bags/user66_2017-05-12.bag"
    "bags/user8_2017-01-31.bag"
    "bags/user8_2017-03-23.bag"
    "bags/user95_2017-02-24.bag"
    "bags/user104_2017-06-20.bag"
    "bags/user2_2017-01-19.bag"
    "bags/user215_2017-02-03.bag"
    "bags/user23_2017-03-10.bag"
    "bags/user350_2017-04-13.bag"
    "bags/user53_2017-01-31.bag"
    "bags/user68_2017-01-26.bag"
)

# Topics to filter (space-separated)
TOPICS="/naoqi_driver_node/camera/front/image_raw /naoqi_driver_node/camera/front/camera_info /naoqi_driver_node/camera/depth/image_raw /naoqi_driver_node/camera/depth/camera_info /tf"

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
echo "sleeping 60 seconds before starting to play"
sleep 60

# Iterate through the list of rosbags
for BAG_FILE in "${ROSBAG_FILES[@]}"; do
    if [ -f "$BAG_FILE" ]; then
        echo "Playing rosbag: $BAG_FILE"

        # Get the total duration of the bag file
        TOTAL_DURATION=$(rosbag info -y -k duration "$BAG_FILE")
        START_TIME=$(date +%s.%N)

        # Start the progress monitor in the background
        print_progress $TOTAL_DURATION $START_TIME &

        # Play the bag file with selected topics and --clock option
        rosbag play "$BAG_FILE" --clock --topics $TOPICS

        # Kill the progress monitor
        wait $!

        echo "Finished playing rosbag: $BAG_FILE"
    else
        echo "Bag file $BAG_FILE does not exist."
    fi
done

echo "Completed sequence of bags. Exiting."
