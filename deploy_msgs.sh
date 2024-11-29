#!/bin/bash

# Source directory
SOURCE_DIR="./grace_common_msgs"

# Destination directories
DEST_DIRS=("eng_0/catkin_ws/src/" "eng_1/catkin_ws/src/", "eng_5/catkin_ws/src/", "compare/catkin_ws/src/")

# Loop through each destination directory and copy the source directory
for DEST_DIR in "${DEST_DIRS[@]}"; do
    # Construct the full path for the destination
    FULL_DEST_DIR="${DEST_DIR}grace_common_msgs"
    
    # Check if the source directory exists
    if [ -d "$SOURCE_DIR" ]; then
        # Remove the destination directory if it exists
        if [ -d "$FULL_DEST_DIR" ]; then
            rm -rf "$FULL_DEST_DIR"
        fi
        
        # Copy the source directory to the destination
        cp -r "$SOURCE_DIR" "$FULL_DEST_DIR"
        
        # Verify if the copy was successful
        if [ $? -eq 0 ]; then
            echo "Successfully copied $SOURCE_DIR to $FULL_DEST_DIR"
        else
            echo "Failed to copy $SOURCE_DIR to $FULL_DEST_DIR"
        fi
    else
        echo "Source directory $SOURCE_DIR does not exist"
    fi
done
