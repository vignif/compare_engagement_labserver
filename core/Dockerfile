# Use the official ROS base image
FROM ros:noetic-ros-core

# Copy the entrypoint script into the container
COPY entrypoint.sh /

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint to the script that sets simtime param and starts roscore
ENTRYPOINT ["/entrypoint.sh"]
