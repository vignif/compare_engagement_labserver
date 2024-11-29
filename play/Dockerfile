FROM ros:noetic

# Install Python 3, pip, and necessary ROS packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ros-noetic-rosbag \
    ros-noetic-rospy \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

COPY *.py /app/

COPY *.sh /app/

# Copy the entrypoint script into the container
COPY entrypoint.sh /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/*.sh


# Set the entrypoint to run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
