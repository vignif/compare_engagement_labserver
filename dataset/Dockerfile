FROM ros:noetic

# Install Python 3, pip, and necessary ROS packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ros-noetic-rosbag \
    ros-noetic-rospy \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

COPY download_manager.py /app/

RUN pip install pandas opencv-python progressbar

CMD [ "/bin/bash" ]