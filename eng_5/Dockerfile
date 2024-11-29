FROM eng_5_base
ARG CACHE_BUST=1

LABEL maintainer="Francesco Vigni <vignif@gmail.com>"


COPY grace /grace

# Set Git access token as an environment variable
ENV GIT_ACCESS_TOKEN
RUN git clone https://${GIT_ACCESS_TOKEN}@github.com/vignif/grace_common_msgs.git /catkin_ws/src/grace_common_msgs

# Install pybind11_catkin and other ROS dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-noetic-pybind11-catkin \
    # Add other ROS dependencies here if needed \
    ros-noetic-geometry-msgs \ 
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /grace

# Build and install the Python package using build
RUN pip install -r /grace/requirements.txt
RUN python -m build
RUN pip install dist/*.whl

WORKDIR /catkin_ws

RUN rm -f /etc/ros/rosdep/sources.list.d/20-default.list

# # Initialize rosdep and install dependencies
# RUN rosdep init && \
#     rosdep update && \
#     rosdep install --from-paths src --ignore-src -y


RUN git clone https://github.com/ros4hri/hri_face_detect.git -b main /catkin_ws/src/hri_face_detect

RUN git clone https://github.com/ros4hri/hri_msgs.git -b master /catkin_ws/src/hri_msgs

COPY grace_ros /catkin_ws/src/grace_ros

COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# RUN /entrypoint.sh
# RUN mv devel/lib/yunet_detector.cpython-38-x86_64-linux-gnu.so devel/lib/python3/dist-packages/

ENTRYPOINT ["/entrypoint.sh"]

# CMD [ "/bin/bash" ]

