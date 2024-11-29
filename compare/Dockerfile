FROM ros:noetic

LABEL maintainer="Francesco Vigni <vignif@gmail.com>"

# Set Git access token as an environment variable
# ENV GIT_ACCESS_TOKEN

# Install apt-utils to prevent delaying package configuration
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

# Install essential build tools and dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    g++-8 \
    cmake \
    git \
    libopenblas-dev \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python-dev \
    python3 \
    python3-pip \
    python3-numpy \
    python3-opencv \
    python3-rosdep \
    python3-catkin-tools \
    python3-catkin \
    python3-pybind11 \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    unzip \
    wget \
    ros-noetic-catkin \
    ros-noetic-cv-bridge \
    libyaml-cpp-dev \
    ros-noetic-tf2* \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 and pip3 are the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

COPY ./catkin_ws /catkin_ws

COPY ./try.py /catkin_ws/

RUN chmod +x /catkin_ws/try.py

RUN pip install --upgrade pip setuptools wheel build

RUN pip install scipy

WORKDIR /catkin_ws

# Install Git and clone private repository
RUN git clone https://${GIT_ACCESS_TOKEN}@github.com/vignif/grace_common_msgs.git /catkin_ws/src/grace_common_msgs


RUN rm -f /etc/ros/rosdep/sources.list.d/20-default.list

# Initialize rosdep and install dependencies
RUN rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y


COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# RUN /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# CMD [ "/bin/bash" ]

