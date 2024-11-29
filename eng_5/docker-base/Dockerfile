FROM ros:noetic

LABEL maintainer="Francesco Vigni <vignif@gmail.com>"

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

RUN pip install --upgrade pip setuptools wheel build
RUN pip install mediapipe


CMD [ "/bin/bash" ]

