# Use an official Ubuntu 20.04 image as the base (use Ubuntu 20.04 for compatibility with ROS Noetic)
FROM ubuntu:20.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Set environment variable to prevent tzdata prompt during install
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies to fetch ROS packages and general utilities
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    ca-certificates \
    build-essential \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-rosdep \
    python3-venv \
    wget \
    lsb-release \
    git

# Add ROS Noetic repository for Ubuntu Focal (20.04) explicitly
RUN curl -sSL http://packages.ros.org/ros.key | apt-key add - \
    && echo "deb http://packages.ros.org/ros/ubuntu focal main" | tee /etc/apt/sources.list.d/ros-latest.list \
    && apt-get update

# Install ROS Noetic Desktop Full
RUN apt-get install -y ros-noetic-desktop-full

# Install Python dependencies using pip3
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install \
    opencv-python-headless \
    torch \
    torchvision \
    pympi \
    numpy \
    pandas \
    matplotlib \
    scikit-learn

# Set up ROS environment (source ROS setup file directly)
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install ROS dependencies using rosdep
RUN rosdep update \
    && rosdep install --from-paths src --ignore-src -r -y \
    && apt-get install -y python3-catkin-pkg-modules python3-rospkg-modules python3-rosdistro-modules

# Define the default command to run
CMD [ "/bin/bash" ]
