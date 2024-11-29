# Use the Nvidia CUDA base image
FROM nvidia/cuda:11.2.2-base-ubuntu20.04

# Set the working directory inside the container
WORKDIR /workspace

# Install necessary packages
RUN apt-get update && \
    apt-get install -y git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/opendr-eu/opendr.git

# Set the working directory to the cloned repository
WORKDIR /workspace/opendr

# Install Python dependencies
# RUN pip3 install -r requirements.txt

# Start an interactive shell
CMD ["/bin/bash"]
