# Use the Nvidia CUDA runtime image
FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04

# Set the working directory inside the container
WORKDIR /workspace

# Install necessary packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    libfreetype6-dev \
    build-essential \
    cmake \
    python3-dev \
    wget \
    libopenblas-dev \
    libsndfile1 \
    libboost-dev \
    libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install wheel==0.38.4 && \
    pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip3 install mxnet-cu112==1.8.0post0 && \
    pip3 install opendr-toolkit-engine && \
    pip3 install opendr-toolkit

# Start an interactive shell
CMD ["/bin/bash"]

