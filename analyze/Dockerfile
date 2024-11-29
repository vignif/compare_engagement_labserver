# Use the official NVIDIA CUDA base image for compatibility with GPU
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies in a single step
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    xgboost \
    scikit-learn \
    optuna \
    matplotlib \
    seaborn \
    cupy-cuda112 \
    && rm -rf /root/.cache

# Copy necessary scripts and data
COPY optim/ /app/optim/
COPY dataset_2weeks.csv /app/dataset.csv
COPY entrypoint.sh /app/entrypoint.sh

# Change working directory
WORKDIR /app

# Ensure the entrypoint script is executable
RUN chmod +x entrypoint.sh

# Set the default command
CMD ["/bin/bash"]
