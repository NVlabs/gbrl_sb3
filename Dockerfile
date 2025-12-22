##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
FROM nvcr.io/nvidia/pytorch:23.11-py3 as base
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install ffmpeg libsm6 libxext6 libxrender-dev -y
RUN apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 -y
# install for gfootball
RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc 
# Install wget and other dependencies
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install -y unzip && \
    apt-get install -y swig
# Create the directory where Mujoco will be installed
RUN mkdir -p ~/.mujoco
# Download and extract Mujoco
RUN wget --no-check-certificate -O mujoco.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xzf mujoco.tar.gz -C ~/.mujoco && \
    rm mujoco.tar.gz
WORKDIR /
RUN git clone https://github.com/NVlabs/gbrl_sb3.git
WORKDIR /gbrl_sb3
# Set CUDA environment for GBRL GPU compilation
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
# Install gfootball first (has conflicting deps)
RUN pip install gfootball
# Install requirements EXCEPT gbrl (we'll build from source)
RUN grep -v "^gbrl" requirements.txt > requirements_no_gbrl.txt && \
    pip install -r requirements_no_gbrl.txt
# Build and install GBRL from source with CUDA support
RUN pip install gbrl --no-binary gbrl --verbose
# Fix numpy/opencv version conflicts
RUN pip uninstall -y opencv-python && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2* && \
    pip install "numpy<1.25" "opencv-python==4.8.1.78"
# Set LD_LIBRARY_PATH with MuJoCo and HPC-X paths
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# Create runs directory and declare as volume mount point
RUN mkdir -p /gbrl_sb3/runs && chmod 777 /gbrl_sb3/runs
VOLUME /gbrl_sb3/runs

# Set working directory
WORKDIR /gbrl_sb3



