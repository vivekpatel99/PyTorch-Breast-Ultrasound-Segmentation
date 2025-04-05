# This assumes the container is running on a system with a CUDA GPU
# https://github.com/pytorch/pytorch#docker-image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

WORKDIR /code

RUN apt-get update -y && \
    apt-get upgrade -y  \
    # Packages need for opencv
    && apt-get install  --no-install-recommends -y curl ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
