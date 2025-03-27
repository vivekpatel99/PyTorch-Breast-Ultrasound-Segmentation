# This assumes the container is running on a system with a CUDA GPU
#FROM tensorflow/tensorflow:nightly-gpu-jupyter


FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

WORKDIR /code

RUN apt-get update -y && \
    apt-get upgrade -y  \
    && apt-get install curl ffmpeg libsm6 libxext6  -y 



