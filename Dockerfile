FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build libopencv-dev python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .
RUN cmake --preset cuda && cmake --build --preset cuda

ENTRYPOINT ["/workspace/build/cuda/cuda-image-processor"]
