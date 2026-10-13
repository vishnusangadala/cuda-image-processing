# CUDA Image Processing Accelerator

A command-line image-processing application with matched C++/OpenCV and custom CUDA implementations of grayscale conversion, Gaussian blur, and Sobel edge detection. It can validate GPU output against the CPU reference and benchmark both paths across multiple image resolutions.

## What is implemented

- **Three filters:** BGR-to-grayscale, configurable Gaussian blur, and Sobel edges.
- **Optimized CUDA kernels:** 16×16 thread blocks, coalesced global-memory access, shared-memory halo tiles, and constant-memory Gaussian coefficients.
- **Fair performance reporting:** warm-up runs, repeated measurements, CUDA-event timing for upload/kernels/download, plus wall-clock end-to-end timing.
- **Correctness checks:** CPU/GPU maximum absolute error validation, non-block-aligned test images, and CPU-only CI.
- **Reproducible tooling:** CMake presets, a CUDA development container, deterministic image generation, and a multi-resolution benchmark runner.

## Architecture

```text
input image (OpenCV)
        |
        +--> CPU reference: cvtColor / GaussianBlur / Sobel
        |
        +--> CUDA: upload BGR --> grayscale kernel --> optional tiled filter --> download
        |
        +--> verification + timing + output image / CSV
```

The CUDA Gaussian and Sobel kernels cooperatively load each block plus its halo into shared memory. Border pixels are replicated, matching the CPU reference. Gaussian weights are generated from `sigma`, normalized on the host, and copied to CUDA constant memory. The implementation supports radii from 1 through 7.

## Requirements

- Linux with a C++17 compiler
- CMake 3.22 or newer and Ninja
- OpenCV 4.5 or newer (`core`, `imgproc`, and `imgcodecs`)
- For GPU execution: an NVIDIA GPU and CUDA toolkit (compute capability 6.0+)
- Python 3 only for sample generation and the benchmark sweep

Ubuntu dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libopencv-dev python3
```

## Build and test

CUDA build (falls back to CPU-only if a CUDA compiler is not found):

```bash
cmake --preset cuda
cmake --build --preset cuda
ctest --preset cuda
```

Explicit CPU-only build, also used by CI:

```bash
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
```

To target a particular GPU architecture, configure directly, for example:

```bash
cmake -S . -B build/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cuda
```

## Run

Generate a deterministic test image and apply a filter:

```bash
python3 tools/generate_sample.py out/sample.ppm --width 1920 --height 1080

./build/cuda/cuda-image-processor \
  --input out/sample.ppm \
  --output out/edges.png \
  --operation sobel \
  --backend both \
  --warmup 3 \
  --iterations 20 \
  --csv benchmark-results/single-run.csv
```

Backends are `cpu`, `cuda`, `both`, and `auto`. `auto` selects CUDA when it was built and a device is available. With `both`, the CLI prints CPU time, CUDA upload/kernel/download time, end-to-end speedup, compute-only speedup, and maximum output difference.

Gaussian-specific options:

```bash
./build/cuda/cuda-image-processor -i input.jpg -o out/blur.png \
  --operation gaussian --radius 3 --sigma 1.5 --backend cuda
```

## Benchmark multiple resolutions

```bash
python3 tools/benchmark.py \
  --executable build/cuda/cuda-image-processor \
  --iterations 20 \
  --resolutions 640x480 1280x720 1920x1080 3840x2160
```

The generated CSV contains CPU latency, CUDA end-to-end latency, kernel latency, and both speedup definitions for every operation and resolution. Run benchmarks with the GPU idle, use a Release build, and report the GPU, CPU, CUDA version, OpenCV version, image sizes, warm-up count, iteration count, and whether speedup is kernel-only or end-to-end.

> Performance is hardware-dependent. This repository intentionally does not commit an invented “4.2×” result. Use the benchmark above on the target CUDA system and cite the resulting CSV. A defensible résumé number should name the hardware and speedup definition.

## Container

Build and run with the NVIDIA Container Toolkit:

```bash
docker build -t cuda-image-processing .
docker run --rm --gpus all -v "$PWD:/data" cuda-image-processing \
  --input /data/input.jpg --output /data/out/edges.png \
  --operation sobel --backend both
```

## Project layout

```text
apps/                  CLI and benchmark reporting
include/cip/           Public processing API
src/cpu_processor.cpp  OpenCV reference implementation
src/cuda_processor.cu  Custom CUDA kernels and event timing
tests/                 Correctness and CPU/GPU agreement tests
tools/                 Sample generator and benchmark sweep
.github/workflows/     CPU build/test automation
```

## Development roadmap

The end-to-end baseline is complete. Natural extensions are pinned host memory and persistent GPU buffers for streaming workloads, separable two-pass Gaussian blur for larger radii, CUDA Graphs for repeated pipelines, and batch/video processing.

## License

[MIT](LICENSE)
