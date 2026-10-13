#include "cip/image_processor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cip {
namespace {

constexpr int kBlock = 16;
constexpr int kMaxRadius = 7;
__constant__ float c_gaussian[2 * kMaxRadius + 1];

void checkCuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) { checkCuda(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

class Event {
public:
    Event() { checkCuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() { if (event_) cudaEventDestroy(event_); }
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_{};
};

__device__ __forceinline__ int clampIndex(int value, int maximum) {
    return max(0, min(value, maximum - 1));
}

__global__ void grayscaleKernel(const uchar3* input, unsigned char* output, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const uchar3 pixel = input[y * width + x];
    // OpenCV input is BGR. Integer BT.601 coefficients avoid per-pixel floating point.
    output[y * width + x] = static_cast<unsigned char>(
        (29 * static_cast<int>(pixel.x) + 150 * static_cast<int>(pixel.y) +
         77 * static_cast<int>(pixel.z) + 128) >> 8);
}

__global__ void gaussianKernel(const unsigned char* input, unsigned char* output,
                               int width, int height, int radius) {
    __shared__ unsigned char tile[kBlock + 2 * kMaxRadius][kBlock + 2 * kMaxRadius];
    const int tile_width = blockDim.x + 2 * radius;
    const int tile_height = blockDim.y + 2 * radius;
    const int origin_x = blockIdx.x * blockDim.x - radius;
    const int origin_y = blockIdx.y * blockDim.y - radius;

    for (int ty = threadIdx.y; ty < tile_height; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tile_width; tx += blockDim.x) {
            const int x = clampIndex(origin_x + tx, width);
            const int y = clampIndex(origin_y + ty, height);
            tile[ty][tx] = input[y * width + x];
        }
    }
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0F;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const float weight = c_gaussian[dy + radius] * c_gaussian[dx + radius];
            sum += weight * tile[threadIdx.y + radius + dy][threadIdx.x + radius + dx];
        }
    }
    output[y * width + x] = static_cast<unsigned char>(fminf(255.0F, fmaxf(0.0F, roundf(sum))));
}

__global__ void sobelKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char tile[kBlock + 2][kBlock + 2];
    const int origin_x = blockIdx.x * blockDim.x - 1;
    const int origin_y = blockIdx.y * blockDim.y - 1;
    for (int ty = threadIdx.y; ty < blockDim.y + 2; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < blockDim.x + 2; tx += blockDim.x) {
            const int x = clampIndex(origin_x + tx, width);
            const int y = clampIndex(origin_y + ty, height);
            tile[ty][tx] = input[y * width + x];
        }
    }
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;
    const int gx = -tile[ty - 1][tx - 1] + tile[ty - 1][tx + 1]
                   - 2 * tile[ty][tx - 1] + 2 * tile[ty][tx + 1]
                   - tile[ty + 1][tx - 1] + tile[ty + 1][tx + 1];
    const int gy = -tile[ty - 1][tx - 1] - 2 * tile[ty - 1][tx] - tile[ty - 1][tx + 1]
                   + tile[ty + 1][tx - 1] + 2 * tile[ty + 1][tx] + tile[ty + 1][tx + 1];
    const int abs_x = min(abs(gx), 255);
    const int abs_y = min(abs(gy), 255);
    output[y * width + x] = static_cast<unsigned char>((abs_x + abs_y + 1) / 2);
}

std::vector<float> gaussianWeights(int radius, float sigma) {
    std::vector<float> weights(2 * radius + 1);
    float sum = 0.0F;
    for (int i = -radius; i <= radius; ++i) {
        const float value = std::exp(-(i * i) / (2.0F * sigma * sigma));
        weights[i + radius] = value;
        sum += value;
    }
    for (float& weight : weights) weight /= sum;
    return weights;
}

float elapsed(cudaEvent_t start, cudaEvent_t end) {
    float milliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, end), "cudaEventElapsedTime");
    return milliseconds;
}

}  // namespace

bool cudaAvailable() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cudaDeviceName() {
    if (!cudaAvailable()) return "No CUDA device available";
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    return properties.name;
}

Result processCuda(const cv::Mat& bgr, const Options& options) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument("input must be a non-empty 8-bit BGR image");
    }
    if (options.radius < 1 || options.radius > kMaxRadius || options.sigma <= 0.0F) {
        throw std::invalid_argument("radius must be 1..7 and sigma must be positive");
    }
    if (!cudaAvailable()) throw std::runtime_error("no CUDA-capable device is available");

    const cv::Mat contiguous = bgr.isContinuous() ? bgr : bgr.clone();
    const int width = contiguous.cols;
    const int height = contiguous.rows;
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    const auto wall_start = std::chrono::steady_clock::now();

    DeviceBuffer<uchar3> device_bgr(pixels);
    DeviceBuffer<unsigned char> device_gray(pixels);
    DeviceBuffer<unsigned char> device_result(pixels);
    Event upload_start, upload_end, kernel_end, download_end;
    checkCuda(cudaEventRecord(upload_start.get()), "record upload start");
    checkCuda(cudaMemcpyAsync(device_bgr.get(), contiguous.ptr(), pixels * sizeof(uchar3),
                              cudaMemcpyHostToDevice), "upload image");
    checkCuda(cudaEventRecord(upload_end.get()), "record upload end");

    const dim3 block(kBlock, kBlock);
    const dim3 grid((width + kBlock - 1) / kBlock, (height + kBlock - 1) / kBlock);
    grayscaleKernel<<<grid, block>>>(device_bgr.get(), device_gray.get(), width, height);
    checkCuda(cudaGetLastError(), "launch grayscale kernel");

    unsigned char* final_device = device_gray.get();
    if (options.operation == Operation::GaussianBlur) {
        const auto weights = gaussianWeights(options.radius, options.sigma);
        checkCuda(cudaMemcpyToSymbol(c_gaussian, weights.data(), weights.size() * sizeof(float)),
                  "upload Gaussian coefficients");
        gaussianKernel<<<grid, block>>>(device_gray.get(), device_result.get(), width, height,
                                        options.radius);
        checkCuda(cudaGetLastError(), "launch Gaussian kernel");
        final_device = device_result.get();
    } else if (options.operation == Operation::SobelEdges) {
        sobelKernel<<<grid, block>>>(device_gray.get(), device_result.get(), width, height);
        checkCuda(cudaGetLastError(), "launch Sobel kernel");
        final_device = device_result.get();
    }
    checkCuda(cudaEventRecord(kernel_end.get()), "record kernel end");

    Result result;
    result.image.create(height, width, CV_8UC1);
    checkCuda(cudaMemcpyAsync(result.image.ptr(), final_device, pixels, cudaMemcpyDeviceToHost),
              "download image");
    checkCuda(cudaEventRecord(download_end.get()), "record download end");
    checkCuda(cudaEventSynchronize(download_end.get()), "wait for CUDA pipeline");

    result.timing.upload_ms = elapsed(upload_start.get(), upload_end.get());
    result.timing.kernel_ms = elapsed(upload_end.get(), kernel_end.get());
    result.timing.download_ms = elapsed(kernel_end.get(), download_end.get());
    const auto wall_end = std::chrono::steady_clock::now();
    result.timing.total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    return result;
}

}  // namespace cip
