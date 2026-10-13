#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace cip {

enum class Operation { Grayscale, GaussianBlur, SobelEdges };

struct Options {
    Operation operation = Operation::Grayscale;
    int radius = 3;
    float sigma = 1.5F;
};

struct Timing {
    double total_ms = 0.0;
    double upload_ms = 0.0;
    double kernel_ms = 0.0;
    double download_ms = 0.0;
};

struct Result {
    cv::Mat image;
    Timing timing;
};

Operation parseOperation(const std::string& value);
const char* operationName(Operation operation);

Result processCpu(const cv::Mat& bgr, const Options& options);
Result processCuda(const cv::Mat& bgr, const Options& options);
bool cudaAvailable();
std::string cudaDeviceName();

double maxAbsoluteDifference(const cv::Mat& expected, const cv::Mat& actual);

}  // namespace cip
