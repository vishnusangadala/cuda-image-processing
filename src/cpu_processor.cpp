#include "cip/image_processor.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <stdexcept>

namespace cip {
namespace {

void validate(const cv::Mat& input, const Options& options) {
    if (input.empty() || input.type() != CV_8UC3) {
        throw std::invalid_argument("input must be a non-empty 8-bit BGR image");
    }
    if (options.radius < 1 || options.radius > 7) {
        throw std::invalid_argument("radius must be between 1 and 7");
    }
    if (options.sigma <= 0.0F) throw std::invalid_argument("sigma must be positive");
}

cv::Mat toGray(const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

}  // namespace

Result processCpu(const cv::Mat& bgr, const Options& options) {
    validate(bgr, options);
    const auto start = std::chrono::steady_clock::now();

    Result result;
    const cv::Mat gray = toGray(bgr);
    if (options.operation == Operation::Grayscale) {
        result.image = gray;
    } else if (options.operation == Operation::GaussianBlur) {
        const int width = options.radius * 2 + 1;
        cv::GaussianBlur(gray, result.image, cv::Size(width, width), options.sigma,
                         options.sigma, cv::BORDER_REPLICATE);
    } else {
        cv::Mat gradient_x;
        cv::Mat gradient_y;
        cv::Sobel(gray, gradient_x, CV_16S, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(gray, gradient_y, CV_16S, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::Mat abs_x;
        cv::Mat abs_y;
        cv::convertScaleAbs(gradient_x, abs_x);
        cv::convertScaleAbs(gradient_y, abs_y);
        cv::addWeighted(abs_x, 0.5, abs_y, 0.5, 0.0, result.image);
    }

    const auto end = std::chrono::steady_clock::now();
    result.timing.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.timing.kernel_ms = result.timing.total_ms;
    return result;
}

}  // namespace cip
