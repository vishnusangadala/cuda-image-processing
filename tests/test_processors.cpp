#include "cip/image_processor.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

cv::Mat testImage(int width = 96, int height = 64) {
    cv::Mat image(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<unsigned char>((x * 3 + y) % 256),
                static_cast<unsigned char>((x + y * 5) % 256),
                static_cast<unsigned char>((x * 7 + y * 2) % 256));
        }
    }
    return image;
}

void testParsing() {
    require(cip::parseOperation("gray") == cip::Operation::Grayscale, "gray parsing failed");
    require(cip::parseOperation("BLUR") == cip::Operation::GaussianBlur, "blur parsing failed");
    require(cip::parseOperation("edges") == cip::Operation::SobelEdges, "edge parsing failed");
    bool threw = false;
    try { (void)cip::parseOperation("not-a-filter"); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "invalid operation was accepted");
}

void testCpuOperations() {
    const cv::Mat input = testImage();
    for (const auto operation : {cip::Operation::Grayscale, cip::Operation::GaussianBlur,
                                 cip::Operation::SobelEdges}) {
        cip::Options options;
        options.operation = operation;
        const auto result = cip::processCpu(input, options);
        require(!result.image.empty(), "CPU result is empty");
        require(result.image.size() == input.size(), "CPU result dimensions changed");
        require(result.image.type() == CV_8UC1, "CPU result is not 8-bit grayscale");
        require(result.timing.total_ms >= 0.0, "CPU timing is invalid");
    }
}

void testUniformImage() {
    const cv::Mat uniform(40, 50, CV_8UC3, cv::Scalar(90, 90, 90));
    cip::Options options;
    options.operation = cip::Operation::SobelEdges;
    const auto result = cip::processCpu(uniform, options);
    require(cv::countNonZero(result.image) == 0, "uniform image should have no edges");
}

void testCudaAgreementWhenAvailable() {
    if (!cip::cudaAvailable()) {
        std::cout << "CUDA device unavailable; GPU agreement checks skipped\n";
        return;
    }
    const cv::Mat input = testImage(257, 193);  // deliberately not block-aligned
    for (const auto operation : {cip::Operation::Grayscale, cip::Operation::GaussianBlur,
                                 cip::Operation::SobelEdges}) {
        cip::Options options;
        options.operation = operation;
        const auto cpu = cip::processCpu(input, options);
        const auto gpu = cip::processCuda(input, options);
        const double tolerance = operation == cip::Operation::SobelEdges ? 8.0 : 4.0;
        require(cip::maxAbsoluteDifference(cpu.image, gpu.image) <= tolerance,
                std::string("CPU/CUDA disagreement for ") + cip::operationName(operation));
    }
}

}  // namespace

int main() {
    try {
        testParsing();
        testCpuOperations();
        testUniformImage();
        testCudaAgreementWhenAvailable();
        std::cout << "All tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
