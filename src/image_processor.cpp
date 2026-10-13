#include "cip/image_processor.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace cip {

Operation parseOperation(const std::string& value) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized == "grayscale" || normalized == "gray") return Operation::Grayscale;
    if (normalized == "gaussian" || normalized == "blur") return Operation::GaussianBlur;
    if (normalized == "sobel" || normalized == "edges" || normalized == "edge") {
        return Operation::SobelEdges;
    }
    throw std::invalid_argument("unknown operation: " + value);
}

const char* operationName(Operation operation) {
    switch (operation) {
        case Operation::Grayscale: return "grayscale";
        case Operation::GaussianBlur: return "gaussian";
        case Operation::SobelEdges: return "sobel";
    }
    return "unknown";
}

double maxAbsoluteDifference(const cv::Mat& expected, const cv::Mat& actual) {
    if (expected.size() != actual.size() || expected.type() != actual.type()) {
        throw std::invalid_argument("images must have identical dimensions and types");
    }
    cv::Mat difference;
    cv::absdiff(expected, actual, difference);
    double maximum = 0.0;
    cv::minMaxLoc(difference.reshape(1), nullptr, &maximum);
    return maximum;
}

}  // namespace cip
