#include "cip/image_processor.hpp"

#include <stdexcept>

namespace cip {

bool cudaAvailable() { return false; }
std::string cudaDeviceName() { return "CUDA backend not built"; }

Result processCuda(const cv::Mat&, const Options&) {
    throw std::runtime_error("CUDA backend not built; configure with a CUDA compiler");
}

}  // namespace cip
