#include "cip/image_processor.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string input;
    std::string output;
    std::string backend = "auto";
    std::string csv;
    cip::Options options;
    int warmup = 2;
    int iterations = 10;
    double tolerance = 8.0;
};

void usage(std::ostream& stream) {
    stream << "Usage: cuda-image-processor --input IMAGE --output IMAGE [options]\n"
              "Options:\n"
              "  --operation grayscale|gaussian|sobel  Filter to apply (default: grayscale)\n"
              "  --backend auto|cpu|cuda|both          Execution backend (default: auto)\n"
              "  --radius N                            Gaussian radius, 1..7 (default: 3)\n"
              "  --sigma N                             Gaussian sigma (default: 1.5)\n"
              "  --warmup N                            Unmeasured runs (default: 2)\n"
              "  --iterations N                        Measured runs (default: 10)\n"
              "  --verify-tolerance N                  Max CPU/GPU difference (default: 8)\n"
              "  --csv PATH                            Append benchmark summary as CSV\n"
              "  --help                                Show this message\n";
}

std::string takeValue(int& index, int argc, char** argv) {
    if (++index >= argc) throw std::invalid_argument("missing value for " + std::string(argv[index - 1]));
    return argv[index];
}

Arguments parseArguments(int argc, char** argv) {
    Arguments arguments;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") { usage(std::cout); std::exit(0); }
        if (flag == "--input" || flag == "-i") arguments.input = takeValue(i, argc, argv);
        else if (flag == "--output" || flag == "-o") arguments.output = takeValue(i, argc, argv);
        else if (flag == "--operation") arguments.options.operation = cip::parseOperation(takeValue(i, argc, argv));
        else if (flag == "--backend") arguments.backend = takeValue(i, argc, argv);
        else if (flag == "--radius") arguments.options.radius = std::stoi(takeValue(i, argc, argv));
        else if (flag == "--sigma") arguments.options.sigma = std::stof(takeValue(i, argc, argv));
        else if (flag == "--warmup") arguments.warmup = std::stoi(takeValue(i, argc, argv));
        else if (flag == "--iterations") arguments.iterations = std::stoi(takeValue(i, argc, argv));
        else if (flag == "--verify-tolerance") arguments.tolerance = std::stod(takeValue(i, argc, argv));
        else if (flag == "--csv") arguments.csv = takeValue(i, argc, argv);
        else throw std::invalid_argument("unknown argument: " + flag);
    }
    if (arguments.input.empty() || arguments.output.empty()) {
        throw std::invalid_argument("--input and --output are required");
    }
    if (arguments.iterations < 1 || arguments.warmup < 0) {
        throw std::invalid_argument("iterations must be positive and warmup cannot be negative");
    }
    if (arguments.backend != "auto" && arguments.backend != "cpu" &&
        arguments.backend != "cuda" && arguments.backend != "both") {
        throw std::invalid_argument("backend must be auto, cpu, cuda, or both");
    }
    return arguments;
}

template <typename Function>
cip::Result benchmark(Function&& function, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) function();
    cip::Result latest;
    cip::Timing total;
    for (int i = 0; i < iterations; ++i) {
        latest = function();
        total.total_ms += latest.timing.total_ms;
        total.upload_ms += latest.timing.upload_ms;
        total.kernel_ms += latest.timing.kernel_ms;
        total.download_ms += latest.timing.download_ms;
    }
    total.total_ms /= iterations;
    total.upload_ms /= iterations;
    total.kernel_ms /= iterations;
    total.download_ms /= iterations;
    latest.timing = total;
    return latest;
}

void printTiming(const std::string& backend, const cip::Timing& timing) {
    std::cout << std::fixed << std::setprecision(3) << "  " << std::setw(5) << backend
              << " total=" << timing.total_ms << " ms";
    if (backend == "CUDA") {
        std::cout << " (upload=" << timing.upload_ms << ", kernels=" << timing.kernel_ms
                  << ", download=" << timing.download_ms << ')';
    }
    std::cout << '\n';
}

void appendCsv(const Arguments& args, const cv::Mat& input, const cip::Result* cpu,
               const cip::Result* gpu) {
    if (args.csv.empty()) return;
    const std::filesystem::path path(args.csv);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    const bool needs_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream file(path, std::ios::app);
    if (!file) throw std::runtime_error("cannot open CSV: " + path.string());
    if (needs_header) {
        file << "operation,width,height,iterations,cpu_ms,cuda_total_ms,cuda_kernel_ms,end_to_end_speedup,kernel_speedup\n";
    }
    const double cpu_ms = cpu ? cpu->timing.total_ms : 0.0;
    const double gpu_ms = gpu ? gpu->timing.total_ms : 0.0;
    const double speedup = cpu && gpu && gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    const double kernel_speedup = cpu && gpu && gpu->timing.kernel_ms > 0.0
                                      ? cpu_ms / gpu->timing.kernel_ms : 0.0;
    file << cip::operationName(args.options.operation) << ',' << input.cols << ',' << input.rows << ','
         << args.iterations << ',' << cpu_ms << ',' << gpu_ms << ','
         << (gpu ? gpu->timing.kernel_ms : 0.0) << ',' << speedup << ',' << kernel_speedup << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parseArguments(argc, argv);
        const cv::Mat input = cv::imread(args.input, cv::IMREAD_COLOR);
        if (input.empty()) throw std::runtime_error("could not read input image: " + args.input);

        std::string backend = args.backend;
        if (backend == "auto") backend = cip::cudaAvailable() ? "cuda" : "cpu";
        if ((backend == "cuda" || backend == "both") && !cip::cudaAvailable()) {
            throw std::runtime_error("CUDA was requested but no CUDA backend/device is available");
        }

        cip::Result cpu;
        cip::Result gpu;
        const bool run_cpu = backend == "cpu" || backend == "both";
        const bool run_gpu = backend == "cuda" || backend == "both";
        std::cout << "Processing " << input.cols << 'x' << input.rows << " image with "
                  << cip::operationName(args.options.operation) << "\n";
        if (run_cpu) {
            cpu = benchmark([&] { return cip::processCpu(input, args.options); }, args.warmup, args.iterations);
            printTiming("CPU", cpu.timing);
        }
        if (run_gpu) {
            std::cout << "  device: " << cip::cudaDeviceName() << '\n';
            gpu = benchmark([&] { return cip::processCuda(input, args.options); }, args.warmup, args.iterations);
            printTiming("CUDA", gpu.timing);
        }
        if (run_cpu && run_gpu) {
            const double difference = cip::maxAbsoluteDifference(cpu.image, gpu.image);
            const double speedup = cpu.timing.total_ms / gpu.timing.total_ms;
            const double kernel_speedup = cpu.timing.total_ms / gpu.timing.kernel_ms;
            std::cout << "  end-to-end speedup=" << speedup << "x, compute speedup="
                      << kernel_speedup << "x, max_difference=" << difference << '\n';
            if (difference > args.tolerance) {
                throw std::runtime_error("CPU/GPU verification exceeded tolerance");
            }
        }

        const cv::Mat& output = run_gpu ? gpu.image : cpu.image;
        const std::filesystem::path output_path(args.output);
        if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
        if (!cv::imwrite(output_path.string(), output)) {
            throw std::runtime_error("could not write output image: " + output_path.string());
        }
        appendCsv(args, input, run_cpu ? &cpu : nullptr, run_gpu ? &gpu : nullptr);
        std::cout << "Wrote " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n\n";
        usage(std::cerr);
        return 1;
    }
}
