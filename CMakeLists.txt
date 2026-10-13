cmake_minimum_required(VERSION 3.22)

project(cuda_image_processing VERSION 1.0.0 LANGUAGES CXX)

option(CIP_ENABLE_CUDA "Build the CUDA backend when a CUDA compiler is available" ON)
option(CIP_BUILD_TESTS "Build correctness tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs)

add_library(cip
    src/cpu_processor.cpp
    src/image_processor.cpp
)
target_include_directories(cip PUBLIC include)
target_link_libraries(cip PUBLIC ${OpenCV_LIBS})
target_compile_options(cip PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

set(CIP_HAS_CUDA OFF)
if(CIP_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
        endif()
        target_sources(cip PRIVATE src/cuda_processor.cu)
        target_link_libraries(cip PRIVATE CUDA::cudart)
        target_compile_definitions(cip PUBLIC CIP_HAS_CUDA=1)
        set_target_properties(cip PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            POSITION_INDEPENDENT_CODE ON
        )
        set(CIP_HAS_CUDA ON)
    else()
        message(WARNING "CUDA compiler not found; building the CPU backend only")
    endif()
endif()

if(NOT CIP_HAS_CUDA)
    target_sources(cip PRIVATE src/cuda_stub.cpp)
endif()

add_executable(cuda-image-processor apps/main.cpp)
target_link_libraries(cuda-image-processor PRIVATE cip)

include(GNUInstallDirs)
install(TARGETS cuda-image-processor RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CIP_BUILD_TESTS)
    enable_testing()
    add_executable(cip-tests tests/test_processors.cpp)
    target_link_libraries(cip-tests PRIVATE cip)
    add_test(NAME processor-correctness COMMAND cip-tests)
endif()
