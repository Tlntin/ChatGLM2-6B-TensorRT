cmake_minimum_required(VERSION 3.0)

project(inference_test)

# Set C++14 standard
set(CMAKE_CXX_STANDARD 17)

# Find and include LibTorch
set(CMAKE_PREFIX_PATH /usr/local/libtorch/share/cmake)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
add_definitions(-DDEBUG)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_KERNEL_FILE ${PROJECT_SOURCE_DIR}/kernel/kernel.cpp)
get_filename_component(CUDA_KERNEL_NAME ${CUDA_KERNEL_FILE} NAME_WE)
CUDA_ADD_LIBRARY(${CUDA_KERNEL_NAME} SHARED ${CUDA_KERNEL_FILE})

# TensorRT install here
find_library(TENSOR_RT_LIB nvinfer /usr/local/cuda/lib64)
# Add the executable
add_executable(inference_test inference_test.cpp)

# Link against LibTorch
target_link_libraries(inference_test ${TORCH_LIBRARIES} ${CUDA_KERNEL_NAME} ${TENSOR_RT_LIB})