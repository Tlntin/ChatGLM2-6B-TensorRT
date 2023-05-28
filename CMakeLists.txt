cmake_minimum_required(VERSION 3.0)
project(inference_project)

set(CMAKE_CXX_STANDARD 17)

# Find and include LibTorch
set(CMAKE_PREFIX_PATH /usr/local/libtorch/share/cmake)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
add_definitions(-DDEBUG)

# Find and include CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
execute_process(
    COMMAND python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())"
    OUTPUT_VARIABLE PYTHON_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${PYTHON_INCLUDE})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fPIC -shared -w")

# Build shared library for pybind11
add_library(cpp_kernel SHARED 
  kernel/kernel.cpp 
  kernel/bind.cpp 
)

# Build shared library for CUDA kernel
set(CUDA_KERNEL_FILE ${PROJECT_SOURCE_DIR}/kernel/kernel.cpp)
get_filename_component(CUDA_KERNEL_NAME ${CUDA_KERNEL_FILE} NAME_WE)
CUDA_ADD_LIBRARY(${CUDA_KERNEL_NAME} SHARED ${CUDA_KERNEL_FILE})

# TensorRT install here
find_library(TENSOR_RT_LIB nvinfer /usr/local/cuda/lib64)

# Link against TensorRT, CUDA and LibTorch libraries
target_link_libraries(cpp_kernel 
  ${TORCH_LIBRARIES} 
  ${CUDA_LIBRARIES}
  ${TENSOR_RT_LIB}
)

# Add the executable
add_executable(inference_test inference_test.cpp)

# Link against LibTorch, CUDA kernel and TensorRT
target_link_libraries(inference_test ${TORCH_LIBRARIES} ${CUDA_KERNEL_NAME} ${TENSOR_RT_LIB})