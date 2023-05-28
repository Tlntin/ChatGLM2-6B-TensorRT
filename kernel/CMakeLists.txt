cmake_minimum_required(VERSION 3.0)
project(kernel.cpython-310-x86_64-linux-gnu)

set(CMAKE_CXX_STANDARD 17)

# Find and include LibTorch
set(CMAKE_PREFIX_PATH /usr/local/libtorch/share/cmake)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
add_definitions(-DDEBUG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fPIC -shared -w")

# Find and include CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
include_directories(/home/tlntin/anaconda3/envs/gea/include/python3.10)

# Build shared library
add_library(kernel.cpython-310-x86_64-linux-gnu SHARED 
  kernel/kernel.cpp 
  kernel/bind.cpp 
)


# TensorRT install here
find_library(TENSOR_RT_LIB nvinfer /usr/local/cuda/lib64)

# Link against TensorRT and CUDA libraries
target_link_libraries(kernel.cpython-310-x86_64-linux-gnu 
  ${TORCH_LIBRARIES} 
  ${CUDA_LIBRARIES}
  ${TENSOR_RT_LIB}
)