cmake_minimum_required(VERSION 3.5)
project(examples)

find_package(glog REQUIRED)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("-- Found CUDA")
    message("---  CUDA_INCLUDE_DIRS - ${CUDA_INCLUDE_DIRS}")
    message("---  CUDA_LIBRARIES - ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY}")
    message("---  CUDA_NVCC_FLAGS - ${CUDA_NVCC_FLAGS}")
    set(DEFINITIONS ${DEFINITIONS} USE_CUDA)
else()
    message(WARNING "-- Find CUDA - failed")
endif()

add_subdirectory(test_voxel)