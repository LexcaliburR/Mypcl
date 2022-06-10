cmake_minimum_required(VERSION 3.5)
project(debug_thrust_sort)

set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -DTV_CUDA -fPIC")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_86,code=compute_86 
                     -gencode=arch=compute_86,code=sm_86 -Xcudafe 
                     -DTV_CUDA 
                     -D__CUDA_NO_HALF_OPERATORS__ 
                     -D__CUDA_NO_HALF_CONVERSIONS__ 
                     -D__CUDA_NO_HALF2_OPERATORS__ 
                     --expt-relaxed-constexpr 
                     -Xcompiler='-fPIC' -Xcompiler='-O3'")

# set(CMAKE_RUNTIME_OUTPUT_DIRECTOR ${PROJECT_SOURCE_DIR}/bin)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

###########################################################
## Find CUDA & CUblas
###########################################################
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

find_package(Torch REQUIRED PATHS /opt/conda/lib/python3.8/site-packages/torch NO_DEFAULT_PATH)
if(TORCH_FOUND)
    message("-- Found torch")
    message("---  TORCH_INCLUDE_DIRS - ${TORCH_INCLUDE_DIRS}")
    message("---  TORCH_LIBRARIES - ${TORCH_LIBRARIES}")
    message("---  TORCH_CXX_FLAGS - ${TORCH_CXX_FLAGS}")
else()
    message(WARNING "-- Find torch - failed")
endif()

add_executable(sort ${PROJECT_SOURCE_DIR}/main.cpp)

target_include_directories(sort
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(sort
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
)


# message("---------\n ${SRC} \n---------")
