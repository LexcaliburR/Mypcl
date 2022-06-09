cmake_minimum_required(VERSION 3.5)
project(voxel)

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
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

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

list(APPEND DEFINITIONS TV_CUDA)

# add_definitions(${DEFINITIONS})

file(GLOB_RECURSE SRC src "src/*.cc" "src/*.cpp" "src/*.cu" "src/*.c")

CUDA_ADD_LIBRARY(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} 
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
)


message("---------\n ${SRC} \n---------")
