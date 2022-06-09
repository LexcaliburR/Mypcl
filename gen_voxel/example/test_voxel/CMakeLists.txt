cmake_minimum_required(VERSION 3.5)
project(test_voxel)

set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -DTV_CUDA -fPIC")

find_package(glog REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/../../include
    ${CUDA_INCLUDE_DIRS}
)

message("${PROJECT_SOURCE_DIR}")
message("----${CUDA_INCLUDE_DIRS} ---")

add_executable(voxel
    ${PROJECT_SOURCE_DIR}/main.cpp
    ../common/binfile_tools.cpp
)

target_link_libraries(voxel
    glog::glog
    ${PROJECT_SOURCE_DIR}/../../lib/libvoxel.a
    ${CUDA_LIBRARIES}

)