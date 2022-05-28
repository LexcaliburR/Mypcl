cmake_minimum_required(VERSION 3.5)
project(filter)

find_package (glog 0.6.0 REQUIRED)

include_directories(
    ${OVERRALL}
)

link_libraries(
    filter
    glog::glog
)

add_executable(testvoxeldownload testVoxelDSample.cpp)






