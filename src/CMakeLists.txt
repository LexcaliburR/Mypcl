cmake_minimum_required(VERSION 3.5)
project(mypclcpp)

set(BUILD_LIB_TYPE STATIC)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "-std=c++14 -g -O2")
set(CMAKE_CXX_FLAGS "-std=c++14 -g")

set(OVERRALL ${PROJECT_SOURCE_DIR})
# find_package (glog 0.6.0 REQUIRED)

# add_subdirectory(base)
# add_subdirectory(common)
# add_subdirectory(cuda)
add_subdirectory(filter)
# add_subdirectory(utils)
add_subdirectory(app)


# file(GLOB_RECURSE SRC src ${PROJECT_SOURCE_DIR}/*.cpp)



