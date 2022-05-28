cmake_minimum_required(VERSION 3.5)
project(filter)

find_package (glog 0.6.0 REQUIRED)

file(GLOB_RECURSE SRC src ${PROJECT_SOURCE_DIR}/*.cpp)

add_library(${PROJECT_NAME} ${BUILD_LIB_TYPE} ${SRC})

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${OVERRALL}
)



