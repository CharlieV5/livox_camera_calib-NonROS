cmake_minimum_required(VERSION 3.0.2)
project(livox_camera_calib)

## Compile as C++14
add_compile_options(-std=c++14)
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL REQUIRED)
find_package(OpenCV)
find_package(Threads)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
)

add_library(calibration STATIC src/Calibration.cpp)

add_executable(lidar_camera_calib src/lidar_camera_calib.cpp)
add_executable(lidar_camera_multi_calib src/lidar_camera_multi_calib.cpp )

target_link_libraries(lidar_camera_calib 
  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES} yaml-cpp calibration
)

target_link_libraries(lidar_camera_multi_calib 
  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES} yaml-cpp calibration
)
