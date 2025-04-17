cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake")

project(livox_camera_calib)

## Compile as C++14
# add_compile_options(-std=c++14)
set(CMAKE_CXX_STANDARD 17)
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


## System dependencies are found with CMake's conventions
find_package(BOOST REQUIRED  COMPONENTS  system filesystem)

include_directories(${Boost_INCLUDE_DIRS})

find_package(PCL CONFIG REQUIRED)
find_package(OpenCV PATHS "D:/opencv/build")
find_package(Threads)
find_package(Ceres PATHS  "D:/third/Install/Ceres1.14/CMake" REQUIRED) 
find_package(Eigen3 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
add_compile_definitions(GLOG_NO_ABBREVIATED_SEVERITIES)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${GLOG_INCLUDE}
  ${CERES_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
)

add_library(calibration STATIC src/Calibration.cpp include/Calibration.h)

add_executable(lidar_camera_calib 
  src/lidar_camera_calib.cpp
  include/common.h
  include/calib_cost.h
 )

add_executable(lidar_camera_multi_calib 
  src/lidar_camera_multi_calib.cpp 
  include/common.h
  include/calib_cost.h
  )

target_link_libraries(lidar_camera_calib 
  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CERES_LIBRARIES} yaml-cpp::yaml-cpp calibration
)

target_link_libraries(lidar_camera_multi_calib 
  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES}  ${CERES_LIBRARIES} yaml-cpp::yaml-cpp calibration
)
