cmake_minimum_required(VERSION 3.5)
project(parallel_parameter_search)

# Add support for C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(rclpy REQUIRED)
find_package(ament_cmake REQUIRED)

set(INCLUDE_DIRS ${rclcpp_INCLUDE_DIRS} ${rclpy_INCLUDE_DIRS}
  ${ament_cmake_INCLUDE_DIRS})
include_directories(${INCLUDE_DIRS})

ament_export_dependencies(rclpy)
ament_export_dependencies(ament_cmake)
ament_export_include_directories(${INCLUDE_DIRS})

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
    scripts/evaluate_walk.py
    scripts/multi_optim.py
    scripts/optimize_dynup.py
    scripts/optimize_walk.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
