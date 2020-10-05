cmake_minimum_required(VERSION 2.8.3)
project(parallel_parameter_search)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosconsole
  rospy
)

catkin_python_setup()

catkin_package()
