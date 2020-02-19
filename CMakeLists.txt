cmake_minimum_required(VERSION 2.8.3)
project(parallel_parameter_search)


find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rosconsole
  rospy
  std_msgs
  bitbots_msgs
)

catkin_python_setup()

## Generate services in the 'srv' folder
add_service_files(
   FILES
   RequestParameters.srv
   SubmitFitness.srv
)
## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
