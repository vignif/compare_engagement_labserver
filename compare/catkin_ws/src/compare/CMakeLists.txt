cmake_minimum_required(VERSION 3.0.2)
project(compare)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
  grace_common_msgs
)

## Add message files
add_message_files(
  FILES
  EngagementValue.msg
  EngValue.msg
)

## Generate added messages
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Catkin specific configuration
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)
