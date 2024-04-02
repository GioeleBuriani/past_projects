cmake_minimum_required(VERSION 3.0.2)
project(custom_msgs)

find_package(
  catkin REQUIRED COMPONENTS
  actionlib_msgs
  std_msgs
  message_generation
  geometry_msgs
)

add_message_files(
  FILES
  Param_list.msg
)

add_service_files(
  FILES
  is_human.srv
)

add_action_files(
  FILES
  Pick.action
  Place.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib_msgs std_msgs message_runtime geometry_msgs
)

include_directories(
# include
 ${catkin_INCLUDE_DIRS}
)
