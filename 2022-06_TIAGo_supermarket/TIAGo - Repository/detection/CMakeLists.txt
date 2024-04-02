cmake_minimum_required(VERSION 3.0.2)
project(detection)

# add_compile_options(-std=c++11)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  actionlib
  move_base_msgs
  custom_msgs
  dyn_reconf_mdp
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES aruco_detection
#  CATKIN_DEPENDS  roscpp
  DEPENDS system_lib custom_msgs dyn_reconf_mdp
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Add the executable of aruco detction
add_executable(aruco_detection src/detection.cpp)
add_dependencies(aruco_detection ${catkin_EXPORTED_TARGETS})
target_link_libraries(aruco_detection ${catkin_LIBRARIES})