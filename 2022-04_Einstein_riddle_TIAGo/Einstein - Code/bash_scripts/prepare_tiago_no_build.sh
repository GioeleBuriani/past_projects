#!/bin/bash

source /opt/ros/melodic/setup.bash
cd ~/ro47014_ws/

#Source the workspace:
source devel/setup.bash

# Only if using the singularity use the following 2 commands
roscd retail_store_simulation
source scripts/set_gazebo_env.sh

# Some useful command to run the simulation
echo ""
echo "roslaunch retail_store_simulation simulation.launch world:=krr tuck_arm:=true rviz:=true gzclient:=true"
echo ""
echo "roslaunch retail_store_planning rosplan_warehouse.launch"
echo ""
