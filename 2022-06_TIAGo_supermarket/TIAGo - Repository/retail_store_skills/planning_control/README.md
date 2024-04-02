# Planning and control package for TIAGo

## Table of contents

1. [Overview](#overview)
2. [Package structure](#package-structure)
    1. [Config](#config)
    2. [Include](#include)
    3. [Launch](#launch)
    4. [PDDL files](#pddl-files)
    5. [Src](#src)
3. [Usage](#usage)
    1. [Check the PDDL file with an online editor to look for sintax errors](#check-the-pddl-file-with-an-online-editor-to-look-for-sintax-errors)    
    2. [Initiate the simulation and set up rosplan](#initiate-the-simulation-and-set-up-rosplan)
    3. [Check the current problem file (optional)](#check-the-current-problem-file-optional)
    4. [Update goal based on parameters and execute plan](#update-goal-based-on-parameters-and-execute-plan)
    5. [Check the new problem file and the computed plan (optional)](#check-the-new-problem-file-and-the-computed-plan-optional)

## Overview

This package manages all the PDDL planning and action description required by TIAGo. Some of the files in this package were borrowed from the course Knowledge Representation and Symbolic Reasoning and modified according to our needs. This will be clearly specified in the folder description.


## Package structure

The structure of the present package is as described in the tree below:

```
├── config
│   ├── objects.yaml
│   └── waypoints.yaml
├── include
│   ├── RPActionInterface.h
│   ├── RPMoveBase.h
│   ├── RPPick.h
│   └── RPPlace.h
├── launch
│   ├── project.launch
│   └── update_goals.launch
├── pddl_files
│   ├── domain.pddl
│   └── problem.pddl
├── src
│   ├── RPActionInterface.cpp
│   ├── RPMoveBase.cpp
│   ├── RPPick.cpp
│   ├── RPPlace.cpp
│   └── update_goals.py
├── CMakeLists.txt
├── package.xml
└── rosplan_executor.bash
```

### Config

Consists of two .yaml files that conatin all the hardcoded information regarding the simulation environment, such as objects information and waypoint coordinates. These two files were borrowed from the KRR repositories.


### Include

Just a simple include folder for the .h files.


### Launch

This folder conatains the launch file `project.launch` that sets up ROSPlan and loads the PDDL file also borrowed from KRR.
It also conatains the `update_goals.launch` file that simply runs the `update_goals` node.


### PDDL files

This folder contains the `domain.pddl` and the `problem.pddl` files. These two files were inspired by the work on KRR and heavily modified to comply with our needs.


### Src

This folder contains the four files `RPActionInterface.cpp`, `RPMoveBase.cpp`, `RPPick.cpp` and `RPPlace.cpp` from the KRR course that manage the actions of the robot. The folder also conatins the `update_goals.py` file that manages the online update of the goals and then generates and executes the plan to reach the new goal.



## Usage

### Check the PDDL file with an online editor to look for sintax errors:

1. Open [editor.planning.domains](http://editor.planning.domains/#)
2. Click `Session` => `load`
3. Copy & paste following hashes

- Basic pick, move and place, with 2 grippers: ZgTlgDhGqqqldxH
- Feel free to modify the problem file and domain file to achieve different tasks


### Initiate the simulation and set up rosplan:

First of all open a terminal and start the simulation with:
``` bash
roslaunch cor_mdp_tiago_gazebo tiago_ahold.launch tuck_arm:=true
```

Open another terminal window and run:
``` bash
roslaunch planning_control project.launch
```
to load the PDDL on ROSPlan.

### Check the current problem file (optional):

If you want to visualize the pddl problem, in a new terminal you can run:
``` bash
rosservice call /rosplan_problem_interface/problem_generation_server
```
In order to generate the problem.
Then, in the same terminal you can run:
``` bash
rostopic echo /rosplan_problem_interface/problem_instance -n 1 -p
```
This should print on screen the current problem.


### Update goal based on parameters and execute plan:

At this point you can manually set the parameters to decide which object to place.
Open another terminal and type:
``` bash
rosparam set place_product 0
```
if you want to place `AH_hagelslag_aruco_0`, or:
``` bash
rosparam set place_product 17
```
if you want to place `AH_hagelslag_aruco_17`.


In the same terminal you can now update and execute the goal by running:
``` bash
roslaunch planning_control update_goals.launch
```
According to the parameter set before, TIAGo will either pick and place `AH_hagelslag_aruco_0` or `AH_hagelslag_aruco_17`


### Check the new problem file and the computed plan (optional):

Once again, to visualize the new problem follow the same steps as before:
``` bash
rosservice call /rosplan_problem_interface/problem_generation_server
```
Followed by:
``` bash
rostopic echo /rosplan_problem_interface/problem_instance -n 1 -p
```
You can also see the current plan by typing:
``` bash
rostopic echo /rosplan_planner_interface/planner_output -p
```

