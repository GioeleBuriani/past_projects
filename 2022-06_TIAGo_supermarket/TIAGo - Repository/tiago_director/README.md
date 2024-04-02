# Tiago Director

## Table of contents

1. [Overview](#overview)
2. [Package structure](#package-structure) 
3. [Usage](#usage)
4. [Code description](#code-description)

## Overview

This is the main file that executes the required tasks by communicating with our own created user interface. 

## Package structure

The structure of the present package is as described in the tree below:

```
├── launch
│   └── director.launch
├── src
│   └── tiago_director.cpp
├── CMakeLists.txt
└── package.xml
```

## Usage 
To start the simulation: 
```bash
roslaunch cor_mdp_tiago_gazebo tiago_ahold.launch
```

Open a new terminal and start our program: 
```bash
roslaunch tiago_director director.launch
```

The GUI consist of 5 main buttons:
- **Request order**: the robot will pick a product detected at the checkout and place it to the correct shelf. After this it will directly go to the charging station. 
- **Go charge**: the robot will move back to the charging station.
- **Scan**: the robot will scan the shelves to detecting available spots in the shelf for products to place.
- **Stop**: the robot will stop performing the action.
- **Resume**: the robot will continue the ongoing action.

## Code description

## move_to_goal
This function replaces 2D Nav Goal button in RViz which publishes the desired goal location to the `move_base/goal` topic.

note that due to the simulation inaccuracies, the following errors could occur:
- It's not able to find the aruco tag.
- It's not able to execute the rosplan
- After done the request order, it can somehow go to checkout again eventhough it should stop at the charging station.
 
