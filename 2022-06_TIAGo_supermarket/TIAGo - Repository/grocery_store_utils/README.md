# Grocery Store utilities package

## Table of contents

1. [Overview](#overview)
2. [Package structure](#package-structure)
    1. [Launch](#launch)
    2. [Scripts](#scripts)
    3. [Srv](#srv)
3. [Server description](#server-description)
4. [Usage](#usage)

## Overview

This repo contains utility servers for robotic manipulation within a grocery store. It was borrowed from the material of the course Knowledge Representation and Symbolic Reasoning

## Package structure

The structure of the present package is as described in the tree below:

```
├── launch
│   └── grocery_utils.launch
├── scripts
│   ├── collision_object_server.py
│   ├── grasp_pose_server.py
│   ├── grocery_list_server.py
│   └── my_grocery.txt
├── srv
│   ├── addCollObj.srv
│   ├── addCollObjByAruco.srv
│   ├── getGraspPose.srv
│   ├── getProduct.srv
│   ├── listInfo.srv
│   ├── removeCollObj.srv
│   ├── removeProduct.srv
│   └── setList.srv
├── CMakeLists.txt
└── package.xml
```

### Launch

This folder only contains the `grocery_utils.launch` file that is used to launch the `grocery_list_server`, the `collision_object_server` and the `grasp_pose_server` nodes.


### Scripts

This folder contains the files that describe the `grocery_list_server`, the `collision_object_server` and the `grasp_pose_server` nodes.


### Srv

This folder contains all the necessary .srv files.



## Server description

The following servers are available:

- *collision_obstacles_server*: This server provides two easy to use services to manage the creation and deletion of collision obstacles in the moveit planningscene. The server uses prior knowledge of the geometry of the object, and placement of the aruco marker. To add new object types, edit the function `register_collision_object`. The services are:
  - **add_collision_object**: Adds a collision object by detected aruco pose, additionally the object type must be provided in the request so the service knows the geometry of the object w.r.t the marker.
  - **remove_collision_object**: Removes a collision object, by requested object id.

- *grasp_pose_server*: This server provides a service called `get_grasp_pose` that returns a predefined grasp pose of a certain object type. The grasp pose is specified in the frame of the detected aruco marker. Note that the aruco marker frame has to exist for the service to work. It is therefore recommended to first register the collision object using the `add_collision_object` service, which will broadcast the aruco marker frame.

- *grocery_list_server*: This server is provides a number of services to manage a grocery list.
  - **set_list_server**: Not Implemented. Currently the inventory is hard coded, and the list is loaded from `my_grocery.txt` during server startup.
  - **list_info_server**: Get info about grocery list. Currently only returns the length of the list.
  - **get_product_server**: Get info about the product at the top of the grocery list.
  - **remove_product_server**: Remove product at the top of the grocery list.


> Further notes: The script `grocery_list_server.py` so far keeps the same order as in the `.txt`. The inventory variable in the script `grocery_list_server.py` contains a dictionary with product names and locations in front of the shelves from where it is possible to pick them. This will be moved into a better handled ontology of the store in the future.
 
## Usage

To run the services use the launch file:

`roslaunch grocery_store_utils grocery_utils.launch`
