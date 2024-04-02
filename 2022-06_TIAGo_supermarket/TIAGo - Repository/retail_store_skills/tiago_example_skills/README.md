# Example skills for TIAGo

## Table of contents

1. [Overview](#overview)
2. [Package structure](#package-structure)
    1. [Launch](#launch)
    2. [Scripts](#scripts)

## Overview

This package contains the description and service creation for all the necessary actions for TIAGo. This package was borrowed from the Knowledge Representation and Symbolic Reasoning course.


## Package structure

The structure of the present package is as described in the tree below:

```
├── launch
│   └── server.launch
├── scripts
│   ├── example_task.py
│   ├── gripper_control.py
│   ├── look_to_point.py
│   ├── move_base.py
│   ├── pick_client.py
│   ├── pick_server.py
│   ├── place_client.py
│   ├── place_server.py
│   ├── plan_arm_ik.py
│   └── task_kb.py
```

### Launch

Only contains the `server.launch` file that is used to launch the pick and place servers.


### Scripts

This folder contains the description of all the actions of TIAGo and their service generation.