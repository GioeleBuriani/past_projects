# Custom message definition package for TIAGo

## Table of contents

1. [Overview](#overview)
2. [Package structure](#package-structure) 
3. [Notes for developers](#notes-for-developers)

## Overview

The present package contains all the necessary files for creating the message used by the custom centralized parameter server to publish to all nodes the updated version of the parameters, reconfigured upon request of a single client.

## Package structure

The structure of the present package is as described in the tree below:

```
├── action
│   ├── Pick.action
│   └── Place.action
├── srv
│   └── is_human.srv
├── msg
│   └── Param_list.msg
├── CMakeLists.txt
└── package.xml
```

### Notes for developers

In order to use this package/messages in your own packages, please modify the file **package.xml** by adding the two following lines:

```xml
  <build_depend>custom_msgs</build_depend>
  <exec_depend>custom_msgs</exec_depend>
```

Also, in the **CMakeLists.txt**, the package should be added to the find_package options:

```txt
find_package(catkin REQUIRED COMPONENTS
  ...
  custom_msgs
  ...
)
```

Now, the message type for the subscriber can be introduced in a simple *include* statement:

```c++
    #include "custom_msgs/Param_list.h"
```

If it is intended to make use of the service, include it as:

```c++
    #include "custom_msgs/is_human.h"
```
