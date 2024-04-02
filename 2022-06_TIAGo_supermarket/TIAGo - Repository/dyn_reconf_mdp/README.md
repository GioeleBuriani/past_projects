# Dynamic Reconfigure package for TIAGo

## Table of contents

1. [Overview](#overview)
2. [Package structure](#package-structure) 
3. [Usage](#usage)
4. [Notes for developers](#notes-for-developers)

## Overview

The present package contains all the necessary files for creating the centralized custom dynamically reconfigurable parameter node of our solution. The idea is that a reconfigurable parameter **server** is created, whose parameters are changed by a number of possible client nodes upon demand. In order to announce the changes to any other node that may be interested in this information, the **server** publishes the contents of its parameters through a topic. In this way the bidirectional communication can be established and all nodes interested in the parameter information will be notified about this without requiring continuous calls to **getParam**.

## Package structure

The structure of the present package is as described in the tree below:

```
├── cfg
│   └── tiago_mdp.cfg
├── include
│   ├── dyn_reconf_mdp
│   │   ├── dyn_pub.h
│   │   └── param_client.h
├── nodes
│   ├── client_test.cpp
│   └── server.cpp
├── src
│   ├── dyn_pub.cpp
│   └── param_client.cpp
├── CMakeLists.txt
└── package.xml
```

## Usage

In order to launch the **server** node, you can run the standalone node by initializing roscore in a new terminal and in a new terminal run:

```bash
rosrun dyn_reconf_mdp server
```

**Note**: remember that you may need to source the environment in the new terminal, if not done already, for running the command.

If everything worked, you should see a message on the screen indicating that the node is spinning.

## Testing

You can verify the correct functioning of the node by listening to the topic to which the server publishes to:

```bash
rostopic echo /params
```

In order to evaluate the correct functioning of the dynamic reconfigure, you can make use of *rqt_reconfigure* to manually adjust the values of the adjustable parameters of the node. To do so, run:

```bash
rosrun rqt_reconfigure rqt_reconfigure
```

Conversely, you can also run a sample client node that will change the value of the parameters every 4 seconds:

```bash
rosrun dyn_reconf_mdp client_test
```

From within the window you can change the values of the different parameters of the server. Upon changes in the values (for example, by ticking the box for a boolean), an informative message should appear in the server's terminal and the contents displayed on rostopic should change accordingly. At all times, the values of the parameters can be monitor via **rosparam**.

```bash
rosparam get /server/<parameter_name>
```

### Notes for developers

In order to make use of the client class provided by the package to establish communication with the server, add the following two lines to the **package.xml** file:

```xml
  <build_depend>dyn_reconf_mdp</build_depend>
  <exec_depend>dyn_reconf_mdp</exec_depend>
```

Also, to the **CMakeLists.txt** file:

```txt
find_package(catkin REQUIRED COMPONENTS
  ...
  dyn_reconf_mdp
  ...
)
```

And depending on your setup you may also need to add it to your **create_package**:

```txt
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS dyn_reconf_mdp 
  DEPENDS system_lib
)
```

In order to make use of the class, import it as in the code snippet below:

```c++
    #include "dyn_reconf_mdp/param_client.h"
```

Please, look into the source code for descriptions on the different functions of the client class. It is also recommended to create a local configuration object external to the class for handling the subscription separately from the configuration within the client class. This can be done by:

```c++
    dyn_reconf_mdp::tiago_mdpConfig my_config;
```

