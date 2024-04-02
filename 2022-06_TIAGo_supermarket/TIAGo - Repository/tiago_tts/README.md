# TTS package for TIAGo

## Table of contents

1. [Overview](#overview)
2. [Package structure](#package-structure) 
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Testing](#testing)

## Overview

The present package contains all the necessary files for running the Text-To-Speech (TTS) used by TIAGo for communicating with its environment. The definition of a single node is contained, which subscribes to a given topic and reproduces the sounds corresponding to the received string.

## Package structure

The structure of the present package is as described in the tree below:

```
├── launch
│   ├── tts.launch
├── scripts
│   ├── tts_client_topic.py
│   └── tts_client_topic_real_robot.py
├── CMakeLists.txt
└── package.xml
```

## Requirements

In order to run this package it is necessary to have installed TIAGo's TTS package from PAL Robotics and the standard soundplay package, both of which should be correctly installed by following the steps in the repository's main README.

## Usage

In order to launch the GUI, plase make use of the roslaunch file to automatically start roscore and the additional packages:

```bash
roslaunch tiago_tts tts.launch
```
A ROSinfo message should appear on screen indicating that the node is active and listening.

## Testing

You can verify the correct functioning of the node by publishing your message to the topic that the node listens to:

```bash
rostopic pub /text_to_say std_msgs/String "data: '<your_message_info>'"
```

Upon publishing, the node will take charge of communicating with the necessary nodes for reproducing your message, providing feedback upon every word reproduction.
