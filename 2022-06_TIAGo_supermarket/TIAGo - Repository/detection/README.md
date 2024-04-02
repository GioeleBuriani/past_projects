# Detection package for TIAGo

## Table of contents

1. [Overview](#overview)
2. [Package structure](#package-structure) 
3. [Requirements](#requirements)
4. [Usage](#usage)

## Overview

This package contains all the files that are required to send the item id from the detected aruco tags to the parameter server. If there are no items detected at the checkout this will be communicated aswell. 
The detection package also communicates whether a human in in front of TIAgo.

## Package structure

The structure of the present package is as described in the tree below:


```
├── src
│   └── detection.cpp
├── CMakeLists.txt
└── package.xml
```

## Requirements

The aruco_ros package is required. 

As this package will publish the detection message containing the aruco tag id and relative location.

## Usage

The package is launched automatically by the director.launch file.  

Located at tiago_director/launch/director.launch

To start the package manually the following command can be used: 
```bash
rosrun detection detection
```
>Note: Be sure to source first



