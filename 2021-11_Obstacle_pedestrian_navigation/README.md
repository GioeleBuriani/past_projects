# Combination of obstacle avoidance and pedestrian detection for a self-driving car

<br>
<img src="./Navigation%20-%20Environment.png" width="300">
<br>

**Location**: Technische Universiteit Delft  
**Period**: Sep 2021 - Nov 2021  
**Collaborators**: Edoardo Panichi

## Context
Within the Robot Software Practicals course, we created a single system that could manage the automated driving of a simulated vehicle in a simple environment. We used ROS to combine different nodes, each managing a different aspect: one node managed car actions (accelerate, brake, turn,…), one node managed obstacle detection and another managed pedestrian recognition. In the end, the car was able to drive inside of a track avoiding the boundaries and stopping in front of pedestrians crossing the road.

## Project Description
This project focused on the development of an autonomous driving software for a simulated Prius vehicle, enabling it to navigate a test track while detecting and avoiding obstacles and pedestrians. The primary objective was to integrate and apply comprehensive skills in Linux, Git, C++, and ROS, alongside leveraging OpenCV for image processing and PCL for point cloud processing. The vehicle utilized front-facing camera images and 360-degree lidar data to identify obstacles and pedestrians, employing this information to make real-time driving decisions.

The solution encompassed three ROS packages: one for detecting pedestrians using OpenCV, another for detecting obstacles like barrels using PCL, and a control package for vehicle maneuvering. The pedestrian detection module utilized a HOG detector to identify humans in camera images, while the obstacle detection module applied Euclidean cluster extraction to identify obstacles in lidar point clouds. The control module then synthesized these detections to navigate the Prius safely around the track, avoiding collisions with both static obstacles and moving pedestrians.

Key challenges included accurately detecting objects in varied environments and ensuring reliable vehicle control under dynamic conditions. Through iterative development and rigorous testing, the project demonstrated the potential of combining advanced image and point cloud processing techniques for autonomous navigation. This endeavor not only showcased the technical prowess in robotics and software engineering but also highlighted the versatility and potential of autonomous vehicles in complex environments.

Unfortunately, the solution files are not available to share.

## Files
- **Navigation - Environment.png**: Image of the simulation environment for the project
- **Navigation - Project Guidelines.pdf**: Guidelines for the project
