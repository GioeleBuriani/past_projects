# Pedestrian recognition by means of multiple sensors

<br>
<img src="./Pedestrian%20-%20Animation.gif" width="300">  
<br>

**Location**: Technische Universiteit Delft  
**Period**: Nov 2021 - Jan 2022  

## Context
Within the Machine Perception course, I designed a system that could detect and recognize pedestrians in front of a self-driving vehicle. Initially, I did it only with a single camera data and ground plane detection: in the end, it was possible to detect closer pedestrians and determine their approximate position in the 3D space. Then, by combining also Lidar detections, the accuracy increased significantly and the exact position and dimension of the pedestrians were possible to determine.

## Project Description
The project focuses on the development of advanced pedestrian detection systems utilizing data from a vehicle equipped with a variety of sensors, including cameras, LiDAR, and radar. The aim is to create algorithms capable of accurately identifying pedestrian locations in three-dimensional space through two distinct approaches: one leveraging only camera data and the other integrating multiple sensor inputs for enhanced accuracy and depth perception.

The camera-only approach involves a multi-stage process beginning with an innovative detection phase. Utilizing the ground plane information and the camera's perspective, the system generates intelligent proposal boxes, significantly improving over traditional methods by estimating pedestrian positions with greater accuracy. The classification phase employs sophisticated, pretrained classifiers and feature extractors, tailored to recognize pedestrians within the proposal boxes without direct depth cues.

Expanding upon this, the multi-sensor approach integrates data from LiDAR and radar alongside camera images to achieve a more comprehensive understanding of the scene. This method capitalizes on the strengths of each sensor type: LiDAR's precision in depth estimation, radar's capability in detecting objects under various weather conditions, and the camera's rich visual information. By synthesizing these data sources, the system can more reliably discern pedestrian shapes and movements, even in complex urban environments.

Crucially, both methods involve intricate data processing and fusion techniques. For the camera-only system, the project exploits the geometric relationships between the camera's viewpoint and the ground plane, alongside advanced image processing to propose likely pedestrian locations. In contrast, the multi-sensor strategy employs clustering algorithms on LiDAR point clouds to identify potential pedestrian forms, which are then correlated with radar signals and visual data from the camera to confirm detections and refine their positions in 3D space.

Throughout the development process, the project faced challenges related to sensor data interpretation, algorithm optimization, and the integration of disparate data sources into a cohesive detection framework. By addressing these issues, the project demonstrates the potential of both mono-camera and multi-sensor approaches to significantly advance pedestrian detection capabilities, crucial for the safety and reliability of autonomous vehicle navigation systems. 

## Files
- **Pedestrian - Setup**: Folder containing configuration files and guides to set up the environment 
- **Pedestrian - assignment**: Folder containing the main ipynb scripts and other important files 
- **Pedestrian - common**: Folder containing some necessary Python scripts
- **Pedestrian - Animation.gif**: Animation of the final result
