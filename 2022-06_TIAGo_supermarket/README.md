# Path planning and control for a mobile manipulator

<br>
<img src="./Planning%20-%20Animation.gif" width="300">  
<br>

**Location**: Technische Universiteit Delft  
**Period**: Nov 2021 - Jan 2022  
**Collaborators**: Amin Berjaoui Tahmaz, Edoardo Panichi, Giovanni Corvi

## Context
Within the Planning & Decision Making course, we created a path planner for a mobile manipulator moving on a differential drive. The robot had to move within a simulated 2D environment with obstacles and reach some randomized goal positions for both its body and its gripper. The path planner was a hybrid form between an RRT and an informed-RRT*, on which a trajectory smoother acted to make the path feasible for the controller (a simple PID).

## Project Description
This project explores the development of a sophisticated motion planning algorithm for a differential drive mobile manipulator operating within a hospital environment. The challenge involves navigating through various rooms, avoiding obstacles, and achieving a cost-effective path to randomized goal positions. Utilizing a unique blend of RRT (Rapidly-exploring Random Tree) and informed-RRT* algorithms, we engineered a solution that balances efficiency with path optimization. The introduction of a trajectory smoothing process further refines the robot's path, making it compatible with its differential drive mechanism. Additionally, the manipulator's movements are precisely controlled via a PID controller, ensuring the robot can perform tasks like transporting medical supplies between locations.

The robot model incorporates a 'roomba-like' differential drive and a 2-degree-of-freedom planar manipulator, specifically designed for navigating the complex layout of a hospital. This design enables the robot to perform simple mechanical tasks, enhancing the efficiency of medical staff. The motion planning strategy begins with the basic RRT algorithm to quickly establish a feasible path, which is then refined through informed-RRT* for optimal path efficiency. This process is tailored to minimize computation time while maintaining path quality, crucial for real-world applications.

The project's results demonstrate the algorithm's capability to adapt to varied goal positions and obstacles efficiently, showcasing a scalable solution for autonomous navigation in healthcare settings. The motion planner's success in a simulated environment highlights its potential for real-world deployment, promising significant benefits in operational efficiency and resource allocation in hospitals.

Through iterative development and testing, the project presents a comprehensive solution that combines advanced algorithmic approaches with practical robotics design. It exemplifies a balance between technical sophistication and practical utility, paving the way for future advancements in robotic applications within healthcare and beyond.

## Files
- **Planning - Code**: Folder containing all the Python scripts to run the project code 
- **Planning - Animation.gif**: Animation of a test
- **Planning - Final report.pdf**: Final report for the project
- **Planning - Presentation.pptx**: Final presentation for the project
