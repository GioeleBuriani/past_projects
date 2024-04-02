# Software to make a TIAGo robot help supermarket employees

<br>
<img src="./TIAGo%20-%20Full%20animation.gif" width="300"><img src="./TIAGo%20-%20Pick%20and%20place%20animation.gif" width="300">
<br>

**Location**: Technische Universiteit Delft  
**Period**: Apr 2022 - Jun 2022  
**Collaborators**: Max Polack, Pim de Ruijter, Ruben Martin-Rodriguez, Yuezhe Zhang

## Context
Within the Multidisciplinary Project course, we had to develop software for the control of a TIAGo robot for Ahold Delhaize, one of the biggest multinational retail and wholesale companies in the world. Our objective was to make TIAGo help supermarket employees to manage their workload, especially during rush hour. We decided to make TIAGo cyclically collect products left at the check-out, recognize them, pick them up and place them back on the correct original shelf. In order to manage this project, our team was divided into Human-Robot Interaction, Navigation, Perception, Planning and Motion Control specialists. I took the latter role. In order to make everything work together, we implemented several ROS nodes communicating with each other.

## Project Description
This project aimed at refining the operational capabilities of the TIAGo robot within the supermarket context, specifically focusing on the automation of returning products left at checkout counters to their designated shelves. The challenge was multifaceted, involving the seamless integration of human-robot interaction, precise navigation through varied and dynamic store layouts, accurate product recognition, and efficient planning and execution of motion control tasks. The essence of the project was to enhance TIAGo’s utility in assisting staff during high-traffic periods, thereby optimizing workflow and ensuring a well-maintained shopping environment.

To achieve this, we developed a complex control system comprising several ROS nodes that facilitated robust communication and coordination among the robot’s subsystems. Key to the system's success was its ability to interact intelligently with both the environment and human operators, ensuring smooth navigation and obstacle avoidance. The project also introduced a sophisticated perception mechanism for TIAGo, enabling it to identify and categorize various products accurately.

A notable aspect of our development was the implementation of advanced motion control algorithms. These were crucial for enabling TIAGo to execute precise pick-and-place operations, a task that required a high degree of accuracy and adaptability. The integration of planning algorithms ensured that TIAGo could autonomously decide on the most efficient routes and actions, adapting in real-time to changes within the supermarket.

However, the project was not without its challenges. Ensuring reliable product detection and handling in a highly variable environment required iterative testing and fine-tuning of the perception algorithms. Additionally, the navigation system had to be optimized to deal with the unpredictable nature of crowded spaces, balancing speed and safety to navigate effectively around shoppers and obstacles.

Ultimately, the project succeeded in elevating TIAGo's capabilities, making it an invaluable asset for supermarket staff. By automating the task of returning misplaced products, we not only streamlined store operations but also contributed to a more pleasant shopping experience. The project stands as a testament to the potential of robotics in retail, showcasing how sophisticated control systems can enhance efficiency, accuracy, and customer satisfaction in real-world applications.

## Files
- **TIAGo - Repository**: All the files coming from the shared GitLab repository for the project
- **TIAGo - Final report.pdf**: Final report for the project
- **TIAGo - Full animation.gif**: Animation of the TIAGo robot performing the full task
- **TIAGo - Pick and place animation.gif**: Animation of the TIAGo robot performing pick and place with a focus view from RViz
- **TIAGo - Presentation.pptx**: Final presentation for the project
