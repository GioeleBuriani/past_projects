# Einstein's riddle for a TIAGo robot in a post office

<br>
<img src="./Einstein%20-%20Riddle%20image.png" width="300"><img src="./Einstein%20-%20TIAGo%20image.png" width="300">  
<br>

**Location**: Technische Universiteit Delft  
**Period**: Feb 2022 - Apr 2022  
**Collaborators**: Edoardo Panichi, Severin Woernle

## Context
Within the Knowledge Representation & Symbolic Reasoning course, we had to implement a high-level reasoning logic on a TIAGo robot working in a simulated store. In order to exploit TIAGo's computational potential, we decided to reformulate Einstein's riddle to simulate a case where only partial information regarding the addressee would be communicated to the post office. TIAGo would then have to infer all the missing information in a manner that would keep a human operator busy for several hours, but would require a robot just a few seconds. According to the inferred information, TIAGo would then have to move the packages to the correct shelves of the office. We used Prolog for the riddle inferring part and PDDL 2.1 for the movement reasoning part.

## Project Description
This project explores the capabilities of the TIAGo robot in a simulated post office environment, tasked with sorting and delivering packages based on incomplete addressee information. Inspired by the logical complexity of Einstein's riddle, this challenge requires the robot to deduce missing details swiftly— a task that might take humans hours, but is solved by the robot in mere seconds.

Our approach utilized Prolog for logic inference, enabling the robot to deduce the correct delivery paths from limited information. We modeled the problem similarly to Einstein's riddle, using Prolog to automatically generate a knowledge base from given hints about the packages' recipients. This setup effectively transformed partial clues into complete delivery instructions.

For the execution of these instructions, we implemented the movement logic using PDDL 2.1, which mapped out the robot's actions within the simulated environment. This allowed TIAGo to physically rearrange packages on the correct shelves, corresponding to the inferred delivery destinations. The entire process was coordinated using ROS, which managed both the reasoning elements and the simulation dynamics.

The project presented unique challenges, particularly in integrating the logical deductions with physical task execution in a dynamic environment. Through iterative development and testing, we refined the robot's perception and motion control systems to handle real-time changes and interactions within the simulation.

The outcome of this project highlights the potential for using advanced logic and planning techniques in robotics to significantly enhance operational efficiency in tasks that involve complex decision-making and physical automation. This work not only demonstrates TIAGo's enhanced utility in logistical applications but also sets a foundational approach for future explorations into intelligent automated systems.

## Files
- **Einstein - Code**: Folder containing all the files to run the project 
- **Einstein - Final report.pdf**: Final report for the project
- **Einstein - Riddle image.png**: Image showing a representation of Einstein's riddle
- **Einstein - Riddle image.png**: Image showing the TIAGo robot in the post office environment
