# 2D modelling and control of a ball-balancing robot

<br>
<img src="./2D_BBR%20-%20Horizontal_rolling/Horizontal_animation.gif" width="300"> <img src="./2D_BBR%20-%20Vertical_rolling/Vertical_animation.gif" width="300">
<br>

**Location**: Alma Mater Studiorum – Università di Bologna  
**Period**: Jan 2021 - Jun 2021  

## Context
As a bachelor thesis, I worked on the modelling and control of a simplified 2D version of a ball-balancing robot. The modelling part consisted of the dynamic analysis of two bodies in relative rotation considering the presence of friction and applied torque, in order to obtain a reliable model of the physical system. As for the control, starting from the modelled plant I applied a state-space control with an LQR, together with an integrator and an anti-windup network on the outer control loop in order to avoid steady-state errors and saturation problems. The final result was a system able to move in a stable way to a reference location, avoiding excessive control inputs that would result in the presence of slippage thus making the system uncontrollable.

## Project Description
The thesis dives into the intricacies of creating a simplified yet highly functional model of a robot that maintains equilibrium on a cylindrical object, facilitating movement across a plane. This exploration is foundational, setting the stage for the eventual realization of a more complex three-dimensional version.

Initially, the thesis outlines the critical dynamics between the robot's wheel and the surface it navigates, emphasizing the pivotal role of friction in motion control. The study commences with an examination under fixed rotational axes, delving into the nuances of static, dynamic, and rolling friction and their cumulative effect on the robot's maneuverability and stability. This phase serves as a groundwork to understand how these frictional forces interact to dictate the robot's movement.

Transitioning to scenarios involving moving rotation axes introduces a new layer of complexity. Here, the focus shifts to understanding how variations in the normal force, induced by changes in the robot's orientation, affect its dynamics. The modelling meticulously accounts for the interplay between these forces, paving the way for developing a robust control mechanism.

The control segment of the thesis is where the theoretical models are translated into actionable strategies aimed at stabilizing and directing the robot's movements. For static scenarios, it proposes a control system designed to swiftly and accurately adjust the robot's position, incorporating techniques to manage potential instabilities and optimize response times. In dynamic scenarios, the thesis escalates the sophistication of the control system, employing state-space representations and stability analyses to fine-tune the robot's balance and trajectory, ensuring precise control over its movements.

Anticipating future advancements, the thesis concludes with proposals for enhancing both the model and control strategies. It suggests incorporating models for additional physical phenomena, like an inverted pendulum, to refine the robot's balance. Furthermore, it outlines potential expansions to the control system, including adapting to three-dimensional movements, thereby broadening the scope of the robot's capabilities.

In summary, the thesis intricately weaves together theoretical models and control strategies to create a comprehensive blueprint for a ball-balancing robot. This detailed study not only demonstrates a commitment to advancing robotic mobility and stability but also lays a solid foundation for future innovations in the field. 

## Files
- **2D_BBR - Horizontal_rolling**: Folder containing the necessary files to run the fixed axes tests 
- **2D_BBR - Vertical_rolling**: Folder containing the necessary files to run the moving axes tests
- **2D_BBR - Presentation.pptx**: The final presentation for the thesis defense
- **2D_BBR - Thesis.pdf**: The final thesis report
