# 2D modelling and control of a ball-balancing robot

<br>
<img src="./2D_BBR%20-%20Horizontal_Rolling/Horizontal_animation.gif" width="300">  
<br>

**Location**: Alma Mater Studiorum – Università di Bologna  
**Period**: Jan 2021 - Jun 2021  

## Context
As a bachelor thesis, I worked on the modelling and control of a simplified 2D version of a ball-balancing robot. The modelling part consisted of the dynamic analysis of two bodies in relative rotation considering the presence of friction and applied torque, in order to obtain a reliable model of the physical system. As for the control, starting from the modelled plant I applied a state-space control with an LQR, together with an integrator and an anti-windup network on the outer control loop in order to avoid steady-state errors and saturation problems. The final result was a system able to move in a stable way to a reference location, avoiding excessive control inputs that would result in the presence of slippage thus making the system uncontrollable.

## Project Description
(FILL) 

## Files
- **2D_BBR - Horizontal_rolling**: Folder containing the necessary files to run the fixed axes tests 
- **2D_BBR - Vertical_rolling**: Folder containing the necessary files to run the moving axes tests
- **2D_BBR - Presentation.pptx**: The final presentation for the thesis defense
- **2D_BBR - Thesis.pdf**: The final thesis report
