# Simulink scheme of an electric drive

<br>
<img src="./LED%20-%20Full%20system.png" width="300">  
<br>

**Location**: Alma Mater Studiorum – Università di Bologna  
**Period**: Sep 2020 - Dec 2020  

## Context
Within the Laboratory of electric drives course, I created the Simulink scheme of an entire electric drive that had to be tuned in order to achieve some specific final performances in terms of output current and voltage.

## Project Description
The project revolves around the design and implementation of a DC/DC drive system for a motor with specified data, including nominal voltage, speed, torque, and other electrical characteristics. The motor's intricate parameters, such as winding resistance, torque constant, and rotor inertia, play a crucial role in the drive's performance.

The drive system aims to achieve optimal control using closed-loop current and speed controls, implemented in a Simulink model. A particular emphasis is placed on the PWM and the Real 4 Quadrant DC/DC subsystems, which are pivotal in achieving the desired control and performance.

Initially, the project parameters were set, focusing on a 4 Quad DC/DC converter with a maximum output current of 1.9A. The expected performance criteria encompassed aspects like output current ripple, speed overshoot in step response, and rise time. All these factors were meticulously monitored and adjusted to ensure optimal system behavior.

Performance diagrams and power measurements in both the source and load sections were scrutinized. Detailed views of the output current, input current, and motor speed were provided to showcase the system's response to varying conditions.

Ultimately, the project achieved results that met the stringent criteria. For instance, the output current ripple in steady state operation was 6.48%, and the rise time to transition from 0 to 50% of nominal speed was 62.7 ms, showcasing the system's efficiency and precision.

In conclusion, this endeavor involved intricate tuning of the drive system, in-depth analysis of the PWM modulation and 4 quadrant DC/DC converter subsystems, and rigorous validation against expected performance metrics. The culmination of these efforts resulted in a robust and efficient DC/DC drive system.

## Files
- **LED - Report.pdf**: The report for the project containing the scheme explanation, parameters and measurements.
- **LED - Simulink scheme.slx**: The Simulink program for the project.
- **LED - Full system.png**: The Simulink scheme of the whole electric drive
