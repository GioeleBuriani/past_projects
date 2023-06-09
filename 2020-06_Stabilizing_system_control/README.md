# Control for a highly unstable system

**Location**: Alma Mater Studiorum – Università di Bologna  
**Period**: Feb 2020 - Jun 2020  
**Collaborators**: Matteo Ambrosini, Vittorio Galletti

## Context
Within the Automatic control 2 course, we designed several control networks using Matlab and Simulink in order to stabilize and obtain some specific performances out of a highly unstable system. The final result made use of anticipatory networks, an anti-windup network and a Smith predictor all tuned in order to obtain the wanted result.

## Project Description
The project entails designing a controller for a system with specific requirements, including zero steady-state error, limited overshoot, fast settling time, and robustness. The plant's transfer function reveals complex conjugate poles and a resonance frequency. The goal is to develop a control solution that improves stability and performance.

Initially, a crossover frequency of 11 rad/s was chosen to exit the critical zone and maintain a 75° phase margin. However, the settling time did not meet the requirements. To address this, a higher crossover frequency of 14.4 rad/s was selected, but it still fell short. To counteract the step response tail, a feedforward controller and damping pre-filter were implemented, along with additional realizable poles.
Further improvements involved introducing three zeros in the controller to control the intervention frequency and increase the crossover frequency. However, this caused pronounced oscillations, which were mitigated using a damping pre-filter. Although the settling time improved, it still didn't meet the requirements.
The crossover frequency was then increased to 23 rad/s, meeting overshoot and settling time specifications. However, the control sensitivity function exhibited a large magnitude at high frequencies. To address this, a controller composed of three coincident lead networks replaced the PID-like structure. An anti-aliasing filter further reduced the control sensitivity function's magnitude.
Optimizing the actuator saturation level enhanced system performance under plant uncertainty, resulting in a solution that met the specifications. Additional considerations included adding a pole to enhance robustness, but this would increase complexity.

An alternative solution involving crossover before the complex conjugate poles posed challenges such as multiple crossings and longer settling times, deviating from the assumed tuning.

Overall, the project involved systematically adjusting the crossover frequency, implementing feedforward and damping techniques, adding realizable poles and zeros, and optimizing actuator saturation to achieve a controller design that met the specifications.

All the documents are in Italian.

## Files
- **Control - Live Script.mlx**: A Matlab Live Script file containing the Matlab code and the theoretical explaination of the project.
- **Control Scenario 3.slx**: A Simulink file containing the behaviour of the third scenary network after simulating uncertainty in the plant.
- **Control Scenario 6.slx**: A Simulink file containing the behaviour of the sixth scenary network after simulating uncertainty in the plant.
