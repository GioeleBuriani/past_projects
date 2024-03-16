# Moving Axes Case Simulation Instructions

Follow these steps to simulate the control for the moving axes case:

1. Open the `Vertical_Rolling_Tongji.m` file.
2. Run the first section to set the parameters.
3. Run the second section to compute the controllers.
4. Open the `Vertical_Rolling_Tongji_Control.slx` file and run the simulation.
5. After the simulation finishes, run the last section of the `Vertical_Roling_Tongji.m` file to see the graphical animation of the behavior.
6. The `theta_1_0` parameter can be modified in the first section: it is currently set at `pi/3`, as in the presentation, but its value can be changed. For example, setting it to `2*pi/3` shows that for a bigger initial displacement, the coupling is not strong enough, and the top wheel falls from the cylinder since the system is not able to stabilize itself.
