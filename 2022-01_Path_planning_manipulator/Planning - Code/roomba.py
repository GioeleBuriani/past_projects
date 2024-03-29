import numpy as np
import pygame


# A roomba object is composed by a base and a manipulator on top of it
class Roomba:
    
    def __init__(self, pos = None, theta = None, q = None):
        '''BASE parameters'''
        # self.mass_w = 2 # Wheel mass
        self.rad_w = 1 # Wheel radius
        self.d = 2 # Distance wheel to center
        self.radius = 10 # Radius of the roomba
        
        # We set a default value
        self.pos = np.zeros([2]) # Default x and y position
        if pos is not None: # if in the input is specified another value, we modify the value of self.pos
            self.pos = pos
        
        self.v = np.zeros([2]) # x and y velocity
        
        # We set a default value
        self.theta = 0 # Default orientation of the roomba
        if theta is not None: # if in the input is specified another value, we modify the value of self.theta
            self.theta = theta

        self.theta_dot = 0 # angular velocity connected with the angle theta

        # Variable for the controller function
        self.I_prev_theta = 0.0 # integrated error
        self.prev_error_theta = 0.0 # previous error
        self.I_prev_qs = np.zeros(2) # integrated error for manipulator
        self.prev_error_qs = np.zeros(2) # previous error for manipulator

        '''MANIPULATOR parameters'''
        self.links = self.radius + 0.5 * self.radius # lenght of the first and second link of the manipulator(in pixels)
        
        # We set a default value
        self.q = np.zeros([2]) # The arm is completely stretch along the y-axis of the body reference frame
        if q is not None: # if in the input is specified another value, we modify the value of self.q
            self.q = q

        self.once = False
        

    '''FUNCTIONS USEFUL FOR THE BASE''' 
    # The control action of the base robot is done by setting, indipendently, the values of the angular velocities of the two wheels.
    # Controlling the robot directly in these two variable is quite challenging though. Therefore, we control omega_L and omega_R through
    # other two variables: omega and v. Where omega is the angular velocity of the base, whereas v is its linear velocity.
    def set_wheel_angular_vel(self, omega, v):
        self.omega_R = (2 * v + omega * 2 * self.d)/(2 * self.rad_w) 
        self.omega_L = (2 * v - omega * 2 * self.d)/(2 * self.rad_w) 

        return self.omega_R, self.omega_L 

    # Kinematics of the base of the robot, gives as output the 3 velocities that describe the behavior of the robot
    def base_kinematics(self):
        # rotation matrix that transform quantities expressed with reference to the body frame, into the inertial frame.
        R_inertial_B = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                                [np.sin(self.theta), np.cos(self.theta), 0],
                                [0, 0, 1]])
        # Velocities expressed with reference to the body frame
        Body_velocities = np.array([self.rad_w * (self.omega_R + self.omega_L)/2, 0, self.rad_w * (self.omega_R - self.omega_L)/(2 * self.d)])
        # Velocities expressed with reference to the inertial frame
        velocities = R_inertial_B @ Body_velocities
        
        # The reference frame for the pygame simulation has y positive towards the bottom of the screen. Therefore is necessary to adjust a bit the 
        # value of the linear velocities through a 2D rotation matrix.
        R_pygame_inertial = np.array([[1, 0],
                                      [0, -1]])

        self.v = R_pygame_inertial @ velocities[:2] 
        self.theta_dot = velocities[2]
        return self.v, self.theta_dot
    
    # Additional function that can be used to set some of the states equal to a particular value
    def base_set_states(self, pos = None, v = None, theta = None, theta_dot = None):
        # The states are updated only if a new value is given as input. This allows us to use the same function to update one, some or all the states.
        if pos is not None:
            self.pos = pos 
        if v is not None:
            self.v=v
        if theta is not None:
            self.theta = theta
        if theta_dot is not None:    
            self.theta_dot = theta_dot

    def base_current_states(self):
        return self.pos, self.v, self.theta, self.theta_dot


    '''FUNCTIONS USEFUL FOR THE MANIPULATOR''' 
    def manipulator_forward_kinematics(self):
        pos_elbow = np.zeros([2]) # elbow position
        pos_end_point = np.zeros([2]) # end point position
        
        pos_elbow[0] = self.pos[0] + self.links * np.cos(self.q[0])
        pos_elbow[1] = self.pos[1] - self.links * np.sin(self.q[0]) # The minus is due to the reference frame of the pygame simulation that points 
        # downwards.

        pos_end_point[0] = pos_elbow[0] + self.links * np.cos(self.q[0] + self.q[1])
        pos_end_point[1] = pos_elbow[1] - self.links * np.sin(self.q[0] + self.q[1]) # The minus is due 
        # to the reference frame of the pygame simulation that points downwards.
        
        return pos_elbow, pos_end_point

    # Forward kinematics based on states given as input. Used to simulate the desired behavior.
    def manipulator_FK(self, x_base, y_base, joint_1, joint_2):
        
        x_1 = x_base + int(self.links * np.cos(joint_1))
        y_1 = y_base - int(self.links * np.sin(joint_1))
        
        x_2 = x_1 + int(self.links * np.cos(joint_1 + joint_2))
        y_2 = y_1 - int(self.links * np.sin(joint_1 + joint_2))
        
        return x_1,y_1,x_2,y_2
    

    def set_joints(self, q0, q1):
        self.q[0] = q0
        self.q[1] = q1

    def manipulator_current_states(self):
        return self.q

    '''FUNCTION FOR THE CONTROL AND MOVEMENT OF THE ROBOT'''
    def calculate_errors_base(self, trajectory, xytheta_roomba):
        error_xy = np.sqrt((trajectory[0] - xytheta_roomba[0])**2 + (trajectory[1] - xytheta_roomba[1])**2) # Linear distance from the current position and the 
        # position of the waypoint we are currently tracking
        errors_theta = trajectory[2] - xytheta_roomba[2] # Angular error
        errors_theta = np.arctan2(np.sin(errors_theta), np.cos(errors_theta)) # Atan2 guarantee that the error is contained between -pi and pi. This is needed if 
        # we work with angles

        errors_base = np.array([error_xy, errors_theta]) # compact notation

        return errors_base
    
    def calculate_errors_manipulator(self, q0, q1):
        errors_q = np.zeros(2)
        errors_q[0] = q0 - self.q[0]
        errors_q[1] = q1 - self.q[1]

        return errors_q

    def controllers_base(self, errors_base, dt):
        '''Simple proportial controller for the linear velocity'''
        Kp_v = 7.5 # proportional gain 
        P_v = np.dot(Kp_v, errors_base[0])

        '''PID controller for the angular velocity'''
        Kp = 10 #10 # proportional gain 
        Ki = 0.5 #1 # integral gain
        Kd = 0.1 #0.5 # derivative gain

        P = np.dot(Kp, errors_base[1])
        I = self.I_prev_theta + Ki * errors_base[1] * dt 
        D = Kd * (errors_base[1] - self.prev_error_theta)/(dt)
        
        PID = P + I + D

        # Assigning some values that will be used in the following iteration of the for cycle
        self.prev_error_theta = errors_base[1]
        self.I_prev_theta = I

        return P_v, PID
    
    def controller_manipulator(self, errors_q, dt):
        '''PID controller for the angular velocity'''
        Kp = 40 # proportional gain 
        Ki = 1  # integral gain
        Kd = 0.3 # derivative gain

        P = np.dot(Kp, errors_q)
        I = self.I_prev_qs + Ki * errors_q * dt 
        D = Kd * (errors_q - self.prev_error_qs)/(dt)
        
        PID_manipulator = P + I + D

        # Assigning some values that will be used in the following iteration of the for cycle
        self.prev_error_qs = errors_q
        self.I_prev_qs = I

        return PID_manipulator
    
    # Given a path is possible to infer the points that compose the path. These are useful to compute the trajectory that the robot has to follow. 
    def points_of_path(self, path, V):
        points = []
        for k in range(len(path)):
            points.append([V[int(path[k, 0])], V[int(path[k, 1])]])
        return np.array(points)

    # Reference_x and Reference_y are two variables that contains a sequence of points that compose the smooth path. The following function decide which is the 
    # point, among the whole list, that should be used as reference according to the time passed since the beginnig of the simulation. 
    def smooth_traj_gen(self, reference_x, reference_y, t_sim, T, step, trajectory, xytheta_roomba):
        steps = len(reference_x)
        stop_v_controller = 1
        
        if step < steps: # If it is false, we are at the goal
            if t_sim < ((T/steps) * step):
                trajectory = np.array([reference_x[steps - step], reference_y[steps - step], 0]) # the angle desired for the moment is set to zero. In the next 
                # line is calculated accurately

                theta_desired = np.arctan2(-(trajectory[1] - xytheta_roomba[1]), (trajectory[0] - xytheta_roomba[0])) # The desired heading angle is computed
                # geometrically. What is needed to computed is just the x and y of the waypoint the roomba has to reach, and the current x and y of the roomba.
                # IMPORTANT: note that the first term has a minus in front of the parenthesis, this is needed because, due to the reference frame of pygame, a
                # point lower in the screen has a greater value of y. Instead in the normal convetion this is not true. The minus allows us to use atan2 normally.
                trajectory[2] = theta_desired
            else: 
                step += 1

        # When we are very close to the goal, we can consider the task completed. Therefore we set the angle of the roomba to a standard configuration.
        # We also deactivate the controller of the linear velocity because we do not want to move from this location, just adjusting the orientation.
        elif (-0.1 < trajectory[1] - xytheta_roomba[1] < 0.1) and (-0.1 < trajectory[0] - xytheta_roomba[0] < 0.1):
            trajectory = np.array([reference_x[0], reference_y[0], 0]) # Goal
            trajectory[2] = np.pi
            stop_v_controller = 0
            self.once = True

        # In this case we are heading towards the goal, therefore the angle theta is chosen in order to have the roomba facing toward the goal.
        # Once we reach the goal, or we are very close to it, calculating theta in this way brings to an unstable behavior because the position of the 
        # robot could be anywhere close to the goal. For this reason when we are very close we prefer to use as reference the configuration specified by
        # the 'elif' block. The follwing code is executed only if the robot haven't already reached, the goal. 
        else:
            if self.once == False: # The follwing code is executed only if the robot haven't already reached, the goal. This extra if, it is to ensure the behavior
                # previously described
                trajectory = np.array([reference_x[0], reference_y[0], 0]) # Goal
                # print("distance from goal: ", trajectory[1] - xytheta_roomba[1], trajectory[0] - xytheta_roomba[0])
                theta_desired = np.arctan2(-(trajectory[1] - xytheta_roomba[1]), (trajectory[0] - xytheta_roomba[0])) # The desired heading angle is computed
                    # geometrically. What is needed to computed is just the x and y of the waypoint the roomba has to reach, and the current x and y of the roomba.
                    # IMPORTANT: note that the first term has a minus in front of the parenthesis, this is needed because, due to the reference frame of pygame, a
                    # point lower in the screen has a greater value of y. Instead in the normal convetion this is not true. The minus allows us to use atan2 normally.
                trajectory[2] = theta_desired

            
        return trajectory, step, stop_v_controller