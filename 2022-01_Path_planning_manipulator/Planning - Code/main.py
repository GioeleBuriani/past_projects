import pygame
import sys
import numpy as np
from rrt_star_complete import RRT_star_complete
from random_goal import create_random_goal
from create_hospital import create_hospital
from drawing_functions import draw_background, draw_edges_weights, draw_path, draw_start_and_goal, draw_counter_nodes, rotate_image_around_center
from roomba import Roomba
from traj_smooth import traj_smooth_mobile, smooth_path_generator


m = 1.2 # Scale for the simulation
start = [m*200, m*120, np.radians(-90), np.radians(0)] # Define start position

goal = create_random_goal(m)  # Define goal position

'''Time variable for the simulation'''
t_sim = 0 # Time past in the simulation
dt = 0.001 # step time for the discrete simulation
step = 0 # Step of the discrete smooth trajectory
step_2 = 0
Execution_Time = 20 # Time in which the roomba has to complete the path

trajectory = np.array([start[0], start[1], -np.pi]) # variable for the base of the robot. (x_position, y_position, theta_orientation)
trajectory_2 = np.array([start[2], start[3], 0]) # variable for the arm. (angle_q0, angle_q1, nothing_relevant)

roomba = Roomba(q=trajectory_2[:2]) # Instance of the robot
roomba.base_set_states(pos = ([start[0], start[1]]), theta = 0)
l = roomba.links

optimization_nodes = 500
max_tot_nodes = 3000
rrt_game = RRT_star_complete(max_tot_nodes, optimization_nodes, start, m, roomba)  # Instance of the class where the algorithm of path planning is managed

obstacles = create_hospital(rrt_game.screen, m)  # Creates the obstacles for the hospital and draws them in the simulation

rrt_game.get_configuration_space_obstacles(obstacles)
# Loading the image of the robot
image_roomba = pygame.image.load('./../images/no_bg.png') 
image_roomba = pygame.transform.scale(image_roomba, (round(m*38), round(m*31)))

once = False # This variable is used to execute a section of the code only once

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  #Click x in window to quit
            pygame.quit()
            sys.exit()
            
    # This function compute one step of the RRT/RRT* algorithm. Since it is in a while loop, the full algorithm is computed. 
    # The input roomba is needed to keep into account the radius of the robot that influences the configuration space for the obstacles.
    rrt_game.launch_sim(start, goal, obstacles, m)
    

    '''Drawing function for the enviroment and the path'''
    if rrt_game.draw_background_bool or (rrt_game.counter == optimization_nodes): # 'draw_background_bool' is variable coming from rrt_game.launch_sim(). 
        # When 'counter' is equal to 'optimization_nodes' it means that the final path is ready, therefore the rrt algorithm does not run anymore. As consequence,
        # draw_background_bool = False, but actually we still want the background printed.
        draw_background(rrt_game.screen, obstacles, start, goal, rrt_game.GOAL_RADIUS, m)

    if rrt_game.draw_edges_weights_bool: # variable coming from rrt_game.launch_sim()
        draw_edges_weights(rrt_game.screen, rrt_game.V[:,:2], rrt_game.E.reshape(-1, 2), m)
    
    if once == False:
        draw_path(rrt_game.screen, rrt_game.path, rrt_game.V[:,:2], (255, 165, 0), m) # path in orange, printed only before the smooth trajectory is computed
    draw_counter_nodes(rrt_game.screen, rrt_game.V) # counter in red
    draw_start_and_goal(rrt_game.screen, start[:2], goal, m, rrt_game.GOAL_RADIUS)
    

    '''States of the base of the robot'''
    pos_base, vel_base, theta_base, theta_dot_base = roomba.base_current_states()
    xytheta_roomba = np.array([pos_base[0], pos_base[1], theta_base])
    
    # We want to smooth the final path only when RRT* is done. This happens when rrt_game.counter == optimization_nodes.
    # Moreover the smoothing step should happen only once, therefore at the second iteration we block this if with once == False. 
    if rrt_game.counter == optimization_nodes and once == False: 
        # The code in this section needs to be executed only once, instead the code inside the following section elif is executed repeatedly
        
        #traj_points = roomba.points_of_path(rrt_game.path, rrt_game.V) # Points that compose the trajectory
        print("shortest path: \n", rrt_game.shortest_path)

        ref_x, ref_y, ref_q0, ref_q1 = traj_smooth_mobile(rrt_game.shortest_path[:, :], rrt_game.V[:,:], rrt_game.screen, obstacles) # Smooth trajectory starting from the output of the RRT* informed

        smooth_path = smooth_path_generator(ref_x) # Variable needed for 'draw_path' function used in the next block of code (elif)
        smooth_path_2 = smooth_path_generator(ref_q0)

        t_sim += dt # Update of the simulation time 
        once = True # This means that this section of the code has been executed once     

    elif once:
        # To see the all the configuration of the robotic arm during the simulation uncomment the following code
        # The following 'for cycle' shows the configuration that the manipulator has to keep to be in a collision free configuration.
        # for i in rrt_game.shortest_path:
        #     pos_elbow1 = np.zeros(2)
        #     pos_end_point1= np.zeros(2)
        #     pos_base1 = np.zeros(2)
        #     pos_base1[0] = i[0]
        #     pos_base1[1] = i[1]

        #     pos_elbow1[0], pos_elbow1[1], pos_end_point1[0], pos_end_point1[1]= roomba.manipulator_FK(i[0], i[1], i[2], i[3])
        #     pygame.draw.circle(rrt_game.screen, (0, 170, 255), ((pos_elbow1[0], pos_elbow1[1])), m*3.5) # draw the elbow of the manipulator
        #     pygame.draw.circle(rrt_game.screen, (0, 170, 255), ((pos_end_point1[0], pos_end_point1[1])), m*3.5) # draw the end-point of the manipulator
        #     pygame.draw.lines(rrt_game.screen, (0, 0, 0), False, [(pos_base1[0], pos_base1[1]), 
        #                                                             (pos_elbow1[0], pos_elbow1[1]), (pos_end_point1[0], pos_end_point1[1])], round(m*2)) # draw links

        t_sim += dt # Update of the simulation time 

        '''Printing the smooth trajectory for the manipulator in red'''
        coor_xs = [] # standing for coordinates
        coor_ys = []
        for i in range(len(ref_q0)):
            _, _, coor_x, coor_y = roomba.manipulator_FK(ref_x[i], ref_y[i], ref_q0[i], ref_q1[i])
            coor_xs.append(coor_x)
            coor_ys.append(coor_y)

        coor_xs = np.asanyarray(coor_xs)
        coor_ys = np.asanyarray(coor_ys)
        draw_path(rrt_game.screen, smooth_path_2, np.array([coor_xs, coor_ys]).T, (255, 5, 0), m/2) # smooth path in red


        '''CONTROL OF THE ROOMBA BASE'''
        # Among all the points that trajectory generates, the following function decide which is the next reference point to reach.
        trajectory, step, stop_v_controller = roomba.smooth_traj_gen(ref_x, ref_y, t_sim, Execution_Time, step, trajectory, xytheta_roomba)
        pygame.draw.circle(rrt_game.screen, (0, 0, 0) , (trajectory[0], trajectory[1]), m*4) # Draw the waypoint to be tracked 
        draw_path(rrt_game.screen, smooth_path, np.array([ref_x, ref_y]).T, (0, 165, 0), m/2) # smooth path in green


        # To apply a controller we first need an error. In this case we will have two controllers and two errors. The first controller is for the linear velocity,
        # the second one for the angular velocity of the base. Hence, the errors are related one to the linear distance and the second one to the angular distance.
        errors_base = roomba.calculate_errors_base(trajectory, xytheta_roomba)

        P_v, PID_theta = roomba.controllers_base(errors_base, dt) # PID is a vector that contains the new values for the velocity computed thru a PID controller
        P_v = P_v * stop_v_controller # the variable 'stop_v_controller' is set to zero when we reach the goal.

        # Even if we build the control around the linear and angular velocity of the base, the real control action is done through the indipended angular 
        # velocities of the wheels. The following function retrieves the values of these two angular velocities starting from the linear and angular velocity 
        # of the base.
        omega_R, omega_L = roomba.set_wheel_angular_vel(PID_theta, P_v)

        # Now that we have set the values of omega_R and omega_L we can predict the development of the kinematic of the body.
        vel_base, theta_dot_base = roomba.base_kinematics()
        
        # update of the velocity of the robot according to the velocities just computed
        pos_base = pos_base + np.dot(vel_base, dt)
        theta_base = theta_base + theta_dot_base * dt
        roomba.base_set_states(pos = pos_base, theta = theta_base)

        '''CONTROL OF THE MANIPULATOR'''
        # Trajectory_2 containts the reference for the manipulator
        trajectory_2, step_2, _ = roomba.smooth_traj_gen(ref_q0, ref_q1, t_sim, Execution_Time, step_2, trajectory_2, xytheta_roomba)
        
        errors_q = roomba.calculate_errors_manipulator(trajectory_2[0], trajectory_2[1])
        # controller for the manipulator
        PID_manipulator = roomba.controller_manipulator(errors_q, dt)
        q_dot = PID_manipulator # controlling directly the velocity of the joints
        # Update of the states for the manipulator
        q = q + q_dot * dt
        roomba.set_joints(q[0], q[1])

    # Updating manipulator variables
    q = roomba.manipulator_current_states()
    pos_elbow, pos_end_point = roomba.manipulator_forward_kinematics()

    '''Managing the visualization of the roomba simulation'''
    # MANIPULATOR
    # pygame.draw.circle(rrt_game.screen, (0, 170, 255), ((pos_base[0], pos_base[1])), m*3) # draw shoulder / base of the manipulator
    pygame.draw.circle(rrt_game.screen, (0, 170, 255), ((pos_elbow[0], pos_elbow[1])), m*3.5) # draw the elbow of the manipulator
    pygame.draw.circle(rrt_game.screen, (0, 170, 255), ((pos_end_point[0], pos_end_point[1])), m*3.5) # draw the end-point of the manipulator
    pygame.draw.lines(rrt_game.screen, (0, 0, 0), False, [(pos_base[0], pos_base[1]),
                                                            (pos_elbow[0], pos_elbow[1]), (pos_end_point[0], pos_end_point[1])], round(m*2)) # draw links

    # Reference point of the manipulator to be followed, printed as a purple dot on the simulation
    _, _, reference_end_point_x, reference_end_point_y = roomba.manipulator_FK(pos_base[0], pos_base[1], trajectory_2[0], trajectory_2[1])#-trajectory_2[0])
    pygame.draw.circle(rrt_game.screen, (255, 10, 255), ((reference_end_point_x, reference_end_point_y)), m*3.5)                                                        
       
                                                              
    # BASE
    rect = image_roomba.get_rect(center = (pos_base[0], pos_base[1])) # it gets the topleft corner of the picture starting from the center of the pic. It is 
    # used because to print the image is necessary the position of the topleft point, not the center
    image, rect = rotate_image_around_center(image_roomba, rect, np.degrees(theta_base))
    rrt_game.screen.blit(image, rect) # print the image

    pygame.display.update()  #update display
    



















