import numpy as np
import random

def create_random_goal(m):
    
    goal_1_1a = [m*850, m*600, -np.pi/2, 0]
    goal_2_3 = [m*840, m*500, 0, 0]
    goal_4_4a = [m*750, m*440, np.pi/2, 0]
    goal_5_6 = [m*840, m*350, 0, 0]
    goal_7_8 = [m*760, m*230, np.pi, 0]
    goal_9_10 = [m*580, m*200, np.pi/2, 0]
    goal_11_12 = [m*440, m*280, 0, 0]
    goal_13_14 = [m*280, m*200, np.pi, 0]
    goal_15_16 = [m*260, m*600, -np.pi/2, 0]
    goal_17_18 = [m*440, m*440, np.pi/2, 0]
    goal_19_19a = [m*350, m*600, -np.pi/2, 0]
    goal_20_21 = [m*600, m*570, 0, 0]

    goal = random.choice([goal_1_1a, goal_2_3, goal_4_4a, goal_5_6, goal_7_8, goal_9_10, goal_11_12, goal_13_14, goal_15_16, goal_17_18, goal_19_19a, goal_20_21])
    print("The goal for the robot is (first two number position of the base, second two angles of the arm): ", goal)

    return goal