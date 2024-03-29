import numpy as np
import random as rnd
import pygame
##------------
from obstacles import create_obstacles_border
from geometry import collision_free,in_free_space,in_ellipse,manipulator_collision_free
from neighbor_functions import closest_neighbor,get_neighbors,get_new_neighbor,solve_neighborhood
from my_path import my_path

'''The following class implements the RRT algorithm. To be more specific the algorithm is RRT until a feasible path is found from the start to the goal.
Once the first way is found, the path is optimized implementing an RRT star informed. Dividing the algorithm in this two scenarios allow us to obtain a 
good trade-off between efficiency and optimality. RRT is faster, but RRT is asymptotically optimal'''

class RRT_star_complete():
    # initialization of the pygame interface
    def __init__(self, max_nodes, rrt_star_nodes, start, m, roomba_object): 
        
        
        self.window_size = (int(m*1040), int(m*720))
        self.GOAL_RADIUS = m*10
        self.screen = pygame.display.set_mode(self.window_size, 0, 32) # The screen variable is defined as Self because it is used also outside the class in main.py

        pygame.init() # start pygame
        pygame.display.set_caption("RRT + Informed-RRT*")
        
        self.max_nodes = max_nodes
        self.distance_array = np.zeros((max_nodes, max_nodes))

        # The following two variables are used to comunicate outside the class the action to be taken. In particular the are related to what we have to draw 
        # in the pygame window
        self.draw_background_bool = False
        self.draw_edges_weights_bool = False
        self.nodes_explored = 0
        self.counter = 0
        self.rrt_star_nodes = rrt_star_nodes
         
        self.V = np.array([start]) # Vertices array
        self.E = np.array([]) # Edges array
        self.path = np.array([])
        self.roomba = roomba_object
    
    def launch_sim(self, start, goal, obstacles, m):


        self.draw_background_bool = False
        self.draw_edges_weights_bool = False

        
        if self.counter < self.rrt_star_nodes and len(self.V) < self.max_nodes:
            if len(self.path) == 0: # The code computes and reprints the background, edges and vertices only if the RRT* is still running
                self.draw_background_bool = True # Instead of using the draw function to draw inside the class, everything is handled outside in main.py
                
                if self.nodes_explored % 100 != 0:
                    n = np.array([rnd.randint(int(m*160), int(m*880)), rnd.randint(int(m*160), int(m*640)),rnd.uniform(0, 2*np.pi),rnd.uniform(0, 2*np.pi)])  # New random node
                else:
                    n = goal
                
                self.nodes_explored += 1 # Simple variable used outside the classe to print the amount of nodes explored

                if in_free_space(n[:2], self.obs_config): # Is node in free space? 
                    neighbor = closest_neighbor(self.V, n, self.distance_array)
                    
                    if collision_free(n[:2], self.V[int(neighbor), :2], self.obs_config):# is connection collision free?
                        if manipulator_collision_free(n, self.V[int(neighbor)], obstacles, self.roomba, self.distance_array[len(self.V),int(neighbor)]):
                            self.V = np.append(self.V, [n], axis=0)
                            self.E = np.append(self.E, np.array([len(self.V)-1,neighbor]), axis=0)
                        

                self.draw_edges_weights_bool = True # Instead of using the draw function to draw inside the class, everything is handled outside in main.py

                self.path, self.shortest_path_length, self.shortest_path = my_path(start, goal, self.V, self.E.reshape(-1,2), self.GOAL_RADIUS, self.distance_array)            
                

            # Once the first path is found, the algorithm implemented is not anymore RRT, but it becomes RRT informed.
            else:
                self.rrt_star(start, goal, self.obs_config, m)

    # RRT start informed
    def rrt_star(self, start, goal, obstacles, m): 
        
        
        self.draw_background_bool = True # Instead of using the draw function to draw inside the class, everything is handled outside in main.py
            
        n = np.array([rnd.randint(int(m*160), int(m*880)), rnd.randint(int(m*160), int(m*640)),rnd.uniform(0, 2*np.pi),rnd.uniform(0, 2*np.pi)])  # New random node
        self.nodes_explored += 1 # Simple variable used outside the classe to print the amount of nodes explored
            
        if in_free_space(n[:2], obstacles) and in_ellipse(n, start, goal, self.shortest_path_length): # Is node in free space?
            neighbor = closest_neighbor(self.V, n, self.distance_array)                     
            neighbors = get_neighbors(self.distance_array, n, self.V, obstacles, self.roomba,self.counter)
                
            self.V = np.append(self.V,[n],axis=0)
                
            if len(neighbors) > 1:
                new_neighbor = get_new_neighbor(n, neighbors, self.E.reshape(-1,2), self.V, obstacles,self.distance_array,self.roomba)
            else:
                new_neighbor = neighbor
                               
            if collision_free(n[:2], self.V[int(new_neighbor),:2], obstacles):# is connection collision free?
                if manipulator_collision_free(n, self.V[int(neighbor)], obstacles,self.roomba,self.distance_array[len(self.V)-1,int(new_neighbor)]):
                    self.E = np.append(self.E,np.array([len(self.V)-1,new_neighbor]),axis=0)

                    # new_neighborhood contains an array of node indexes. The nodes (indexes) contained are the new nodes to which the vertices in the neighborhood have
                    # have to be connected to obtained the shortest path to those nodes.
                    new_neighborhood = solve_neighborhood(len(self.V)-1, neighbors, self.E.reshape(-1,2), self.V, self.distance_array, obstacles,self.roomba)
                    for i in range(len(new_neighborhood)):
                        if new_neighborhood[i, 1] != 0:
                            self.E[2*int(new_neighborhood[i, 0]) - 1] = new_neighborhood[i, 1]
                
                    self.counter += 1
                    # print(self.counter)
                    
                else:
                    self.V = np.delete(self.V, len(self.V)-1, axis=0)                
            else:
                self.V = np.delete(self.V, len(self.V)-1, axis=0)
                     
        self.draw_edges_weights_bool = True # Instead of using the draw function to draw inside the class, everything is handled outside in main.py
        self.path, self.shortest_path_length, self.shortest_path = my_path(start, goal, self.V, self.E.reshape(-1, 2), self.GOAL_RADIUS, self.distance_array)
        # The path is drawn on the window in the main

    def get_configuration_space_obstacles(self,obstacles):
        self.obs_config = create_obstacles_border(self.roomba, obstacles, self.screen)
