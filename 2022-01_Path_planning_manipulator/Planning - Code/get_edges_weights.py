import numpy as np
from geometry import collision_free,my_dist,manipulator_collision_free,my_dist_joints
# Given a neigborhood and a goal (the goal is basically the node under analisys by RRT*), it computes a simplified graph that Dijkstra is going to use to find the 
# shortest path.


def get_edges_weights(goal_ind, neighbors, E, V, dist_arr, obstacles,roomba):
    edges = np.array([])
    weights = []
    
    
    for n in neighbors:    
        if collision_free(V[n,:2],V[goal_ind,:2],obstacles):
            if manipulator_collision_free(V[n],V[goal_ind],obstacles,roomba,dist_arr[n,goal_ind]):
                edges = np.append(edges,np.array([n,0])) 
                prec = n
                weight=0
                
                while prec != 0:
                    weight += dist_arr[int(prec),int(E[int(prec-1),1])]
                    prec = E[int(prec-1),1]
                weights.append(weight)            
                
                edges = np.append(edges,np.array([goal_ind,n]))
                weights.append(np.sqrt(my_dist(V[goal_ind,:2], V[n,:2])+my_dist_joints(V[goal_ind,2:], V[n,2:])))
    
    return edges , weights