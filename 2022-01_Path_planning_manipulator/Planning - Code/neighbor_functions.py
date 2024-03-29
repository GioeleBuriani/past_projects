import numpy as np
from geometry import my_dist,my_dist_joints,collision_free,manipulator_collision_free
from graph_search import my_dijkstra
from get_edges_weights import get_edges_weights


# It finds the neighbor and saves the distances among the nodes in a matrix in order to calculate it just once        
def closest_neighbor(V, n, dist_arr):
    min_dist = np.inf # at the beginning is set to infinity because we have no limitation on the minimum distance for the neighbor
    n_ind = len(V)
    for i in range(n_ind):
        dist_arr[n_ind, i] = dist_arr[i, n_ind] = np.sqrt(my_dist(V[i,:2], n[:2]) + my_dist_joints(V[i,2:], n[2:]))  
        if dist_arr[n_ind, i] < min_dist:
            min_dist = dist_arr[n_ind, i]
            neighbor = i
    return neighbor


# It finds the neighbors in the neighborhood defined thro(ugh a radius
def get_neighbors(dist_arr, q, V, obstacles,roomba,n):
    neighbors = np.array([-1])
    q_ind = len(V)
    neighborhood_radius = 100*(np.log(n+1)/(n+1))**0.25
    
    for i in range(q_ind):
        if dist_arr[q_ind,i] < neighborhood_radius:
            if collision_free(V[i,:2],q[:2],obstacles):
                if manipulator_collision_free(q,V[i],obstacles,roomba,dist_arr[q_ind,i]):
                    neighbors = np.append(neighbors,i)
                
    return np.delete(neighbors,0,axis=0)


# From the neighborhood the function selects the node that gives the shortest connection from start. It can happen that the connection thru the closest neighbor
# is not the fastest way to get from the start to the node.
def get_new_neighbor(q, neighbors, E, V, obstacles, distance_array,roomba):
    edges, weights = get_edges_weights(len(V)-1, neighbors, E, V, distance_array, obstacles,roomba)
    path = my_dijkstra(0,len(V)-1, edges.reshape(-1,2), weights)
    new_neighbor = path[0,1] # path[0, 1] is the neighbor to which the node under analisys has to be connected to obtained the shortest connection from start
    
    return new_neighbor

# Every node in the neighborhood is analyzed to understand if there is a shortest way to reach him now that a new node has been added to the list
def solve_neighborhood(q,neighbors,E,V,dist_arr,obstacles,roomba):
    new_neighborhood = np.array([])
    for n in neighbors:
        n_neighbors = np.append(neighbors[neighbors!=n],q)
        
        edges,weights = get_edges_weights(n,n_neighbors,E,V,dist_arr,obstacles,roomba)
        prec = n
        weight=0
            
        while prec != 0:
            weight += dist_arr[int(prec),int(E[int(prec-1),1])]
            prec = E[int(prec-1),1]
        
        
        edges = np.append(edges,np.array([n,0]))
        weights.append(weight)
        
        path=my_dijkstra(0,n,edges.reshape(-1,2),weights)
        new_neighborhood = np.append(new_neighborhood,path[0])
        
    return new_neighborhood.reshape(-1,2)
