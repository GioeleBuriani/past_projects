import numpy as np
from geometry import my_dist

def my_path(start, goal, V, E, GOAL_RADIUS, distance_arr):
    path = np.array([])
    shortest_path_length = np.inf
    shortest_path = np.array([])
    for i in range(len(V)):    
        if np.sqrt(my_dist(V[i,:2], goal[:2])) < GOAL_RADIUS:
                pred = i
                path_length = 0
                current_path=np.array([V[int(i)]])
                while pred != 0:
                    path = np.append(path, [pred, E[int(pred-1), 1]])
                    path_length += distance_arr[int(pred),int(E[int(pred-1),1])]
                    pred = E[int(pred-1),1]
                    current_path = np.append(current_path,[V[int(pred)]])
                
                if path_length < shortest_path_length:
                    shortest_path_length=path_length
                    shortest_path=current_path
                    
    return path.reshape(-1,2), shortest_path_length,shortest_path.reshape(-1,4)