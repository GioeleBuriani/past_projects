import numpy as np
from scipy import interpolate
from geometry import in_free_space
from obstacles import create_obstacles_border
from roomba import Roomba

# Checking for the closest node in the RRT output
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

# Applying the B-spline using sklearn's splprep
def get_spline(data_points):

    tck, u = interpolate.splprep(data_points.transpose(), s=0)
    unew = np.arange(0, 1.00125, 0.00125)
    out = interpolate.splev(unew, tck)
    
    return out

#%%
def traj_smooth_mobile(path, V, screen, obstacles):

    
# This part will deal with the differential drive trajectory smoothing

    # after extracting the co-ordinates, we assign them to variables x and y
    x = path[:, 0]
    y = path[:, 1]

    # we initialize an empty array to store our soon-to-be smoothened points in
    points = []
    
    for i in range(0, len(x)-1):

        # in order to make the curve less wavy, we add a midpoint between every two points
        # since it will naturally be along the desired edge and will help straighten the path
        points = points + [(x[i], y[i]), ((x[i+1] + x[i])/2, (y[i+1] + y[i])/2)]

    i+=1
    points = points + [(x[i], y[i])]
    
    data_points = np.array(points)
    data_points = np.round(data_points,3)

    # setting up obstacles    
    roomba = Roomba()
    obs_config = create_obstacles_border(roomba, obstacles, screen)

    # applying the B-spline from the data points
    out = get_spline(data_points)
    
    midpoint_index = 0

    for i in range(20):
        
        for i in range(len(out[0])-1):
            spline_point = (out[0][i], out[1][i]) # sample spline point from the 2D configuration space representing the differential drive's position
            if in_free_space(spline_point,obs_config)==False: # Checking if the spline points are in free space
                
                midpoint_index = closest_node(spline_point,points) # Getting the index for the midpoint
                
                # If a collision in the configuration space exists, then insert a midpoint between the current node and the previous one, and 
                # between the current node and the one following it
                try:
                    p1x, p1y = points[midpoint_index]
                    p2x, p2y = points[midpoint_index+1]
                    p3x, p3y = points[midpoint_index-1]

                    new_midpoint1 = ((p1x+p2x)/2, (p1y+p2y)/2)
                    new_midpoint2 = ((p1x+p3x)/2, (p1y+p3y)/2)

                    points.insert(midpoint_index+1, new_midpoint1)
                    points.insert(midpoint_index, new_midpoint2)
                    
                # If an error persists, then it is either from the start spline or end spline, so both are clamped to fix this
                except:
                    p1x, p1y = points[0]
                    p2x, p2y = points[1]

                    p3x, p3y = points[-1]
                    p4x, p4y = points[-2]

                    new_midpoint1 = ((p1x+p2x)/2, (p1y+p2y)/2)
                    points.insert(1, new_midpoint1)

                    new_midpoint2 = ((p3x+p4x)/2, (p3y+p4y)/2)
                    points.insert(-1, new_midpoint2)
                
                data_points = np.round(np.array(points),3)
                out = get_spline(data_points)
                    
                break
    
    # We apply the spline extraction again
    out = get_spline(data_points)
        
#%%

# This part of the function deals with the manipulator

    count=0
    steps_arr = []
    steps = 0
    for i in range(len(out[0])):
        for step in path[count:]:
            if np.allclose([out[0][i],out[1][i]],step[:2],rtol=2e-02):
                steps_arr.append(steps)
                count+=1
                steps = 0
        steps+=1
    
# Storing the manipulator joint angle values in j1_arr and j2_arr
    j1_arr = [path[0,2]]
    j2_arr = [path[0,3]]
    for i in range(len(path)-1):
        
        step1 = path[i,2:]
        step2 = path[i+1,2:]
        
        # We also account for the rotation of the manipulator (clockwise/counterclockwise) to reach the reference configuration
        
        #each dir is positive if the connection is counter-clockwise, negative if clockwise
        diff_j1 = (step2[0]-step1[0])%(2*np.pi)
        diff_j2 = (step2[1]-step1[1])%(2*np.pi)
        
        dir_1 = np.sign(np.sin(diff_j1))
        dir_2 = np.sign(np.sin(diff_j2))

        #distance from point_a to point_b for each joint
        dist_j1 = min(abs(diff_j1),2*np.pi-abs(diff_j1))
        dist_j2 = min(abs(diff_j2),2*np.pi-abs(diff_j2))

        for c in range(1,steps_arr[i+1]+1): #checking for collisions num_steps times

            j1_step= (step1[0] + dir_1*dist_j1/steps_arr[i+1]*c)%(2*np.pi)
            j2_step= (step1[1] + dir_2*dist_j2/steps_arr[i+1]*c)%(2*np.pi)
            
            j1_arr.append(j1_step)
            j2_arr.append(j2_step)
    
    return out[0], out[1], j1_arr, j2_arr
#%%

# This function generates a variable used to exploit the function draw_path, also for the smooth trajectory
def smooth_path_generator(reference_x):
    
    smooth_path = [val for val in range(len(reference_x)) for _ in (0, 1)] # Create a list that contains increasing numbers (starting from 0) up to a number equal to the
    # len(reference_x). Each number is present twice (e.g. 0, 0, 1, 1, 2, 2, ...)
    smooth_path = (np.array(smooth_path)[1:-1]) # From list to an array. Moreover the first and last element are dropped since there is no need of having them
    # doubled
    smooth_path = np.flip(smooth_path) # Flipping because the structure wanted by the function 'draw_path' is from the last node to the first
    smooth_path = smooth_path.reshape(-1, 2) # In each row there are the two nodes between which we want to draw a line
    return smooth_path