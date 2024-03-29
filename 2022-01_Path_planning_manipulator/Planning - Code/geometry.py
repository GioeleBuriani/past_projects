import numpy as np
from roomba import Roomba

EPSILON=0.01

def my_dist(a,b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def my_dist_joints(a,b): 
    dist=np.array([min(abs(a[0]-b[0]),2*np.pi-abs(a[0]-b[0])),min(abs(a[1]-b[1]),2*np.pi-abs(a[1]-b[1]))])
    return np.sum(dist**2)/100


def in_free_space(point,obs):
    for ob in obs:
        if ob.x<point[0]<ob.x+ob.x_dim and ob.y<point[1]<ob.y+ob.y_dim: 
            return False
    return True

def in_ellipse(q,f1,f2,best_path):
    dist1 = np.sqrt(my_dist(f1,q)+my_dist_joints(f1[2:],q[2:]))
    dist2 = np.sqrt(my_dist(f2,q)+my_dist_joints(f2[2:],q[2:]))
    if dist1 + dist2 > max(best_path,1.25*np.sqrt(my_dist(f1, f2))): 
        return False
    return True



def collision_free(point_1,point_2,obs):
    p1,p2 = sorted([point_1,point_2],key=lambda x: x[1])
    
    
    if p1[0]!=p2[0] and p1[1]!=p2[1]:    
        slope = (p2[0]-p1[0])/(p2[1]-p1[1])
        
        for ob in obs:
            y_intersect = np.zeros(2)
            x_intersect = p1[0] + (ob.y-p1[1]) * slope
            y_intersect[0] = p1[1] + (ob.x-p1[0]) / slope 
            y_intersect[1] = p1[1] + (ob.x + ob.x_dim-p1[0]) / slope
    
            if ob.x< x_intersect < ob.x + ob.x_dim and min(p1[0],p2[0]) < x_intersect < max(p1[0],p2[0]):
                return False
            
            for y in y_intersect:
                if ob.y < y < ob.y + ob.y_dim and p1[1] < y < p2[1]:
                    return False                     
    
    elif p1[0]==p2[0]:
        for ob in obs:
            if p1[1] <= ob.y <= p2[1] and ob.x <= p1[0] <= ob.x + ob.x_dim:
                return False
    
    else:
        for ob in obs:
            if min(p1[0],p2[0]) <= ob.x < max(p1[0],p2[0]) and ob.y <= p1[1] <= ob.y + ob.y_dim:
                return False
    
    
    
    return True


def manipulator_collision_free(point_a, point_b, obs,roomba,distance):
    
    num_steps=int(10+distance//10)
    
    diff_j1 = point_b[2]-point_a[2]
    diff_j2 = point_b[3]-point_a[3]
    diff_x = point_b[0]-point_a[0]
    diff_y = point_b[1]-point_a[1]
    
    #each dir is positive if the connection is counter-clockwise, negative if clockwise
    dir_1=np.sign(np.sin(diff_j1)) 
    dir_2=np.sign(np.sin(diff_j2))
    #distance from point_a to point_b for each joint
    dist_j1 = min(abs(diff_j1),2*np.pi-abs(diff_j1))
    dist_j2 = min(abs(diff_j2),2*np.pi-abs(diff_j2))
    
    for c in range(num_steps): #checking for collisions num_steps times
        
        j1_step= point_a[2] + dir_1*dist_j1/num_steps*c
        j2_step= point_a[3] + dir_2*dist_j2/num_steps*c
        
        x_step = point_a[0]+ diff_x/num_steps*c
        y_step = point_a[1]+ diff_y/num_steps*c
        
        
        x_j1,y_j1,x_j2,y_j2 = roomba.manipulator_FK(x_step, y_step, j1_step, j2_step)
        
        if not collision_free([x_step,y_step], [x_j1,y_j1], obs) or not collision_free([x_j1,y_j1], [x_j2,y_j2], obs):  
            return False
    
    return True






