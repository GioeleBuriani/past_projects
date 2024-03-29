import numpy as np

def my_dijkstra(start,goal,edges,weights):
    Q = np.array([[start,0,0]])   #Q[0]:node ; Q[1]:predec ; Q[2]:cost
    Q_seen = np.array([[0,0]])
    
    
    while Q[0,0] != goal: #Keep looking till I find goal
    
        current = Q[0,0]
        
        succ,indx = get_successors(current, edges) #Get successors of current and respective indexes
        
        #add successors of current to Q, c(node) = c(current)+weight(current-->node)
        for i in range(len(succ)): 
            Q = np.append(Q,np.array([[succ[i],current, Q[0,2]+weights[indx[i]]]]),axis=0) 
            
        
        #eliminate edges and weights already seen
        edges=np.delete(edges,indx,axis=0)
        weights = np.delete(weights,indx)
        
        #Adding edges seen to Q_seen
        Q_seen = np.append(Q_seen,np.array([[Q[0,0],Q[0,1]]]),axis=0)
        
        #remove first index from queue (since it's already seen)
        Q = np.delete(Q,0,axis=0)
        
        ##remove edges that go to current from Q, edges, weights (to avoid repetitions)
        rm_indx=[]
        for i in range(len(Q)):
            if Q[i,0]==current:
                rm_indx.append(int(i))
        Q = np.delete(Q,rm_indx,axis=0)
        
        rm_indx=[]
        for i in range(len(edges)):
            if edges[i,0]==current:
                rm_indx.append(int(i))
                
        edges = np.delete(edges,rm_indx,axis=0)
        weights = np.delete(weights,rm_indx)
         
                 
        #sort Q
        Q= Q.reshape(-1,3)
        Q = np.array(sorted(Q,key= lambda x:x[2]))
        
    Q_seen = np.append(Q_seen,np.array([[Q[0,0],Q[0,1]]]),axis=0)
    path = get_dijkstra_path(Q_seen)    
    
    return path

def get_dijkstra_path(Q_seen):
    prec=Q_seen[-1,1]
    prec_ind = -1
    path = np.array([[Q_seen[-1,0],prec]])
    while prec != 0:
        for i in range(len(Q_seen)):
            if Q_seen[i,0] == prec:
                prec_ind = i
        path = np.append(path,np.array([Q_seen[prec_ind]]),axis=0)
        prec = Q_seen[prec_ind,1]
    return path

def get_successors(node,edges):
    succ = []
    indx = []
    for i in range(len(edges)):
        if edges[i,1] == node:
            succ.append(int(edges[i,0]))
            indx.append(int(i))
    return succ,indx
    









