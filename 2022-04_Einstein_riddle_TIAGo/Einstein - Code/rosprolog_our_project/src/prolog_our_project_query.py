#!/usr/bin/env python

import rospy
from rosplan_knowledge_msgs.srv import *
from rosplan_knowledge_msgs.msg import *

import roslib; roslib.load_manifest('rosprolog')
from rosprolog_client import PrologException, Prolog
#import std_srvs.srv


def generate_the_problem():
    rospy.wait_for_service('/rosplan_problem_interface/problem_generation_server') # blocking until the service is available.
    print '\nGenerating a Problem'
    try:
        generating_prob_proxy = rospy.ServiceProxy('/rosplan_problem_interface/problem_generation_server', GenerateProblemService) # Create a callable proxy 
        # to a service. In this case the proxy generated, if it is called, generates the PDDL problem in ROSplan
        
        generating_prob_proxy() # Calling the proxy to generate the PDDL problem
        print '\nProblem generated with success!'
        
        ## PRINT THE PROBLEM ?? rostopic echo /rosplan_problem_interface/problem_instance -n 1 -p
        
    except rospy.ServiceException as exc:
      print 'Service for generatign the problem did not process request:' + str(exc)
    
        

def update_goal(new_goal):
    rospy.wait_for_service('/rosplan_knowledge_base/update') # blocking until the service is available.
    print '\nUpdating the goal'
    try:
        update_goal_proxy = rospy.ServiceProxy('/rosplan_knowledge_base/update', KnowledgeUpdateService) # Create a callable proxy 
        # to a service. In this case the proxy generated, if it is called, updates the PDDL goal in ROSplan
        
        update_goal_proxy(1, new_goal) # Calling the proxy to generate the PDDL problem. The first argument is the type of update. With 1 means update goal.
        # To explore other possible type of updates of the KB execute in the terminal: rossrv show rosplan_knowledge_msgs/KnowledgeUpdateServiceArray
        print '\nGoal updated with success!'
        
    except rospy.ServiceException as exc:
      print 'Service for generatign the problem did not process request:' + str(exc)
      
    

if __name__ == "__main__":
    rospy.init_node('prolog_our_project_query')
    
    # Find the solution with Prolog
    prolog = Prolog()
    query = prolog.query("find_cities(City_tablet, City_watch, City_phone, City_ring, City_bracelet)")
    for solution in query.solutions():
        print 'Object1: tablet, City1: %s' % (solution['City_tablet'])
        print 'Object2: watch, City2: %s' % (solution['City_watch'])
        print 'Object3: phone, City3: %s' % (solution['City_phone'])
        print 'Object4: ring, City4: %s' % (solution['City_ring'])
        print 'Object5: bracelet, City5: %s' % (solution['City_bracelet'])

        # The following lists corresponds to the objects of the simulation.
        cubes = ['aruco_cube_111', 'aruco_cube_222', 'aruco_cube_333', 'aruco_cube_444', 'aruco_cube_582']
        objects = ['bracelet', 'phone', 'ring', 'tablet', 'watch']
        
        generate_the_problem()
        
        for i in range(len(cubes)):
            
            new_goal = KnowledgeItem()
            new_goal.knowledge_type = 1
            new_goal.attribute_name = str('placed-on')
            new_goal.values.append(diagnostic_msgs.msg.KeyValue("?obj", cubes[i]))
            
            city = str(solution['City_' + objects[i]])
            if city == 'amsterdam':
                new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'shelf_1'))
            elif city == 'denHaag':
                new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'shelf_2'))
            elif city == 'eindhoven':
                new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'shelf_3'))
            elif city == 'rotterdam':
                new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'shelf_4'))
            elif city == 'utrecht':
                new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'shelf_5'))
            # if city == 'amsterdam':
            #     new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'wp_cabinet_1'))
            # elif city == 'denHaag':
            #     new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'wp_cabinet_2'))
            # elif city == 'eindhoven':
            #     new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'wp_table_3'))
            # elif city == 'rotterdam':
            #     new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'wp_table_2'))
            # elif city == 'utrecht':
            #     new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", 'wp_table_1'))
            
            update_goal(new_goal)
    
    
    
    query.finish()
    sys.exit(1) # killing the node
    rospy.spin()
