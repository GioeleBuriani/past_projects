#!/usr/bin/env python

# This node is used to modify the goals of the PDDL problem online.

import rospy
import os
from rosplan_knowledge_msgs.srv import *
from rosplan_knowledge_msgs.msg import *


# This function removes a goal from the ones already present in the problem
def remove_goal(old_goal):
    rospy.wait_for_service('/rosplan_knowledge_base/update') # blocking until the service is available.
    print '\nRemoving the goal'
    try:
        remove_goal_proxy = rospy.ServiceProxy('/rosplan_knowledge_base/update', KnowledgeUpdateService) # Create a callable proxy to a service.
        # In this case the proxy generated, if it is called, removes a PDDL goal in ROSplan.
        
        remove_goal_proxy(3, old_goal) # Calling the proxy to generate the PDDL problem. The first argument is the type of update. With 3 means remove goal.
        # To explore other possible type of updates of the KB execute in the terminal: rossrv show rosplan_knowledge_msgs/KnowledgeUpdateServiceArray
        print '\nGoal removed with success!'
        
    except rospy.ServiceException as exc:
      print 'Service for removing the goal did not process request:' + str(exc)
    
# This function adds a new goal to the problem
def add_goal(new_goal):
    rospy.wait_for_service('/rosplan_knowledge_base/update') # blocking until the service is available.
    print '\nAdding the goal'
    try:
        add_goal_proxy = rospy.ServiceProxy('/rosplan_knowledge_base/update', KnowledgeUpdateService) # Create a callable proxy to a service.
        # In this case the proxy generated, if it is called, adds a PDDL goal in ROSplan
        
        add_goal_proxy(1, new_goal) # Calling the proxy to generate the PDDL problem. The first argument is the type of update. With 1 means add goal.
        # To explore other possible type of updates of the KB execute in the terminal: rossrv show rosplan_knowledge_msgs/KnowledgeUpdateServiceArray
        print '\nGoal added with success!'
        
    except rospy.ServiceException as exc:
      print 'Service for adding the goal did not process request:' + str(exc)


if __name__ == "__main__":
  rospy.init_node('update_goals')


  # Get the type of product to be placed from the parameter
  # leave_product = rospy.get_param('leave_product')
  place_product = rospy.get_param('/server/product')


  # Manage the objects to be placed back

  # Get the information of the previous goal from the parameters (if there is any previous goal) and remove it
  if rospy.has_param('pddl_goal_kt'):
    old_goal = KnowledgeItem()
    old_goal.knowledge_type = rospy.get_param('pddl_goal_kt')
    old_goal.attribute_name = rospy.get_param('pddl_goal_an')
    old_goal.values.append(diagnostic_msgs.msg.KeyValue(rospy.get_param('pddl_goal_vt_1'), rospy.get_param('pddl_goal_vn_1')))
    old_goal.values.append(diagnostic_msgs.msg.KeyValue(rospy.get_param('pddl_goal_vt_2'), rospy.get_param('pddl_goal_vn_2')))
    # Call the function to remove the goal
    remove_goal(old_goal)
  
  # We define the new goal to be achieved when a misplaced product is recognized
  # Define the characteristics of a goal to be added and store their values in a parameter
  new_goal = KnowledgeItem()
  new_goal.knowledge_type = 1
  rospy.set_param('pddl_goal_kt', 1)
  new_goal.attribute_name = str('is_classified')
  rospy.set_param('pddl_goal_an', str('is_classified'))
  # First product
  if place_product == 0:
    new_goal.values.append(diagnostic_msgs.msg.KeyValue("?obj", "AH_hagelslag_aruco_0"))
    rospy.set_param('pddl_goal_vt_1', "?obj")
    rospy.set_param('pddl_goal_vn_1', "AH_hagelslag_aruco_0")
    new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", "wp_shelf_1"))
    rospy.set_param('pddl_goal_vt_2', "?wp")
    rospy.set_param('pddl_goal_vn_2', "wp_shelf_1")
  # Second product
  elif place_product == 17:
    new_goal.values.append(diagnostic_msgs.msg.KeyValue("?obj", "AH_hagelslag_aruco_17"))
    rospy.set_param('pddl_goal_vt_1', "?obj")
    rospy.set_param('pddl_goal_vn_1', "AH_hagelslag_aruco_17")
    new_goal.values.append(diagnostic_msgs.msg.KeyValue("?wp", "wp_shelf_1"))
    rospy.set_param('pddl_goal_vt_2', "?wp")
    rospy.set_param('pddl_goal_vn_2', "wp_shelf_1")
  # We add the goal
  add_goal(new_goal)

 
  # Write the bash command to generate and execute the plan
  os.system("rosservice call /rosplan_problem_interface/problem_generation_server")
  os.system("rosservice call /rosplan_planner_interface/planning_server")
  os.system("rosservice call /rosplan_parsing_interface/parse_plan")
  os.system("rosservice call /rosplan_plan_dispatcher/dispatch_plan")

  
  sys.exit(1) # killing the node

  rospy.spin()

  
