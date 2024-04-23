#!bin/bash

## ---------- UPDATE ----------------------------------------------------
# THIS FILE IS NOT IN USE ANYMORE. WE WILL KEEP IT FOR BACKUP
# TO SET THE GOALS NOW THERE IS ROS NODE THAT INTERFACE PDDL AND PROLOG
## ---------- UPDATE ----------------------------------------------------

# This bash scripts updates the problem.pddl through ROSplan

# Uplouding the problem into ROSplan. The goal is still the default one. Once the problem is online, we can add new goals
echo "Generating a Problem"
rosservice call /rosplan_problem_interface/problem_generation_server

# Let's print on the terminal the problem.pddl file to check it before the modifications
echo "Original Problem:"
echo ""
rostopic echo /rosplan_problem_interface/problem_instance -n 1 -p


echo "Updating accoding to the new goals:"
echo ""
# Updating the goals. For each 'rosservice ... update' a new goal is added. 
# First Goal
rosservice call /rosplan_knowledge_base/update "update_type: 1
knowledge:
  knowledge_type: 1
  initial_time: 
    secs: 0
    nsecs: 0
  is_negative: false
  instance_type: ''
  instance_name: ''
  attribute_name: 'object-at'
  values:
    - {key: '?obj', value: 'aruco_cube_333'}
    - {key: '?wp', value: 'wp_cabinet_2'}
  function_value: 0.0
  assign_op: 0
  optimization: ''
  expr:
    tokens: []
  ineq:
    comparison_type: 0
    LHS:
      tokens: []
    RHS:
      tokens: []
    grounded: false"

# Second Goal
rosservice call /rosplan_knowledge_base/update "update_type: 1
knowledge:
  knowledge_type: 1
  initial_time: 
    secs: 0
    nsecs: 0
  is_negative: false
  instance_type: ''
  instance_name: ''
  attribute_name: 'object-at'
  values:
    - {key: '?obj', value: 'aruco_cube_333'}
    - {key: '?wp', value: 'wp_cabinet_2'}
  function_value: 0.0
  assign_op: 0
  optimization: ''
  expr:
    tokens: []
  ineq:
    comparison_type: 0
    LHS:
      tokens: []
    RHS:
      tokens: []
    grounded: false"

#Third Goal
rosservice call /rosplan_knowledge_base/update "update_type: 1
knowledge:
  knowledge_type: 1
  initial_time: 
    secs: 0
    nsecs: 0
  is_negative: false
  instance_type: ''
  instance_name: ''
  attribute_name: 'object-at'
  values:
    - {key: '?obj', value: 'aruco_cube_333'}
    - {key: '?wp', value: 'wp_cabinet_2'}
  function_value: 0.0
  assign_op: 0
  optimization: ''
  expr:
    tokens: []
  ineq:
    comparison_type: 0
    LHS:
      tokens: []
    RHS:
      tokens: []
    grounded: false"

# Fourth Goal
rosservice call /rosplan_knowledge_base/update "update_type: 1
knowledge:
  knowledge_type: 1
  initial_time: 
    secs: 0
    nsecs: 0
  is_negative: false
  instance_type: ''
  instance_name: ''
  attribute_name: 'object-at'
  values:
    - {key: '?obj', value: 'aruco_cube_333'}
    - {key: '?wp', value: 'wp_cabinet_2'}
  function_value: 0.0
  assign_op: 0
  optimization: ''
  expr:
    tokens: []
  ineq:
    comparison_type: 0
    LHS:
      tokens: []
    RHS:
      tokens: []
    grounded: false"

#Fifth Goal
rosservice call /rosplan_knowledge_base/update "update_type: 1
knowledge:
  knowledge_type: 1
  initial_time: 
    secs: 0
    nsecs: 0
  is_negative: false
  instance_type: ''
  instance_name: ''
  attribute_name: 'object-at'
  values:
    - {key: '?obj', value: 'aruco_cube_333'}
    - {key: '?wp', value: 'wp_cabinet_2'}
  function_value: 0.0
  assign_op: 0
  optimization: ''
  expr:
    tokens: []
  ineq:
    comparison_type: 0
    LHS:
      tokens: []
    RHS:
      tokens: []
    grounded: false"

# When the new goals have been added we need to update the problem to apply the changes to the problem.pddl file.
rosservice call /rosplan_problem_interface/problem_generation_server

# Let's print on the terminal the new problem.pddl file to check if the goal are updated
echo "New Problem:"
echo ""
rostopic echo /rosplan_problem_interface/problem_instance -n 1 -p

# Now that the goal is update we can solve the pddl file to find a sequence of actions able to accomplish the task
echo "Planning on the new problem"
rosservice call /rosplan_planner_interface/planning_server

# Let's check the plan found by the solver
echo "The plan found is the following:"
echo ""
rostopic echo /rosplan_planner_interface/planner_output -p

echo "If you want TIAGo to execute the task execute the folliwing commands:"
echo "rosservice call /rosplan_parsing_interface/parse_plan"
echo "rosservice call /rosplan_plan_dispatcher/dispatch_plan"