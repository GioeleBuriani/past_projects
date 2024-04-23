# README
## Introduction
In this file, we will list all the steps necessary to run the simulation we implement correctly. Before proceeding with the steps, here there are some prerequisites you should have on your pc for the simulation:
- Ubuntu 18.04
- singularity ro47014-22-3.simg (follow Practicum 1.2) (Note: the singularity has to stay in your home directory)
- Knowrob: *installed in the folder ~/ro47014_ws*
- Rosprolog: *installed in the folder ~/ro47014_ws*
- Rosplan (Follow Practicum 5)
- Terminator

Once the aforementioned packages/programs are installed we can proceed with specific steps for our simulation.

## Phase 1
**Phase 1** has to be executed only once. If you have already followed **phase 1** once, go directly **to phase 2**.

- open a terminal window
- execute the followings in the terminal: 
    - mkdir -p ~/ro47014_ws/src
    - cd ~/ro47014_ws/src
    - git clone https://gitlab.tudelft.nl/cor/ro47014/2022_course_projects/group_06/retail_store_skills.git
    - git clone https://gitlab.tudelft.nl/cor/ro47014/2022_course_projects/group_06/retail_store_simulation.git
    - git clone https://gitlab.tudelft.nl/cor/ro47014/2022_course_projects/group_06/bash_scripts.git
    - git clone https://gitlab.tudelft.nl/cor/ro47014/2022_course_projects/group_06/rosprolog_our_project.git

After these commands, you should have a folder in your home directory called 'ro47014_ws', inside a folder called 'src', and inside it four different folders: 
1. **retail_store_skills** &#8594; useful packages for the simulation, including the pddl files.
2. **retail_store_simulation** &#8594; this package contains gazebo worlds of (parts of) retail stores, as well as launch files to launch gazebo with TIAGo in it.
3. **bash_scripts** &#8594; it contains some bash scripts (explained later) to automatize some setup procedures of the simulation.
4. **rosprolog_our_project** &#8594; it mainly contain a ros_node (prolog_our_project_query.py) that calls Prolog to solve 'our_riddle.pl' (the riddle on which the project is based on). According to the solution, the node sets the goals of the pddl problem file.

## Phase 2
The following instruction can be executed every time you want to run the simulation.

- open a terminator window
- execute the followings in the terminator window: 
    - cd ~/ro47014_ws/src/bash_scripts
    - source run_singularity.sh
    - ctrl + shift + E (To split the window vertically)
    - ctrl + shift + O (To split the window horizontally)
- In the new 2 section of the terminal execute again 'source run_singularity.sh'

Now you should have three terminals in the same window. So in the following instruction, you will find the commands divide per terminal. Execute them in order, 

**TERMINAL 1**
- source prepare_tiago_build_catkin.sh
- roslaunch retail_store_simulation simulation.launch world:=krr_warehouse tuck_arm:=true rviz:=true gzclient:=true

Now follow TERMINAL 2.

**TERMINAL 2**
- source prepare_tiago_no_build.sh
- roslaunch retail_store_planning rosplan_warehouse.launch

Wait until you see in TERMINAL 2 'Ready to receive', then go to TERMINAL 3.

**TERMINAL 3**
- source prepare_tiago_no_build.sh 
- roslaunch rosprolog_our_project prolog_our_project_query.launch
- WAIT FOR THE ERROR SYSTEM (exit code 1)
- press ctrl + C

The problem should now be updated with the new goal, it is possible to see the updates printed out in TERMINAL 2.

Now again, in this same terminal execute:
- cd ~/ro47014_ws/src/bash_scripts
- source solve_the_planning.sh

After this command, the simulation should start, and TIAGo will organize the post office.


## Extra Useful Commands for the Simulation
To get some insight into what is going on during the steps above open a fourth terminal in terminator.

**TERMINAL 4**
Execute the following commands in any case:
- cd ~/ro47014_ws/src/bash_scripts
- source run_singularity.sh
- source prepare_tiago_no_build.sh

Then, here below you find some optional commands to do some checks on the code (you do not need to execute all of them, nor in the same order are listed below):

1. If you want to see the pddl problem that is currently in ROSplan:
    - rostopic echo /rosplan_problem_interface/problem_instance -n 1 -p
2. If you want to see the plan found by ROSplan to execute the task:
    - rostopic echo /rosplan_planner_interface/planner_output -p
