(define (domain waypoint_following)

    (:requirements
        :typing
        :disjunctive-preconditions
        :durative-actions
        )

    (:types
        waypoint
        robot
        object
        gripper
        shelf
        location
    )
    
    (:predicates
        (visited ?wp - waypoint)
        (robot-at ?v - robot ?wp - waypoint)
        (object-at ?obj - object ?wp - waypoint)
        (is_holding ?g - gripper ?obj - object)
        (free ?g - gripper)
        (free_loc ?loc - location)
        (occupied_loc ?loc - location)
        (last ?loc - location)
        (previous ?loc1 ?loc2 - location)
        (location-of ?loc - location ?s - shelf)
        (waypoint-of ?g - gripper ?wp - waypoint ?loc - location)
        (placed-on ?obj - object ?s - shelf)
        (stored-at ?obj - object ?loc - location)
        (pick-right ?g - gripper ?obj - object)
        (pick-left ?g - gripper ?obj - object)
        (no_actions_in_exec ?v - robot) ;; This predicate is used to avoid 
        ;; parallel execution of two or more actions. Sometimes this could be 
        ;; possible and advisible to improve the efficiency of the code, but 
        ;; since the simulation is not so reliable we prefer to make tiago do 
        ;; one action at the time.
    )
    
    ;; action to allow the robot to move from one location (or waypoint) to
    ;; another one
    (:durative-action move
        :parameters (?v - robot ?from ?to - waypoint)
        :duration ( = ?duration 2)
        :condition (and
            (at start (robot-at ?v ?from))
            (at start (no_actions_in_exec ?v)) ;; checking if other actions are going on
        )
        :effect (and
            (at end (visited ?to))
            (at end (robot-at ?v ?to))
            
            ;; as soon as the action starts, we update the KB saying that the
            ;; robot is not located anymore in the original position
            (at start (not
                (robot-at ?v ?from)
            ))
            
            ;; as soon as the action starts, we update the KB saying that the an
            ;; action is in execution; in the end the action is not in execution anymore
            (at start (not 
                (no_actions_in_exec ?v)
            ))
            (at end (no_actions_in_exec ?v))
        )
    )
    
    ;; action to control the picking task with right gripper
    (:durative-action pick_right
        :parameters (?v - robot ?wp - waypoint ?obj - object ?g - gripper ?loc - location)
        :duration ( = ?duration 2)
        :condition (and
            (at start (robot-at ?v ?wp))
            (at start (object-at ?obj ?wp))
            (at start (free ?g)) ;; the gripper is free
            (at start (pick-right ?g ?obj))
            (at start (stored-at ?obj ?loc))
            (at start (no_actions_in_exec ?v)) ;; checking if other actions are going on
        )
        :effect (and 
            (at end (is_holding ?g ?obj)) ;; the gripper is holding the object
            (at end (not 
                (free ?g)
            )) ;; the gripper is not free anymore
                    
            (at end (not
                (object-at ?obj ?wp)
            )) ;; once finished picking the object is not in its original location anymore
                    
            (at end (free_loc ?loc)) ;; location of object is free
            (at end (not
                (occupied_loc ?loc)
            )) ;; location of object is not occupied anymore
            (at end (not
                (stored-at ?obj ?loc)
            )) ;; object is not stored at the location anymore
                    
            ;; as soon as the action starts, we update the KB saying that the an
            ;; action is in execution; in the end the action is not in execution anymore
            (at start (not 
                (no_actions_in_exec ?v)
            ))
            (at end (no_actions_in_exec ?v))
        )
    )
    
    ;; action to control the picking task with left gripper
    (:durative-action pick_left
        :parameters (?v - robot ?wp - waypoint ?obj - object ?g - gripper ?loc - location)
        :duration ( = ?duration 2)
        :condition (and
            (at start (robot-at ?v ?wp))
            (at start (free ?g)) ;; the gripper is free
            (at start (object-at ?obj ?wp))
            (at start (pick-left ?g ?obj))
            (at start (stored-at ?obj ?loc))
            (at start (no_actions_in_exec ?v)) ;; checking if other actions are going on
        )
        :effect (and 
            (at end (is_holding ?g ?obj)) ;; the gripper is holding the object
            (at end (not 
                (free ?g)
            )) ;; the gripper is not free anymore
                    
            (at end (not
                (object-at ?obj ?wp)
            )) ;; once finished picking the object is not in its original location anymore
            
            (at end (free_loc ?loc)) ;; location of object is free
            (at end (not
                (occupied_loc ?loc)
            )) ;; location of object is not occupied anymore
            (at end (not
                (stored-at ?obj ?loc)
            )) ;; object is not stored at the location anymore
                    
            ;; as soon as the action starts, we update the KB saying that the an
            ;; action is in execution; in the end the action is not in execution anymore
            (at start (not 
                (no_actions_in_exec ?v)
            ))
            (at end (no_actions_in_exec ?v))
        )
    )
    
    ;; action to control the placing task
    (:durative-action place
        :parameters (?v - robot ?wp - waypoint ?obj - object ?g - gripper ?loc1 ?loc2 - location ?s - shelf)
        :duration ( = ?duration 2)
        :condition (and
            (at start (robot-at ?v ?wp))
            (at start (is_holding ?g ?obj)) ;; the robot is already holding something
    
            (at start (location-of ?loc1 ?s)) 
            (at start (location-of ?loc2 ?s))
            (at start (occupied_loc ?loc1))
            (at start (free_loc ?loc2))
            (at start (previous ?loc1 ?loc2)) ;; find free location in order; for the first position an
                                              ;; imaginary occupied position is added in 'front' of it
            (at start (waypoint-of ?g ?wp ?loc2)) ;; to make sure Tiago uses the right waypoint to place
                                                  ;; something with/at the corresponding gripper/position
            (at start (no_actions_in_exec ?v)) ;; checking if other actions are going on
        )
        :effect (and 
            (at end (free ?g)) ;; object released, hence free gripper
            (at end 
                (not (is_holding ?g ?obj)
            )) ;; the gripper does not hold the object anymore
            (at end (object-at ?obj ?wp)) ;; update the position of the object
                    
            (at end (stored-at ?obj ?loc2)) ;; update the location where the object is stored
            (at end (placed-on ?obj ?s)) ;; update the shelf/table where the object is placed
            
            (at end (occupied_loc ?loc2)) ;; location of object is occupied 
            (at end (not
                (free_loc ?loc2)
            )) ;; location of object is not free anymore
            
            ;; as soon as the action starts, we update the KB saying that the an
            ;; action is in execution; in the end the action is not in execution anymore
            (at start (not 
                (no_actions_in_exec ?v)
            ))
            (at end (no_actions_in_exec ?v))
        )
    )
    
    ;; full shelf - leave cube on table
    (:action full_shelf
        :parameters (?v - robot ?obj - object ?wp - waypoint ?s - shelf ?loc1 ?loc2 ?loc3 - location)
        :precondition (and
            ;; Check if all three locations are occupied
            (location-of ?loc1 ?s)
            (location-of ?loc2 ?s)
            (location-of ?loc3 ?s)
            (occupied_loc ?loc1)
            (occupied_loc ?loc2)
            (occupied_loc ?loc3)
            (previous ?loc1 ?loc2)
            (previous ?loc2 ?loc3)
            (last ?loc3)
            
            (robot-at ?v ?wp) ;; needed to get the waypoint which is used to run the action as a 
                              ;; moved action in the simulation, thereby Tiago tries to move to the 
                              ;; position where he already is and succeds without performing any 
                              ;; action
            (no_actions_in_exec ?v) ;; checking if other actions are going ons
        )
        :effect (and
            (placed-on ?obj ?s) ;; The action returns that the item is placed on the desired 
                                ;; location, although in reality it is not. That is necessary 
                                ;; to get a solution from the PDDL calculator, in reality TIAGo 
                                ;; does not pick the object and leave it on the table. To be 
                                ;; able to check if in the end the object was placed correctly 
                                ;; or not, a new predicate can be added, called 'placed_correctly' 
                                ;; and which switches to TRUE as soon TIAGo places the object on 
                                ;; the shelf.
            
            ;; as soon as the action starts, we update the KB saying that the an
            ;; action is in execution; in the end the action is not in execution anymore
            (not 
                (no_actions_in_exec ?v)
            )
            (no_actions_in_exec ?v)
        )
    )  
)