(define (domain waypoint_following)

    (:requirements
        :typing
        :durative-actions
        )

    (:types
        waypoint
        robot
        object
        gripper
    )
    
    (:predicates
        (visited ?wp - waypoint)
        (robot_at ?v - robot ?wp - waypoint)
        (object_at ?obj - object ?wp - waypoint)
        (is_holding ?g - gripper ?obj - object)
        (free ?g - gripper)
        (can_pick ?g - gripper ?obj - object)
        (can_move ?v - robot)      
        (is_classified ?obj - object ?wp - waypoint)
    )
    
    (:durative-action move
        :parameters (?v - robot ?from ?to - waypoint)
        :duration (= ?duration 2)
        :condition (and
            (at start (can_move ?v))
            (at start (robot_at ?v ?from))
        )
        :effect (and
            (at end (visited ?to))
            (at end (robot_at ?v ?to))
            (at start(not 
                (can_move ?v)
                ))
            (at start (not
                (robot_at ?v ?from)
            ))
        )
    )

    (:durative-action pick
        :parameters (?v - robot ?wp - waypoint ?obj - object ?g - gripper)
        :duration (= ?duration 2)
        :condition (and
            (at start (robot_at ?v ?wp))
            (at start (object_at ?obj ?wp))
            (at start (can_pick ?g ?obj))
            (at start (free ?g))
        )
        :effect (and
            (at end (robot_at ?v ?wp))
            (at end (is_holding ?g ?obj))
            (at start (not 
                (free ?g)
                )
            )
            (at start (not
                (object_at ?obj ?wp)
                )
            )
            (at end (can_move ?v))
        )
    )
    
    (:durative-action place
        :parameters (?v - robot ?wp - waypoint ?obj -object ?g - gripper)
        :duration (= ?duration 2)
        :condition (and
            (at start (robot_at ?v ?wp))
            (at start (is_holding ?g ?obj))
        )
        :effect (and
            (at start (robot_at ?v ?wp))
            (at end (free ?g))
            (at start (not (is_holding ?g ?obj)))
            (at end (object_at ?obj ?wp))
            (at end (can_move ?v))
            (at end (is_classified ?obj ?wp))
        )
    )
)