(define (problem waypoint_following)
    (:domain waypoint_following)
    (:requirements :strips :typing)

    (:objects   tiago - robot
                wp0 wp_table_1 wp_shelf_1 - waypoint
                leftgrip rightgrip - gripper
                AH_hagelslag_melk_tag36_11_00011 AH_hagelslag_puur_tag36_11_00019 AH_hagelslag_aruco_17 AH_hagelslag_aruco_0 - object
    )
    (:init    
        ; INIT STATES
        (visited wp0)
        (robot_at tiago wp0) 
        (free leftgrip)
        (free rightgrip)
        (can_move tiago)
        (object_at AH_hagelslag_melk_tag36_11_00011 wp_table_1)        
        (object_at AH_hagelslag_puur_tag36_11_00019 wp_table_1)            
        (object_at AH_hagelslag_aruco_17 wp_table_1)        
        (object_at AH_hagelslag_aruco_0 wp_table_1)
        (can_pick rightgrip AH_hagelslag_aruco_17)
        (can_pick rightgrip AH_hagelslag_aruco_0)
    )
    
    (:goal (and
        (is_classified AH_hagelslag_aruco_0 wp_shelf_1)
	)
	)
)
