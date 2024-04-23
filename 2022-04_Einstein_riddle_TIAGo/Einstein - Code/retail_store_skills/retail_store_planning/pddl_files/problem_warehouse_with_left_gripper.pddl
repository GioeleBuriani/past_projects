(define (problem waypoint_following)
    (:domain waypoint_following)
    (:requirements :strips :typing)

    (:objects   tiago - robot
                leftgrip rightgrip - gripper
		        no_shelf shelf_1 shelf_2 shelf_3 shelf_4 shelf_5 table_3 - shelf
                ;; ADD YOUR OBJECTS AS IN THE CORRESPONDING .yaml FILE - object
                aruco_cube_111 aruco_cube_222 aruco_cube_333 aruco_cube_444 aruco_cube_582 - object
		        ;; ADD YOUR WAYPOINTS AS IN THE CORRESPONDING .yaml FILE - waypoint
		        wp0 wp_table_1 wp_table_2 wp_table_3 - waypoint
		        ;; different locations of the gripper to place cubes with both the left and
		        ;; right gripper into all locations of the shelves
		        wp_cabinet_1 wp_cabinet_1_left wp_cabinet_1_right - waypoint
		        wp_cabinet_2 wp_cabinet_2_left wp_cabinet_2_right - waypoint
		        ;; ADD YOUR WAYPOINTS AS IN THE CORRESPONDING .yaml FILE - location
		        no_location - location
		        ;; The start locations (e.g. shelf_1_start) are imaginary locations ehich are used
		        ;; to define a relation of the imaginary location and the first 'real' location 
		        ;; (e.g. shelf_1_loc_1). The imaginary location is so to say in 'front' of the 
		        ;; first 'real' location. Thereby we do not need to define another place action 
		        ;; for the first location where no (previous) relation would exist
		        shelf_1_start shelf_1_loc_1 shelf_1_loc_2 shelf_1_loc_3 - location
		        shelf_2_start shelf_2_loc_1 shelf_2_loc_2 shelf_2_loc_3 - location
		        shelf_3_start shelf_3_loc_1 shelf_3_loc_2 shelf_3_loc_3 - location
		        shelf_4_start shelf_4_loc_1 shelf_4_loc_2 shelf_4_loc_3 - location
		        shelf_5_start shelf_5_loc_1 shelf_5_loc_2 shelf_5_loc_3 - location
    )
    
    (:init
    ;; DEFINE YOUR INITIAL STATE
        (visited wp0)
        (robot-at tiago wp0)
        (free rightgrip)
        (free leftgrip)
        
        ;; At start no action are already going on
        (no_actions_in_exec tiago) 
        
        ;; Define locations of objects
        (object-at aruco_cube_111 wp_table_2)
        (object-at aruco_cube_222 wp_table_3)
        (object-at aruco_cube_333 wp_table_3)
        (object-at aruco_cube_444 wp_table_3)
        (object-at aruco_cube_582 wp_table_2)
        
        ;; Objects are placed on NO shelf
        (placed-on aruco_cube_111 no_shelf)
        (placed-on aruco_cube_222 no_shelf)
        (placed-on aruco_cube_333 no_shelf)
        (placed-on aruco_cube_444 no_shelf)
        (placed-on aruco_cube_582 no_shelf)
        
        ;; Objects are stored at NO exact location
        (stored-at aruco_cube_111 no_location)
        (stored-at aruco_cube_222 no_location)
        (stored-at aruco_cube_333 no_location)
        (stored-at aruco_cube_444 no_location)
        (stored-at aruco_cube_582 no_location)
        
        ;; Define which locations are free/ occupied
        ;; Only the imaginary prefixed locations are occupied
        (occupied_loc shelf_1_start)
        (occupied_loc shelf_2_start)
        (occupied_loc shelf_3_start)
        (occupied_loc shelf_4_start)
        (occupied_loc shelf_5_start)
        (free_loc shelf_1_loc_1)
        (free_loc shelf_1_loc_2)
        (free_loc shelf_1_loc_3)
        (free_loc shelf_2_loc_1)
        (free_loc shelf_2_loc_2)
        (free_loc shelf_2_loc_3)
        (free_loc shelf_3_loc_1)
        (free_loc shelf_3_loc_2)
        (free_loc shelf_3_loc_3)
        (free_loc shelf_4_loc_1)
        (free_loc shelf_4_loc_2)
        (free_loc shelf_4_loc_3)
        (free_loc shelf_5_loc_1)
        (free_loc shelf_5_loc_2)
        (free_loc shelf_5_loc_3)
        
        ;; Define which cube should be picked with the right gripper
        (pick-right rightgrip aruco_cube_111)
        (pick-right rightgrip aruco_cube_222)
        (pick-right rightgrip aruco_cube_333)
        
        ;; Define which cube should be picked with the left gripper
        (pick-left leftgrip aruco_cube_444)
        (pick-left leftgrip aruco_cube_582)
        
        ;; Define waypoint of locations for right gripper
        (waypoint-of rightgrip wp_cabinet_2_left shelf_1_loc_1)
        (waypoint-of rightgrip wp_cabinet_2 shelf_1_loc_2)
        (waypoint-of rightgrip wp_cabinet_2 shelf_1_loc_3)
        (waypoint-of rightgrip wp_cabinet_2_left shelf_3_loc_1)
        (waypoint-of rightgrip wp_cabinet_2 shelf_3_loc_2)
        (waypoint-of rightgrip wp_cabinet_2 shelf_3_loc_3)
        (waypoint-of rightgrip wp_cabinet_2_left shelf_5_loc_1)
        (waypoint-of rightgrip wp_cabinet_2 shelf_5_loc_2)
        (waypoint-of rightgrip wp_cabinet_2 shelf_5_loc_3)
        (waypoint-of rightgrip wp_cabinet_1_left shelf_2_loc_1)
        (waypoint-of rightgrip wp_cabinet_1 shelf_2_loc_2)
        (waypoint-of rightgrip wp_cabinet_1 shelf_2_loc_3)
        (waypoint-of rightgrip wp_cabinet_1_left shelf_4_loc_1)
        (waypoint-of rightgrip wp_cabinet_1 shelf_4_loc_2)
        (waypoint-of rightgrip wp_cabinet_1 shelf_4_loc_3)
        
        ;; Define waypoint of locations for left gripper
        (waypoint-of leftgrip wp_cabinet_2 shelf_1_loc_1)
        (waypoint-of leftgrip wp_cabinet_2 shelf_1_loc_2)
        (waypoint-of leftgrip wp_cabinet_2_right shelf_1_loc_3)
        (waypoint-of leftgrip wp_cabinet_2 shelf_3_loc_1)
        (waypoint-of leftgrip wp_cabinet_2 shelf_3_loc_2)
        (waypoint-of leftgrip wp_cabinet_2_right shelf_3_loc_3)
        (waypoint-of leftgrip wp_cabinet_2 shelf_5_loc_1)
        (waypoint-of leftgrip wp_cabinet_2 shelf_5_loc_2)
        (waypoint-of leftgrip wp_cabinet_2_right shelf_5_loc_3)
        (waypoint-of leftgrip wp_cabinet_1 shelf_2_loc_1)
        (waypoint-of leftgrip wp_cabinet_1 shelf_2_loc_2)
        (waypoint-of leftgrip wp_cabinet_1_right shelf_2_loc_3)
        (waypoint-of leftgrip wp_cabinet_1 shelf_4_loc_1)
        (waypoint-of leftgrip wp_cabinet_1 shelf_4_loc_2)
        (waypoint-of leftgrip wp_cabinet_1_right shelf_4_loc_3)
        
        ;; Define which location belongs to which shelf
        (location-of shelf_1_start shelf_1)
        (location-of shelf_1_loc_1 shelf_1)
        (location-of shelf_1_loc_2 shelf_1)
        (location-of shelf_1_loc_3 shelf_1)
        (location-of shelf_2_start shelf_2)
        (location-of shelf_2_loc_1 shelf_2)
        (location-of shelf_2_loc_2 shelf_2)
        (location-of shelf_2_loc_3 shelf_2)
        (location-of shelf_3_start shelf_3)
        (location-of shelf_3_loc_1 shelf_3)
        (location-of shelf_3_loc_2 shelf_3)
        (location-of shelf_3_loc_3 shelf_3)
        (location-of shelf_4_start shelf_4)
        (location-of shelf_4_loc_1 shelf_4)
        (location-of shelf_4_loc_2 shelf_4)
        (location-of shelf_4_loc_3 shelf_4)
        (location-of shelf_5_start shelf_5)
        (location-of shelf_5_loc_1 shelf_5)
        (location-of shelf_5_loc_2 shelf_5)
        (location-of shelf_5_loc_3 shelf_5)
        
        ;; Define order of the locations
        (previous shelf_1_start shelf_1_loc_1)
        (previous shelf_1_loc_1 shelf_1_loc_2)
        (previous shelf_1_loc_2 shelf_1_loc_3)
        (previous shelf_2_start shelf_2_loc_1)
        (previous shelf_2_loc_1 shelf_2_loc_2)
        (previous shelf_2_loc_2 shelf_2_loc_3)
        (previous shelf_3_start shelf_3_loc_1)
        (previous shelf_3_loc_1 shelf_3_loc_2)
        (previous shelf_3_loc_2 shelf_3_loc_3)
        (previous shelf_4_start shelf_4_loc_1)
        (previous shelf_4_loc_1 shelf_4_loc_2)
        (previous shelf_4_loc_2 shelf_4_loc_3)
        (previous shelf_5_start shelf_5_loc_1)
        (previous shelf_5_loc_1 shelf_5_loc_2)
        (previous shelf_5_loc_2 shelf_5_loc_3)
        
        ;;Define which locations are the last in the shelf
        (last shelf_1_loc_3)
        (last shelf_2_loc_3)
        (last shelf_3_loc_3)
        (last shelf_4_loc_3)
        (last shelf_5_loc_3)
    )
    
    (:goal (and
    ;; DEFINE YOUR GOAL
        (placed-on aruco_cube_111 shelf_1)
        (placed-on aruco_cube_444 shelf_1)
        ;;(placed-on aruco_cube_333 shelf_1)
        ;;(placed-on aruco_cube_582 shelf_1)
        ;;(placed-on aruco_cube_222 shelf_1)
	))
)