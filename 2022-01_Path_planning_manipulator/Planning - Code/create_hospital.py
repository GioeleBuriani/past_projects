import numpy as np
from obstacles import FixedObstacle

def create_hospital(screen, m):
    
    # Create the walls rectangles
    out_wall_1 = FixedObstacle(screen, m*159, m*79, m*2, m*562)
    out_wall_2 = FixedObstacle(screen, m*159, m*639, m*722, m*2)
    out_wall_3 = FixedObstacle(screen, m*879, m*159, m*2, m*482)
    out_wall_4 = FixedObstacle(screen, m*239, m*159, m*642, m*2)
    out_wall_5 = FixedObstacle(screen, m*239, m*79, m*2, m*82)
    out_wall_6 = FixedObstacle(screen, m*159, m*79, m*82, m*2)

    in_wall_1 = FixedObstacle(screen, m*239, m*159, m*2, m*162)
    in_wall_2 = FixedObstacle(screen, m*479, m*159, m*2, m*162)
    in_wall_3 = FixedObstacle(screen, m*719, m*399, m*162, m*2)
    in_wall_4 = FixedObstacle(screen, m*639, m*399, m*2, m*242)
    in_wall_5 = FixedObstacle(screen, m*319, m*399, m*2, m*242)

    door_wall_1 = FixedObstacle(screen, m*159, m*159, m*22, m*2)
    door_wall_2 = FixedObstacle(screen, m*219, m*159, m*22, m*2)
    door_wall_3 = FixedObstacle(screen, m*239, m*319, m*102, m*2)
    door_wall_4 = FixedObstacle(screen, m*379, m*319, m*202, m*2)
    door_wall_5 = FixedObstacle(screen, m*619, m*319, m*102, m*2)
    door_wall_6 = FixedObstacle(screen, m*719, m*159, m*2, m*182)
    door_wall_7 = FixedObstacle(screen, m*719, m*379, m*2, m*122)
    door_wall_8 = FixedObstacle(screen, m*719, m*539, m*2, m*102)
    door_wall_9 = FixedObstacle(screen, m*499, m*399, m*142, m*2)
    door_wall_10 = FixedObstacle(screen, m*219, m*399, m*242, m*2)
    door_wall_11 = FixedObstacle(screen, m*159, m*399, m*22, m*2)

    box_wall_1 = FixedObstacle(screen, m*859, m*619, m*2, m*22)
    box_wall_1a = FixedObstacle(screen, m*839, m*619, m*2, m*22)
    box_wall_2 = FixedObstacle(screen, m*859, m*509, m*22, m*2)
    box_wall_3 = FixedObstacle(screen, m*859, m*489, m*22, m*2)
    box_wall_4 = FixedObstacle(screen, m*739, m*399, m*2, m*22)
    box_wall_4a = FixedObstacle(screen, m*759, m*399, m*2, m*22)
    box_wall_5 = FixedObstacle(screen, m*859, m*359, m*22, m*2)
    box_wall_6 = FixedObstacle(screen, m*859, m*339, m*22, m*2)
    box_wall_7 = FixedObstacle(screen, m*719, m*239, m*22, m*2)
    box_wall_8 = FixedObstacle(screen, m*719, m*219, m*22, m*2)
    box_wall_9 = FixedObstacle(screen, m*589, m*159, m*2, m*22)
    box_wall_10 = FixedObstacle(screen, m*569, m*159, m*2, m*22)
    box_wall_11 = FixedObstacle(screen, m*459, m*289, m*22, m*2)
    box_wall_12 = FixedObstacle(screen, m*459, m*269, m*22, m*2)
    box_wall_13 = FixedObstacle(screen, m*239, m*209, m*22, m*2)
    box_wall_14 = FixedObstacle(screen, m*239, m*189, m*22, m*2)
    box_wall_15 = FixedObstacle(screen, m*249, m*619, m*2, m*22)
    box_wall_16 = FixedObstacle(screen, m*269, m*619, m*2, m*22)
    box_wall_17 = FixedObstacle(screen, m*429, m*399, m*2, m*22)
    box_wall_18 = FixedObstacle(screen, m*449, m*399, m*2, m*22)
    box_wall_19 = FixedObstacle(screen, m*339, m*619, m*2, m*22)
    box_wall_19a = FixedObstacle(screen, m*359, m*619, m*2, m*22)
    box_wall_20 = FixedObstacle(screen, m*619, m*559, m*22, m*2)
    box_wall_21 = FixedObstacle(screen, m*619, m*579, m*22, m*2)
    
    

    # Save the rectangles in an array
    obstacles = np.array([out_wall_1, out_wall_2, out_wall_3, out_wall_4, out_wall_5, out_wall_6,
                        in_wall_1, in_wall_2, in_wall_3, in_wall_4, in_wall_5,
                        door_wall_1, door_wall_2, door_wall_3, door_wall_4, door_wall_5, door_wall_6, door_wall_7, door_wall_8, door_wall_9, door_wall_10, door_wall_11,
                        box_wall_1, box_wall_1a, box_wall_2, box_wall_3, box_wall_4, box_wall_4a, box_wall_5, box_wall_6, box_wall_7, box_wall_8, box_wall_9, box_wall_10, box_wall_11, box_wall_12,
                            box_wall_13, box_wall_14, box_wall_15, box_wall_16, box_wall_17, box_wall_18, box_wall_19, box_wall_19a, box_wall_20, box_wall_21])

    return obstacles