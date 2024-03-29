import pygame
import numpy as np

def draw_edges_weights(screen, V, E, m):
    for i in range(len(E)):
        pygame.draw.circle(screen, (255, 255, 0) , (V[i + 1, 0],V[i + 1, 1]), m*5)
        pygame.draw.line(screen, (0, 0, 255), V[int(E[i, 0])], V[int(E[i, 1])], width = (int(1)))
        
        
def draw_background(screen, obstacles, start, goal, GOAL_RADIUS, m):
    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (0, 255, 0) , (start[0], start[1]), m*10) #Draw start
    pygame.draw.circle(screen, (255, 0, 0) , (goal[0], goal[1]), GOAL_RADIUS) #Draw goal
    for obs in obstacles:
        obs.draw()
        
def draw_path(screen, path, V, RGB, m):
    for k in range(len(path)):
        pygame.draw.line(screen, RGB, V[int(path[k, 0])], V[int(path[k, 1])], width = (int(m*5)))


def draw_start_and_goal(screen, start, goal, m, GOAL_RADIUS):
    pygame.draw.circle(screen, (0, 255, 0), (start[0], start[1]), m*10) # Draw start
    pygame.draw.circle(screen, (255, 0, 0), (goal[0], goal[1]), GOAL_RADIUS) # Draw goal
    
    if goal[2] == 0:
        pygame.draw.circle(screen, (255, 0, 0), (goal[0] + m*30, goal[1]), m*5)
    elif goal[2] == np.pi/2:
        pygame.draw.circle(screen, (255, 0, 0), (goal[0], goal[1] - m*30), m*5)
    elif goal[2] == -np.pi/2:
        pygame.draw.circle(screen, (255, 0, 0), (goal[0], goal[1] + m*30), m*5)
    elif goal[2] == np.pi:
        pygame.draw.circle(screen, (255, 0, 0), (goal[0] - m*30, goal[1]), m*5)

def draw_counter_nodes(screen, V):
    # The following functions are necessary to print the counter of nodes explored on the screen
    nodes_explored_font = pygame.font.Font(None, 50)
    nodes_explored_surf = nodes_explored_font.render(str(len(V)), 1, (255, 0, 0))
    nodes_explored_pos = [70, 100]
    screen.blit(nodes_explored_surf, nodes_explored_pos)

def rotate_image_around_center(image, rect, angle): # rect is a tuple containing the position of the topleft pixel of image. It can be computed using image.get_rect 
    """Rotate the image while keeping its center."""
    # Rotate the original image without modifying it.
    new_image = pygame.transform.rotate(image, angle)
    # Get a new rect with the center of the old rect.
    rect = new_image.get_rect(center=rect.center)
    return new_image, rect


