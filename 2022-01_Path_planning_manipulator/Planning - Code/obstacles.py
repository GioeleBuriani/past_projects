import pygame


class FixedObstacle:
    def __init__(self, screen, x, y, x_dim, y_dim):  # Gather obstacle position and dimension
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x = x
        self.y = y
        self.color = (0, 0, 0)    # Determine obstacle's colour
        self.screen = screen    # Define the screen on which the obstacle has to appear
    def draw(self):
        obs_rect = pygame.Rect(self.x,self.y,self.x_dim,self.y_dim) # Create a rectangle for the obstacle
        pygame.draw.rect(self.screen, self.color, obs_rect) # Draw the rectangle

def create_obstacles_border(robot, obstacles, screen):  # Creates borders for the obstacles depending on the robot configuration
    obstacles_border = []    # Create an array equal to the obstacles
    for i in range(obstacles.shape[0]):
        obstacles_border.append(FixedObstacle(screen, 
                                            obstacles[i].x - robot.radius, obstacles[i].y - robot.radius,   # Define border's position
                                            obstacles[i].x_dim + 2*robot.radius, obstacles[i].y_dim + 2*robot.radius))   # Define border's dimension
    return obstacles_border