# Code for the generation of a Sierpiński triangle

### Imports ###
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter


### Functions ###

# Function to generate the initial points in the Sierpiński triangle.
def generate_initial_points(center, side_length):

    # Calculate the height of the equilateral triangle.
    height = (math.sqrt(3) / 2) * side_length
    
    # Calculate the coordinates of the three vertices of the triangle.
    point_1 = (center[0], center[1] + (2/3) * height)
    point_2 = (center[0] + side_length / 2, center[1] - height / 3)
    point_3 = (center[0] - side_length / 2, center[1] - height / 3)
    
    # Return the list of vertices.
    return [point_1, point_2, point_3]


# Function to check if a point is within a triangle.
def is_within_triangle(point, points):
    
        # Extract the x and y coordinates of each vertex into separate lists.
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
    
        # Calculate the area of the triangle.
        triangle_area = 0.5 * abs((x_coords[1] - x_coords[0]) * (y_coords[2] - y_coords[0]) - (x_coords[2] - x_coords[0]) * (y_coords[1] - y_coords[0]))
    
        # Calculate the area of the three sub-triangles formed by the point and the vertices.
        sub_triangle_1_area = 0.5 * abs((x_coords[0] - point[0]) * (y_coords[1] - point[1]) - (x_coords[1] - point[0]) * (y_coords[0] - point[1]))
        sub_triangle_2_area = 0.5 * abs((x_coords[1] - point[0]) * (y_coords[2] - point[1]) - (x_coords[2] - point[0]) * (y_coords[1] - point[1]))
        sub_triangle_3_area = 0.5 * abs((x_coords[2] - point[0]) * (y_coords[0] - point[1]) - (x_coords[0] - point[0]) * (y_coords[2] - point[1]))
    
        # Check if the sum of the areas of the sub-triangles is equal to the area of the triangle.
        if triangle_area == sub_triangle_1_area + sub_triangle_2_area + sub_triangle_3_area:
            return True
        else:
            return False


# Function to generate the first point that falls within the triangle
def generate_first_point(points):

    # Calculate the bounding box
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    # Generate a random point within the bounding box
    point = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

    # Check if the point is within the triangle
    if is_within_triangle(point, points):
        return point
    else:
        return generate_first_point(points)


# Function to calculate the middle point between two points
def calculate_middle_point(point_1, point_2):
    return ((point_1[0] + point_2[0]) / 2, (point_1[1] + point_2[1]) / 2)


# Function to generate the next point in the Sierpiński triangle
def generate_next_point(points):
    return calculate_middle_point(points[random.randint(0, 2)], points[-1])



### Visualization ###

# Create a figure and axis for the animation
fig, ax = plt.subplots()

# Set the limits of the plot
ax.set_xlim(-67.8, 67.8)
ax.set_ylim(-33.2, 62.1)

# Generate the initial points
points = generate_initial_points((0, 0), 100)
points.append(generate_first_point(points))

# Define a function to update the plot with each new point
def update(frame):

    if len(points) >= 10000:
        ani.event_source.stop() # stop the animation when the number of points is equal to or greater than 10000
        return []

    if frame > 0:
        for i in range(round(frame/10)):
            points.append(generate_next_point(points))

    ax.clear()
    ax.set_xlim(-67.8, 67.8)
    ax.set_ylim(-33.2, 62.1)
    ax.set_aspect('equal')

    # Plot all the points generated so far
    ax.scatter([p[0] for p in points], [p[1] for p in points], s=0.5, color='black')

    # Add text annotation for the point counter
    ax.text(-60, 50, f'Points: {len(points)}', fontsize=10)

    # Return an empty iterable to prevent the plot from showing a second time
    return []

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=1000, interval=10, repeat=False)

# Save the animation by creating a new .mp4 file
# Save the animation as a GIF
writer = animation.PillowWriter(fps=60)
ani.save('sierpinski_triangle.gif', writer=writer)

# Show the plot
plt.show()