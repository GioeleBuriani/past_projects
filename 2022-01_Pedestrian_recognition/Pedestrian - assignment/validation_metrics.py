import math

import numpy as np


# Compute min distance to obstacles
def min_dist_to_obstacles(pedestrians, candidate):
    # number of timesteps
    nsteps = min(candidate.states.size, len(pedestrians[0].prediction))
    min_dist = 1e6

    collisions = 0

    # Model vehicle with N discs
    n_discs = 15  # The number of collision disc that model the vehicle region
    vehicle_length = 4.5
    vehicle_width = 2.0

    # Let's position the discs equally spaced
    # Then define the offset
    disc_offset = np.zeros(n_discs)
    for disc in range(n_discs):
        disc_offset[disc] = -vehicle_length / 2.0 + (vehicle_length) / (n_discs * 2.0) + vehicle_length / n_discs * disc

    disc_radius = math.sqrt((vehicle_length / n_discs / 2.0) ** 2 + (vehicle_width / 2.0) ** 2)
    safe_dist_to_ped = disc_radius + 1.0

    # print('Disc offsets (from the vehicle center): {} m\nDisc Radius: {} m'.format(disc_offset, disc_radius))

    # compute cost for each time step
    for step in range(nsteps):

        s = candidate.states[step]


        in_collision = False
        for pedestrian in pedestrians:
            ox = pedestrian.prediction[step, 0]
            oy = pedestrian.prediction[step, 1]

            for d in range(n_discs):
                disc_x = s.x + disc_offset[d] * np.sin(s.theta)
                disc_y = s.y + disc_offset[d] * np.cos(s.theta)

                # distance to object
                obstacle_dist = math.sqrt((disc_x - ox) ** 2 + (disc_y - oy) ** 2)
                if obstacle_dist < min_dist:
                    min_dist = obstacle_dist
                    # print("{}: ({}, {}) \t| ({}, {}) \t {}".format(d, disc_x, disc_y, ox, oy, min_dist))

                if obstacle_dist < safe_dist_to_ped and (not in_collision):
                    collisions += 1
                    in_collision = True

    return min_dist, collisions


# Compute progress to goal
def progress_to_goal(s_candidate, max_travel):
    return (s_candidate.states[-1].y - s_candidate.states[0].y) / (max_travel.y - s_candidate.states[0].y)


# Compute lateral offset
def lateral_offset(s_candidate, road):
    return (s_candidate.states[-1].x - road.init_x) / road.rwidth


# Compute off road time
def check_offroad(s_candidate, road, dt):
    # number of timesteps
    nsteps = s_candidate.states.size
    offroads = 0

    half_width = 2.25
    half_length = 1

    offroad = lambda x: x > road.init_x + road.rwidth / 2 or x < road.init_x - road.rwidth / 2

    # compute cost for each time step
    for step in range(nsteps):

        s = s_candidate.states[step]
        pos = np.array([s.x, s.y])
        theta = s.theta + np.pi / 2
        rotation_matrix = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])

        # Check if the x of one of the vehicle corners is outside of the road
        top_right = pos + rotation_matrix.dot(np.array([half_width, half_length]).transpose())
        if offroad(top_right[0]):
            offroads += 1
            continue

        top_left = pos + rotation_matrix.dot(np.array([-half_width, half_length]).transpose())
        if offroad(top_left[0]):
            offroads += 1
            continue

        bot_right = pos + rotation_matrix.dot(np.array([half_width, -half_length]).transpose())
        if offroad(bot_right[0]):
            offroads += 1
            continue

        bot_left = pos + rotation_matrix.dot(np.array([-half_width, -half_length]).transpose())
        if offroad(bot_left[0]):
            offroads += 1
            continue

    return offroads * dt


def score(validation):

    points_progress = 100
    lateral_penalty = 10

    collision_penalty = 200
    intrusion_penalty = 50
    offroad_penalty = 150
    safe_distance = 3.0

    score = 0
    score += validation["progress"] * points_progress
    score -= collision_penalty * (validation["collisions"] > 0)
    score -= validation["lat_offset"] * lateral_penalty
    score -= abs(min(validation["min_distance"] - safe_distance, 0.0) / (-safe_distance) * intrusion_penalty)

    if validation["offroads"] > 0 and validation["offroads"] < 1.0:
        score -= offroad_penalty / 4.0
    elif validation["offroads"] > 0 and validation["offroads"] < 5.0:
        score -= offroad_penalty / 2.0
    elif validation["offroads"] > 0:
        score -= offroad_penalty

    return score
