##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import numpy as np 

orientation_mapping = {
    'left and below': 0,
    'right and above': 1,
    'left and above': 2,
    'right and below': 3,
    'x_aligned and above': 4,
    'x_aligned and below': 5,
    'left and y_aligned': 6,
    'right and y_aligned': 7,
    'x_aligned and y_aligned': 8
}

x_orientation_mapping = {
    'left': 0,
    'right': 1,
    'below': 2,
    'None': 3,
}

alien_orientation_mapping = {
    'above': 0,
    'below': 1,
    'right': 2,
    'left': 3,
    'above and right': 4,
    'above and left': 5,
    'below and right': 6,
    'below and left': 7,
    'encircled': 8
}

def orientation_to_one_hot(orientation: np.ndarray):
    orientation_mapped = np.vectorize(orientation_mapping.get)(orientation)
    # Create one-hot encoding
    num_classes = len(orientation_mapping)
    one_hot_encoded = np.eye(num_classes)[orientation_mapped]
    # Flatten the one-hot encoded matrix
    return one_hot_encoded.flatten()

def x_orientation_to_one_hot(orientation: np.ndarray):
    orientation_mapped = np.vectorize(x_orientation_mapping.get)(orientation)
    # Create one-hot encoding
    num_classes = len(x_orientation_mapping)
    one_hot_encoded = np.eye(num_classes)[orientation_mapped]
    # Flatten the one-hot encoded matrix
    return one_hot_encoded.flatten()

def alien_orientation_to_one_hot(orientation: np.ndarray):
    orientation_mapped = np.vectorize(alien_orientation_mapping.get)(orientation)
    # Create one-hot encoding
    num_classes = len(alien_orientation_mapping)
    one_hot_encoded = np.eye(num_classes)[orientation_mapped]
    # Flatten the one-hot encoded matrix
    return one_hot_encoded.flatten()


def get_orientation(player_position: np.ndarray, other_positions: np.ndarray):
    if other_positions.ndim > 1:
        delta_x = player_position[0] - other_positions[:, 0]
        delta_y = player_position[1] - other_positions[:, 1]
    else:
        delta_x = player_position[0] - other_positions[0]
        delta_y = player_position[1] - other_positions[1]
    delta = np.zeros_like(delta_x).astype(f'<U50')
    delta[(delta_x < 0) & (delta_y < 0)] = 'left and below'
    delta[(delta_x > 0) & (delta_y > 0)] = 'right and above'
    delta[(delta_x < 0) & (delta_y > 0)] = 'left and above'
    delta[(delta_x > 0) & (delta_y < 0)] = 'right and below'
    delta[(delta_x == 0) & (delta_y > 0)] = 'x_aligned and above'
    delta[(delta_x == 0) & (delta_y < 0)] = 'x_aligned and below'
    delta[(delta_x < 0) & (delta_y == 0)] = 'left and y_aligned'
    delta[(delta_x > 0) & (delta_y == 0)] = 'right and y_aligned'
    delta[(delta_x == 0) & (delta_y == 0)] = 'x_aligned and y_aligned'
    return delta

def get_x_orientation(player_x: float, object_x: float):
    if player_x > object_x:
        return 'right'
    elif player_x < object_x:
        return 'left'
    else:
        return 'below'

def gopher_extraction(positions: np.ndarray, prev_positions: np.ndarray, is_mixed: bool) -> np.ndarray:
    player_position = positions[0]
    gopher_position = positions[1]
    delta_player_position = player_position - prev_positions[0]
    delta_gopher_position = gopher_position - prev_positions[1]
    gopher_distance = np.linalg.norm(player_position - gopher_position)
    gopher_orientation = get_orientation(player_position, gopher_position)
    empty_block_first_idx = 5
    empty_blocks = positions[empty_block_first_idx:]
    unique_x, unique_idx = np.unique(empty_blocks[:, 0], return_index=True)
    counts = np.sum(empty_blocks[:, 0][:, None] == empty_blocks[unique_idx, 0], axis=0)
    counts[empty_blocks[unique_idx][:, 0] == 0] = 0
    aligned_blocks_exist = (counts == 3).any()
    almost_aligned_block_exist = (counts == 2).any()

    nearest_aligned_pos = np.array([0, 0])
    aligned_orientation = 'None'
    aligned_dist = 0
    if aligned_blocks_exist:
        aligned_x = unique_x[counts == 3]
        aligned_pos = empty_blocks[np.isin(empty_blocks[:, 0], aligned_x)]
        aligned_dist = np.linalg.norm(player_position - aligned_pos, axis=1)
        nearest_aligned_pos = aligned_pos[np.argmin(aligned_dist)]
        aligned_dist = np.min(aligned_dist)
        aligned_orientation = get_x_orientation(player_position[0], aligned_x[0])

    nearest_almost_aligned_pos = np.array([0, 0])
    almost_aligned_orientation = 'None'
    almost_aligned_dist = 0
    if almost_aligned_block_exist:
        almost_aligned_x = unique_x[counts == 2]
        almost_aligned_pos = empty_blocks[np.isin(empty_blocks[:, 0], almost_aligned_x)]
        almost_aligned_dist = np.linalg.norm(player_position - almost_aligned_pos, axis=1)
        nearest_almost_aligned_pos = almost_aligned_pos[np.argmin(almost_aligned_dist)]
        almost_aligned_dist = np.min(almost_aligned_dist)
        almost_aligned_orientation = get_x_orientation(player_position[0], almost_aligned_x[0])
    
    block_count_below = (player_position[0] == empty_blocks[:, 0]).sum()
    prev_block_count_below = (prev_positions[0, 0] == prev_positions[empty_block_first_idx:, 0]).sum()
    delta_prev_count_below = block_count_below - prev_block_count_below
    
    if is_mixed:
        return np.array([player_position[0], player_position[1], delta_player_position[0], delta_player_position[1],
                        gopher_position[0], gopher_position[1], delta_gopher_position[0], delta_gopher_position[1], gopher_distance, 
                        str(gopher_orientation),str(aligned_blocks_exist), nearest_aligned_pos[0], nearest_aligned_pos[1],
                        aligned_dist, aligned_orientation, str(almost_aligned_block_exist), nearest_almost_aligned_pos[0], 
                        nearest_almost_aligned_pos[1], almost_aligned_dist, almost_aligned_orientation, block_count_below, prev_block_count_below, delta_prev_count_below], dtype=object)
    
    return np.concatenate([player_position, delta_player_position, gopher_position, delta_gopher_position, np.array([gopher_distance]), 
                           orientation_to_one_hot(gopher_orientation), np.array([int(aligned_blocks_exist)]), nearest_aligned_pos,
                           np.array([aligned_dist]), x_orientation_to_one_hot(aligned_orientation), np.array([int(almost_aligned_block_exist)]),
                           nearest_almost_aligned_pos, np.array([almost_aligned_dist]), x_orientation_to_one_hot(almost_aligned_orientation), np.array([block_count_below, prev_block_count_below, delta_prev_count_below])], axis=0, dtype=np.single)

def breakout_extraction(positions: np.ndarray, prev_positions: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    ball_position = positions[1]
    prev_ball_position = prev_positions[1]
    # return player_position
    ball_distance = np.linalg.norm(player_position - ball_position)
    prev_ball_distance = np.linalg.norm(prev_positions[0] - prev_ball_position)
    ball_orientation = get_orientation(player_position, ball_position)
    ball_velocity = ball_position - prev_ball_position
    n_rows  = 50

    empty_block_first_idx = 2
    empty_blocks = positions[empty_block_first_idx:]
    unique_x, unique_idx = np.unique(empty_blocks[:, 0], return_index=True)
    counts = np.sum(empty_blocks[:, 0][:, None] == empty_blocks[unique_idx, 0], axis=0)
    counts[empty_blocks[unique_idx][:, 0] == 0] = 0
    columns = np.zeros((n_rows, 3), dtype=object)
    try:
        columns[:len(unique_idx), 0] = unique_x
        columns[:len(unique_idx), 2] = counts
    except:
        print(f"unique_x: {unique_x}, empty_blocks: {empty_blocks}")
        
    for i, x in enumerate(unique_x):
        columns[i, 1] = np.max(empty_blocks[empty_blocks[:, 0] == x, 1])

    columns[:, 0] = columns[:, 0].astype(float)
    columns[:, 1] = columns[:, 0].astype(float)
    columns[:, 2] = columns[:, 2].astype(float)
    columns = columns.flatten()
    delta_player = player_position - prev_positions[0]
    delta_distance = ball_distance - prev_ball_distance

    if is_mixed:
        info = np.array([player_position[0], player_position[1], delta_player[0], delta_player[1], ball_position[0], ball_position[1], ball_distance, delta_distance,
                        str(ball_orientation), ball_velocity[0], ball_velocity[1]], dtype=object)
    else:
        info = np.concatenate([player_position, delta_player, ball_position, np.array([ball_distance, delta_distance]),
                                orientation_to_one_hot(ball_orientation), ball_velocity
                               ], axis=0, dtype=np.single)
    return np.append(info, columns)

def pong_extraction(positions: np.ndarray, prev_positions: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    distances[(positions[1:, 0] == 0) & (positions[1:, 1] == 0)] = np.max(distances)
    orientation = get_orientation(player_position, positions[1:, :])
    prev_orientation = get_orientation(prev_positions[1], prev_positions[1:, :])
    velocity = positions - prev_positions
    ball_position = positions[1]
    player_orientation = get_orientation(player_position, prev_positions[0])
    distance_ball_enemy =  np.array([np.linalg.norm(positions[2] - positions[1])])
    enemy_ball_orientation = get_orientation(positions[2], ball_position)
    if is_mixed:
        return np.concatenate([player_position, player_orientation[np.newaxis], distance_ball_enemy, enemy_ball_orientation[np.newaxis], ball_position, distances, orientation, prev_orientation, velocity[:, 0], velocity[:, 1]], axis=0, dtype=object)
    
    return np.concatenate([player_position, orientation_to_one_hot(player_orientation), distance_ball_enemy, orientation_to_one_hot(enemy_ball_orientation), ball_position, distances, orientation_to_one_hot(orientation), orientation_to_one_hot(prev_orientation), velocity[:, 0], velocity[:, 1]], axis=0, dtype=np.single)

def alien_extraction(positions: np.ndarray, prev_positions: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    prev_player_position = prev_positions[0]
    pulsar_index = 4
    alien_positions = positions[1:pulsar_index].copy()
    prev_alien_positions = prev_positions[1:pulsar_index].copy()
    egg_start_index = 5
    egg_positions = positions[egg_start_index:].copy()
    pulsar_position = positions[pulsar_index].copy()
    positions[(positions[:, 0] == 0) & (positions[:, 1] == 0)] = np.nan
    prev_positions[(prev_positions[:, 0] == 0) & (prev_positions[:, 1] == 0)] = np.nan
    distances = np.linalg.norm(player_position - positions, axis=1)
    prev_distances = np.linalg.norm(prev_player_position - prev_positions, axis=1)
    distances[np.isnan(distances)] = np.inf
    prev_distances[np.isnan(prev_distances)] = np.inf
    
    def get_cluster_orientation(player_position, alien_positions):
        alien_cluster_orientation = ''
        alien_encircled = True
        if (player_position[1] > alien_positions[:, 1]).all():
            alien_cluster_orientation = 'above'
            alien_encircled = False 
        elif (player_position[1] < alien_positions[:, 1]).all():
            alien_cluster_orientation = 'below'
            alien_encircled = False 

        if (player_position[0] > alien_positions[:, 0]).all():
            if alien_cluster_orientation:
                alien_cluster_orientation += ' and '
            alien_cluster_orientation += 'right'
            alien_encircled = False 
        elif (player_position[0] < alien_positions[:, 0]).all():
            if alien_cluster_orientation:
                alien_cluster_orientation += ' and '
            alien_cluster_orientation += 'left'
            alien_encircled = False 

        if alien_encircled:
            alien_cluster_orientation = 'encircled'
        alien_cluster_orientation = np.array([alien_cluster_orientation], dtype=str)
        return alien_cluster_orientation
    
    pulsar_distance = np.array([distances[pulsar_index]])
    prev_pulsar_distance = np.array([prev_distances[pulsar_index]])
    if np.isinf(pulsar_distance).any():
        pulsar_distance = np.array([distances[distances < np.inf].max()])
    delta_pulsar = pulsar_distance - prev_pulsar_distance
    if np.isnan(delta_pulsar).any():
        delta_pulsar = np.array([0])
    pulsar_orientation = get_orientation(player_position, pulsar_position)
    pulsar_orientation = pulsar_orientation[np.newaxis]
    egg_distances = distances[egg_start_index:]
    prev_egg_distances = prev_distances[egg_start_index:]
    closest_egg = np.argmin(egg_distances)
    closest_egg_distance = np.array([egg_distances[closest_egg]])
    prev_closest_egg = np.argmin(prev_egg_distances)
    prev_closest_egg_distance = np.array([prev_egg_distances[prev_closest_egg]])
    delta_closest_egg = closest_egg_distance - prev_closest_egg_distance
    closest_egg_position = egg_positions[closest_egg]
    closest_egg_orientation = get_orientation(player_position, closest_egg_position)
    closest_egg_orientation = closest_egg_orientation[np.newaxis]
    n_eggs = np.array([np.sum(~np.isnan(egg_positions[:, 0]))])
    n_aliens = np.array([np.sum(~np.isnan(alien_positions[:, 0]))])

    alien_orientation = get_orientation(player_position, alien_positions)
    prev_alien_orientation = get_orientation(prev_player_position, prev_alien_positions)
    alien_cluster_orientation = get_cluster_orientation(player_position, alien_positions)
    alien_distances = np.linalg.norm(player_position - alien_positions, axis=1)
    prev_alien_distances = np.linalg.norm(prev_player_position - prev_alien_positions, axis=1)
    delta_distances = alien_distances - prev_alien_distances

    player_orientation = get_orientation(player_position, prev_positions[0])
    player_orientation = player_orientation[np.newaxis]

    if is_mixed:
        info = np.concatenate([player_position, player_orientation, pulsar_distance, delta_pulsar, pulsar_orientation, closest_egg_position, closest_egg_orientation,
                            closest_egg_distance, delta_closest_egg, n_eggs, alien_positions[:, 0], alien_positions[:, 1], alien_distances, alien_orientation, delta_distances, prev_alien_orientation, n_aliens, alien_cluster_orientation], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, orientation_to_one_hot(player_orientation), pulsar_distance, delta_pulsar, orientation_to_one_hot(pulsar_orientation), closest_egg_position, orientation_to_one_hot(closest_egg_orientation),
                            closest_egg_distance, delta_closest_egg, n_eggs, alien_positions[:, 0], alien_positions[:, 1], alien_distances, orientation_to_one_hot(alien_orientation), delta_distances, orientation_to_one_hot(prev_alien_orientation), n_aliens, alien_orientation_to_one_hot(alien_cluster_orientation)], axis=0, dtype=np.single)
    return info

def kangaroo_extraction(positions: np.ndarray, prev_positions: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    # valid_positions = positions.copy()

    child_idx = 1
    fruit_idx = 2
    bell_idx = 5
    platform_idx = 5
    ladder_idx = 26
    monkey_idx = 32
    coconut_idx = 36
    life_idx = 40
    orientations = get_orientation(player_position, positions[1:life_idx])

    prev_orientation = get_orientation(player_position, prev_positions[0])
    prev_orientation = prev_orientation[np.newaxis]

    distances = np.linalg.norm(player_position - positions[1:life_idx], axis=1)
    prev_distances = np.linalg.norm(prev_positions[0] - prev_positions[1:life_idx], axis=1)
    delta_distances = distances - prev_distances

    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, distances, delta_distances, orientations], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), distances, delta_distances, orientation_to_one_hot(orientations)], axis=0, dtype=np.single)
    return info

# def space_invaders_extraction(positions: np.ndarray, prev_positions: np.ndarray) -> np.ndarray:
#     player_position = positions[0]
#     # valid_positions = positions.copy()
#     orientations = get_orientation(player_position, positions[1:life_idx])

#     prev_orientation = get_orientation(player_position, prev_positions[0])
#     prev_orientation = prev_orientation[np.newaxis]

#     distances = np.linalg.norm(player_position - positions[1:life_idx], axis=1)
#     prev_distances = np.linalg.norm(player_position - positions[1:life_idx], axis=1)
#     delta_distances = distances - prev_distances

#     info = np.concatenate([player_position, prev_orientation, distances, delta_distances, orientations], axis=0, dtype=object)
#     return info.flatten()

def general_extraction(positions: np.ndarray, prev_positions: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    # valid_positions = positions.copy()
    orientations = get_orientation(player_position, positions[1:])

    prev_orientation = get_orientation(player_position, prev_positions[0])
    prev_orientation = prev_orientation[np.newaxis]

    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    prev_distances = np.linalg.norm(prev_positions[0] - prev_positions[1:], axis=1)
    delta_distances = distances - prev_distances

    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, distances, delta_distances, orientations], axis=0, dtype=object)
    else:
         info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), distances, delta_distances, orientation_to_one_hot(orientations)], axis=0, dtype=object)
    return info





    
    
