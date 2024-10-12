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
    'x_aligned and y_aligned': 8,
    'No object': 9,
    'semi left and below': 10,
    'semi right and below': 11,
    'semi left and above': 12,
    'semi right and above': 13,
    'left and semi above': 14,
    'right and semi above': 15,
    'left and semi below': 16,
    'right and semi below': 17,
    'semi right and semi below': 18,
    'semi right and semi above': 19,
    'semi left and semi below': 20,
    'semi left and semi above': 21,
    'semi left and y_aligned': 22,
    'semi right and y_aligned': 23,
    'x_aligned and semi below': 24,
    'x_aligned and semi above': 25,
    'fully aligned': 26,
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

def alien_orientation_to_one_hot(orientation: np.ndarray):
    orientation_mapped = np.vectorize(alien_orientation_mapping.get)(orientation)
    # Create one-hot encoding
    num_classes = len(alien_orientation_mapping)
    one_hot_encoded = np.eye(num_classes)[orientation_mapped]
    # Flatten the one-hot encoded matrix
    return one_hot_encoded.flatten()


def get_orientation(player_position: np.ndarray, other_positions: np.ndarray, player_size: np.ndarray, other_sizes: np.ndarray):
    # Calculate the bottom-right and top-left corners of the player

    player_top_left = player_position
    player_bottom_right = player_position + player_size
    other_top_left = other_positions
    other_bottom_right = other_positions + other_sizes
    player_center = (player_top_left + player_bottom_right) / 2
    other_center = (other_top_left + other_bottom_right) / 2
    
    if other_positions.ndim > 1:
        is_right = (other_top_left[:, 0] - player_bottom_right[0]) > 0 
        is_semi_right = ((other_top_left[:, 0] - player_top_left[0]) >= 0) & ~is_right 
        is_left = (player_top_left[0] - other_bottom_right[:, 0]) > 0 
        is_semi_left = ((player_top_left[0] - other_top_left[:, 0]) >= 0) & ~is_left
        is_x_aligned = (is_semi_right & is_semi_left) | ((other_top_left[:, 0] == player_top_left[0]) & (other_bottom_right[:, 0] == player_bottom_right[0]))
        is_above = (player_top_left[1] - other_bottom_right[:, 1]) > 0 
        is_semi_above = ((player_top_left[1] - other_top_left[:, 1]) >= 0) & ~is_above 
        is_below = (other_top_left[:, 1] - player_bottom_right[1]) > 0 
        is_semi_below = ((other_top_left[:, 1] - player_top_left[1]) >= 0 ) & ~is_below
        is_y_aligned = is_semi_above & is_semi_below
        is_center = ((player_center[0] - other_center[:, 0]) == 0) & ((player_center[1] - other_center[:, 1]) == 0)
        no_object = (other_top_left[:, 0] == 0) & (other_top_left[:, 1] == 0)
        nan = (np.isnan(other_top_left[:, 0])) & (np.isnan(other_top_left[:, 1]))
        n_samples =  len(other_positions)
    else:
        is_right = (other_top_left[0] - player_bottom_right[0]) > 0 
        is_semi_right = ((other_top_left[0] - player_top_left[0]) >= 0) & ~is_right 
        is_left = (player_top_left[0] - other_bottom_right[0]) > 0 
        is_semi_left = ((player_top_left[0] - other_top_left[0]) >= 0) & ~is_left
        is_x_aligned = is_semi_right & is_semi_left | (other_top_left[0] == player_top_left[0] and other_bottom_right[0] == player_bottom_right[0])

        is_above = (player_top_left[1] - other_bottom_right[1]) > 0 
        is_semi_above = ((player_top_left[1] - other_top_left[1]) >= 0) & ~is_above 
        is_below = (other_top_left[1] - player_bottom_right[1]) > 0 
        is_semi_below = ((other_top_left[1] - player_top_left[1]) >= 0 ) & ~is_below
        is_y_aligned = is_semi_above & is_semi_below
        no_object = (other_top_left[0] == 0) & (other_top_left[1] == 0)
        nan = (np.isnan(other_top_left[0])) & (np.isnan(other_top_left[1]))
        is_center = ((player_center[0] - other_center[0]) == 0) & ((player_center[1] - other_center[1]) == 0)
        n_samples = 1

    delta = np.zeros(n_samples).astype('<U50')
    # Define the relative positions using the bounding box comparisons
    delta[is_right & is_above] = 'right and above'
    delta[is_right & is_below] = 'right and below'
    delta[is_left & is_above]  = 'left and above'
    delta[is_left & is_below]  = 'left and below'
    delta[is_semi_right & is_above] = 'semi right and above'
    delta[is_semi_right & is_below] = 'semi right and below'
    delta[is_semi_left & is_above]  = 'semi left and above'
    delta[is_semi_left & is_below]  = 'semi left and below'
    delta[is_semi_left & is_semi_above]  = 'semi left and semi above'
    delta[is_semi_left & is_semi_below]  = 'semi left and semi below'
    delta[is_semi_right & is_semi_above]  = 'semi right and semi above'
    delta[is_semi_right & is_semi_below]  = 'semi right and semi below'
    delta[is_right & is_semi_above] = 'right and semi above'
    delta[is_right & is_semi_below] = 'right and semi below'
    delta[is_left & is_semi_above]  = 'left and semi above'
    delta[is_left & is_semi_below]  = 'left and semi below'
    delta[is_x_aligned & is_below]  = 'x_aligned and below'
    delta[is_x_aligned & is_above]  = 'x_aligned and above'
    delta[is_x_aligned & is_semi_below]  = 'x_aligned and semi below'
    delta[is_x_aligned & is_semi_above]  = 'x_aligned and semi above'
    delta[is_left & is_y_aligned]  = 'left and y_aligned'
    delta[is_right & is_y_aligned]  = 'right and y_aligned'
    delta[is_semi_left & is_y_aligned]  = 'semi left and y_aligned'
    delta[is_semi_right & is_y_aligned]  = 'semi right and y_aligned'
    delta[is_x_aligned & is_y_aligned]  = 'x_aligned and y_aligned'
    delta[is_center] = 'fully aligned'
    # No object cases
    delta[no_object] = 'No object'
    delta[nan] = 'No object'
    return delta


def gopher_extraction(positions: np.ndarray, prev_positions: np.ndarray, object_sizes: np.ndarray, is_mixed: bool) -> np.ndarray:
    player_position = positions[0]
    player_size = object_sizes[0]
    gopher_position = positions[1]
    gopher_size = object_sizes[1]
    delta_player_position = player_position - prev_positions[0]
    delta_gopher_position = gopher_position - prev_positions[1]
    gopher_distance = np.linalg.norm(player_position - gopher_position)
    gopher_orientation = get_orientation(player_position, gopher_position, player_size, gopher_size)
    empty_block_first_idx = 5
    empty_blocks = positions[empty_block_first_idx:]
    block_size = object_sizes[empty_block_first_idx]
    unique_x, unique_idx = np.unique(empty_blocks[:, 0], return_index=True)
    counts = np.sum(empty_blocks[:, 0][:, None] == empty_blocks[unique_idx, 0], axis=0)
    counts[empty_blocks[unique_idx][:, 0] == 0] = 0
    aligned_blocks_exist = (counts == 3).any()
    almost_aligned_block_exist = (counts == 2).any()

    nearest_aligned_pos = np.array([0, 0])
    aligned_orientation = np.array(['No object']).astype('<U50')
    aligned_dist = 0
    if aligned_blocks_exist:
        aligned_x = unique_x[counts == 3]
        aligned_pos = empty_blocks[np.isin(empty_blocks[:, 0], aligned_x)]
        aligned_dist = np.linalg.norm(player_position - aligned_pos, axis=1)
        min_arg = np.argmin(aligned_dist)
        nearest_aligned_pos = aligned_pos[min_arg]
        aligned_dist = aligned_dist[min_arg]
        aligned_orientation = get_orientation(player_position, nearest_aligned_pos, player_size, block_size)
    
    if aligned_orientation.ndim > 1:
        print()
    nearest_almost_aligned_pos = np.array([0, 0])
    almost_aligned_orientation = np.array(['No object']).astype('<U50')
    almost_aligned_dist = 0
    if almost_aligned_block_exist:
        almost_aligned_x = unique_x[counts == 2]
        almost_aligned_pos = empty_blocks[np.isin(empty_blocks[:, 0], almost_aligned_x)]
        almost_aligned_dist = np.linalg.norm(player_position - almost_aligned_pos, axis=1)
        min_arg = np.argmin(almost_aligned_dist)
        nearest_almost_aligned_pos = almost_aligned_pos[min_arg]
        almost_aligned_dist = almost_aligned_dist[min_arg]
        almost_aligned_orientation = get_orientation(player_position, nearest_almost_aligned_pos, player_size, block_size)
    
    if almost_aligned_orientation.ndim > 1:
        print()
    block_count_below = (player_position[0] == empty_blocks[:, 0]).sum()
    prev_block_count_below = (prev_positions[0, 0] == prev_positions[empty_block_first_idx:, 0]).sum()
    delta_prev_count_below = block_count_below - prev_block_count_below
    
    if is_mixed:
        info = np.concatenate([player_position, delta_player_position, gopher_position, delta_gopher_position, np.array([gopher_distance]), 
                            gopher_orientation, np.array([int(aligned_blocks_exist)]), nearest_aligned_pos,
                            np.array([aligned_dist]), aligned_orientation, np.array([int(almost_aligned_block_exist)]),
                            nearest_almost_aligned_pos, np.array([almost_aligned_dist]), almost_aligned_orientation, np.array([block_count_below, prev_block_count_below, delta_prev_count_below])], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, delta_player_position, gopher_position, delta_gopher_position, np.array([gopher_distance]), 
                           orientation_to_one_hot(gopher_orientation), np.array([int(aligned_blocks_exist)]), nearest_aligned_pos,
                           np.array([aligned_dist]), orientation_to_one_hot(aligned_orientation), np.array([int(almost_aligned_block_exist)]),
                           nearest_almost_aligned_pos, np.array([almost_aligned_dist]), orientation_to_one_hot(almost_aligned_orientation), np.array([block_count_below, prev_block_count_below, delta_prev_count_below])], axis=0, dtype=np.single)
    return info

def breakout_extraction(positions: np.ndarray, prev_positions: np.ndarray, object_sizes: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    player_size = object_sizes[0]
    
    ball_position = positions[1]
    prev_ball_position = prev_positions[1]
    ball_size = object_sizes[1]
    # return player_position
    ball_distance = np.linalg.norm(player_position - ball_position)
    prev_ball_distance = np.linalg.norm(prev_positions[0] - prev_ball_position)
    ball_orientation = get_orientation(player_position, ball_position, player_size, ball_size)
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
        info = np.concatenate([player_position, delta_player, ball_position, np.array([ball_distance, delta_distance]),
                               ball_orientation, ball_velocity
                               ], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, delta_player, ball_position, np.array([ball_distance, delta_distance]),
                                orientation_to_one_hot(ball_orientation), ball_velocity
                               ], axis=0, dtype=np.single)
    return np.append(info, columns)

def pong_extraction(positions: np.ndarray, prev_positions: np.ndarray, object_sizes: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    player_size = object_sizes[0]
    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    distances[(positions[1:, 0] == 0) & (positions[1:, 1] == 0)] = np.max(distances)
    orientation = get_orientation(player_position, positions[1:, :], player_size, object_sizes[1:])
    prev_orientation = get_orientation(prev_positions[1], prev_positions[1:, :], object_sizes[1], object_sizes[1:])
    velocity = positions - prev_positions
    ball_position = positions[1]
    ball_size = object_sizes[1]
    player_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)
    distance_ball_enemy =  np.array([np.linalg.norm(positions[2] - positions[1])])
    enemy_ball_orientation = get_orientation(positions[2], ball_position, object_sizes[2], ball_size)
    if is_mixed:
        info = np.concatenate([player_position, player_orientation, distance_ball_enemy, enemy_ball_orientation, ball_position, distances, orientation, prev_orientation, velocity[:, 0], velocity[:, 1]], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, orientation_to_one_hot(player_orientation), distance_ball_enemy, orientation_to_one_hot(enemy_ball_orientation), ball_position, distances, orientation_to_one_hot(orientation), orientation_to_one_hot(prev_orientation), velocity[:, 0], velocity[:, 1]], axis=0, dtype=np.single)
    return info

def alien_extraction(positions: np.ndarray, prev_positions: np.ndarray, object_sizes: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    player_size = object_sizes[0]
    prev_player_position = prev_positions[0]
    pulsar_index = 4
    pulsar_size = object_sizes[4]
    alien_size = object_sizes[1]
    alien_positions = positions[1:pulsar_index].copy()
    prev_alien_positions = prev_positions[1:pulsar_index].copy()
    egg_start_index = 5
    egg_size = object_sizes[5]
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
    if np.isinf(pulsar_distance).any() or np.isnan(pulsar_distance).any():
        pulsar_distance = np.array([distances[distances < np.inf].max()])
    delta_pulsar = pulsar_distance - prev_pulsar_distance
    if np.isnan(delta_pulsar).any() or np.isinf(delta_pulsar).any():
        delta_pulsar = np.array([0])

    pulsar_orientation = get_orientation(player_position, pulsar_position, player_size, pulsar_size)
    egg_distances = distances[egg_start_index:]
    prev_egg_distances = prev_distances[egg_start_index:]
    closest_egg = np.argmin(egg_distances)
    closest_egg_distance = np.array([egg_distances[closest_egg]])
    prev_closest_egg = np.argmin(prev_egg_distances)
    prev_closest_egg_distance = np.array([prev_egg_distances[prev_closest_egg]])
    delta_closest_egg = closest_egg_distance - prev_closest_egg_distance
    closest_egg_position = egg_positions[closest_egg]
    closest_egg_orientation = get_orientation(player_position, closest_egg_position, player_size, egg_size)
    n_eggs = np.array([np.sum(~np.isnan(egg_positions[:, 0]))])
    n_aliens = np.array([np.sum(~np.isnan(alien_positions[:, 0]))])

    alien_orientation = get_orientation(player_position, alien_positions, player_size, alien_size)
    prev_alien_orientation = get_orientation(prev_player_position, prev_alien_positions, player_size, alien_size)
    alien_cluster_orientation = get_cluster_orientation(player_position, alien_positions)
    alien_distances = np.linalg.norm(player_position - alien_positions, axis=1)
    prev_alien_distances = np.linalg.norm(prev_player_position - prev_alien_positions, axis=1)
    delta_distances = alien_distances - prev_alien_distances

    player_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)

    if is_mixed:
        info = np.concatenate([player_position, player_orientation, pulsar_distance, delta_pulsar, pulsar_orientation, closest_egg_position, closest_egg_orientation,
                            closest_egg_distance, delta_closest_egg, n_eggs, alien_positions[:, 0], alien_positions[:, 1], alien_distances, alien_orientation, delta_distances, prev_alien_orientation, n_aliens, alien_cluster_orientation], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, orientation_to_one_hot(player_orientation), pulsar_distance, delta_pulsar, orientation_to_one_hot(pulsar_orientation), closest_egg_position, orientation_to_one_hot(closest_egg_orientation),
                            closest_egg_distance, delta_closest_egg, n_eggs, alien_positions[:, 0], alien_positions[:, 1], alien_distances, orientation_to_one_hot(alien_orientation), delta_distances, orientation_to_one_hot(prev_alien_orientation), n_aliens, alien_orientation_to_one_hot(alien_cluster_orientation)], axis=0, dtype=np.single)
    if np.isnan(info).any() or np.isinf(info).any():
        print()
    return info

def kangaroo_extraction(positions: np.ndarray, prev_positions: np.ndarray, object_sizes: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    player_size = object_sizes[0]
    # valid_positions = positions.copy()

    child_idx = 1
    fruit_idx = 2
    bell_idx = 5
    platform_idx = 6
    ladder_idx = 26
    monkey_idx = 32
    coconut_idx = 36
    life_idx = 40
    orientations = get_orientation(player_position, positions[:life_idx], player_size, object_sizes[:life_idx])

    prev_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)

    distances = np.linalg.norm(player_position - positions[:life_idx], axis=1)
    prev_distances = np.linalg.norm(prev_positions[0] - prev_positions[:life_idx], axis=1)
    delta_distances = distances - prev_distances

    child_distance = distances[child_idx:fruit_idx]
    child_delta = delta_distances[child_idx:fruit_idx]
    child_orientation = orientations[child_idx:fruit_idx]
    fruit_distance = distances[fruit_idx:bell_idx]
    fruit_delta = distances[fruit_idx:bell_idx]
    fruit_orientation = orientations[fruit_idx:bell_idx]

    bell_distance = distances[bell_idx:platform_idx]
    bell_delta = delta_distances[bell_idx:platform_idx]
    bell_orientation = orientations[bell_idx:platform_idx]
    platform_distance = distances[platform_idx:ladder_idx]
    platform_delta = delta_distances[platform_idx:ladder_idx]
    platform_orientation = orientations[platform_idx:ladder_idx]

    ladder_distance = distances[ladder_idx:monkey_idx]
    ladder_delta = delta_distances[ladder_idx:monkey_idx]
    ladder_orientation = orientations[ladder_idx:monkey_idx]
    monkey_distance = distances[monkey_idx:coconut_idx]
    monkey_delta = delta_distances[monkey_idx:coconut_idx]
    monkey_orientation = orientations[monkey_idx:coconut_idx]

    coconut_distance = distances[coconut_idx:]
    coconut_delta = delta_distances[coconut_idx:]
    coconut_orientation = orientations[coconut_idx:]


    player_y = player_position[1]
    closest_ladder = np.argmin(player_y - positions[ladder_idx:monkey_idx, 1])

    closest_ladder_distance = ladder_distance[closest_ladder]
    closest_ladder_delta = ladder_delta[closest_ladder]
    closest_ladder_orientation = ladder_orientation[closest_ladder]

    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, child_distance, child_delta, child_orientation,
                               fruit_distance, fruit_delta, fruit_orientation, bell_distance, bell_delta, bell_orientation,
                               platform_distance, platform_delta, platform_orientation, 
                               np.array([closest_ladder_distance, closest_ladder_delta, closest_ladder_orientation], dtype=object),
                               monkey_distance, monkey_delta, monkey_orientation, coconut_distance, coconut_delta, coconut_orientation,
                               ], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), child_distance, 
                        child_delta, orientation_to_one_hot(child_orientation), fruit_distance, fruit_delta, 
                        orientation_to_one_hot(fruit_orientation), bell_distance, bell_delta, orientation_to_one_hot(bell_orientation),
                        platform_distance, platform_delta, orientation_to_one_hot(platform_orientation), 
                        np.array([closest_ladder_distance, closest_ladder_delta], dtype=np.single), orientation_to_one_hot(closest_ladder_orientation), 
                        monkey_distance, monkey_delta, orientation_to_one_hot(monkey_orientation), coconut_distance, coconut_delta, orientation_to_one_hot(coconut_orientation),
                        ], axis=0, dtype=np.single)
    return info

def general_extraction(positions: np.ndarray, prev_positions: np.ndarray, object_sizes: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    player_size = object_sizes[0]

    orientations = get_orientation(player_position, positions[1:], player_size, object_sizes[1:])

    prev_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)

    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    prev_distances = np.linalg.norm(prev_positions[0] - prev_positions[1:], axis=1)
    delta_distances = distances - prev_distances

    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, distances, delta_distances, orientations], axis=0, dtype=object)
    else:
         info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), distances, delta_distances, orientation_to_one_hot(orientations)], axis=0, dtype=object)
    return info

def space_invaders_extraction(positions: np.ndarray, prev_positions: np.ndarray, object_sizes: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    player_position = positions[0]
    player_size = object_sizes[0]
    orientations = get_orientation(player_position, positions[1:], player_size, object_sizes[1:])

    prev_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)

    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    prev_distances = np.linalg.norm(prev_positions[0] - prev_positions[1:], axis=1)
    delta_distances = distances - prev_distances

    shield_idx = 1
    alien_idx = 4
    bullet_idx = 40

    below_shield = np.isin('x_aligned_and_above', orientations[shield_idx:alien_idx]).any()
    below_alien = np.isin('x_aligned_and_above', orientations[alien_idx:bullet_idx]).any()
    below_bullet = np.isin('x_aligned_and_above', orientations[bullet_idx:]).any()
    below_info = np.array([below_shield, below_alien, below_bullet]).astype(str if is_mixed else int)

    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, distances, delta_distances, orientations, below_info], axis=0, dtype=object)
    else:
         info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), distances, delta_distances, orientation_to_one_hot(orientations), below_info], axis=0, dtype=object)
    return info





    
    
