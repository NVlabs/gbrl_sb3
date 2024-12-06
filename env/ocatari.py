##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import numpy as np

MIXED_ATARI_ENVS = ['Gopher', 'Breakout', 'Alien', 'Kangaroo', 'SpaceInvaders', 'Pong', 'Assault', 'Asterix', 'Bowling', 'Tennis', 'Freeway', 'Boxing']
ATARI_GENERAL_EXTRACTION = ['Assault', 'Asterix', 'Bowling', 'Freeway', 'Boxing']


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

x_momentum_mapping = {
    'no change': 0,
    'left consistently': 1,
    'left steady': 2,
    'left frequent_moving': 3,
    'right consistently': 4,
    'right steady': 5,
    'right frequent_moving': 6,
    'no object': 7,
}

y_momentum_mapping = {
    'no change': 0,
    'up consistently': 1,
    'up steady': 2,
    'up frequent_moving': 3,
    'down consistently': 4,
    'down steady': 5,
    'down frequent_moving': 6,
    'no object': 7,
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


def x_momentum_to_one_hot(x_momentum: np.ndarray):
    momentum_mapped = np.vectorize(x_momentum_mapping.get)(x_momentum)
    # Create one-hot encoding
    num_classes = len(x_momentum_mapping)
    one_hot_encoded = np.eye(num_classes)[momentum_mapped]
    # Flatten the one-hot encoded matrix
    return one_hot_encoded.flatten()

def y_momentum_to_one_hot(y_momentum: np.ndarray):
    momentum_mapped = np.vectorize(y_momentum_mapping.get)(y_momentum)
    # Create one-hot encoding
    num_classes = len(y_momentum_mapping)
    one_hot_encoded = np.eye(num_classes)[momentum_mapped]
    # Flatten the one-hot encoded matrix
    return one_hot_encoded.flatten()

def get_x_momentum(positions: np.ndarray):
    # assuming index -1 is current frame and -4 is final frame
    momentum = np.diff(positions, axis=0)
    sum_momentum = np.sum(momentum, axis=0)
    x_change = 'no change'
    if sum_momentum[0] < 0:
        x_change = 'left'
        if (momentum[:, 0] < 0).all():
            x_change += ' consistently'
        elif (momentum[:, 0] <= 0).sum() > 1:
            x_change += ' steady'
        else:
            x_change += ' frequent_moving'
    elif sum_momentum[0] > 0:
        x_change = 'right'
        if (momentum[:, 0] > 0).all():
            x_change += ' consistently'
        elif (momentum[:, 0] >= 0).sum() > 1:
            x_change += ' steady'
        else:
            x_change += ' frequent_moving'
    if np.array_equal(positions, np.zeros_like(positions)):
        x_change = 'no object'
    return x_change 

def get_y_momentum(positions: np.ndarray):
    # assuming index -1 is current frame and -4 is final frame
    momentum = np.diff(positions, axis=0)
    sum_momentum = np.sum(momentum, axis =0)
    y_change = 'no change'
    if sum_momentum[1] < 0:
        y_change = 'up'
        if (momentum[:, 1] < 0).all():
            y_change += ' consistently'
        elif (momentum[:, 1] <= 0).sum() > 1:
            y_change += ' steady'
        else:
            y_change += ' frequent_moving'
    elif sum_momentum[1] > 0:
        y_change = 'down'
        if (momentum[:, 1] > 0).all():
            y_change += ' consistently'
        elif (momentum[:, 1] >= 0).sum() > 1:
            y_change += ' steady'
        else:
            y_change += ' frequent_moving'
    if np.array_equal(positions, np.zeros_like(positions)):
        y_change = 'no object'
    return y_change




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


def gopher_extraction(observation: np.ndarray, is_mixed: bool) -> np.ndarray:
    positions = observation[-1, :, :2]
    prev_positions = observation[-2, :, :2]
    object_sizes = observation[-1, :, 2:]

    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])
    
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
    
    # if aligned_orientation.ndim > 1:
    #     print()
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
        info = np.concatenate([player_position, delta_player_position, np.array([x_momentum, y_momentum]), gopher_position, delta_gopher_position, np.array([gopher_distance]), 
                            gopher_orientation, np.array([int(aligned_blocks_exist)]), nearest_aligned_pos,
                            np.array([aligned_dist]), aligned_orientation, np.array([int(almost_aligned_block_exist)]),
                            nearest_almost_aligned_pos, np.array([almost_aligned_dist]), almost_aligned_orientation, np.array([block_count_below, prev_block_count_below, delta_prev_count_below])], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, delta_player_position,x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), gopher_position, delta_gopher_position, np.array([gopher_distance]), 
                           orientation_to_one_hot(gopher_orientation), np.array([int(aligned_blocks_exist)]), nearest_aligned_pos,
                           np.array([aligned_dist]), orientation_to_one_hot(aligned_orientation), np.array([int(almost_aligned_block_exist)]),
                           nearest_almost_aligned_pos, np.array([almost_aligned_dist]), orientation_to_one_hot(almost_aligned_orientation), np.array([block_count_below, prev_block_count_below, delta_prev_count_below])], axis=0, dtype=np.single)
    return info

def breakout_extraction(observation: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    positions = observation[-1, :, :2]
    prev_positions = observation[-2, :, :2]
    object_sizes = observation[-1, :, 2:]

    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])
    
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
        info = np.concatenate([player_position, delta_player, np.array([x_momentum, y_momentum]), ball_position, np.array([ball_distance, delta_distance]),
                               ball_orientation, ball_velocity
                               ], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, delta_player, x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), ball_position, np.array([ball_distance, delta_distance]),
                                orientation_to_one_hot(ball_orientation), ball_velocity
                               ], axis=0, dtype=np.single)
    return np.append(info, columns)

def pong_extraction(observation: np.ndarray, is_mixed: bool = True, useless_value: float = 0) -> np.ndarray:
    positions = observation[-1, :, :2]
    prev_positions = observation[-2, :, :2]
    object_sizes = observation[-1, :, 2:]

    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])

    player_position = positions[0]
    player_size = object_sizes[0]
    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    distances[(positions[1:, 0] == 0) & (positions[1:, 1] == 0)] = np.max(distances)
    orientation = get_orientation(player_position, positions[1:, :], player_size, object_sizes[1:])
    prev_orientation = get_orientation(prev_positions[0], prev_positions[1:, :], player_size, object_sizes[1:])
    velocity = positions - prev_positions
    ball_position = positions[1]
    ball_size = object_sizes[1]
    player_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)
    distance_ball_enemy =  np.array([np.linalg.norm(positions[2] - positions[1])])
    enemy_ball_orientation = get_orientation(positions[2], ball_position, object_sizes[2], ball_size)
    # distraction_mapping = {0: 'blue', 1: 'red', 2: 'green'}
    # # distraction_idx = 0 if num_timesteps < 1250000 else 1
    # distraction_idx = 2
    # # distraction_idx = np.random.choice(len(distraction_mapping))
    # def one_hot_distraction(idx, distraction_mapping):
    #     # Create one-hot encoding
    #     num_classes = len(distraction_mapping)
    #     one_hot_encoded = np.eye(num_classes)[idx]
    #     # Flatten the one-hot encoded matrix
    #     return one_hot_encoded.flatten()
    # min_value = 0
    # max_value = 128
    # min_value = 128
    # max_value = 255

    useless_feature = np.array([useless_value])
    # useless_feature = np.array([float(np.random.uniform(low=min_value, high=max_value))])
    # print(useless_feature)
    if is_mixed:
        info = np.concatenate([player_position, player_orientation, np.array([x_momentum, y_momentum]), distance_ball_enemy, enemy_ball_orientation, ball_position, distances, orientation, prev_orientation, velocity[:, 0], velocity[:, 1],
                               useless_feature], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, orientation_to_one_hot(player_orientation), x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), distance_ball_enemy, orientation_to_one_hot(enemy_ball_orientation), ball_position, distances, orientation_to_one_hot(orientation), orientation_to_one_hot(prev_orientation), velocity[:, 0], velocity[:, 1],
                               useless_feature], axis=0, dtype=np.single)
    return info

def tennis_extraction(observation: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    positions = observation[-1, :-1, :2]
    prev_positions = observation[-2, :-1, :2]
    object_sizes = observation[-1, :-1, 2:]


    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])

    player_position = positions[0]
    player_size = object_sizes[0]
    pos_diff = np.abs(player_position - positions[1:])
    prev_pos_diff = np.abs(prev_positions[0] - prev_positions[1:])

    x_distances = pos_diff[:, 0]
    x_distances[(positions[1:, 0] == 0) & (positions[1:, 1] == 0)] = np.max(pos_diff[:, 0])
    prev_x_distances = prev_pos_diff[:, 0]
    prev_x_distances[(prev_positions[1:, 0] == 0) & (prev_positions[1:, 1] == 0)] = np.max(pos_diff[:, 0])
    
    y_distances = pos_diff[:, 1]
    y_distances[(positions[1:, 0] == 0) & (positions[1:, 1] == 0)] = np.max(pos_diff[:, 1])
    prev_y_distances = prev_pos_diff[:, 1]
    prev_y_distances[(prev_positions[1:, 0] == 0) & (prev_positions[1:, 1] == 0)] = np.max(pos_diff[:, 1])
    
    delta_x_distances = x_distances - prev_x_distances
    delta_y_distances = y_distances - prev_y_distances

    orientation = get_orientation(player_position, positions[1:, :], player_size, object_sizes[1:])
    prev_orientation = get_orientation(prev_positions[0], prev_positions[1:, :], player_size, object_sizes[1:])
    ball_position = positions[2]
    ball_size = object_sizes[2]
    enemy_position = positions[1]
    enemy_size = object_sizes[1]
    is_upper_player = np.array([positions[0, 1] < positions[1, 1]])
    

    x_ball_momentum = get_x_momentum(observation[:, 2, :2])
    y_ball_momentum = get_y_momentum(observation[:, 2, :2])
    x_enemy_momentum = get_x_momentum(observation[:, 1, :2])
    y_enemy_momentum = get_y_momentum(observation[:, 1, :2])

    distance_ball_enemy =  np.array([np.linalg.norm(positions[2] - positions[1])])
    enemy_ball_orientation = get_orientation(enemy_position, ball_position, enemy_size, ball_size)
    # ball_in_my_court = distances[1] < distance_ball_enemy

    if is_mixed:
        info = np.concatenate([player_position, np.array([x_momentum, y_momentum]), np.array([x_ball_momentum, y_ball_momentum]), 
                               np.array([x_enemy_momentum, y_enemy_momentum]), distance_ball_enemy, enemy_ball_orientation, ball_position, 
                               x_distances, y_distances, orientation, prev_orientation,
                               is_upper_player.astype(str), delta_x_distances, delta_y_distances], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), x_momentum_to_one_hot(x_ball_momentum), 
                               y_momentum_to_one_hot(y_ball_momentum), x_momentum_to_one_hot(x_enemy_momentum), 
                               y_momentum_to_one_hot(y_enemy_momentum), distance_ball_enemy, 
                               orientation_to_one_hot(enemy_ball_orientation), ball_position, x_distances, y_distances, 
                               orientation_to_one_hot(orientation), orientation_to_one_hot(prev_orientation), 
                               is_upper_player, delta_x_distances, delta_y_distances], axis=0, dtype=np.single)
    return info

def alien_extraction(observation: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    positions = observation[-1, :, :2]
    prev_positions = observation[-2, :, :2]
    object_sizes = observation[-1, :, 2:]

    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])

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
        info = np.concatenate([player_position, player_orientation, np.array([x_momentum, y_momentum]), pulsar_distance, delta_pulsar, pulsar_orientation, closest_egg_position, closest_egg_orientation,
                            closest_egg_distance, delta_closest_egg, n_eggs, alien_positions[:, 0], alien_positions[:, 1], alien_distances, alien_orientation, delta_distances, prev_alien_orientation, n_aliens, alien_cluster_orientation], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, orientation_to_one_hot(player_orientation), x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), pulsar_distance, delta_pulsar, orientation_to_one_hot(pulsar_orientation), closest_egg_position, orientation_to_one_hot(closest_egg_orientation),
                            closest_egg_distance, delta_closest_egg, n_eggs, alien_positions[:, 0], alien_positions[:, 1], alien_distances, orientation_to_one_hot(alien_orientation), delta_distances, orientation_to_one_hot(prev_alien_orientation), n_aliens, alien_orientation_to_one_hot(alien_cluster_orientation)], axis=0, dtype=np.single)

    return info

def kangaroo_extraction(observation: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    positions = observation[-1, :, :2]
    prev_positions = observation[-2, :, :2]
    object_sizes = observation[-1, :, 2:]

    player_position = positions[0]
    player_size = object_sizes[0]
    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])
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
    delta_distances[(positions[:life_idx, 0] == 0) & (positions[:life_idx, 1] == 0)] = 0

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
    below_coconut = True if 'x_aligned' in coconut_orientation[0] or 'fully aligned' in coconut_orientation[0] or 'semi left' in coconut_orientation[0] or 'semi right' in coconut_orientation[1] else False
    below_coconut &= coconut_delta[0] > 0
    coconut_thrown_at = ((positions[coconut_idx+1:life_idx, 0] > 0) & (positions[coconut_idx+1:life_idx, 1] > 0)) & (coconut_delta[1:] > 0)
    coconut_speeds = positions[coconut_idx:life_idx] - prev_positions[coconut_idx:life_idx]
    coconut_speeds[coconut_speeds == 0] = 1
    coconut_time = np.array([np.min((coconut_distance[:, np.newaxis] / coconut_speeds).flatten())])
    
    player_y = player_position[1]
    closest_ladder = np.argmin(player_y - positions[ladder_idx:monkey_idx, 1])

    closest_ladder_distance = ladder_distance[closest_ladder]
    closest_ladder_delta = ladder_delta[closest_ladder]
    closest_ladder_orientation = ladder_orientation[closest_ladder]

    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, np.array([x_momentum, y_momentum]), child_distance, child_delta, child_orientation,
                               fruit_distance, fruit_delta, fruit_orientation, bell_distance, bell_delta, bell_orientation,
                               platform_distance, platform_delta, platform_orientation, 
                               np.array([closest_ladder_distance, closest_ladder_delta, closest_ladder_orientation], dtype=object),
                               monkey_distance, monkey_delta, monkey_orientation, coconut_distance, coconut_delta, coconut_orientation,
                               np.array([below_coconut], dtype=str), coconut_thrown_at.astype(str), coconut_time], axis=0, dtype=object)
    else:
        info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), child_distance, 
                        child_delta, orientation_to_one_hot(child_orientation), fruit_distance, fruit_delta, 
                        orientation_to_one_hot(fruit_orientation), bell_distance, bell_delta, orientation_to_one_hot(bell_orientation),
                        platform_distance, platform_delta, orientation_to_one_hot(platform_orientation), 
                        np.array([closest_ladder_distance, closest_ladder_delta], dtype=np.single), orientation_to_one_hot(closest_ladder_orientation), 
                        monkey_distance, monkey_delta, orientation_to_one_hot(monkey_orientation), coconut_distance, coconut_delta, orientation_to_one_hot(coconut_orientation),
                        np.array([below_coconut], dtype=np.single), coconut_thrown_at.astype(np.single), coconut_time], axis=0, dtype=np.single)
    return info

def general_extraction(observation: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    positions = observation[-1, :, :2]
    prev_positions = observation[-2, :, :2]
    object_sizes = observation[-1, :, 2:]

    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])
    
    player_position = positions[0]
    player_size = object_sizes[0]

    orientations = get_orientation(player_position, positions[1:], player_size, object_sizes[1:])

    prev_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)

    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    prev_distances = np.linalg.norm(prev_positions[0] - prev_positions[1:], axis=1)
    delta_distances = distances - prev_distances
    delta_distances[(positions[1:, 0] == 0) & (positions[1:, 1] == 0)] = 0

    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, np.array([x_momentum, y_momentum]), distances, delta_distances, orientations], axis=0, dtype=object)
    else:
         info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), distances, delta_distances, orientation_to_one_hot(orientations)], axis=0, dtype=object)
    return info

def asterix_extraction(observation: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    positions = observation[-1, :, :2]
    prev_positions = observation[-2, :, :2]
    object_sizes = observation[-1, :, 2:]

    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])

    enemy_idx = 1
    reward_idx = 9
    
    player_position = positions[0]
    player_size = object_sizes[0]

    orientations = get_orientation(player_position, positions[1:], player_size, object_sizes[1:])

    prev_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)

    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    prev_distances = np.linalg.norm(prev_positions[0] - prev_positions[1:], axis=1)
    delta_distances = distances - prev_distances
    delta_distances[(positions[1:, 0] == 0) & (positions[1:, 1] == 0)] = 0

    closest_enemy = np.argmin(distances[enemy_idx - 1:reward_idx - 1]) + enemy_idx - 1
    closest_enemy_distance = np.array([distances[closest_enemy]])
    closest_enemy_orientation = np.array([orientations[closest_enemy]])
    closest_enemy_x_momentum = np.array([get_x_momentum(observation[:, closest_enemy + 1, :2])])


    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, np.array([x_momentum, y_momentum]), distances, delta_distances, orientations,
                               closest_enemy_distance, closest_enemy_orientation, closest_enemy_x_momentum], axis=0, dtype=object)
    else:
         info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), distances, delta_distances, orientation_to_one_hot(orientations),
                               closest_enemy_distance, orientation_to_one_hot(closest_enemy_orientation),
                               x_momentum_to_one_hot(closest_enemy_x_momentum)], axis=0, dtype=object)
    return info

def bowling_extraction(observation: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    positions = observation[-1, :, :2]
    prev_positions = observation[-2, :, :2]
    object_sizes = observation[-1, :, 2:]

    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])

    pin_idx = 1

    pin_positions = positions[pin_idx:]
    valid_pins = pin_positions[(pin_positions[:, 0] > 0) & (pin_positions[:, 1] > 0)]
    n_pins = len(valid_pins)
    center_pin_pos = np.mean(valid_pins, axis=0)
    
    player_position = positions[0]
    player_size = object_sizes[0]

    orientations = get_orientation(player_position, positions[1:], player_size, object_sizes[1:])

    prev_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)

    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    prev_distances = np.linalg.norm(prev_positions[0] - prev_positions[1:], axis=1)
    delta_distances = distances - prev_distances
    delta_distances[(positions[1:, 0] == 0) & (positions[1:, 1] == 0)] = 0

    orientation_center_pin = get_orientation(player_position, center_pin_pos, player_size, object_sizes[1])
    closest_distance_to_center = np.min(np.linalg.norm(center_pin_pos - positions[1:], axis=1))

    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, np.array([x_momentum, y_momentum]), 
                               distances, delta_distances, orientations, orientation_center_pin, np.array([n_pins, closest_distance_to_center])], axis=0, dtype=object)
    else:
         info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), distances, delta_distances, orientation_to_one_hot(orientations)
                               , orientation_to_one_hot(orientation_center_pin), np.array([n_pins, closest_distance_to_center])], axis=0, dtype=object)
    return info


def space_invaders_extraction(observation: np.ndarray, is_mixed: bool = True) -> np.ndarray:
    positions = observation[-1, :, :2]
    prev_positions = observation[-2, :, :2]
    object_sizes = observation[-1, :, 2:]

    x_momentum = get_x_momentum(observation[:, 0, :2])
    y_momentum = get_y_momentum(observation[:, 0, :2])

    player_position = positions[0]
    player_size = object_sizes[0]
    orientations = get_orientation(player_position, positions[1:], player_size, object_sizes[1:])

    prev_orientation = get_orientation(player_position, prev_positions[0], player_size, player_size)

    distances = np.linalg.norm(player_position - positions[1:], axis=1)
    prev_distances = np.linalg.norm(prev_positions[0] - prev_positions[1:], axis=1)
    delta_distances = distances - prev_distances
    delta_distances[(positions[1:, 0] == 0) & (positions[1:, 1] == 0)] = 0

    shield_idx = 1
    alien_idx = 4
    bullet_idx = 40
    aliens_per_row = 6
    alien_rows = 6

    below = (orientations == 'x_aligned and above') |  (orientations == 'semi left and above') | (orientations == 'semi right and above') 

    below_shield = below[shield_idx-1:alien_idx-1].any()
    below_alien = below[alien_idx-1:bullet_idx-1].any()
    below_bullet = below[bullet_idx-1:].any()
    
    alien_positions = positions[alien_idx:bullet_idx]
    below_aliens = alien_positions[below[alien_idx-1:bullet_idx-1]]
    below_first_row_alien = True if below_alien and below_aliens.size > 0 and np.max(below_aliens[:, 1] == np.max(alien_positions[:, 1])) else False

    def below_mapping(below_shield: bool, below_alien: bool, below_first_row_alien: bool, below_bullet: bool):
        below_info = 'no object'
        if below_shield:
            below_info = 'shield'
        elif below_alien: 
            if below_first_row_alien:
                below_info = 'alien first row'
            below_info = 'alien'
        elif below_bullet:
            below_info = 'bullet'
        return np.array([below_info])
    
    below_info = below_mapping(below_shield, below_alien, below_first_row_alien, below_bullet)

    def below_mapping_to_onehot(below_info: str):
        space_invaders_mapping = {
        'no object': 0,
        'alien' : 1,
        'alien first row': 2,
        'bullet': 3,
        'shield': 4
        }
        below_info_mapped = np.vectorize(space_invaders_mapping.get)(below_info)
        # Create one-hot encoding
        num_classes = len(space_invaders_mapping)
        one_hot_encoded = np.eye(num_classes)[below_info_mapped]
        # Flatten the one-hot encoded matrix
        return one_hot_encoded.flatten()

    j = 0
    counts = np.zeros(alien_rows)
    empty_rows = alien_rows
    for i in range(0, aliens_per_row*alien_rows, aliens_per_row):
        y_s = alien_positions[i:i+aliens_per_row, 1]
        counts[j] = aliens_per_row - np.sum(y_s == 0)
        j += 1
    empty_rows -= np.sum(counts == 0)

    counts = np.append(counts, empty_rows)

    if is_mixed:
        info = np.concatenate([player_position, prev_orientation, np.array([x_momentum, y_momentum]), distances, delta_distances, orientations, below_info, counts], axis=0, dtype=object)
    else:
         info = np.concatenate([player_position, orientation_to_one_hot(prev_orientation), x_momentum_to_one_hot(x_momentum), 
                               y_momentum_to_one_hot(y_momentum), distances, delta_distances, orientation_to_one_hot(orientations), below_mapping_to_onehot(below_info), counts], axis=0, dtype=object)
    return info