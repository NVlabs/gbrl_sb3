##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import numpy as np 

def get_x_orientation(player_x, object_x):
    if player_x > object_x:
        return 'right'
    elif player_x < object_x:
        return 'left'
    else:
        return 'below'
    

def gopher_extraction(positions: np.ndarray) -> np.ndarray:


    player_position = positions[0]
    gopher_position = positions[1]
    gopher_distance = np.linalg.norm(player_position - gopher_position)
    gopher_orientation = get_x_orientation(player_position[0], gopher_position[0])
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
    
    return np.array([player_position[0], player_position[1], gopher_position[0], gopher_position[1], gopher_distance, 
                     str(gopher_orientation),str(aligned_blocks_exist), nearest_aligned_pos[0], nearest_aligned_pos[1],
                     aligned_dist, aligned_orientation, str(almost_aligned_block_exist), nearest_almost_aligned_pos[0], 
                     nearest_almost_aligned_pos[1], almost_aligned_dist, almost_aligned_orientation], dtype=object)

def breakout_extraction(positions: np.ndarray, prev_positions: np.ndarray) -> np.ndarray:
    player_position = positions[0]
    ball_position = positions[1]
    prev_ball_position = prev_positions[1]
    # return player_position
    ball_distance = np.linalg.norm(player_position - ball_position)
    ball_orientation = get_x_orientation(player_position[0], ball_position[0])
    ball_velocity = ball_position - prev_ball_position
    n_rows  = 7 # + 0

    empty_block_first_idx = 2
    empty_blocks = positions[empty_block_first_idx:]
    unique_x, unique_idx = np.unique(empty_blocks[:, 0], return_index=True)
    counts = np.sum(empty_blocks[:, 0][:, None] == empty_blocks[unique_idx, 0], axis=0)
    counts[empty_blocks[unique_idx][:, 0] == 0] = 0
    columns = np.zeros((n_rows, 3), dtype=object)
    columns[:len(unique_idx), 0] = unique_x
    columns[:len(unique_idx), 2] = counts
    
    for i, x in enumerate(unique_x):
        columns[i, 1] = np.max(empty_blocks[empty_blocks[:, 0] == x, 1])

    columns[:, 0] = columns[:, 0].astype(float)
    columns[:, 1] = columns[:, 0].astype(float)
    columns[:, 2] = columns[:, 2].astype(float)
    columns = columns.flatten()

    info = np.array([player_position[0], player_position[1], ball_position[0], ball_position[1], ball_distance, 
                     str(ball_orientation), ball_velocity[0], ball_velocity[1]], dtype=object)

    return np.append(info, columns)

def general_extraction(positions: np.ndarray) -> np.ndarray:
    player_position = positions[0]
    distances = np.linalg.norm(player_position - positions, axis=1)
    delta_x = player_position[0] - positions[:, 0]
    delta_x[delta_x < 0] = -1
    delta_x[delta_x > 0] = 1
    delta_y = player_position[1] - positions[:, 1]
    delta_y[delta_y < 0] = -1
    delta_y[delta_y > 0] = 1
    delta_x = delta_x.astype(str)
    delta_x[delta_x == '-1.0'] = 'left'
    delta_x[delta_x == '1.0'] = 'right'
    delta_y = delta_y.astype(str)
    delta_y[delta_y == '-1.0'] = 'below'
    delta_y[delta_y == '1.0'] = 'above'
    info = np.concatenate([player_position, distances, delta_x, delta_y], axis=0, dtype=object)
    return info.flatten()

    
    
