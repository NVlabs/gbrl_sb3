##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
#!/bin/bash

# List of environments
ENVS=(
    'MiniGrid-DoorKey-5x5-v0'
    'MiniGrid-Empty-Random-5x5-v0'
    'MiniGrid-Fetch-5x5-N2-v0'
    'MiniGrid-FourRooms-v0'
    'MiniGrid-GoToDoor-5x5-v0'
    'MiniGrid-KeyCorridorS3R1-v0'
    'MiniGrid-PutNear-6x6-N2-v0'
    'MiniGrid-RedBlueDoors-6x6-v0'
    'MiniGrid-Unlock-v0'
)

# Short environments list
short_envs=(
    'MiniGrid-KeyCorridorS3R1-v0'
    'MiniGrid-Empty-Random-5x5-v0'
    'MiniGrid-RedBlueDoors-6x6-v0'
    'MiniGrid-GoToDoor-5x5-v0'
    'MiniGrid-Unlock-v0'
    'MiniGrid-DoorKey-5x5-v0'
)

# Function to check if an environment is valid
is_valid_env() {
    for ENV in "${ENVS[@]}"; do
        if [ "$ENV" == "$1" ]; then
            return 0
        fi
    done
    return 1
}

# Check if an environment argument is provided
if [ -z "$1" ]; then
    echo "Error: No environment provided."
    echo "Usage: $0 <env_name> [seed]"
    echo "Valid environments are:"
    for ENV in "${ENVS[@]}"; do
        echo "  $ENV"
    done
    exit 1
fi

# Check if the provided environment is valid
USER_ENV=$1
if ! is_valid_env "$USER_ENV"; then
    echo "Error: Invalid environment name."
    echo "Valid environments are:"
    for ENV in "${ENVS[@]}"; do
        echo "  $ENV"
    done
    exit 1
fi

# Check if the environment is in the short_envs list
if [[ " ${short_envs[@]} " == *" $USER_ENV "* ]]; then
    TOTAL_N_STEPS=1000000
else
    TOTAL_N_STEPS=10000000
fi

# Check if a seed argument is provided
SEED=""
if [ "$2" ]; then
    SEED=$2
fi

echo "Running training for environment: $USER_ENV"

COMMAND="python scripts/train.py --batch_size=64 \
    --clip_range=0.2 \
    --device=cuda \
    --ent_coef=0 \
    --env_name=$USER_ENV \
    --gae_lambda=0.95 \
    --gamma=0.993 \
    --learning_rate=0.00025 \
    --max_grad_norm=0.5 \
    --n_epochs=10 \
    --n_steps=128 \
    --num_envs=8 \
    --total_n_steps=$TOTAL_N_STEPS \
    --algo_type=ppo_nn \
    --env_type=minigrid"

# Add the seed argument if provided
if [ ! -z "$SEED" ]; then
    COMMAND+=" --seed=$SEED"
fi

# Run the command
eval $COMMAND
