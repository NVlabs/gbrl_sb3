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
    'academy_3_vs_1_with_keeper'
    'academy_corner'
    'academy_counterattack_easy'
    'academy_counterattack_hard'
    'academy_empty_goal'
    'academy_empty_goal_close'
    'academy_pass_and_shoot_with_keeper'
    'academy_run_pass_and_shoot_with_keeper'
    'academy_run_to_score'
    'academy_run_to_score_with_keeper'
    'academy_single_goal_versus_lazy'
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

# Check if a seed argument is provided
SEED=""
if [ "$2" ]; then
    SEED=$2
fi

echo "Running training for environment: $USER_ENV"

COMMAND="python scripts/train.py --actor_learning_rate=0.0005454700890743699 \
    --algo_type=awr_nn \
    --batch_size=64 \
    --buffer_size=100000 \
    --critic_learning_rate=0.00513208543832738 \
    --device=cuda \
    --ent_coef=0.00023871925619838431 \
    --env_kwargs=\"{\\\"rewards\\\": \\\"scoring,checkpoints\\\", \\\"representation\\\": \\\"simple115v2\\\"}\" \
    --env_name=$USER_ENV \
    --env_type=football \
    --gae_lambda=0.95 \
    --gamma=0.993 \
    --learning_starts=10000 \
    --max_grad_norm=1.650169566704384 \
    --n_steps=2048 \
    --num_envs=1 \
    --total_n_steps=10000000 \
    --wrapper=None"

# Add the seed argument if provided
if [ ! -z "$SEED" ]; then
    COMMAND+=" --seed=$SEED"
fi

# Run the command
eval $COMMAND
