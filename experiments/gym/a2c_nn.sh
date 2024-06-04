##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
#!/bin/bash

# Check if environment name and seed are provided
if [ -z "$1" ]; then
    echo "Usage: $0 <env_name> [seed]"
    exit 1
fi

ENVS=(
    'Acrobot-v1'
    'CartPole-v1'
    'LunarLander-v2'
    'Pendulum-v1'
    'MountainCarContinuous-v0'
    'MountainCar-v0'
)


ENV_NAME=$1
SEED=""
if [ "$2" ]; then
    SEED=$2
fi

# Define commands for each environment
if [ "$ENV_NAME" == "CartPole-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_nn \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --num_envs=8 \
        --total_n_steps=500000 \
        --policy='MlpPolicy'"

elif [ "$ENV_NAME" == "LunarLander-v2" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_nn \
        --device=cuda \
        --ent_coef=0.00001 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gamma=0.995 \
        --n_steps=5 \
        --num_envs=8 \
        --total_n_steps=200000 \
        --learning_rate=lin_0.00083 \
        --policy='MlpPolicy'"

elif [ "$ENV_NAME" == "MountainCar-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_nn \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --num_envs=16 \
        --total_n_steps=1000000 \
        --policy='MlpPolicy' \
        --wrapper=normalize"

elif [ "$ENV_NAME" == "Acrobot-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_nn \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --num_envs=16 \
        --total_n_steps=500000 \
        --policy='MlpPolicy' \
        --wrapper=normalize"

elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_nn \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --max_grad_norm=0.5 \
        --n_steps=8 \
        --gae_lambda=0.9 \
        --vf_coef=0.4 \
        --gamma=0.9 \
        --use_rms_prop=True \
        --normalize_advantage=False \
        --num_envs=8 \
        --total_n_steps=1000000 \
        --learning_rate=lin_7e-4 \
        --use_sde=True \
        --policy_kwargs=\"{\\\"log_std_init\\\": -2, \\\"ortho_init\\\": false}\" \
        --wrapper=normalize"

elif [ "$ENV_NAME" == "MountainCarContinuous-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_nn \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --n_steps=100 \
        --use_sde=True \
        --sde_sample_freq=16 \
        --num_envs=4 \
        --total_n_steps=100000 \
        --policy_kwargs=\"{\\\"log_std_init\\\": 0.0, \\\"ortho_init\\\": false}\" \
        --wrapper=normalize"

else
    echo "Unknown environment name: $ENV_NAME"
    echo "Valid environments are:"
    for ENV in "${ENVS[@]}"; do
        echo "  $ENV"
    done
    exit 1
fi

# Add the seed argument if provided
if [ ! -z "$SEED" ]; then
    COMMAND+=" --seed=$SEED"
fi

# Run the command
eval $COMMAND
