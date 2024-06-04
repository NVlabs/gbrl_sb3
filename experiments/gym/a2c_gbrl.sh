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

ENV_NAME=$1
SEED=""
if [ "$2" ]; then
    SEED=$2
fi

ENVS=(
    'Acrobot-v1'
    'CartPole-v1'
    'LunarLander-v2'
    'Pendulum-v1'
    'MountainCarContinuous-v0'
    'MountainCar-v0'
)

# Define commands for each environment
if [ "$ENV_NAME" == "Acrobot-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_gbrl \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=1 \
        --gamma=0.99 \
        --n_steps=8 \
        --normalize_advantage=True \
        --split_score_func=Cosine \
        --num_envs=4 \
        --policy_lr=0.7941291726157867 \
        --total_n_steps=1000000 \
        --vf_coef=0.5 \
        --value_lr=0.03156352960142528 \
        --wrapper=None \
        --wrapper_kwargs=None"

elif [ "$ENV_NAME" == "CartPole-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_gbrl \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --split_score_func=Cosine \
        --gae_lambda=1 \
        --gamma=0.99 \
        --n_steps=8 \
        --normalize_advantage=True \
        --num_envs=16 \
        --policy_lr=0.13522782541918751 \
        --total_n_steps=1000000 \
        --vf_coef=0.5 \
        --value_lr=0.04741932245744507 \
        --wrapper=None \
        --wrapper_kwargs=None"

elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_gbrl \
        --device=cuda \
        --ent_coef=1e-05 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --split_score_func=Cosine \
        --gae_lambda=0.9 \
        --gamma=0.9 \
        --n_steps=10 \
        --normalize_advantage=True \
        --num_envs=32 \
        --policy_lr=0.0031713221315178178 \
        --log_std_lr=0.000184599955396231 \
        --log_std_init=-2 \
        --total_n_steps=1000000 \
        --vf_coef=0.5 \
        --value_lr=0.05696582939066114 \
        --wrapper=None \
        --wrapper_kwargs=None"

elif [ "$ENV_NAME" == "MountainCar-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_gbrl \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=1 \
        --gamma=0.99 \
        --n_steps=8 \
        --normalize_advantage=True \
        --num_envs=16 \
        --policy_lr=0.6402117929865448 \
        --total_n_steps=1000000 \
        --vf_coef=0.5 \
        --value_lr=0.0327367056021962 \
        --wrapper=None \
        --wrapper_kwargs=None"

elif [ "$ENV_NAME" == "MountainCarContinuous-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_gbrl \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=1 \
        --gamma=0.95 \
        --log_std_init=-1 \
        --log_std_lr=0.0004139447331747008 \
        --n_steps=128 \
        --normalize_advantage=True \
        --num_envs=1 \
        --policy_lr=0.000892124281617261 \
        --grow_policy=oblivious \
        --total_n_steps=1000000 \
        --vf_coef=0.4 \
        --value_lr=2.816670478037442e-04 \
        --wrapper=normalize \
        --wrapper_kwargs=\"{\\\"training\\\": true, \\\"norm_obs\\\": false, \\\"norm_reward\\\": true}\""

elif [ "$ENV_NAME" == "LunarLander-v2" ]; then
    COMMAND="python scripts/train.py --algo_type=a2c_gbrl \
        --device=cuda \
        --ent_coef=1e-05 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=1 \
        --gamma=0.995 \
        --n_steps=5 \
        --normalize_advantage=True \
        --num_envs=32 \
        --policy_lr=0.16027457635754927 \
        --total_n_steps=1500000 \
        --vf_coef=0.5 \
        --value_lr=0.04232447250591277 \
        --wrapper=None \
        --wrapper_kwargs=None"

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
