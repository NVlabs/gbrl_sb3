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
if [ "$ENV_NAME" == "MountainCar-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=awr_gbrl \
        --batch_size=64 \
        --buffer_size=50000 \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gradient_steps=150 \
        --grow_policy=oblivious \
        --num_envs=1 \
        --policy_lr=0.08938480943840346 \
        --reward_mode=gae \
        --total_n_steps=1000000 \
        --train_freq=2000 \
        --value_lr=0.08317628198321741 \
        --wrapper=None"

elif [ "$ENV_NAME" == "MountainCarContinuous-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=awr_gbrl \
        --batch_size=64 \
        --buffer_size=50000 \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gradient_steps=150 \
        --grow_policy=oblivious \
        --max_policy_grad_norm=150 \
        --num_envs=1 \
        --policy_lr=0.0003938480943840346 \
        --reward_mode=gae \
        --total_n_steps=1000000 \
        --train_freq=2000 \
        --value_lr=0.08317628198321741 \
        --wrapper=None"

elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=awr_gbrl \
        --device=cuda \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.9 \
        --gamma=0.9 \
        --gradient_steps=50 \
        --grow_policy=oblivious \
        --log_std_init=-2 \
        --log_std_lr=0.0005318970196570411 \
        --normalize_advantage=True \
        --num_envs=1 \
        --policy_lr=0.00378772024172414 \
        --total_n_steps=1000000 \
        --train_freq=1000 \
        --value_lr=0.07353926885096224 \
        --wrapper=None \
        --wrapper_kwargs=None"

elif [ "$ENV_NAME" == "LunarLander-v2" ]; then
    COMMAND="python scripts/train.py --algo_type=awr_gbrl \
        --batch_size=1024 \
        --buffer_size=50000 \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gradient_steps=150 \
        --num_envs=1 \
        --policy_lr=0.05073960309815198 \
        --reward_mode=gae \
        --total_n_steps=1500000 \
        --train_freq=2000 \
        --value_lr=0.10526702677422366 \
        --wrapper=normalize"

elif [ "$ENV_NAME" == "Acrobot-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=awr_gbrl \
        --batch_size=1024 \
        --buffer_size=50000 \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gradient_steps=150 \
        --num_envs=1 \
        --policy_lr=0.05073960309815198 \
        --reward_mode=gae \
        --total_n_steps=1500000 \
        --train_freq=2000 \
        --value_lr=0.10526702677422366 \
        --wrapper=normalize"

elif [ "$ENV_NAME" == "CartPole-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=awr_gbrl \
        --batch_size=1024 \
        --buffer_size=50000 \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gradient_steps=150 \
        --num_envs=1 \
        --policy_lr=0.05073960309815198 \
        --reward_mode=gae \
        --total_n_steps=1500000 \
        --train_freq=2000 \
        --value_lr=0.10526702677422366 \
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
