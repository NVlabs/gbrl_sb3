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
if [ "$ENV_NAME" == "LunarLander-v2" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_gbrl \
        --batch_size=256 \
        --clip_range=0.2 \
        --device=cuda \
        --ent_coef=0.033440814554543896 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.98 \
        --gamma=0.999 \
        --n_epochs=20 \
        --n_steps=512 \
        --num_envs=16 \
        --policy_lr=0.03113195249121072 \
        --total_n_steps=1500000 \
        --value_lr=0.0031781512018050338"

elif [ "$ENV_NAME" == "MountainCar-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_gbrl \
        --batch_size=256 \
        --clip_range=0.2 \
        --device=cuda \
        --ent_coef=0.033440814554543896 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.98 \
        --gamma=0.999 \
        --max_policy_grad_norm=100 \
        --max_value_grad_norm=10 \
        --n_epochs=20 \
        --n_steps=512 \
        --num_envs=16 \
        --policy_lr=0.03113195249121072 \
        --total_n_steps=1000000 \
        --value_lr=0.0031781512018050338"

elif [ "$ENV_NAME" == "CartPole-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_gbrl \
        --batch_size=64 \
        --clip_range=0.2 \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.8 \
        --gamma=0.98 \
        --max_policy_grad_norm=100 \
        --max_value_grad_norm=10 \
        --n_epochs=1 \
        --n_steps=128 \
        --num_envs=8 \
        --policy_lr=0.029 \
        --total_n_steps=1000000 \
        --value_lr=0.015"

elif [ "$ENV_NAME" == "MountainCarContinuous-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_gbrl \
        --batch_size=256 \
        --clip_range=0.2 \
        --device=cuda \
        --ent_coef=0.033440814554543896 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.98 \
        --gamma=0.999 \
        --max_policy_grad_norm=100 \
        --max_value_grad_norm=10 \
        --n_epochs=20 \
        --n_steps=512 \
        --num_envs=16 \
        --policy_lr=0.03113195249121072 \
        --total_n_steps=1000000 \
        --value_lr=0.0031781512018050338"

elif [ "$ENV_NAME" == "Acrobot-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_gbrl \
        --batch_size=512 \
        --clip_range=0.2 \
        --ent_coef=0 \
        --device=cuda \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.94 \
        --gamma=0.99 \
        --n_epochs=20 \
        --n_steps=128 \
        --num_envs=16 \
        --policy_lr=0.160275707585934 \
        --total_n_steps=1000000 \
        --value_lr=0.034185252451467855"

elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_gbrl \
        --batch_size=512 \
        --clip_range=0.2 \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --device=cuda \
        --gae_lambda=0.9375823088645378 \
        --gamma=0.9147245889494424 \
        --max_policy_grad_norm=100 \
        --max_value_grad_norm=10 \
        --n_epochs=20 \
        --n_steps=256 \
        --num_envs=16 \
        --policy_lr=0.031246793805561623 \
        --total_n_steps=1000000 \
        --value_lr=0.01315225738736567"
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
