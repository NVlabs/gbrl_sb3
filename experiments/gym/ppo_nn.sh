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
    COMMAND="python scripts/train.py --algo_type=ppo_nn \
        --batch_size=64 \
        --device=cuda \
        --ent_coef=0.01 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.98 \
        --gamma=0.999 \
        --n_epochs=4 \
        --n_steps=1024 \
        --num_envs=16 \
        --total_n_steps=1000000 \
        --use_sde=False \
        --wrapper=None"

elif [ "$ENV_NAME" == "Acrobot-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_nn \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.94 \
        --gamma=0.99 \
        --n_epochs=4 \
        --n_steps=256 \
        --num_envs=16 \
        --total_n_steps=1000000 \
        --wrapper=normalize"

elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_nn \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.95 \
        --gamma=0.9 \
        --n_epochs=10 \
        --n_steps=1024 \
        --num_envs=4 \
        --total_n_steps=100000 \
        --learning_rate=0.001 \
        --clip_range=0.2 \
        --use_sde=True \
        --sde_sample_freq=4 \
        --wrapper=None"

elif [ "$ENV_NAME" == "CartPole-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_nn \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.8 \
        --gamma=0.98 \
        --n_epochs=20 \
        --n_steps=32 \
        --num_envs=8 \
        --total_n_steps=100000 \
        --batch_size=256 \
        --learning_rate=lin_0.001 \
        --clip_range=lin_0.2 \
        --wrapper=None"

elif [ "$ENV_NAME" == "MountainCar-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_nn \
        --device=cuda \
        --ent_coef=0.0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.98 \
        --gamma=0.99 \
        --n_epochs=4 \
        --n_steps=16 \
        --num_envs=16 \
        --total_n_steps=1000000 \
        --wrapper=normalize"

elif [ "$ENV_NAME" == "MountainCarContinuous-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=ppo_nn \
        --device=cuda \
        --ent_coef=0.00429 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gae_lambda=0.9 \
        --gamma=0.9999 \
        --n_epochs=10 \
        --n_steps=8 \
        --num_envs=1 \
        --total_n_steps=20000 \
        --batch_size=256 \
        --learning_rate=7.77e-05 \
        --clip_range=0.1 \
        --max_grad_norm=5 \
        --vf_coef=0.19 \
        --use_sde=True \
        --policy_kwargs=\"{\\\"log_std_init\\\": -3.29, \\\"ortho_init\\\": false}\" \
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
