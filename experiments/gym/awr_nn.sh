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

is_valid_env() {
    for ENV in "${ENVS[@]}"; do
        if [ "$ENV" == "$ENV_NAME" ]; then
            return 0
        fi
    done
    return 1
}

if ! is_valid_env "$ENV_NAME"; then
    echo "Error: Invalid environment name."
    echo "Valid environments are:"
    for ENV in "${ENVS[@]}"; do
        echo "  $ENV"
    done
    exit 1
fi

# Define total_n_steps based on environment
if [ "$ENV_NAME" == "LunarLander-v2" ]; then
    TOTAL_N_STEPS=1500000
else
    TOTAL_N_STEPS=1000000
fi

# Define the command for all environments
COMMAND="python scripts/train.py --actor_learning_rate=5e-05 \
    --algo_type=awr_nn \
    --batch_size=64 \
    --buffer_size=50000 \
    --critic_learning_rate=0.0001 \
    --device=cuda \
    --ent_coef=4.044271827905092e-05 \
    --env_name=$ENV_NAME \
    --env_type=gym \
    --gae_lambda=0.95 \
    --gamma=0.99 \
    --learning_starts=10000 \
    --max_grad_norm=0.5 \
    --n_steps=2000 \
    --num_envs=1 \
    --total_n_steps=$TOTAL_N_STEPS \
    --wrapper=normalize"

# Add the seed argument if provided
if [ ! -z "$SEED" ]; then
    COMMAND+=" --seed=$SEED"
fi

# Run the command
eval $COMMAND
