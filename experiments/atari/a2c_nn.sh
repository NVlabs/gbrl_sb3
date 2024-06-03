#!/bin/bash

# List of environments
ENVS=(
    'Alien-ramNoFrameskip-v4'
    'Amidar-ramNoFrameskip-v4'
    'Asteroids-ramNoFrameskip-v4'
    'Breakout-ramNoFrameskip-v4'
    'Gopher-ramNoFrameskip-v4'
    'Kangaroo-ramNoFrameskip-v4'
    'Krull-ramNoFrameskip-v4'
    'MsPacman-ramNoFrameskip-v4'
    'Pong-ramNoFrameskip-v4'
    'SpaceInvaders-ramNoFrameskip-v4'
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

COMMAND="python scripts/train.py --algo_type=a2c_nn \
    --atari_wrapper_kwargs=\"{\\\"clip_reward\\\": true}\" \
    --batch_size=256 \
    --device=cuda \
    --ent_coef=0.0013840164699101328 \
    --env_name=$USER_ENV \
    --env_type=atari \
    --gae_lambda=0.95 \
    --gamma=0.993 \
    --learning_rate=0.0009574409467192832 \
    --max_grad_norm=1.213928417332135 \
    --n_steps=64 \
    --num_envs=8 \
    --total_n_steps=10000000 \
    --use_rms_prop=True \
    --wrapper=normalize \
    --wrapper_kwargs=\"{\\\"training\\\": true, \\\"norm_obs\\\": false, \\\"norm_reward\\\": true}\""

# Add the seed argument if provided
if [ ! -z "$SEED" ]; then
    COMMAND+=" --seed=$SEED"
fi

# Run the command
eval $COMMAND
