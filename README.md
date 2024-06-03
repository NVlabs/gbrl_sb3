# Gradient Boosting Reinforcement Learning in stable_baselines3 (GBRL_SB3)
GBRL is a Python-based GBT library designed and optimized for reinforcement learning (RL). GBRL is implemented in C++/CUDA aimed to seamlessly integrate within popular RL libraries.

***This repository contains a GBRL based wrapper for stable_baselines3 [1] algorithm.***

### Key Features:
- GBTs Tailored for RL: GBRL adapts the power of Gradient Boosting Trees to the unique challenges of RL environments, including non-stationarity and delayed feedback.
- Optimized Actor-Critic Architecture: GBRL features a shared tree-based structure for policy and value functions. This significantly reduces memory and computational overhead, enabling it to tackle complex, high-dimensional RL problems.
- Hardware Acceleration: GBRL leverages CUDA for hardware-accelerated computation, ensuring efficiency and speed.
- Seamless Integration: GBRL is designed for easy integration with popular RL libraries, making it readily accessible for practitioners.

## Features
GBRL based implementation of  
- PPO 
- A2C
- AWR

GBRL SB3 supports the following environments:  
- Gymansium environments [2]
- Atari-ram [3]
- Football [4]
- MiniGrid [5] including a custom categorical feature wrapper


## Getting started

## Getting started
### DOCKER USAGE 
#### Prerequisites
- Docker 19 or newer.
- Access to NVIDIA Docker Catalog. Visit the [NGC website](https://ngc.nvidia.com/signup) and follow the instructions. This will grant you access to the base docker image (from the Dockerfile) and ability to run on NVIDIA GPU using the nvidia runtime flag.

#### INSTALLATION
building docker
```
docker build -f Dockerfile -t <your-image-name:tag> .
```  
Running docker
```
docker run --runtime=nvidia -it <your-image-name:tag> /bin/bash
```  

### Local GBRL_SB3 installation
GBRL_SB3 is based on [the GBRL library](https://github.com/NVlabs/gbrl), stable_baselines3, and other popular python libraries. To run GBRL_SB3 locally please install the necessary dependencies by running:
```
pip install -r requirements.txt
``` 

The Google Research Football installation is not part of the requirements due to additionaly non-python dependencies and should be installed separetly (see [gfootball repository](https://github.com/google-research/football/tree/master)).

For GPU support GBRL looks for `CUDA_PATH` or `CUDA_HOME` environment variables. Unless found, GBRL will automatically compile only for CPU.

Verify that GPU is visible by running  
```
import gbrl

gbrl.cuda_available()
```

*OPTIONAL*  
For GBRL tree visualization make sure graphviz is installed before installing GBRL.


## Training
A general training script is located at `scripts/train.py`.  
 Configuration (not tuned hyperparameters) yaml is provided at `config/defaults.yaml`.  
valid CLI arguments are found at `config/args.py`.  

Example - running from project root directory
```
python3 scripts/train.py --algo_type=ppo_gbrl --batch_size=512 --clip_range=0.2 --device=cuda --ent_coef=0 --env_name=MiniGrid-Unlock-v0 --env_type=minigrid --gae_lambda=0.95 --gamma=0.99 --grow_policy=oblivious  --n_epochs=20 --n_steps=256 --num_envs=16 --pg_lr=0.17  --total_n_steps=1000000 --vf_lr=0.01
```

For tracking with weights and biases use CLI args:
- project=<project_name> 
- wandb=true
- group_name=<group_name>
- run_name=<run_name>

## Experiments Reproducibility
Exact training reproduction with GBRL is not possible as GPU training is non-deterministic. This is due to the non-deterministic nature of floating point summation. However, running the training scripts with the reported hyperparameters are expected to produce similar results.   

Experiment scripts are located at `experiments/`.  
Running is done via a bash script per algorithm per environment with the following two arguments: `scenario_name` and `seed`. For example, the run command for `CartPole-v1`, with `seed=0` GBRL PPO is:
```
experiments/gym/ppo_gbrl.sh CartPole-v1 0
```

### Beta Implementations (not tested)
- SAC
- DQN

## References
[1] Raffin et al. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine
Learning Research, 22(268):1–8, 2021. URL http://jmlr.org/papers/v22/20-1364.html.  

[2] Towers et al.  Gymnasium,March 2023. URL https://zenodo.org/record/8127025.

[3] Bellemare et al. The arcade learning environment: An
evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:253–279,
June 2013. ISSN 1076-9757. doi: 10.1613/jair.3912. URL http://dx.doi.org/10.1613/jair.
3912.

[4] Kurach et al. Google
research football: A novel reinforcement learning environment, 2020

[5] Chevalier-Boisvert et al. Minigrid & miniworld: Modular &
customizable reinforcement learning environments for goal-oriented tasks. CoRR, abs/2306.13831,
2023

# Cite