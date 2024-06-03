# Gradient Boosting Reinforcement Learning for stable_baselines3 (GBRL_SB3)
GBRL is a Python-based GBT library designed and optimized for reinforcement learning (RL). GBRL is implemented in C++/CUDA. 

This repository contains a GBRL based wrapper for stable_baselines3 [1] algorithm.

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
### Docker 
Building cpu only docker
```
docker build -f Dockerfile.cpu -t <your-image-name:cpu-tag> .
```  
Running cpu only docker
```
docker run --runtime=nvidia -it <your-image-name:cpu-tag> /bin/bash
```  
Building gpu docker
```
docker build -f Dockerfile.gpu -t <your-image-name:gpu-tag> .
```  
Running gpu docker
```
docker run --runtime=nvidia -it <your-image-name:gpu-tag> /bin/bash
```  

### Training
A general training script is located at `scripts/train.py`.  
Default configurations for all algorithms are provided at `config/defaults.yaml`.  
CLI arguments are found at `config/args.py`.  

## Extra 

Beta GBRL implementations (not tested):  
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

