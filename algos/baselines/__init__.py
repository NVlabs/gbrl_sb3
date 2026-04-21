"""Baseline hybrid IL+RL algorithms for comparison with Split-AWR.

Available baselines (from `imitation` library):
- SQIL: Soft Q Imitation Learning (wraps SB3 DQN)
- BC: Behavioral Cloning (+ optional PPO fine-tuning)

Custom baselines:
- RLPD: RL with Prior Data (demos mixed into DQN replay buffer)
"""
