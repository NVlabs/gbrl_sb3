import sys
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import explained_variance, safe_mean
import copy
from torch.nn import functional as F

from algos.safety.ppo_lag import PPOLag
from policies.cost_actor_critic import CostActorCriticPolicy

SelfCUP = TypeVar("SelfCUP", bound="CUP")

class CUP(PPOLag):
    """
    Constrained Update Projection (CUP) Approach to Safe Policy Optimization.
    
    CUP performs a two-step policy update:
    1. Standard PPO update to improve reward performance
    2. Additional projection step to satisfy cost constraints using KL-regularized optimization
    """
    
    def __init__(
        self,
        policy: Union[str, Type[CostActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        clip_range_cf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        cf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        cost_limit: float = 25.0,
        lagrangian_multiplier_init: float = 0.001,
        lambda_lr: float = 0.035,
        lambda_optimizer: str = 'Adam',
        lagrangian_upper_bound: float | None = None,
        # CUP-specific parameters
        cup_update_iters: int = 10,
        cup_kl_early_stop: bool = True,
        cup_target_kl: float = 0.02,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            clip_range_cf=clip_range_cf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            cf_coef=cf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
            lagrangian_upper_bound=lagrangian_upper_bound,
        )
        
        self.cup_update_iters = cup_update_iters
        self.cup_kl_early_stop = cup_kl_early_stop
        self.cup_target_kl = cup_target_kl
        
    def train(self) -> None:
        """
        Update policy using the CUP two-step approach:
        1. Update Lagrange multiplier
        2. First step: Standard PPOLag update (Reward maximization)
        3. Second step: CUP projection update (Cost constraint projection)
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        if self.clip_range_cf is not None:
            clip_range_cf = self.clip_range_cf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, cost_losses = [], [], []
        clip_fractions = []

        continue_training = True
        
        # --- Update Lagrange multiplier ---
        self.lambda_optimizer.zero_grad()
        lambda_loss = -self.lagrangian_multiplier * (self.ep_cost_mean - self.cost_limit)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0.0, self.lagrangian_upper_bound)

        # --- Step 1: Standard PPO update (reward-only, no cost scalarization) ---
        # NOTE: Unlike PPOLag, CUP uses PURE REWARD advantage in Step 1
        # The cost constraint is handled in Step 2 (projection step)
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, costs, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                costs = costs.flatten()

                # Normalize REWARD advantage only (pure PPO step)
                # CUP Step 1 uses ONLY reward advantage - cost handling is in Step 2
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Cost loss
                if self.clip_range_cf is None:
                    costs_pred = costs
                else:
                    costs_pred = rollout_data.old_value_costs + th.clamp(
                        costs - rollout_data.old_value_costs, -clip_range_cf, clip_range_cf
                    )
                cost_loss = F.mse_loss(rollout_data.cost_returns, costs_pred)
                cost_losses.append(cost_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.cf_coef * cost_loss

                # Calculate approximate KL for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/cost_loss", np.mean(cost_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        # --- Step 2: CUP projection step ---
        self._cup_projection_update()
        
    def _get_actor_parameters(self) -> List[th.nn.Parameter]:
        """
        Get all actor parameters for CUP Step 2 optimization.
        Matches omnisafe's actor.parameters() which includes mean network + log_std.
        """
        actor_params = []
        
        # Action net (output layer)
        actor_params.extend(self.policy.action_net.parameters())
        
        # Policy MLP (hidden layers)
        if hasattr(self.policy.mlp_extractor, "policy_net"):
            actor_params.extend(self.policy.mlp_extractor.policy_net.parameters())
        
        # Log std (if DiagGaussian - learnable std)
        if hasattr(self.policy, "log_std") and self.policy.log_std is not None:
            actor_params.append(self.policy.log_std)
        
        # Policy feature extractor (only if NOT shared with critics)
        if hasattr(self.policy, "share_features_extractor") and not self.policy.share_features_extractor:
            if hasattr(self.policy, "pi_features_extractor"):
                actor_params.extend(self.policy.pi_features_extractor.parameters())
        
        return list(actor_params)
        
    def _cup_projection_update(self) -> None:
        """
        Perform the second-step CUP projection update.
        Safely freezes Critic/Features to prevent drift and anchors to Step 1 policy.
        """
        self.policy.set_training_mode(True)

        # 1. Freeze EVERYTHING to stop Adam optimizer momentum from Step 1 affecting the Critic
        for param in self.policy.parameters():
            param.requires_grad = False

        # 2. Unfreeze only the Actor components (matching omnisafe's actor.parameters())
        actor_params = self._get_actor_parameters()
        for param in actor_params:
            param.requires_grad = True
        
        # 3. Create the Anchor Policy (The "Old" Distribution)
        # We copy the policy state exactly as it is after Step 1.
        # This gives us a frozen reference point for KL calculation.
        old_mlp_extractor = copy.deepcopy(self.policy.mlp_extractor)
        old_action_net = copy.deepcopy(self.policy.action_net)
        
        second_step_losses = []
        second_step_ratios = []
        second_step_entropies = []
        final_iter = self.n_epochs
        
        # 4. CUP Optimization Loop
        for i in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                    
                with th.no_grad():
                    old_distribution = self.policy.get_distribution_from_extractor(rollout_data.observations, old_mlp_extractor, old_action_net)

                # B. Get Current Distribution (The one changing in Step 2)
                # We use get_distribution to skip Value network computation entirely
                new_distribution = self.policy.get_distribution(rollout_data.observations)
                
                # C. Calculate Ratio vs Sampling Distribution (Step 0)
                # We must use rollout_data.old_log_prob (from Step 0) as denominator
                new_log_prob = new_distribution.log_prob(actions)
                if len(new_log_prob.shape) > 1:
                    new_log_prob = new_log_prob.sum(dim=1)
                
                ratio = th.exp(new_log_prob - rollout_data.old_log_prob)
                
                # D. Calculate KL Divergence vs Anchor Distribution (Step 1)
                # KL(new || old) - penalize new policy for deviating from Step 1 anchor
                # Keep per-sample KL (sum over action dims, keep batch dim)
                kl = th.distributions.kl_divergence(new_distribution.distribution, old_distribution.distribution).sum(-1, keepdim=True)
                
                # E. Calculate Loss
                # CUP projection: minimize cost surrogate + KL penalty
                # Coef = (1 - gamma * gae_lambda) / (1 - gamma) - from omnisafe reference
                coef = (1 - self.gamma * self.gae_lambda) / (1 - self.gamma)
                
                # Loss = E[lambda * coef * ratio * A_c + KL] (per-sample KL, then mean)
                # Note: omnisafe does NOT normalize cost advantages in the second step
                adv_c = rollout_data.advantages_costs
                if len(adv_c.shape) == 1:
                    adv_c = adv_c.unsqueeze(-1)
                loss = (self.lagrangian_multiplier * coef * ratio.unsqueeze(-1) * adv_c + kl).mean()

                # F. Optimize (only actor parameters, matching omnisafe)
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Only clip gradients for actor parameters (matching omnisafe actor.parameters())
                th.nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                self.policy.optimizer.step()
                
                # Logging stats
                second_step_losses.append(loss.item())
                second_step_ratios.append(ratio.mean().item())
                second_step_entropies.append(new_distribution.entropy().mean().item())

            # 5. Early Stopping Check
            if self.cup_kl_early_stop:
                # Approximate check using the last batch's KL to avoid full pass
                # kl has shape [batch, 1], need to take mean first
                mean_kl = kl.mean().item()
                if mean_kl > self.cup_target_kl:
                    final_iter = i + 1
                    if self.verbose >= 1:
                        print(f"CUP early stopping at iter {i + 1} due to KL: {mean_kl:.4f}")
                    break
        
        del old_mlp_extractor
        del old_action_net
        # 6. Unfreeze everything for the next rollout phase
        for param in self.policy.parameters():
            param.requires_grad = True

        # Logging
        self.logger.record("train/cup_loss_pi_cost", np.mean(second_step_losses))
        self.logger.record("train/cup_stop_iter", final_iter)
        self.logger.record("train/cup_entropy", np.mean(second_step_entropies))
        self.logger.record("train/cup_policy_ratio", np.mean(second_step_ratios))
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0 and "s" in self.ep_info_buffer[0]:
            self.logger.record("rollout/ep_scalarization_mean", safe_mean([ep_info["s"] for ep_info in self.ep_info_buffer]))
        self.logger.record("train/lagrange_multiplier", self.lagrangian_multiplier.item())

    def learn(
        self: SelfCUP,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "CUP",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfCUP:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )