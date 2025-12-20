import copy
import sys
import time
from functools import partial
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import TRPO
from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (explained_variance,
                                            get_schedule_fn, obs_as_tensor,
                                            safe_mean)
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from buffers.rollout_buffer import CostRolloutBuffer, CostRolloutBufferSamples
from policies.cost_actor_critic import CostActorCriticPolicy

SelfCPO = TypeVar("SelfCPO", bound="CPO")

class CPO(TRPO):
    def __init__(
        self,
        policy: Union[str, Type[CostActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = True,
        target_kl: float = 0.01,
        sub_sampling_factor: int = 1,
        cost_limit: float = 25.0,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        
        self.cost_limit = cost_limit
        self.ep_cost_mean = 0.0
        
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
            batch_size=batch_size,
            use_sde=use_sde,
            target_kl=target_kl,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            sub_sampling_factor=sub_sampling_factor,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            cg_max_steps=cg_max_steps,
            cg_damping=cg_damping,
            line_search_shrinking_factor=line_search_shrinking_factor,
            line_search_max_iter=line_search_max_iter,
            n_critic_updates=n_critic_updates,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )
        
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = CostRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)
        
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []
        value_losses = []
        cost_losses = []

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            # Optional: sub-sample data for faster computation
            if self.sub_sampling_factor > 1:
                rollout_data = CostRolloutBufferSamples(
                    rollout_data.observations[:: self.sub_sampling_factor],
                    rollout_data.actions[:: self.sub_sampling_factor],
                    None,  # type: ignore[arg-type]  # old values, not used here
                    None,  # type: ignore[arg-type]  # old values_costs, not used here
                    rollout_data.old_log_prob[:: self.sub_sampling_factor],
                    rollout_data.advantages[:: self.sub_sampling_factor],
                    rollout_data.advantages_costs[:: self.sub_sampling_factor],
                    None,  # type: ignore[arg-type]  # returns, not used here
                    None,  # type: ignore[arg-type]  # returns, not used here
                    None,  # type: ignore[arg-type]  # returns, not used here
                )

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                # batch_size is only used for the value function
                self.policy.reset_noise(actions.shape[0])

            with th.no_grad():
                # Note: is copy enough, no need for deepcopy?
                # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
                # directly to avoid PyTorch errors.
                old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

            distribution = self.policy.get_distribution(rollout_data.observations)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # surrogate policy objective
            policy_objective = (advantages * ratio).mean()

            # KL divergence
            kl_div = kl_divergence(distribution, old_distribution).mean()

            # Surrogate & KL gradient
            self.policy.optimizer.zero_grad()

            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

            # Hessian-vector dot product function used in the conjugate gradient step
            hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl, retain_graph=True)

            # Computing search direction
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )
            
            # Extract x^T H x (TRPO uses this implicitly in line_search_max_step_size)
            xHx = th.matmul(search_direction, hessian_vector_product_fn(search_direction))
            q = xHx  # CPO needs this for LQCLP
            
            cost_advantages = rollout_data.advantages_costs
            if self.normalize_advantage:
                cost_advantages = cost_advantages - cost_advantages.mean()

            with th.enable_grad():
                dist_c = self.policy.get_distribution(rollout_data.observations)
                log_prob_c = dist_c.log_prob(actions)
                cost_ratio = th.exp(log_prob_c - rollout_data.old_log_prob)
                cost_objective = (cost_ratio * cost_advantages).mean()
                loss_cost_before = cost_objective.item()

                # Compute cost gradient for each actor parameter
                cost_grads = th.autograd.grad(cost_objective, actor_params, retain_graph=True)
                b_grads = th.cat([g.flatten() for g in cost_grads])
            # 2. Solve for cost direction: p = H^-1 * b
            p = conjugate_gradient_solver(
                hessian_vector_product_fn,
                b_grads,
                max_iter=self.cg_max_steps,
            )

            # 3. Compute LQCLP terms
            ep_costs = self.ep_cost_mean - self.cost_limit  # negative means safe
            r = th.matmul(policy_objective_gradients, p)
            s = th.matmul(b_grads, p)

            # 4. Determine optimization case (see _determine_case below)
            optim_case, A, B = self._determine_case(b_grads, ep_costs, q, r, s, self.target_kl)

            # 5. Compute CPO constrained step direction (see _compute_cpo_step_direction below)
            step_direction, lambda_star, nu_star = self._compute_cpo_step_direction(
                optim_case, search_direction, p, q, r, s, A, B, ep_costs, self.target_kl
            )

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            with th.no_grad():
                # Line-search (backtracking)
                for _ in range(self.line_search_max_iter):
                    start_idx = 0
                    # Applying the scaled step direction
                    # CPO uses: line_search_backtrack_coeff * step_direction (already constrained)
                    for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * step_direction[start_idx : (start_idx + n_params)].view(shape)
                        )
                        start_idx += n_params

                    # Recomputing the policy log-probabilities
                    distribution = self.policy.get_distribution(rollout_data.observations)
                    log_prob = distribution.log_prob(actions)

                    # New policy objective
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    new_policy_objective = (advantages * ratio).mean()
                    # ===== CPO ADDITION: New cost objective =====
                    new_cost_objective = (cost_advantages * ratio).mean()

                    # New KL-divergence
                    kl_div = kl_divergence(distribution, old_distribution).mean()

                    # ========== CPO MODIFIED CONSTRAINT CRITERIA ==========
                    # TRPO checks: kl < target_kl and reward improved
                    # CPO checks: kl < target_kl AND reward improved AND cost constraint
                    
                    policy_improve = new_policy_objective - policy_objective
                    cost_diff = new_cost_objective - loss_cost_before
                    
                    accept = True
                    if policy_improve < 0 and optim_case > 1:
                        accept = False  # Reward didn't improve
                    elif cost_diff > max(-ep_costs, 0):  # CPO cost check
                        accept = False  # Cost constraint violated
                    elif kl_div > self.target_kl:
                        accept = False  # KL constraint violated
                    
                    if accept:
                        is_line_search_success = True
                        break

                    # Reducing step size if line-search wasn't successful
                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if not is_line_search_success:
                    # If the line-search wasn't successful we revert to the original parameters
                    for param, original_param in zip(actor_params, original_actor_params):
                        param.data = original_param.data.clone()

                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0.0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())

        # Critic update
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(rollout_data.observations)
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())
                # Cost critic (reuse from PPOLag)
                cost_values_pred = self.policy.predict_costs(rollout_data.observations)
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values_pred.flatten())
                cost_losses.append(cost_value_loss.item())
                
                critic_loss = value_loss + cost_value_loss
                
                self.policy.optimizer.zero_grad()
                critic_loss.backward()
                # Removing gradients of parameters shared with the actor
                # otherwise it defeats the purposes of the KL constraint
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/cost_value_loss", np.mean(cost_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def _compute_cpo_step_direction(
        self,
        optim_case: int,
        search_direction: th.Tensor,
        p: th.Tensor,
        q: th.Tensor,
        r: th.Tensor,
        s: th.Tensor,
        A: th.Tensor,
        B: th.Tensor,
        ep_costs: float,
        target_kl: float,
    ) -> tuple[th.Tensor, float, float]:
        """
        Compute CPO step direction using analytical solution to LQCLP.
        
        Returns:
            step_direction: The constrained step direction
            lambda_star: Lagrange multiplier for KL constraint
            nu_star: Lagrange multiplier for cost constraint
        """
        if optim_case in (3, 4):
            # Safe: use TRPO update
            alpha = th.sqrt(2 * target_kl / (q + 1e-8))
            nu_star = 0.0
            lambda_star = 1.0 / (alpha + 1e-8)
            step_direction = alpha * search_direction
            
        elif optim_case in (1, 2):
            # Partially safe: solve constrained optimization
            # Compute candidate lambdas
            lambda_a = th.sqrt(A / (B + 1e-8))
            lambda_b = th.sqrt(q / (2 * target_kl + 1e-8))
            
            # Project to feasible region based on r/c
            # If ep_costs < 0 (currently safe), we want lambda > r/c to keep it safe
            # If ep_costs >= 0 (currently unsafe), we need lambda in valid range
            if ep_costs < 0:
                lambda_a_star = th.clamp(lambda_a, 0.0, r / (ep_costs + 1e-8))      # [0, r/c]
                lambda_b_star = th.clamp(lambda_b, r / (ep_costs + 1e-8), float('inf'))  # [r/c, ∞]
            else:
                lambda_a_star = th.clamp(lambda_a, r / (ep_costs + 1e-8), float('inf'))  # [r/c, ∞]
                lambda_b_star = th.clamp(lambda_b, 0.0, r / (ep_costs + 1e-8))      # [0, r/c]
            
            # Evaluate objectives for both candidates and choose best
            def objective_a(lam):
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)
            
            def objective_b(lam):
                return -0.5 * (q / (lam + 1e-8) + 2 * target_kl * lam)
            
            if objective_a(lambda_a_star) >= objective_b(lambda_b_star):
                lambda_star = lambda_a_star
            else:
                lambda_star = lambda_b_star
            
            # Compute nu_star from lambda_star
            nu_star = th.max((lambda_star * ep_costs - r) / (s + 1e-8), th.tensor(0.0))
            
            # Compute step direction
            step_direction = (1.0 / (lambda_star + 1e-8)) * (search_direction - nu_star * p)
            
            lambda_star = lambda_star.item()
            nu_star = nu_star.item()
            
        else:  # Case 0: purely decrease costs
            lambda_star = 0.0
            nu_star = th.sqrt(2 * target_kl / (s + 1e-8))
            step_direction = -nu_star * p
            nu_star = nu_star.item()
            
        return step_direction, lambda_star, nu_star

    def _determine_case(
        self,
        b_grads: th.Tensor,
        ep_costs: float,
        q: th.Tensor,
        r: th.Tensor,
        s: th.Tensor,
        target_kl: float,
    ) -> tuple[int, th.Tensor, th.Tensor]:
        """
        Determine which optimization case based on feasibility.
        
        Returns:
            optim_case: Integer 0-4 indicating the optimization case
            A: Quadratic term for LQCLP
            B: Linear term for LQCLP
        """
        # Case 4: Cost gradient negligible and currently safe
        if th.matmul(b_grads, b_grads) <= 1e-6 and ep_costs < 0:
            return 4, th.zeros(1), th.zeros(1)
        
        # Compute A and B for constrained cases
        A = q - r**2 / (s + 1e-8)
        B = 2 * target_kl - ep_costs**2 / (s + 1e-8)
        
        if ep_costs < 0 and B < 0:
            # Case 3: Entire trust region feasible
            return 3, A, B
        elif ep_costs < 0 and B >= 0:
            # Case 2: Partially feasible
            return 2, A, B
        elif ep_costs >= 0 and B >= 0:
            # Case 1: Feasible recovery
            return 1, A, B
        else:
            # Case 0: Infeasible recovery
            return 0, A, B
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CostRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, value_costs, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            costs = th.tensor([info.get('cost', 0.0) for info in infos])

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                value_costs,
                log_probs,
                costs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            new_obs_tensor = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(new_obs_tensor)  # type: ignore[arg-type]
            value_costs = self.policy.predict_costs(new_obs_tensor)  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, last_value_costs=value_costs, dones=dones)

        callback.on_rollout_end()

        return True

    def learn(
        self: SelfCPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "CPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        ) -> SelfCPO:
        
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.ep_cost_mean = safe_mean([ep_info["c"] for ep_info in self.ep_info_buffer])
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_cost_mean", self.ep_cost_mean)
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_scalarization_mean", safe_mean([ep_info["s"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()
