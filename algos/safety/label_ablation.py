"""Label-ablation controls for the Split-RL fairness / robustness experiments.

Three independent mechanisms, all inert unless explicitly switched on:

1. ``apply_label_noise`` -- corrupt guidance labels once, at rollout-storage
   time (used by Split-RL).  Answers "how robust is Split-RL to mislabelled
   guidance?".
2. ``masked_lagrangian_advantages`` -- route the reward and cost advantages by
   the guidance label instead of blending them into one stream (used by PPO-Lag
   NN and PPO-Lag GBT).  This is the label-aware control for the baselines.
3. ``LabelObsWrapper`` (see ``env/label_obs_wrapper.py``) -- append the one-hot
   guidance label to the observation.

Scalar label convention, shared with GBRL's ``obj_labels``:

    0 -> reward objective only
    1 -> cost objective only
    2 -> blended (both objectives active)
"""
from typing import Optional, Tuple, Union

import numpy as np
import torch as th

# Scalar guidance-label values.
LABEL_REWARD_ONLY = 0
LABEL_COST_ONLY = 1
LABEL_BLENDED = 2


def apply_label_noise(
    labels: np.ndarray,
    prob: float,
    rng: np.random.Generator,
    n_label_values: int = 2,
) -> Tuple[np.ndarray, float]:
    """Corrupt a batch of guidance labels with probability ``prob``.

    With probability ``prob`` a label is discarded and redrawn uniformly from
    ``{0, ..., n_label_values - 1}``.  For the binary case this is exactly the
    "redraw as Bernoulli(0.5)" scheme, so the *effective* flip rate is
    ``prob * (1 - 1/n_label_values)`` -- i.e. ``prob / 2`` when binary.  Report
    the returned ``flip_rate`` rather than ``prob`` when plotting noise levels.

    Must be called once, where labels enter the rollout buffer -- not in the
    training loop.  Re-drawing every epoch would let the corruption average out
    across passes over the same rollout and understate its effect.

    :param labels: scalar labels, any shape
    :param prob: probability of redrawing a label
    :param rng: seeded generator, so a run stays reproducible
    :param n_label_values: size of the label alphabet (2, or 3 when blending)
    :return: (noisy labels, realised fraction of labels actually changed)
    """
    labels = np.asarray(labels, dtype=np.float32)
    if prob <= 0.0:
        return labels, 0.0

    corrupt = rng.random(labels.shape) < prob
    if not corrupt.any():
        return labels, 0.0

    redrawn = rng.integers(0, n_label_values, size=labels.shape).astype(np.float32)
    noisy = np.where(corrupt, redrawn, labels)
    return noisy, float(np.mean(noisy != labels))


def label_objective_weights(
    labels: Union[np.ndarray, th.Tensor],
) -> Tuple[Union[np.ndarray, th.Tensor], Union[np.ndarray, th.Tensor]]:
    """Split scalar labels into per-objective indicator weights.

    Returns ``(w_reward, w_cost)`` where a sample contributes to an objective
    iff its weight is 1.  Label 2 (blended) contributes to both, which keeps
    this consistent with Split-RL's blended third objective.
    """
    if isinstance(labels, th.Tensor):
        w_reward = (labels != LABEL_COST_ONLY).float()
        w_cost = (labels != LABEL_REWARD_ONLY).float()
    else:
        w_reward = (labels != LABEL_COST_ONLY).astype(np.float32)
        w_cost = (labels != LABEL_REWARD_ONLY).astype(np.float32)
    return w_reward, w_cost


def label_objective_rates(
    safety_labels: np.ndarray,
    min_rate: float = 0.01,
) -> Tuple[float, float]:
    """Fraction of the *whole rollout* active for each objective.

    These are the ``1/p`` rescaling constants used by
    :func:`masked_lagrangian_advantages`.  They are deliberately computed over
    the full rollout buffer rather than per minibatch: with a low label rate a
    minibatch can contain zero or one cost-labelled sample, and a per-minibatch
    ``N / N_cost`` factor would then divide by zero or hand a single sample the
    weight of the entire batch.  A rollout-level constant has the same
    expectation with far lower variance.

    :param safety_labels: scalar labels for the full rollout, any shape
    :param min_rate: floor applied to both rates to bound the rescaling
    :return: (reward-active rate, cost-active rate)
    """
    labels = np.asarray(safety_labels)
    if labels.size == 0:
        return 1.0, 1.0
    p_reward = float(np.mean(labels != LABEL_COST_ONLY))
    p_cost = float(np.mean(labels != LABEL_REWARD_ONLY))
    return (
        float(np.clip(p_reward, min_rate, 1.0)),
        float(np.clip(p_cost, min_rate, 1.0)),
    )


def _normalize_on_subset(
    advantages: th.Tensor,
    weights: th.Tensor,
    center_only: bool,
) -> th.Tensor:
    """Normalise ``advantages`` using statistics of the active subset only.

    Standardising across the full minibatch would re-couple the two objectives
    through a shared mean/std, which is exactly the aggregation the mask is
    meant to remove.  Falls back to the unnormalised tensor when the subset is
    too small for a meaningful statistic.
    """
    selected = weights > 0
    if int(selected.sum().item()) < 2:
        return advantages
    subset = advantages[selected]
    advantages = advantages - subset.mean()
    if not center_only:
        advantages = advantages / (subset.std() + 1e-8)
    return advantages


def masked_lagrangian_advantages(
    advantages_reward: th.Tensor,
    advantages_costs: th.Tensor,
    labels: th.Tensor,
    penalty: float,
    p_reward: float,
    p_cost: float,
    normalize_advantage: bool = True,
) -> th.Tensor:
    """Label-routed replacement for the blended Lagrangian advantage.

    The standard PPO-Lag update aggregates both objectives into every sample::

        A = (A_reward - lambda * A_cost) / (1 + lambda)

    Here each sample instead contributes to only the objective its guidance
    label selects, and each term is rescaled by the inverse of its own
    activation rate so that the subsequent ``.mean()`` over the minibatch
    recovers a *subset* mean rather than a diluted full-batch mean::

        A = (w_r * A_reward / p_r - lambda * w_c * A_cost / p_c) / (1 + lambda)

    Without the ``1/p`` rescaling a low label rate would silently shrink the
    cost term by that same rate -- effectively switching the constraint off and
    making the control fail for a reason that has nothing to do with labels.

    The ``/(1 + lambda)`` denominator is kept so that this differs from the
    unmasked baseline in the mask alone, not in effective step size.

    :param advantages_reward: reward advantages, unnormalised
    :param advantages_costs: cost advantages, unnormalised
    :param labels: scalar guidance labels for the minibatch
    :param penalty: current Lagrange multiplier
    :param p_reward: rollout-level reward activation rate
    :param p_cost: rollout-level cost activation rate
    :param normalize_advantage: normalise each term within its own subset
    :return: the routed advantage, same shape as the inputs
    """
    w_reward, w_cost = label_objective_weights(labels)
    w_reward = w_reward.reshape(advantages_reward.shape)
    w_cost = w_cost.reshape(advantages_costs.shape)

    if normalize_advantage:
        advantages_reward = _normalize_on_subset(advantages_reward, w_reward, center_only=False)
        advantages_costs = _normalize_on_subset(advantages_costs, w_cost, center_only=True)

    reward_term = w_reward * advantages_reward / p_reward
    cost_term = w_cost * advantages_costs / p_cost
    return (reward_term - penalty * cost_term) / (1.0 + penalty)


def buffer_safety_labels(rollout_buffer) -> Optional[np.ndarray]:
    """Return the rollout's scalar labels, or ``None`` if it does not carry any."""
    labels = getattr(rollout_buffer, "safety_labels", None)
    if labels is None:
        return None
    return np.asarray(labels)
