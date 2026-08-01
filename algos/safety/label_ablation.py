"""Guidance-label corruption for the Split-RL robustness experiments.

Both functions answer "how robust is Split-RL to mislabelled guidance?".  They
are inert unless the corresponding probability is set above zero.

The baselines' label-mask control is *not* here: it is six inline lines in each
of ppo_lag.py, ppo_lag_gbrl.py, ipo.py, cup.py and cpo.py, so that each
algorithm's use of the label can be read and checked in place.

Scalar label convention, shared with GBRL's ``obj_labels``:

    0 -> reward objective only
    1 -> cost objective only
    2 -> blended (both objectives active)
"""
from typing import Tuple

import numpy as np


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


def apply_label_inversion(
    labels: np.ndarray,
    prob: float,
    rng: np.random.Generator,
    n_label_values: int = 2,
) -> Tuple[np.ndarray, float]:
    """Structured (adversarial) corruption: replace a label with a *different* one.

    Distinct from :func:`apply_label_noise`, which redraws uniformly and may
    return the original value.  Inversion guarantees the label is wrong, so the
    purity term is actively misdirected rather than merely uninformative --
    states needing cost-routing are marked reward-relevant and vice versa.

    This distinction is the point of the ablation.  Under symmetric noise every
    candidate split sees roughly the same label composition, so the purity term
    becomes a near-constant offset and split selection falls back on the
    standard gain.  Under inversion the purity term instead prefers splits that
    mix conflicting gradients, which is destructive rather than inert.

    For binary labels this is ``label -> 1 - label``; for ``n_label_values > 2``
    the replacement is drawn uniformly from the other values.  The effective
    flip rate is exactly ``prob`` (not ``prob/2`` as under symmetric noise).

    :param labels: scalar labels, any shape
    :param prob: probability of inverting a label
    :param rng: seeded generator, so a run stays reproducible
    :param n_label_values: size of the label alphabet
    :return: (corrupted labels, realised fraction of labels changed)
    """
    labels = np.asarray(labels, dtype=np.float32)
    if prob <= 0.0:
        return labels, 0.0

    corrupt = rng.random(labels.shape) < prob
    if not corrupt.any():
        return labels, 0.0

    if n_label_values == 2:
        flipped = 1.0 - labels
    else:
        # A non-zero offset modulo K guarantees a different label.
        offset = rng.integers(1, n_label_values, size=labels.shape)
        flipped = (labels + offset) % n_label_values

    noisy = np.where(corrupt, flipped.astype(np.float32), labels)
    return noisy, float(np.mean(noisy != labels))
