"""Reward -> discounted return -> advantage -> binary label.

These are pure NumPy functions (no torch, no lerobot) so they are trivial to
unit-test and reason about. They sit in the middle of the RECAP chain:

    rewards r_t  --(this module)-->  returns R_t
    returns R_t + V(s_t)  --(this module)-->  advantages A_t
    advantages A_t  --(this module)-->  labels I_t in {0, 1}

Key design points
------------------
* Returns are computed PER EPISODE. A frame's future never leaks across an
  episode boundary, so we need episode ids aligned with the rewards.
* The advantage is the actor-critic baseline form: A_t = R_t - V(s_t).
  V is the *expected* return from a state; the advantage says how much better
  the actually-observed return was than expected. That is the temporal credit
  assignment a per-frame classifier cannot do on its own.
* The good/bad threshold epsilon is set by a PERCENTILE of advantages, so a
  target fraction (default 30%, matching the RECAP paper) is labeled positive.
  This is robust to the arbitrary scale of the classifier's reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdvantageConfig:
    """Hyperparameters for the reward -> label pipeline."""

    gamma: float = 0.99
    # Fraction of frames to label positive. The RECAP paper tunes epsilon so that
    # ~30% of demonstration data has positive advantage.
    positive_fraction: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0.0 < self.positive_fraction < 1.0:
            raise ValueError(
                f"positive_fraction must be in (0, 1), got {self.positive_fraction}"
            )


def discounted_returns(
    rewards: np.ndarray,
    episode_ids: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute per-frame discounted return-to-go, reset at episode boundaries.

    R_t = sum_{k>=0} gamma^k * r_{t+k}, where the sum stops at the end of the
    episode that frame t belongs to.

    Args:
        rewards: float array, shape (N,). Per-frame rewards in dataset order.
        episode_ids: int array, shape (N,). Episode index for each frame. Frames
            of the same episode must be contiguous and in time order.
        gamma: discount factor in (0, 1].

    Returns:
        float32 array, shape (N,), of discounted returns-to-go.
    """
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    episode_ids = np.asarray(episode_ids).reshape(-1)
    if rewards.shape[0] != episode_ids.shape[0]:
        raise ValueError(
            f"rewards ({rewards.shape[0]}) and episode_ids ({episode_ids.shape[0]}) "
            "must have the same length."
        )
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    n = rewards.shape[0]
    returns = np.zeros(n, dtype=np.float64)
    running = 0.0
    # Walk backwards; reset the accumulator whenever the episode changes.
    for t in range(n - 1, -1, -1):
        if t == n - 1 or episode_ids[t] != episode_ids[t + 1]:
            running = 0.0  # start of a new episode (seen from the right)
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns.astype(np.float32)


def labels_from_advantages(
    advantages: np.ndarray,
    positive_fraction: float = 0.30,
) -> tuple[np.ndarray, float]:
    """Threshold advantages into binary labels using a percentile.

    epsilon is chosen so that approximately `positive_fraction` of frames have
    advantage strictly greater than epsilon and are labeled 1 (positive).

    Returns:
        (labels, epsilon) where labels is an int8 array in {0, 1}, shape (N,).
    """
    advantages = np.asarray(advantages, dtype=np.float64).reshape(-1)
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError(
            f"positive_fraction must be in (0, 1), got {positive_fraction}"
        )
    # The (1 - f) quantile is the cutoff above which the top f fraction lies.
    epsilon = float(np.quantile(advantages, 1.0 - positive_fraction))
    labels = (advantages > epsilon).astype(np.int8)
    return labels, epsilon


def compute_advantage_labels(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_ids: np.ndarray,
    config: AdvantageConfig | None = None,
) -> dict[str, np.ndarray]:
    """Full reward -> label computation.

    Args:
        rewards: (N,) per-frame rewards from the classifier.
        values: (N,) per-frame value estimates V(s_t) from the value function.
        episode_ids: (N,) episode index per frame (contiguous, time-ordered).
        config: AdvantageConfig (gamma, positive_fraction).

    Returns:
        dict with arrays (all length N):
            "return"          : discounted return-to-go R_t
            "advantage"       : A_t = R_t - V(s_t)
            "advantage_label" : int8 in {0, 1}
        and scalar "epsilon" (the chosen threshold).
    """
    config = config or AdvantageConfig()
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if rewards.shape[0] != values.shape[0]:
        raise ValueError(
            f"rewards ({rewards.shape[0]}) and values ({values.shape[0]}) "
            "must have the same length."
        )

    returns = discounted_returns(rewards, episode_ids, gamma=config.gamma)
    advantages = returns.astype(np.float64) - values
    labels, epsilon = labels_from_advantages(advantages, config.positive_fraction)

    return {
        "return": returns.astype(np.float32),
        "advantage": advantages.astype(np.float32),
        "advantage_label": labels,
        "epsilon": np.float32(epsilon),
    }
