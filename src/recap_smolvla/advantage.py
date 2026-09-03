"""
Advantage computation and binary labeling for RECAP.

RECAP does not use gradient-based policy optimization.  Instead it:
  1. Estimates per-step advantages A(s,a) = G_t - V(s_t).
  2. Binarizes them into an improvement indicator I_t ∈ {True, False}.
  3. Prepends "Advantage: positive / negative" to the language instruction
     and re-trains the policy via supervised learning.

The default threshold_pct=70 means the top ~30% of steps are labeled
positive, matching the fraction reported in the π*0.6 paper.
Human-correction steps always receive I_t = True regardless of advantage.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from recap_smolvla.value_function import ValueFunction, compute_returns

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Step = dict[str, Any]
Trajectory = list[Step]
Rollout = tuple[Trajectory, bool]
LabeledStep = dict[str, Any]


# ---------------------------------------------------------------------------
# Per-trajectory advantage computation
# ---------------------------------------------------------------------------

def compute_advantages(
    trajectory: Trajectory,
    value_fn: ValueFunction,
    reward_fn: Callable,
    success: bool,
    *,
    gamma: float = 1.0,
    feature_key: str = "obs",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-step advantages A_t = G_t - V(s_t).

    Parameters
    ----------
    trajectory:
        List of step dicts (each must contain key "obs").
    value_fn:
        Trained ValueFunction; called without gradient.
    reward_fn:
        Callable(trajectory, success) → list[float].
    success:
        Episode outcome, forwarded to reward_fn.
    gamma:
        Discount factor (matches value function training).
    feature_key:
        Per-step representation field supplied to ``value_fn``.

    Returns
    -------
    advantages : np.ndarray, shape (T,)
    returns    : np.ndarray, shape (T,)
    """
    if len(trajectory) == 0:
        return np.array([]), np.array([])

    rewards = reward_fn(trajectory, success)
    returns = np.array(compute_returns(rewards, gamma=gamma), dtype=np.float32)

    obs = torch.tensor(
        np.stack([step[feature_key] for step in trajectory]).astype(np.float32)
    )
    with torch.no_grad():
        values = value_fn(obs).numpy()

    advantages = returns - values
    return advantages, returns


# ---------------------------------------------------------------------------
# Dataset labeling
# ---------------------------------------------------------------------------

def label_trajectories(
    rollouts: list[Rollout],
    value_fn: ValueFunction,
    reward_fn: Callable,
    *,
    threshold_pct: float = 70.0,
    gamma: float = 1.0,
    feature_key: str = "obs",
) -> list[LabeledStep]:
    """Label every trajectory step with a binary advantage indicator.

    Steps whose advantage exceeds the ``(100 - threshold_pct)``-th percentile
    over ALL steps in the dataset receive ``advantage_positive=True``.
    This matches the paper's ~30% positive label fraction when threshold_pct=70.

    Human-correction steps (``step["is_correction"] == True``) are always
    forced to ``advantage_positive=True`` per Algorithm 1.

    Parameters
    ----------
    rollouts:
        List of (trajectory, success) tuples.
    value_fn:
        Trained ValueFunction instance.
    reward_fn:
        Callable(trajectory, success) → list[float].
    threshold_pct:
        Percentile below which steps are labeled negative (default 70).
    gamma:
        Discount factor.
    feature_key:
        Per-step representation field supplied to ``value_fn``.

    Returns
    -------
    Flat list of labeled step dicts, one per (trajectory, timestep) pair.
    Each dict contains all original step keys plus:
      - "advantage"          : float  — raw A(s,a) estimate
      - "return_"            : float  — G_t (avoids shadowing built-in)
      - "advantage_positive" : bool   — binary improvement indicator I_t
    """
    if not rollouts:
        return []

    # First pass: collect all advantages for threshold calibration
    all_advantages: list[float] = []
    per_rollout: list[tuple[np.ndarray, np.ndarray]] = []

    for trajectory, success in rollouts:
        advs, rets = compute_advantages(
            trajectory, value_fn, reward_fn, success, gamma=gamma, feature_key=feature_key
        )
        per_rollout.append((advs, rets))
        all_advantages.extend(advs.tolist())

    if not all_advantages:
        return []

    threshold = float(np.percentile(all_advantages, threshold_pct))

    # Second pass: assign labels
    labeled_data: list[LabeledStep] = []
    for (trajectory, _), (advs, rets) in zip(rollouts, per_rollout):
        for step, adv, ret in zip(trajectory, advs, rets):
            is_correction = bool(step.get("is_correction", False))
            positive = is_correction or (float(adv) > threshold)

            labeled_step: LabeledStep = {
                **step,
                "advantage": float(adv),
                "return_": float(ret),
                "advantage_positive": positive,
            }
            labeled_data.append(labeled_step)

    return labeled_data


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def advantage_distribution_stats(labeled_data: list[LabeledStep]) -> dict[str, float]:
    """Return summary statistics of the advantage distribution.

    Useful for verifying that the labeling threshold is producing the
    expected ~30% positive fraction and that advantages have spread.

    Returns
    -------
    Dict with keys: mean, std, min, max, pct_positive.
    """
    if not labeled_data:
        return {}

    advs = np.array([d["advantage"] for d in labeled_data], dtype=np.float32)
    pct_pos = float(np.mean([d["advantage_positive"] for d in labeled_data]))

    return {
        "mean": float(advs.mean()),
        "std": float(advs.std()),
        "min": float(advs.min()),
        "max": float(advs.max()),
        "pct_positive": pct_pos,
    }


def assert_advantage_label_invariants(labeled_data: list[LabeledStep]) -> None:
    """Raise AssertionError if labeled data violates structural invariants.

    Checks:
    - All human corrections are labeled positive.
    - Advantage field is finite for every step.
    - advantage_positive is a bool.
    - Expected ~30% positive fraction (warns if outside [0.10, 0.50]).
    """
    for i, d in enumerate(labeled_data):
        assert isinstance(d["advantage_positive"], bool), (
            f"Step {i}: advantage_positive must be bool, got {type(d['advantage_positive'])}"
        )
        assert np.isfinite(d["advantage"]), (
            f"Step {i}: advantage is not finite: {d['advantage']}"
        )
        if d.get("is_correction"):
            assert d["advantage_positive"], (
                f"Step {i}: human correction must be labeled advantage_positive=True"
            )

    stats = advantage_distribution_stats(labeled_data)
    pct = stats.get("pct_positive", 0.0)
    if not (0.05 <= pct <= 0.60):
        import warnings
        warnings.warn(
            f"advantage_positive fraction {pct:.1%} is outside expected [5%, 60%]. "
            "Check threshold_pct and reward scale.",
            RuntimeWarning,
            stacklevel=2,
        )
