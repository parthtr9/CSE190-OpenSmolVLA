"""
VLA self-supervised trajectory scoring (Direction 2).

Instead of a separately-trained value function, this module uses the VLA's
own language-model backbone to score whether the robot appears to be making
progress toward the task goal.

This is viable because SmolVLA's VLM backbone already understands spatial
relationships and motion; we exploit that understanding to generate a proxy
advantage signal without any additional training labels.

Usage with a real SmolVLA policy
---------------------------------
    from recap_smolvla.scoring import vla_score_trajectory, scores_to_advantages

    scores = vla_score_trajectory(trajectory, "push block to goal", policy)
    advantages = scores_to_advantages(scores)

Usage with a mock policy (unit tests)
---------------------------------------
    from recap_smolvla.scoring import MockScoringPolicy

    mock = MockScoringPolicy(obs_dim=4, fixed_score=0.7)
    scores = vla_score_trajectory(trajectory, "push block to goal", mock)
"""

from __future__ import annotations

import re
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Protocol: scoring policy interface
# ---------------------------------------------------------------------------

class ScoringPolicy(Protocol):
    """Policy that can additionally score an observation + instruction pair."""

    def score(
        self,
        image: np.ndarray | None,
        prompt: str,
        obs: np.ndarray | None = None,
    ) -> float:
        """Return a scalar score in [0, 1]."""
        ...


# ---------------------------------------------------------------------------
# Scoring prompt template
# ---------------------------------------------------------------------------

SCORING_PROMPT_TEMPLATE = """\
Task: {instruction}

Rate whether the robot is making progress toward completing this task.
0.0 = moving away from goal or stuck
0.5 = neutral / unclear progress
1.0 = clearly making progress toward task completion

Respond with only a number between 0.0 and 1.0."""


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def vla_score_trajectory(
    trajectory: list[dict[str, Any]],
    instruction: str,
    policy: ScoringPolicy,
    *,
    stride: int = 1,
) -> list[float]:
    """Score every (or every stride-th) step using the VLA backbone.

    For steps not scored (due to stride > 1), the score is linearly
    interpolated from its neighbors.

    Parameters
    ----------
    trajectory:
        List of step dicts from collect_rollout.
    instruction:
        Natural-language task description.
    policy:
        Any object with a ``.score(image, prompt, obs)`` method.
    stride:
        Score every ``stride`` steps to save compute (default 1 = all).

    Returns
    -------
    List of float scores in [0, 1], length == len(trajectory).
    """
    if not trajectory:
        return []

    prompt = SCORING_PROMPT_TEMPLATE.format(instruction=instruction)
    T = len(trajectory)
    raw_scores: dict[int, float] = {}

    for t in range(0, T, stride):
        step = trajectory[t]
        image = step.get("image")
        obs = step.get("obs")
        score = policy.score(image, prompt, obs)
        raw_scores[t] = float(np.clip(score, 0.0, 1.0))

    # Fill gaps via linear interpolation when stride > 1
    if stride > 1:
        filled = _interpolate_scores(raw_scores, T)
    else:
        filled = [raw_scores[t] for t in range(T)]

    return filled


def _interpolate_scores(raw: dict[int, float], T: int) -> list[float]:
    """Linear interpolation to fill missing timestep scores."""
    filled = [0.0] * T
    keys = sorted(raw.keys())
    for i, t in enumerate(range(T)):
        if t in raw:
            filled[t] = raw[t]
        else:
            # Find nearest known neighbors
            left = max((k for k in keys if k <= t), default=keys[0])
            right = min((k for k in keys if k >= t), default=keys[-1])
            if left == right:
                filled[t] = raw[left]
            else:
                alpha = (t - left) / (right - left)
                filled[t] = (1 - alpha) * raw[left] + alpha * raw[right]
    return filled


# ---------------------------------------------------------------------------
# Convert VLA scores → advantages
# ---------------------------------------------------------------------------

def scores_to_advantages(
    scores: list[float],
    *,
    center: bool = True,
) -> np.ndarray:
    """Convert per-step VLA scores to advantage-like values.

    Simple approach: subtract the mean so that above-average steps are
    positive and below-average steps are negative (matches the spirit of
    A(s,a) = Q(s,a) - V(s)).

    Parameters
    ----------
    scores:
        Output of vla_score_trajectory.
    center:
        If True, subtract mean (recommended).

    Returns
    -------
    np.ndarray of shape (T,).
    """
    arr = np.array(scores, dtype=np.float32)
    if center and len(arr) > 0:
        arr = arr - arr.mean()
    return arr


# ---------------------------------------------------------------------------
# Mock scoring policy for unit tests
# ---------------------------------------------------------------------------

class MockScoringPolicy(nn.Module):
    """Deterministic mock that returns a fixed or obs-derived score.

    Parameters
    ----------
    obs_dim:
        Expected obs dimensionality.
    fixed_score:
        If provided, always return this scalar.
    score_from_obs:
        If True (and fixed_score is None), return the mean of the obs
        vector clamped to [0, 1].
    """

    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 2,
        fixed_score: float | None = 0.5,
        score_from_obs: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fixed_score = fixed_score
        self.score_from_obs = score_from_obs
        # Tiny learnable layer so gradient tests can run
        self._scorer = nn.Linear(obs_dim, 1)
        self._action_head = nn.Linear(obs_dim, action_dim)

    def score(
        self,
        image: np.ndarray | None,
        prompt: str,
        obs: np.ndarray | None = None,
    ) -> float:
        if self.fixed_score is not None:
            return self.fixed_score
        if self.score_from_obs and obs is not None:
            return float(np.clip(np.mean(obs), 0.0, 1.0))
        return 0.5

    def select_action(self, obs_dict: dict[str, Any]) -> np.ndarray:
        obs = _obs_from_dict(obs_dict, self.obs_dim)
        with torch.no_grad():
            return self._action_head(obs).numpy()

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        obs = _obs_from_dict(batch, self.obs_dim)
        pred = self._action_head(obs)
        target = torch.zeros(self.action_dim)
        return nn.functional.mse_loss(pred, target)


def _obs_from_dict(d: dict[str, Any], obs_dim: int) -> torch.Tensor:
    """Extract and tensorify obs from various dict formats."""
    for key in ("obs", "observation.state"):
        if key in d:
            v = d[key]
            if isinstance(v, torch.Tensor):
                return v.float().reshape(-1)[:obs_dim]
            return torch.tensor(np.asarray(v, dtype=np.float32)).reshape(-1)[:obs_dim]
    return torch.zeros(obs_dim)
