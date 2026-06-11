"""
Reward functions for RECAP training.

Two variants are provided for the ablation study:

sparse_reward_fn
    Direct implementation of Equation 5 from the π*0.6 paper.
    Terminal-only signal: 0 on success, -C_fail on failure, -1 per step.

dense_reward_fn
    Augments the sparse signal with inverse-distance proximity bonuses.
    The alpha parameter (default 0.1) keeps the sparse component dominant
    so the dense bonus guides without enabling reward hacking.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Step = dict[str, Any]
Trajectory = list[Step]


# ---------------------------------------------------------------------------
# Sparse reward  (π*0.6 paper, Equation 5)
# ---------------------------------------------------------------------------

def sparse_reward_fn(
    trajectory: Trajectory,
    success: bool,
    *,
    C_fail: float = 100.0,
) -> list[float]:
    """Return the per-step sparse reward from Equation 5.

    Parameters
    ----------
    trajectory:
        List of step dicts produced by collect_rollout.
    success:
        Whether the episode ended successfully.
    C_fail:
        Large negative constant applied at the terminal failure step.

    Returns
    -------
    List of per-step rewards, length == len(trajectory).
    """
    if len(trajectory) == 0:
        return []

    T = len(trajectory)
    rewards: list[float] = [-1.0] * T
    rewards[-1] = 0.0 if success else -C_fail
    return rewards


# ---------------------------------------------------------------------------
# Dense reward  (Direction 1 from project spec)
# ---------------------------------------------------------------------------

def dense_reward_fn(
    trajectory: Trajectory,
    success: bool,
    goal_pos: np.ndarray | list[float],
    *,
    C_fail: float = 100.0,
    alpha: float = 0.1,
    agent_pos_key: str = "agent_pos",
    block_pos_key: str = "block_pos",
) -> list[float]:
    """Sparse reward augmented with inverse-distance proximity bonuses.

    The dense bonus is bounded by ``2 * alpha`` (both distances → 0), so the
    sparse component stays dominant for any ``alpha`` << ``C_fail``.

    Parameters
    ----------
    trajectory:
        List of step dicts; each step["obs"] must expose agent_pos and block_pos
        (or the keys specified by agent_pos_key / block_pos_key), or alternatively
        step["obs"] is a flat numpy array where the first two elements are agent
        x,y and elements [2:4] are block x,y.
    success:
        Whether the episode ended successfully.
    goal_pos:
        2-D target position for the block (task goal).
    C_fail:
        Large negative constant for terminal failure (matches sparse_reward_fn).
    alpha:
        Weight of the dense bonus.  Keep small (0.05–0.2) to prevent hacking.
    agent_pos_key:
        Key to look for in step["obs"] dict for agent position.
    block_pos_key:
        Key to look for in step["obs"] dict for block position.

    Returns
    -------
    List of per-step rewards, length == len(trajectory).
    """
    if len(trajectory) == 0:
        return []

    goal = np.asarray(goal_pos, dtype=np.float32)
    T = len(trajectory)
    rewards: list[float] = []

    for t, step in enumerate(trajectory):
        # --- sparse component ---
        if t == T - 1:
            sparse = 0.0 if success else -C_fail
        else:
            sparse = -1.0

        # --- dense proximity bonus ---
        agent_pos, block_pos = _extract_positions(step, agent_pos_key, block_pos_key)

        dist_agent_to_block = float(np.linalg.norm(agent_pos - block_pos))
        dist_block_to_goal = float(np.linalg.norm(block_pos - goal))

        proximity_bonus = alpha * (
            1.0 / (1.0 + dist_agent_to_block)
            + 1.0 / (1.0 + dist_block_to_goal)
        )

        rewards.append(sparse + proximity_bonus)

    return rewards


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_positions(
    step: Step,
    agent_pos_key: str,
    block_pos_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract agent and block positions from a step dict.

    Supports two obs formats:
    1. step["obs"] is a dict with named keys (preferred).
    2. step["obs"] is a flat array: [:2] = agent x,y; [2:4] = block x,y.
    """
    obs = step["obs"]

    if isinstance(obs, dict):
        agent_pos = np.asarray(obs[agent_pos_key], dtype=np.float32)
        block_pos = np.asarray(obs[block_pos_key], dtype=np.float32)
    else:
        obs_arr = np.asarray(obs, dtype=np.float32)
        agent_pos = obs_arr[:2]
        block_pos = obs_arr[2:4]

    return agent_pos, block_pos


# ---------------------------------------------------------------------------
# Reward registry  (used by ablation experiment scripts)
# ---------------------------------------------------------------------------

REWARD_REGISTRY: dict[str, Any] = {
    "sparse": sparse_reward_fn,
    "dense": dense_reward_fn,
}
