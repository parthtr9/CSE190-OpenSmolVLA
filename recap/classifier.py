"""Reward classifier interface.

This is the SEAM your teammate fills in. The RECAP training loop only depends on
this interface, never on a concrete reward implementation. That is what lets the
team swap a VLM-based, state-based, or learned reward in and out without touching
the training loop (the independent variable in the project's research question).

Contract
--------
A reward classifier maps a *frame* (one timestep of a LeRobot v3 episode) to a
scalar reward r_t. Higher = better. The downstream pipeline turns these per-frame
rewards into discounted returns, advantages, and finally good/bad labels.

A "frame" is a dict with at least these keys (LeRobot v3 demo format):
    "observation.state"          : np.ndarray | torch.Tensor, shape (state_dim,)
    "observation.images.<cam>"   : np.ndarray | torch.Tensor, shape (H, W, 3) or (3, H, W)
    "action"                     : np.ndarray | torch.Tensor, shape (action_dim,)
    "task"                       : str
Plus bookkeeping keys (episode_index, frame_index, index, ...). A classifier may
use any subset of these.

For batched efficiency, implementations receive a *list* of frames and return a
1-D array of rewards (one per frame). A naive implementation can just loop.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import numpy as np

Frame = dict[str, Any]


class RewardClassifier(abc.ABC):
    """Maps frames to scalar rewards. Implement `predict_rewards`."""

    @abc.abstractmethod
    def predict_rewards(self, frames: Sequence[Frame]) -> np.ndarray:
        """Return a float32 array of shape (len(frames),) of per-frame rewards.

        Higher reward = better action/state. The absolute scale does not matter
        much: advantages are computed relative to a learned baseline V(s), and the
        good/bad threshold is set by a percentile (see recap.advantages). What
        matters is the *ordering* and that "good" frames score higher than "bad".
        """
        raise NotImplementedError

    def __call__(self, frames: Sequence[Frame]) -> np.ndarray:
        rewards = self.predict_rewards(frames)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if rewards.shape[0] != len(frames):
            raise ValueError(
                f"predict_rewards returned {rewards.shape[0]} rewards for "
                f"{len(frames)} frames; must be one per frame."
            )
        return rewards


class StubRewardClassifier(RewardClassifier):
    """Deterministic placeholder reward for development before a real model exists.

    It is NOT a real reward model. It exists so the whole pipeline
    (returns -> value function -> advantages -> labels -> SmolVLA training) can run
    end-to-end on the synthetic fixture and on real demos with no reward columns.

    The reward is a smooth function of within-episode progress: later frames in an
    episode score higher, with a small deterministic per-frame perturbation derived
    from the state vector. This gives the value function and advantage computation a
    non-trivial, reproducible signal to chew on.

    Replace this with your team's VLM / state / learned reward by subclassing
    RewardClassifier and implementing `predict_rewards`.
    """

    def __init__(self, progress_weight: float = 1.0, state_weight: float = 0.1):
        self.progress_weight = float(progress_weight)
        self.state_weight = float(state_weight)

    def predict_rewards(self, frames: Sequence[Frame]) -> np.ndarray:
        rewards = np.empty(len(frames), dtype=np.float32)
        for i, frame in enumerate(frames):
            # Progress signal: normalized position within the episode, if available.
            frame_index = float(frame.get("frame_index", i))
            episode_len = float(frame.get("_episode_len", max(frame_index + 1.0, 1.0)))
            progress = frame_index / max(episode_len - 1.0, 1.0)

            # Deterministic per-frame perturbation from the state (no RNG -> reproducible).
            state = frame.get("observation.state")
            if state is not None:
                state = np.asarray(state, dtype=np.float64).reshape(-1)
                perturb = float(np.cos(state.sum())) if state.size else 0.0
            else:
                perturb = 0.0

            rewards[i] = self.progress_weight * progress + self.state_weight * perturb
        return rewards
