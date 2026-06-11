"""
Shared pytest fixtures for SmolVLA + RECAP test suite.

Fixtures are organized by scope:
  session   — expensive objects created once (model weights, large datasets)
  module    — medium cost, recreated per test file
  function  — cheap objects recreated for each test (default)

All fixtures here use only NumPy + PyTorch — no gym_pusht or lerobot required.
Integration tests that need those dependencies guard themselves with
pytest.importorskip or the @pytest.mark.integration marker.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from recap_smolvla.rollout import MockEnv, make_dummy_rollouts, make_dummy_trajectory
from recap_smolvla.value_function import ValueFunction
from recap_smolvla.rewards import sparse_reward_fn, dense_reward_fn


# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACTION_DIM = 2
HIDDEN_DIM = 64        # smaller than production 256 to keep tests fast
TRAJ_LENGTH = 10
N_ROLLOUTS = 5
GOAL_POS = np.array([0.8, 0.8], dtype=np.float32)


# ---------------------------------------------------------------------------
# MockSmolVLAPolicy
# ---------------------------------------------------------------------------

class MockSmolVLAPolicy(nn.Module):
    """Minimal policy for testing the RECAP training loop.

    Interface mirrors the real SmolVLA policy:
      - select_action(obs_dict) → np.ndarray
      - compute_loss(batch_dict) → torch.Tensor (scalar, with grad_fn)

    The advantage token in the instruction string is parsed and used to
    add a small bias — this makes the gradient tests meaningful (the
    advantage conditioning is not ignored).
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        # Learnable advantage bias — initialized non-zero so that positive vs
        # negative advantage conditioning produces measurably different outputs.
        self.advantage_bias = nn.Parameter(torch.ones(hidden_dim) * 0.5)

    def _parse_advantage_positive(self, instruction: str) -> bool:
        """Return True if instruction starts with 'Advantage: positive'."""
        return instruction.strip().startswith("Advantage: positive")

    def _encode(self, obs: torch.Tensor, instruction: str) -> torch.Tensor:
        h = self.encoder(obs)
        if self._parse_advantage_positive(instruction):
            h = h + self.advantage_bias
        return h

    def select_action(self, obs_dict: dict[str, Any]) -> np.ndarray:
        obs = _tensorify_obs(obs_dict, self.obs_dim)
        instruction = str(obs_dict.get("instruction", ""))
        with torch.no_grad():
            h = self._encode(obs, instruction)
            return self.action_head(h).numpy()

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        obs = _tensorify_obs(batch, self.obs_dim)
        instruction = str(batch.get("instruction", ""))
        action_target = _tensorify_action(batch, self.action_dim)

        h = self._encode(obs, instruction)
        pred = self.action_head(h)
        return nn.functional.mse_loss(pred, action_target)


def _tensorify_obs(d: dict[str, Any], obs_dim: int) -> torch.Tensor:
    for key in ("obs", "observation.state"):
        if key in d:
            v = d[key]
            if isinstance(v, torch.Tensor):
                return v.float().reshape(-1)[:obs_dim]
            return torch.tensor(np.asarray(v, dtype=np.float32)).reshape(-1)[:obs_dim]
    return torch.zeros(obs_dim)


def _tensorify_action(d: dict[str, Any], action_dim: int) -> torch.Tensor:
    v = d.get("action", np.zeros(action_dim))
    if isinstance(v, torch.Tensor):
        return v.float().reshape(-1)[:action_dim]
    return torch.tensor(np.asarray(v, dtype=np.float32)).reshape(-1)[:action_dim]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_policy() -> MockSmolVLAPolicy:
    """Fresh MockSmolVLAPolicy for each test."""
    return MockSmolVLAPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)


@pytest.fixture
def mock_env() -> MockEnv:
    """Seeded MockEnv for each test.
    Uses success_prob=0.4 so unit tests get a non-trivial mix of outcomes
    without requiring a policy that actually learns contact physics."""
    return MockEnv(max_steps=TRAJ_LENGTH, success_prob=0.4, seed=42)


@pytest.fixture
def value_fn() -> ValueFunction:
    """Fresh ValueFunction (small hidden dim) for each test."""
    return ValueFunction(obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM)


@pytest.fixture
def trained_value_fn() -> ValueFunction:
    """ValueFunction that has been briefly trained on dummy rollouts."""
    vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM)
    rollouts = make_dummy_rollouts(n=5, obs_dim=OBS_DIM, success_rate=0.4, seed=7)
    from recap_smolvla.value_function import train_value_function
    train_value_function(rollouts, vf, lambda t, s: sparse_reward_fn(t, s), epochs=10, verbose=False)
    return vf


@pytest.fixture
def dummy_trajectory() -> tuple[list[dict], bool]:
    """Single synthetic trajectory (10 steps, failure)."""
    return make_dummy_trajectory(length=TRAJ_LENGTH, obs_dim=OBS_DIM, action_dim=ACTION_DIM, success=False)


@pytest.fixture
def dummy_trajectory_success() -> tuple[list[dict], bool]:
    """Single synthetic trajectory (10 steps, success)."""
    return make_dummy_trajectory(length=TRAJ_LENGTH, obs_dim=OBS_DIM, action_dim=ACTION_DIM, success=True)


@pytest.fixture
def dummy_rollouts() -> list[tuple[list[dict], bool]]:
    """Small set of mixed rollouts (5 total, ~40% success)."""
    return make_dummy_rollouts(n=N_ROLLOUTS, obs_dim=OBS_DIM, success_rate=0.4, seed=0)


@pytest.fixture
def sparse_reward_fn_fixture():
    """Sparse reward function with default C_fail=100."""
    return lambda traj, suc: sparse_reward_fn(traj, suc)


@pytest.fixture
def dense_reward_fn_fixture():
    """Dense reward function with GOAL_POS and default alpha=0.1."""
    return lambda traj, suc: dense_reward_fn(traj, suc, GOAL_POS)


@pytest.fixture
def labeled_data(dummy_rollouts, trained_value_fn, sparse_reward_fn_fixture):
    """Pre-labeled dataset from dummy rollouts, ready for finetune_smolvla."""
    from recap_smolvla.advantage import label_trajectories
    return label_trajectories(dummy_rollouts, trained_value_fn, sparse_reward_fn_fixture)


# ---------------------------------------------------------------------------
# Scope: session — heavy shared resources
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def session_rollouts() -> list[tuple[list[dict], bool]]:
    """Larger rollout set shared across the whole test session."""
    return make_dummy_rollouts(n=20, obs_dim=OBS_DIM, success_rate=0.35, seed=99)


@pytest.fixture(scope="session")
def session_value_fn() -> ValueFunction:
    """Session-scoped trained VF (trained once, reused)."""
    rollouts = make_dummy_rollouts(n=20, obs_dim=OBS_DIM, success_rate=0.35, seed=99)
    vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM)
    from recap_smolvla.value_function import train_value_function
    train_value_function(rollouts, vf, lambda t, s: sparse_reward_fn(t, s), epochs=20, verbose=False)
    return vf
