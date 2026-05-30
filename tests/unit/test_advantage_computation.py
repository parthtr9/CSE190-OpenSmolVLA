"""
Unit tests: compute_advantages — mathematical correctness.

A(s,a) = G_t - V(s_t)

We verify this identity by constructing a ValueFunction with known weights
or by comparing against a manually computed baseline.
"""

import numpy as np
import pytest
import torch

from recap_smolvla.advantage import compute_advantages
from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.rollout import make_dummy_trajectory
from recap_smolvla.value_function import ValueFunction, compute_returns


OBS_DIM = 4


def _zero_value_fn() -> ValueFunction:
    """ValueFunction that always predicts 0 (V(s) = 0 for all s)."""
    vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=16)
    with torch.no_grad():
        for p in vf.parameters():
            p.zero_()
    return vf


# ---------------------------------------------------------------------------
# Correctness with zero value function
# ---------------------------------------------------------------------------

class TestComputeAdvantagesZeroVF:
    """When V(s) = 0, advantages should equal the raw returns."""

    def test_failure_trajectory(self):
        traj, _ = make_dummy_trajectory(length=5, obs_dim=OBS_DIM, success=False)
        vf = _zero_value_fn()
        advs, rets = compute_advantages(traj, vf, lambda t, s: sparse_reward_fn(t, s), False)
        np.testing.assert_allclose(advs, rets, atol=1e-5)

    def test_success_trajectory(self):
        traj, _ = make_dummy_trajectory(length=5, obs_dim=OBS_DIM, success=True)
        vf = _zero_value_fn()
        advs, rets = compute_advantages(traj, vf, lambda t, s: sparse_reward_fn(t, s), True)
        np.testing.assert_allclose(advs, rets, atol=1e-5)

    def test_returns_equal_manually_computed(self):
        traj, _ = make_dummy_trajectory(length=4, obs_dim=OBS_DIM, success=False)
        rewards = sparse_reward_fn(traj, False)
        expected_returns = compute_returns(rewards, gamma=1.0)
        vf = _zero_value_fn()
        advs, rets = compute_advantages(traj, vf, lambda t, s: sparse_reward_fn(t, s), False)
        np.testing.assert_allclose(rets, expected_returns, atol=1e-5)


# ---------------------------------------------------------------------------
# Advantage = return - value (general)
# ---------------------------------------------------------------------------

class TestComputeAdvantagesGeneral:
    def test_advantage_equals_return_minus_value(self):
        traj, _ = make_dummy_trajectory(length=6, obs_dim=OBS_DIM, success=False)
        vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)

        rewards = sparse_reward_fn(traj, False)
        returns = np.array(compute_returns(rewards), dtype=np.float32)
        obs = torch.tensor(np.stack([s["obs"] for s in traj]).astype(np.float32))
        with torch.no_grad():
            values = vf(obs).numpy()
        expected_advs = returns - values

        advs, _ = compute_advantages(traj, vf, lambda t, s: sparse_reward_fn(t, s), False)
        np.testing.assert_allclose(advs, expected_advs, atol=1e-5)

    def test_advantage_shape_matches_trajectory(self):
        T = 8
        traj, _ = make_dummy_trajectory(length=T, obs_dim=OBS_DIM)
        vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)
        advs, rets = compute_advantages(traj, vf, lambda t, s: sparse_reward_fn(t, s), False)
        assert advs.shape == (T,)
        assert rets.shape == (T,)

    def test_advantages_finite(self):
        traj, _ = make_dummy_trajectory(length=10, obs_dim=OBS_DIM)
        vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)
        advs, _ = compute_advantages(traj, vf, lambda t, s: sparse_reward_fn(t, s), False)
        assert np.all(np.isfinite(advs))

    def test_empty_trajectory(self):
        vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)
        advs, rets = compute_advantages([], vf, lambda t, s: sparse_reward_fn(t, s), False)
        assert len(advs) == 0
        assert len(rets) == 0

    def test_no_gradient_through_value_fn(self):
        """compute_advantages should not leave gradient on VF parameters."""
        traj, _ = make_dummy_trajectory(length=5, obs_dim=OBS_DIM)
        vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)
        compute_advantages(traj, vf, lambda t, s: sparse_reward_fn(t, s), False)
        for p in vf.parameters():
            assert p.grad is None, "compute_advantages should not accumulate gradients"
