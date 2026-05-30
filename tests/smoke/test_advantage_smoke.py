"""
Smoke tests: advantage computation and labeling don't crash and return
sensible shapes.
"""

import numpy as np
import pytest

from recap_smolvla.advantage import (
    compute_advantages,
    label_trajectories,
    advantage_distribution_stats,
    assert_advantage_label_invariants,
)
from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.rollout import make_dummy_rollouts, make_dummy_trajectory
from recap_smolvla.value_function import ValueFunction


OBS_DIM = 4


@pytest.fixture
def vf():
    return ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)


@pytest.fixture
def rollouts():
    return make_dummy_rollouts(n=4, obs_dim=OBS_DIM, success_rate=0.5, seed=0)


def reward_fn(traj, suc):
    return sparse_reward_fn(traj, suc)


# ---------------------------------------------------------------------------
# compute_advantages smoke
# ---------------------------------------------------------------------------

def test_compute_advantages_returns_two_arrays(vf, rollouts):
    traj, success = rollouts[0]
    advs, rets = compute_advantages(traj, vf, reward_fn, success)
    assert isinstance(advs, np.ndarray)
    assert isinstance(rets, np.ndarray)


def test_compute_advantages_length_matches_trajectory(vf, rollouts):
    traj, success = rollouts[0]
    advs, rets = compute_advantages(traj, vf, reward_fn, success)
    assert len(advs) == len(traj)
    assert len(rets) == len(traj)


def test_compute_advantages_empty_trajectory(vf):
    advs, rets = compute_advantages([], vf, reward_fn, False)
    assert len(advs) == 0
    assert len(rets) == 0


def test_compute_advantages_all_finite(vf, rollouts):
    for traj, success in rollouts:
        advs, _ = compute_advantages(traj, vf, reward_fn, success)
        assert np.all(np.isfinite(advs)), "Advantages must be finite"


# ---------------------------------------------------------------------------
# label_trajectories smoke
# ---------------------------------------------------------------------------

def test_label_trajectories_returns_list(vf, rollouts):
    labeled = label_trajectories(rollouts, vf, reward_fn)
    assert isinstance(labeled, list)


def test_label_trajectories_total_length(vf, rollouts):
    total_steps = sum(len(t) for t, _ in rollouts)
    labeled = label_trajectories(rollouts, vf, reward_fn)
    assert len(labeled) == total_steps


def test_label_trajectories_has_required_keys(vf, rollouts):
    labeled = label_trajectories(rollouts, vf, reward_fn)
    for step in labeled:
        assert "advantage" in step
        assert "advantage_positive" in step
        assert "return_" in step


def test_label_trajectories_advantage_positive_is_bool(vf, rollouts):
    labeled = label_trajectories(rollouts, vf, reward_fn)
    for step in labeled:
        assert isinstance(step["advantage_positive"], bool)


def test_label_trajectories_empty_rollouts(vf):
    labeled = label_trajectories([], vf, reward_fn)
    assert labeled == []


def test_label_trajectories_corrections_forced_positive(vf):
    """Steps with is_correction=True must always be labeled positive."""
    traj, _ = make_dummy_trajectory(length=5, obs_dim=OBS_DIM, success=False)
    traj[2]["is_correction"] = True
    rollouts = [(traj, False)]
    labeled = label_trajectories(rollouts, vf, reward_fn)
    correction_step = labeled[2]
    assert correction_step["advantage_positive"] is True


# ---------------------------------------------------------------------------
# advantage_distribution_stats smoke
# ---------------------------------------------------------------------------

def test_advantage_distribution_stats_keys(vf, rollouts):
    labeled = label_trajectories(rollouts, vf, reward_fn)
    stats = advantage_distribution_stats(labeled)
    for key in ("mean", "std", "min", "max", "pct_positive"):
        assert key in stats


def test_advantage_distribution_stats_empty():
    stats = advantage_distribution_stats([])
    assert stats == {}


def test_advantage_distribution_stats_pct_positive_in_range(vf, rollouts):
    labeled = label_trajectories(rollouts, vf, reward_fn)
    stats = advantage_distribution_stats(labeled)
    assert 0.0 <= stats["pct_positive"] <= 1.0


# ---------------------------------------------------------------------------
# assert_advantage_label_invariants smoke
# ---------------------------------------------------------------------------

def test_assert_invariants_passes_on_valid_data(vf, rollouts):
    labeled = label_trajectories(rollouts, vf, reward_fn)
    # Should not raise
    assert_advantage_label_invariants(labeled)
