"""
Smoke tests: collect_rollout and trajectory factory functions.
"""

import numpy as np
import pytest

from recap_smolvla.rollout import (
    MockEnv,
    collect_rollout,
    make_dummy_trajectory,
    make_dummy_rollouts,
)
from tests.conftest import MockSmolVLAPolicy, OBS_DIM, ACTION_DIM


# ---------------------------------------------------------------------------
# make_dummy_trajectory
# ---------------------------------------------------------------------------

def test_make_dummy_trajectory_length():
    traj, _ = make_dummy_trajectory(length=7, obs_dim=OBS_DIM)
    assert len(traj) == 7


def test_make_dummy_trajectory_success_flag():
    _, suc = make_dummy_trajectory(success=True)
    assert suc is True
    _, suc = make_dummy_trajectory(success=False)
    assert suc is False


def test_make_dummy_trajectory_step_keys():
    traj, _ = make_dummy_trajectory(length=3, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    for step in traj:
        for key in ("t", "obs", "action", "image", "reward", "success", "instruction"):
            assert key in step, f"Missing key '{key}' in step"


def test_make_dummy_trajectory_obs_shape():
    traj, _ = make_dummy_trajectory(length=5, obs_dim=OBS_DIM)
    for step in traj:
        assert step["obs"].shape == (OBS_DIM,)


def test_make_dummy_trajectory_action_shape():
    traj, _ = make_dummy_trajectory(length=5, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    for step in traj:
        assert step["action"].shape == (ACTION_DIM,)


def test_make_dummy_trajectory_timestep_indices():
    traj, _ = make_dummy_trajectory(length=4)
    assert [s["t"] for s in traj] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# make_dummy_rollouts
# ---------------------------------------------------------------------------

def test_make_dummy_rollouts_count():
    rollouts = make_dummy_rollouts(n=6, obs_dim=OBS_DIM)
    assert len(rollouts) == 6


def test_make_dummy_rollouts_each_is_tuple():
    rollouts = make_dummy_rollouts(n=3, obs_dim=OBS_DIM)
    for item in rollouts:
        traj, suc = item
        assert isinstance(traj, list)
        assert isinstance(suc, bool)


def test_make_dummy_rollouts_success_rate_approximate():
    rollouts = make_dummy_rollouts(n=100, obs_dim=OBS_DIM, success_rate=0.6, seed=0)
    actual_sr = np.mean([s for _, s in rollouts])
    assert 0.4 <= actual_sr <= 0.8, f"Success rate {actual_sr:.2f} far from 0.6"


# ---------------------------------------------------------------------------
# collect_rollout with MockEnv + MockSmolVLAPolicy
# ---------------------------------------------------------------------------

def test_collect_rollout_returns_tuple():
    env = MockEnv(max_steps=5, seed=0)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    result = collect_rollout(policy, env, max_steps=5)
    assert len(result) == 2
    traj, suc = result
    assert isinstance(traj, list)
    assert isinstance(suc, bool)


def test_collect_rollout_trajectory_not_empty():
    env = MockEnv(max_steps=10, seed=0)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    traj, _ = collect_rollout(policy, env, max_steps=10)
    assert len(traj) > 0


def test_collect_rollout_step_has_required_keys():
    env = MockEnv(max_steps=5, seed=0)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    traj, _ = collect_rollout(policy, env, max_steps=5)
    for step in traj:
        for key in ("t", "obs", "action", "reward", "instruction", "is_correction"):
            assert key in step, f"Missing key '{key}'"


def test_collect_rollout_respects_max_steps():
    env = MockEnv(max_steps=100, success_prob=0.0, seed=0)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    traj, _ = collect_rollout(policy, env, max_steps=7)
    assert len(traj) <= 7


def test_collect_rollout_obs_shape():
    env = MockEnv(max_steps=8, seed=0)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    traj, _ = collect_rollout(policy, env, max_steps=8)
    for step in traj:
        assert step["obs"].shape == (OBS_DIM,)


def test_collect_rollout_is_correction_false_by_default():
    env = MockEnv(max_steps=5, seed=0)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    traj, _ = collect_rollout(policy, env, max_steps=5)
    for step in traj:
        assert step["is_correction"] is False
