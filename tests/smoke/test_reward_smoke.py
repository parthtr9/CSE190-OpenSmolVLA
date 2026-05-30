"""
Smoke tests: reward functions return the right shape and don't crash.

Each test runs in under 10 ms — no training, no model weights.
"""

import numpy as np
import pytest

from recap_smolvla.rewards import sparse_reward_fn, dense_reward_fn, REWARD_REGISTRY
from recap_smolvla.rollout import make_dummy_trajectory

GOAL = np.array([0.8, 0.8], dtype=np.float32)


# ---------------------------------------------------------------------------
# Sparse reward smoke
# ---------------------------------------------------------------------------

def test_sparse_reward_runs_on_failure():
    traj, _ = make_dummy_trajectory(length=5, success=False)
    r = sparse_reward_fn(traj, False)
    assert len(r) == 5


def test_sparse_reward_runs_on_success():
    traj, _ = make_dummy_trajectory(length=5, success=True)
    r = sparse_reward_fn(traj, True)
    assert len(r) == 5


def test_sparse_reward_empty_trajectory():
    r = sparse_reward_fn([], False)
    assert r == []


def test_sparse_reward_returns_list_of_floats():
    traj, _ = make_dummy_trajectory(length=8, success=False)
    r = sparse_reward_fn(traj, False)
    assert all(isinstance(v, float) for v in r)


# ---------------------------------------------------------------------------
# Dense reward smoke
# ---------------------------------------------------------------------------

def test_dense_reward_runs():
    traj, _ = make_dummy_trajectory(length=5, success=False)
    r = dense_reward_fn(traj, False, GOAL)
    assert len(r) == 5


def test_dense_reward_empty_trajectory():
    r = dense_reward_fn([], False, GOAL)
    assert r == []


def test_dense_reward_returns_list_of_floats():
    traj, _ = make_dummy_trajectory(length=6, success=True)
    r = dense_reward_fn(traj, True, GOAL)
    assert all(isinstance(v, float) for v in r)


# ---------------------------------------------------------------------------
# Registry smoke
# ---------------------------------------------------------------------------

def test_reward_registry_has_sparse_and_dense():
    assert "sparse" in REWARD_REGISTRY
    assert "dense" in REWARD_REGISTRY


def test_reward_registry_callables():
    for name, fn in REWARD_REGISTRY.items():
        assert callable(fn), f"REWARD_REGISTRY['{name}'] is not callable"
