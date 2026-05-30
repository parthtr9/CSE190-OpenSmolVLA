"""
Smoke tests: ValueFunction and DistributionalValueFunction basic forward passes.
"""

import numpy as np
import pytest
import torch

from recap_smolvla.value_function import (
    ValueFunction,
    DistributionalValueFunction,
    compute_returns,
    train_value_function,
)
from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.rollout import make_dummy_rollouts


# ---------------------------------------------------------------------------
# ValueFunction forward pass
# ---------------------------------------------------------------------------

def test_value_fn_forward_single_obs():
    vf = ValueFunction(obs_dim=4, hidden_dim=32)
    obs = torch.randn(4)
    out = vf(obs)
    assert out.shape == torch.Size([]), "Single obs should give scalar output"


def test_value_fn_forward_batch():
    vf = ValueFunction(obs_dim=4, hidden_dim=32)
    obs = torch.randn(8, 4)
    out = vf(obs)
    assert out.shape == torch.Size([8])


def test_value_fn_predict_numpy():
    vf = ValueFunction(obs_dim=4, hidden_dim=32)
    obs = np.random.randn(4).astype(np.float32)
    out = vf.predict(obs)
    assert isinstance(out, np.ndarray)
    assert out.shape == ()  # scalar


def test_value_fn_predict_batch_numpy():
    vf = ValueFunction(obs_dim=4, hidden_dim=32)
    obs = np.random.randn(6, 4).astype(np.float32)
    out = vf.predict(obs)
    assert out.shape == (6,)


# ---------------------------------------------------------------------------
# DistributionalValueFunction smoke
# ---------------------------------------------------------------------------

def test_distributional_vf_forward():
    dvf = DistributionalValueFunction(obs_dim=4, n_bins=11, hidden_dim=32)
    obs = torch.randn(3, 4)
    log_probs = dvf(obs)
    assert log_probs.shape == (3, 11)
    # Should be valid log-probabilities (all <= 0)
    assert torch.all(log_probs <= 0)


def test_distributional_vf_expected_value_shape():
    dvf = DistributionalValueFunction(obs_dim=4, n_bins=11, hidden_dim=32)
    obs = torch.randn(5, 4)
    ev = dvf.expected_value(obs)
    assert ev.shape == (5,)


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------

def test_compute_returns_no_discount():
    rewards = [1.0, 2.0, 3.0]
    returns = compute_returns(rewards, gamma=1.0)
    assert returns == [6.0, 5.0, 3.0]


def test_compute_returns_with_discount():
    returns = compute_returns([1.0, 1.0, 1.0], gamma=0.9)
    assert abs(returns[0] - (1 + 0.9 + 0.81)) < 1e-5


def test_compute_returns_empty():
    assert compute_returns([], gamma=1.0) == []


def test_compute_returns_single():
    assert compute_returns([5.0], gamma=1.0) == [5.0]


# ---------------------------------------------------------------------------
# train_value_function smoke (very few epochs)
# ---------------------------------------------------------------------------

def test_train_value_function_runs():
    rollouts = make_dummy_rollouts(n=3, obs_dim=4, success_rate=0.4, seed=0)
    vf = ValueFunction(obs_dim=4, hidden_dim=32)
    losses = train_value_function(
        rollouts, vf,
        lambda t, s: sparse_reward_fn(t, s),
        epochs=3,
        verbose=False,
    )
    assert len(losses) == 3
    assert all(isinstance(l, float) for l in losses)


def test_train_value_function_loss_is_finite():
    rollouts = make_dummy_rollouts(n=3, obs_dim=4, success_rate=0.4, seed=1)
    vf = ValueFunction(obs_dim=4, hidden_dim=32)
    losses = train_value_function(
        rollouts, vf,
        lambda t, s: sparse_reward_fn(t, s),
        epochs=5,
        verbose=False,
    )
    assert all(np.isfinite(l) for l in losses), "Losses must be finite"
