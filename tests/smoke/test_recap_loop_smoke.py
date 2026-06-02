"""
Smoke tests: end-to-end RECAP training loop with mocks.

These tests run the full pipeline (collect → VF train → label → finetune)
with tiny settings to confirm nothing crashes.  They are the most important
"is it wired up?" tests.
"""

import numpy as np
import pytest
import torch

from recap_smolvla.rewards import sparse_reward_fn, dense_reward_fn
from recap_smolvla.rollout import MockEnv
from recap_smolvla.value_function import ValueFunction
from recap_smolvla.training import finetune_smolvla, recap_training_iteration
from recap_smolvla.advantage import label_trajectories
from tests.conftest import MockSmolVLAPolicy, OBS_DIM, ACTION_DIM

GOAL = np.array([0.8, 0.8], dtype=np.float32)


# ---------------------------------------------------------------------------
# finetune_smolvla smoke
# ---------------------------------------------------------------------------

def test_finetune_smolvla_runs(labeled_data, mock_policy):
    """finetune_smolvla completes without raising."""
    losses = finetune_smolvla(mock_policy, labeled_data, n_epochs=2, verbose=False)
    assert isinstance(losses, list)
    assert len(losses) == 2


def test_finetune_smolvla_empty_data(mock_policy):
    losses = finetune_smolvla(mock_policy, [], n_epochs=3)
    assert losses == []


def test_finetune_smolvla_losses_finite(labeled_data, mock_policy):
    losses = finetune_smolvla(mock_policy, labeled_data, n_epochs=3, verbose=False)
    assert all(np.isfinite(l) for l in losses), "All finetune losses must be finite"


def test_finetune_smolvla_policy_params_change(labeled_data, mock_policy):
    """Policy weights should change after fine-tuning."""
    before = {n: p.clone().detach() for n, p in mock_policy.named_parameters()}
    finetune_smolvla(mock_policy, labeled_data, n_epochs=5, lr=1e-3, verbose=False)
    changed = False
    for name, p in mock_policy.named_parameters():
        if not torch.allclose(before[name], p.detach()):
            changed = True
            break
    assert changed, "Policy parameters should change after fine-tuning"


def test_finetune_smolvla_subsamples_large_data(mock_policy):
    """max_labeled_steps caps FT dataset size (SmolVLA throughput fix)."""
    from recap_smolvla.rollout import make_dummy_trajectory

    big = []
    for i in range(2000):
        traj, _ = make_dummy_trajectory(length=2)
        for step in traj:
            step["instruction"] = "test task"
            step["advantage_positive"] = i % 2 == 0
            step["advantage"] = 1.0 if step["advantage_positive"] else -1.0
            big.append(step)

    n_calls = {"count": 0}
    original = mock_policy.compute_loss

    def counting_loss(batch):
        n_calls["count"] += 1
        return original(batch)

    mock_policy.compute_loss = counting_loss
    finetune_smolvla(
        mock_policy, big, n_epochs=1, batch_size=32,
        max_labeled_steps=64, verbose=False,
    )
    mock_policy.compute_loss = original
    # 64 steps / batch_size 32 = 2 batches per epoch
    assert n_calls["count"] == 64


def test_finetune_smolvla_max_labeled_steps_none_uses_all(mock_policy):
    """max_labeled_steps=None disables subsampling."""
    from recap_smolvla.rollout import make_dummy_trajectory

    data = []
    for _ in range(40):
        traj, _ = make_dummy_trajectory(length=1)
        for step in traj:
            step["instruction"] = "task"
            step["advantage_positive"] = True
            step["advantage"] = 1.0
            data.append(step)

    n_calls = {"count": 0}
    original = mock_policy.compute_loss

    def counting_loss(batch):
        n_calls["count"] += 1
        return original(batch)

    mock_policy.compute_loss = counting_loss
    finetune_smolvla(
        mock_policy, data, n_epochs=1, batch_size=10,
        max_labeled_steps=None, verbose=False,
    )
    mock_policy.compute_loss = original
    assert n_calls["count"] == 40


def test_smolvla_wrapper_normalizes_action_in_compute_loss():
    """FT must normalize rollout actions before forward (raw [-100,100] → unit scale)."""
    import torch
    from experiments.ablation_sparse_vs_dense import _SmolVLAWrapper

    mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    std = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32)

    class _FakeModel(torch.nn.Module):
        def forward(self, batch):
            self.last_action = batch["action"].detach().clone()
            return torch.tensor(1.0, requires_grad=True)

    wrapper = _SmolVLAWrapper(
        _FakeModel(),
        action_mean=mean,
        action_std=std,
    )
    raw = np.full(6, 100.0, dtype=np.float32)
    wrapper.compute_loss({
        "observation.state": np.zeros(6, dtype=np.float32),
        "action": raw,
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "instruction": "test",
    })
    expected = (torch.tensor(raw) - torch.tensor(mean)) / torch.tensor(std)
    # compute_loss tiles action to (B, n_action_steps, 6) — check one row
    assert torch.allclose(
        wrapper._model.last_action.reshape(-1, 6)[0], expected, atol=1e-4
    )


def test_finetune_smolvla_advantage_token_in_instruction(labeled_data, mock_policy):
    """The advantage token is actually prepended to at least some instructions."""
    seen_tokens = set()
    original_compute_loss = mock_policy.compute_loss

    instructions_seen = []

    def patched_compute_loss(batch):
        instructions_seen.append(batch.get("instruction", ""))
        return original_compute_loss(batch)

    mock_policy.compute_loss = patched_compute_loss
    finetune_smolvla(mock_policy, labeled_data, n_epochs=1, advantage_dropout=0.0, verbose=False)
    mock_policy.compute_loss = original_compute_loss

    has_positive = any("Advantage: positive" in i for i in instructions_seen)
    has_negative = any("Advantage: negative" in i for i in instructions_seen)
    assert has_positive or has_negative, "No advantage token was prepended to any instruction"


# ---------------------------------------------------------------------------
# recap_training_iteration smoke
# ---------------------------------------------------------------------------

def test_recap_iteration_runs_sparse():
    """Full RECAP iteration with sparse reward completes end-to-end."""
    env = MockEnv(max_steps=8, success_prob=0.3, seed=0)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)
    reward_fn = lambda t, s: sparse_reward_fn(t, s)

    sr, policy_out, vf_out, stats = recap_training_iteration(
        policy, vf, env, reward_fn,
        n_rollouts=5, vf_epochs=3, ft_epochs=2,
        max_steps=8, verbose=False,
    )
    assert isinstance(sr, float)
    assert 0.0 <= sr <= 1.0
    assert isinstance(stats, dict)
    assert "pct_positive" in stats


def test_recap_iteration_runs_dense():
    """Full RECAP iteration with dense reward completes end-to-end."""
    env = MockEnv(max_steps=8, success_prob=0.3, seed=1)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)
    reward_fn = lambda t, s: dense_reward_fn(t, s, GOAL)

    sr, _, _, stats = recap_training_iteration(
        policy, vf, env, reward_fn,
        n_rollouts=5, vf_epochs=3, ft_epochs=2,
        max_steps=8, verbose=False,
    )
    assert 0.0 <= sr <= 1.0
    assert all(np.isfinite(v) for v in stats["ft_losses"])


def test_recap_iteration_stats_structure():
    env = MockEnv(max_steps=8, seed=2)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)

    _, _, _, stats = recap_training_iteration(
        policy, vf, env, lambda t, s: sparse_reward_fn(t, s),
        n_rollouts=4, vf_epochs=2, ft_epochs=2,
        max_steps=8, verbose=False,
    )
    assert "success_rate" in stats
    assert "n_rollouts" in stats
    assert "n_labeled_steps" in stats
    assert "ft_losses" in stats
    assert "pct_positive" in stats


def test_two_recap_iterations_do_not_crash():
    """Two consecutive RECAP iterations complete without error."""
    env = MockEnv(max_steps=8, seed=3)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)
    reward_fn = lambda t, s: sparse_reward_fn(t, s)

    for _ in range(2):
        sr, policy, vf, stats = recap_training_iteration(
            policy, vf, env, reward_fn,
            n_rollouts=4, vf_epochs=2, ft_epochs=2,
            max_steps=8, verbose=False,
        )
    assert 0.0 <= sr <= 1.0
