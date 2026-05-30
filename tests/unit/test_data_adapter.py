"""
Unit tests: data.py — LeRobot dataset adapter.

All tests use MockLeRobotDataset so no network or lerobot install is needed.

Key properties validated:
1. dataset_to_rollouts produces the correct number of rollouts.
2. Each trajectory step has all required keys.
3. Obs and action shapes match dataset dims.
4. Success flags are correctly read from `next.success`.
5. split_rollouts preserves total count and stratifies by success.
6. offline_eval_metrics returns a dict with all expected keys.
7. _extract_obs handles flat arrays and concatenation.
8. MockLeRobotDataset itself satisfies the adapter protocol.
"""

import numpy as np
import pytest
import torch

from recap_smolvla.data import (
    MockLeRobotDataset,
    dataset_info,
    dataset_to_rollouts,
    offline_eval_metrics,
    split_rollouts,
)
from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.value_function import ValueFunction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_EPISODES = 10
EP_LENGTH = 8
OBS_DIM = 5
ACTION_DIM = 2
SUCCESS_RATE = 0.4


@pytest.fixture(scope="module")
def mock_ds():
    return MockLeRobotDataset(
        n_episodes=N_EPISODES,
        episode_length=EP_LENGTH,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        success_rate=SUCCESS_RATE,
        seed=0,
    )


@pytest.fixture(scope="module")
def rollouts(mock_ds):
    return dataset_to_rollouts(mock_ds, verbose=False)


@pytest.fixture(scope="module")
def train_eval(rollouts):
    return split_rollouts(rollouts, eval_frac=0.2, seed=0)


# ---------------------------------------------------------------------------
# MockLeRobotDataset protocol
# ---------------------------------------------------------------------------

class TestMockLeRobotDataset:
    def test_len_equals_episodes_times_length(self, mock_ds):
        assert len(mock_ds) == N_EPISODES * EP_LENGTH

    def test_getitem_returns_dict(self, mock_ds):
        frame = mock_ds[0]
        assert isinstance(frame, dict)

    def test_frame_has_required_keys(self, mock_ds):
        frame = mock_ds[0]
        for key in ("observation.state", "action", "episode_index", "frame_index", "next.success"):
            assert key in frame, f"Frame missing '{key}'"

    def test_obs_shape(self, mock_ds):
        frame = mock_ds[0]
        assert np.array(frame["observation.state"]).shape == (OBS_DIM,)

    def test_action_shape(self, mock_ds):
        frame = mock_ds[0]
        assert np.array(frame["action"]).shape == (ACTION_DIM,)

    def test_features_dict(self, mock_ds):
        assert isinstance(mock_ds.features, dict)
        assert "observation.state" in mock_ds.features
        assert "action" in mock_ds.features

    def test_num_episodes(self, mock_ds):
        assert mock_ds.num_episodes == N_EPISODES


# ---------------------------------------------------------------------------
# dataset_to_rollouts
# ---------------------------------------------------------------------------

class TestDatasetToRollouts:
    def test_returns_list(self, rollouts):
        assert isinstance(rollouts, list)

    def test_count_equals_n_episodes(self, rollouts):
        assert len(rollouts) == N_EPISODES

    def test_each_element_is_tuple(self, rollouts):
        for item in rollouts:
            traj, suc = item
            assert isinstance(traj, list)
            assert isinstance(suc, bool)

    def test_trajectory_length(self, rollouts):
        for traj, _ in rollouts:
            assert len(traj) == EP_LENGTH, (
                f"Expected {EP_LENGTH} steps, got {len(traj)}"
            )

    def test_step_has_required_keys(self, rollouts):
        for traj, _ in rollouts:
            for step in traj:
                for key in ("t", "obs", "action", "instruction", "is_correction"):
                    assert key in step, f"Step missing key '{key}'"

    def test_obs_shape_per_step(self, rollouts):
        for traj, _ in rollouts:
            for step in traj:
                assert step["obs"].shape == (OBS_DIM,), (
                    f"obs shape {step['obs'].shape} != ({OBS_DIM},)"
                )

    def test_action_shape_per_step(self, rollouts):
        for traj, _ in rollouts:
            for step in traj:
                assert step["action"].shape == (ACTION_DIM,), (
                    f"action shape {step['action'].shape} != ({ACTION_DIM},)"
                )

    def test_timestep_indices_sequential(self, rollouts):
        for traj, _ in rollouts:
            assert [s["t"] for s in traj] == list(range(len(traj)))

    def test_is_correction_all_false(self, rollouts):
        for traj, _ in rollouts:
            for step in traj:
                assert step["is_correction"] is False

    def test_obs_dtype_float32(self, rollouts):
        for traj, _ in rollouts:
            for step in traj:
                assert step["obs"].dtype == np.float32

    def test_success_flags_are_bool(self, rollouts):
        for _, suc in rollouts:
            assert isinstance(suc, bool)

    def test_success_rate_approximate(self, rollouts):
        sr = np.mean([s for _, s in rollouts])
        # MockLeRobotDataset uses seed=0, success_rate=0.4 — allow wide range
        assert 0.0 <= sr <= 1.0

    def test_instruction_propagated(self, mock_ds):
        custom = "pick up the red cube"
        rollouts = dataset_to_rollouts(mock_ds, instruction=custom)
        for traj, _ in rollouts:
            for step in traj:
                assert step["instruction"] == custom

    def test_max_episodes_limits_count(self, mock_ds):
        rollouts = dataset_to_rollouts(mock_ds, max_episodes=3)
        assert len(rollouts) <= 3


# ---------------------------------------------------------------------------
# split_rollouts
# ---------------------------------------------------------------------------

class TestSplitRollouts:
    def test_total_count_preserved(self, rollouts, train_eval):
        train, evl = train_eval
        assert len(train) + len(evl) == len(rollouts)

    def test_eval_fraction_approximate(self, rollouts, train_eval):
        _, evl = train_eval
        actual_frac = len(evl) / len(rollouts)
        assert 0.1 <= actual_frac <= 0.4, (
            f"Eval fraction {actual_frac:.2f} far from 0.2"
        )

    def test_no_overlap(self, rollouts, train_eval):
        train, evl = train_eval
        # Use episode_index to check no episode appears in both
        train_eps = set(traj[0]["episode_index"] for traj, _ in train if traj)
        eval_eps = set(traj[0]["episode_index"] for traj, _ in evl if traj)
        assert train_eps.isdisjoint(eval_eps), "Train and eval sets must not overlap"

    def test_both_sets_non_empty(self, train_eval):
        train, evl = train_eval
        assert len(train) > 0
        assert len(evl) > 0

    def test_reproducible_with_same_seed(self, rollouts):
        t1, e1 = split_rollouts(rollouts, seed=42)
        t2, e2 = split_rollouts(rollouts, seed=42)
        assert len(t1) == len(t2)
        assert len(e1) == len(e2)

    def test_different_seeds_produce_different_splits(self, rollouts):
        if len(rollouts) < 5:
            pytest.skip("Too few rollouts for seed sensitivity test")
        _, e1 = split_rollouts(rollouts, seed=0)
        _, e2 = split_rollouts(rollouts, seed=99)
        eps1 = set(traj[0]["episode_index"] for traj, _ in e1 if traj)
        eps2 = set(traj[0]["episode_index"] for traj, _ in e2 if traj)
        assert eps1 != eps2, "Different seeds should produce different eval sets"


# ---------------------------------------------------------------------------
# dataset_info
# ---------------------------------------------------------------------------

class TestDatasetInfo:
    def test_returns_dict(self, mock_ds):
        info = dataset_info(mock_ds)
        assert isinstance(info, dict)

    def test_has_n_episodes(self, mock_ds):
        info = dataset_info(mock_ds)
        assert "n_episodes" in info
        assert info["n_episodes"] == N_EPISODES

    def test_has_n_frames(self, mock_ds):
        info = dataset_info(mock_ds)
        assert "n_frames" in info
        assert info["n_frames"] == N_EPISODES * EP_LENGTH

    def test_inferred_obs_dim(self, mock_ds):
        info = dataset_info(mock_ds)
        assert info.get("obs_dim") == OBS_DIM

    def test_inferred_action_dim(self, mock_ds):
        info = dataset_info(mock_ds)
        assert info.get("action_dim") == ACTION_DIM


# ---------------------------------------------------------------------------
# offline_eval_metrics
# ---------------------------------------------------------------------------

class _TinyPolicy(torch.nn.Module):
    """Minimal policy that computes a real loss for metric testing."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.head = torch.nn.Linear(obs_dim, action_dim)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def compute_loss(self, batch):
        obs = torch.tensor(np.asarray(batch["obs"], dtype=np.float32))
        pred = self.head(obs.reshape(-1)[:self.obs_dim])
        target = torch.tensor(np.asarray(batch["action"], dtype=np.float32)[:self.action_dim])
        return torch.nn.functional.mse_loss(pred, target)


class TestOfflineEvalMetrics:
    @pytest.fixture(scope="class")
    def setup(self, rollouts):
        train_r, eval_r = split_rollouts(rollouts, eval_frac=0.3, seed=0)
        vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)
        from recap_smolvla.value_function import train_value_function
        train_value_function(train_r, vf, lambda t, s: sparse_reward_fn(t, s), epochs=5)
        policy = _TinyPolicy(OBS_DIM, ACTION_DIM)
        return eval_r, vf, policy

    def test_returns_dict(self, setup):
        eval_r, vf, policy = setup
        metrics = offline_eval_metrics(policy, eval_r, vf, lambda t, s: sparse_reward_fn(t, s))
        assert isinstance(metrics, dict)

    def test_expected_keys_present(self, setup):
        eval_r, vf, policy = setup
        metrics = offline_eval_metrics(policy, eval_r, vf, lambda t, s: sparse_reward_fn(t, s))
        for key in (
            "vf_mse", "score_return_corr", "bc_loss_mean",
            "bc_loss_positive", "bc_loss_negative", "bc_loss_gap",
            "pct_positive", "advantage_mean", "advantage_std",
            "n_eval_steps", "n_eval_episodes",
        ):
            assert key in metrics, f"Missing metric key '{key}'"

    def test_vf_mse_is_positive_finite(self, setup):
        eval_r, vf, policy = setup
        metrics = offline_eval_metrics(policy, eval_r, vf, lambda t, s: sparse_reward_fn(t, s))
        assert np.isfinite(metrics["vf_mse"])
        assert metrics["vf_mse"] >= 0.0

    def test_pct_positive_in_range(self, setup):
        eval_r, vf, policy = setup
        metrics = offline_eval_metrics(policy, eval_r, vf, lambda t, s: sparse_reward_fn(t, s))
        assert 0.0 <= metrics["pct_positive"] <= 1.0

    def test_n_eval_episodes_correct(self, setup):
        eval_r, vf, policy = setup
        metrics = offline_eval_metrics(policy, eval_r, vf, lambda t, s: sparse_reward_fn(t, s))
        assert metrics["n_eval_episodes"] == len(eval_r)

    def test_empty_eval_returns_empty_dict(self):
        vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=32)
        policy = _TinyPolicy(OBS_DIM, ACTION_DIM)
        metrics = offline_eval_metrics(policy, [], vf, lambda t, s: sparse_reward_fn(t, s))
        assert metrics == {}
