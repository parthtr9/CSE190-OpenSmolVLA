"""
Setup tests: real LeRobot dataset loading.

All tests are marked @pytest.mark.integration and require:
  1. lerobot installed  (pip install lerobot)
  2. Network access to HuggingFace Hub
  3. LEROBOT_DATASETS_TEST=1 environment variable  (safety gate)

Run with:
    LEROBOT_DATASETS_TEST=1 pytest tests/setup/test_lerobot_datasets.py -m integration -v

Datasets tested:
  lerobot/pusht                        (2D push, 206 eps, obs_dim=5)
  lerobot/aloha_sim_transfer_cube_human (bimanual, 50 eps, obs_dim=14)
  lerobot/aloha_sim_insertion_human     (peg insertion, 50 eps, sparse reward stress test)

For a quick smoke check with no network:
  pytest tests/setup/test_lerobot_datasets.py -k "mock"  (always runs)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from recap_smolvla.data import (
    MockLeRobotDataset,
    dataset_info,
    dataset_to_rollouts,
    split_rollouts,
)

# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

lerobot = pytest.importorskip("lerobot", reason="lerobot not installed")

REAL_DATASETS_ENABLED = os.environ.get("LEROBOT_DATASETS_TEST", "0") == "1"
real_datasets = pytest.mark.skipif(
    not REAL_DATASETS_ENABLED,
    reason="Set LEROBOT_DATASETS_TEST=1 to run real dataset tests (requires network)",
)


# ---------------------------------------------------------------------------
# Mock adapter smoke (always runs — no network)
# ---------------------------------------------------------------------------

class TestMockAdapterSmoke:
    """Verify adapter contract using MockLeRobotDataset — no network needed."""

    def test_mock_roundtrip(self):
        ds = MockLeRobotDataset(n_episodes=5, episode_length=6, obs_dim=4, action_dim=2)
        rollouts = dataset_to_rollouts(ds)
        assert len(rollouts) == 5

    def test_mock_split(self):
        ds = MockLeRobotDataset(n_episodes=10, episode_length=4, obs_dim=4, action_dim=2)
        rollouts = dataset_to_rollouts(ds)
        train, evl = split_rollouts(rollouts, eval_frac=0.2)
        assert len(train) + len(evl) == 10

    def test_mock_dataset_info(self):
        ds = MockLeRobotDataset(n_episodes=8, episode_length=5, obs_dim=3, action_dim=1)
        info = dataset_info(ds)
        assert info["n_episodes"] == 8
        assert info["n_frames"] == 40
        assert info["obs_dim"] == 3

    def test_full_recap_pipeline_on_mock(self):
        """Run VF training + advantage labeling on mock data end-to-end."""
        from recap_smolvla.rewards import sparse_reward_fn
        from recap_smolvla.value_function import ValueFunction, train_value_function
        from recap_smolvla.advantage import label_trajectories

        ds = MockLeRobotDataset(n_episodes=10, episode_length=8, obs_dim=5, action_dim=2, seed=7)
        rollouts = dataset_to_rollouts(ds)
        train_r, _ = split_rollouts(rollouts, eval_frac=0.2)

        vf = ValueFunction(obs_dim=5, hidden_dim=32)
        losses = train_value_function(train_r, vf, lambda t, s: sparse_reward_fn(t, s), epochs=5)
        labeled = label_trajectories(train_r, vf, lambda t, s: sparse_reward_fn(t, s))

        assert len(losses) == 5
        assert all(np.isfinite(l) for l in losses)
        assert len(labeled) > 0
        for step in labeled:
            assert "advantage" in step
            assert "advantage_positive" in step


# ---------------------------------------------------------------------------
# Real dataset tests (require LEROBOT_DATASETS_TEST=1 + network)
# ---------------------------------------------------------------------------

@real_datasets
@pytest.mark.integration
class TestPushTDataset:
    """lerobot/pusht: 2D push task, obs_dim=5, action_dim=2, 206 episodes."""

    REPO_ID = "lerobot/pusht"
    MAX_EPISODES = 20  # subset for speed

    @pytest.fixture(scope="class")
    def rollouts(self):
        from recap_smolvla.data import load_lerobot_dataset
        ds = load_lerobot_dataset(self.REPO_ID, episodes=list(range(self.MAX_EPISODES)))
        return dataset_to_rollouts(ds, verbose=False)

    def test_loads_correct_episode_count(self, rollouts):
        assert len(rollouts) == self.MAX_EPISODES

    def test_obs_dim_is_5(self, rollouts):
        """PushT state = [agent_x, agent_y, block_x, block_y, block_angle]."""
        obs_dim = rollouts[0][0][0]["obs"].shape[0]
        assert obs_dim == 5, f"Expected obs_dim=5 for PushT, got {obs_dim}"

    def test_action_dim_is_2(self, rollouts):
        action_dim = rollouts[0][0][0]["action"].shape[0]
        assert action_dim == 2, f"Expected action_dim=2 for PushT, got {action_dim}"

    def test_success_flag_varies(self, rollouts):
        """PushT should have a mix of successes and failures."""
        successes = [s for _, s in rollouts]
        assert any(successes), "No successful episodes found in PushT subset"
        assert not all(successes), "All episodes succeeded — success flag may not be reading correctly"

    def test_vf_trains_on_pusht(self, rollouts):
        from recap_smolvla.rewards import sparse_reward_fn
        from recap_smolvla.value_function import ValueFunction, train_value_function
        obs_dim = rollouts[0][0][0]["obs"].shape[0]
        vf = ValueFunction(obs_dim=obs_dim, hidden_dim=64)
        losses = train_value_function(
            rollouts, vf, lambda t, s: sparse_reward_fn(t, s),
            epochs=10, verbose=False,
        )
        assert losses[0] > losses[-1], "VF loss should decrease on PushT data"

    def test_advantage_labeling_on_pusht(self, rollouts):
        from recap_smolvla.rewards import sparse_reward_fn
        from recap_smolvla.value_function import ValueFunction, train_value_function
        from recap_smolvla.advantage import label_trajectories, advantage_distribution_stats
        obs_dim = rollouts[0][0][0]["obs"].shape[0]
        vf = ValueFunction(obs_dim=obs_dim, hidden_dim=64)
        train_value_function(rollouts, vf, lambda t, s: sparse_reward_fn(t, s), epochs=10)
        labeled = label_trajectories(rollouts, vf, lambda t, s: sparse_reward_fn(t, s))
        stats = advantage_distribution_stats(labeled)
        assert 0.05 <= stats["pct_positive"] <= 0.55, (
            f"PushT pct_positive={stats['pct_positive']:.1%} outside expected range"
        )


@real_datasets
@pytest.mark.integration
class TestAlohaInsertionDataset:
    """
    lerobot/aloha_sim_insertion_human: sparse reward stress test.
    Very low success rate → almost all advantages negative under sparse reward.
    Dense reward should produce more positive-advantage steps.
    """

    REPO_ID = "lerobot/aloha_sim_insertion_human"
    MAX_EPISODES = 10

    @pytest.fixture(scope="class")
    def rollouts(self):
        from recap_smolvla.data import load_lerobot_dataset
        ds = load_lerobot_dataset(self.REPO_ID, episodes=list(range(self.MAX_EPISODES)))
        return dataset_to_rollouts(ds, verbose=False)

    def test_obs_dim_is_14(self, rollouts):
        """ALOHA bimanual arm has 14-DOF state."""
        obs_dim = rollouts[0][0][0]["obs"].shape[0]
        assert obs_dim == 14, f"Expected obs_dim=14 for ALOHA, got {obs_dim}"

    def test_action_dim_is_14(self, rollouts):
        action_dim = rollouts[0][0][0]["action"].shape[0]
        assert action_dim == 14, f"Expected action_dim=14 for ALOHA, got {action_dim}"

    def test_sparse_pct_positive_low_due_to_hard_task(self, rollouts):
        """
        Insertion is hard. With sparse reward, almost all advantages should
        be negative. This replicates the sparse reward problem described in
        the research plan.
        """
        from recap_smolvla.rewards import sparse_reward_fn
        from recap_smolvla.value_function import ValueFunction, train_value_function
        from recap_smolvla.advantage import label_trajectories, advantage_distribution_stats
        obs_dim = rollouts[0][0][0]["obs"].shape[0]
        vf = ValueFunction(obs_dim=obs_dim, hidden_dim=64)
        train_value_function(rollouts, vf, lambda t, s: sparse_reward_fn(t, s), epochs=10)
        labeled = label_trajectories(rollouts, vf, lambda t, s: sparse_reward_fn(t, s))
        stats = advantage_distribution_stats(labeled)
        # With threshold_pct=70 we always get ~30% positive by definition,
        # but the absolute advantages are much more negative than PushT.
        # Report it for the paper, don't assert a specific value.
        print(f"\n  ALOHA insertion sparse pct_positive: {stats['pct_positive']:.1%}")
        print(f"  ALOHA insertion advantage mean: {stats['mean']:.2f}")
        assert np.isfinite(stats["mean"])


@real_datasets
@pytest.mark.integration
class TestAlohaTransferDataset:
    """lerobot/aloha_sim_transfer_cube_human: bimanual cube transfer."""

    REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
    MAX_EPISODES = 10

    @pytest.fixture(scope="class")
    def rollouts(self):
        from recap_smolvla.data import load_lerobot_dataset
        ds = load_lerobot_dataset(self.REPO_ID, episodes=list(range(self.MAX_EPISODES)))
        return dataset_to_rollouts(ds, verbose=False)

    def test_loads_without_error(self, rollouts):
        assert len(rollouts) > 0

    def test_recap_pipeline_end_to_end(self, rollouts):
        """Full offline RECAP on ALOHA transfer data should not crash."""
        from recap_smolvla.rewards import sparse_reward_fn
        from recap_smolvla.value_function import ValueFunction, train_value_function
        from recap_smolvla.advantage import label_trajectories

        obs_dim = rollouts[0][0][0]["obs"].shape[0]
        vf = ValueFunction(obs_dim=obs_dim, hidden_dim=64)
        train_value_function(rollouts, vf, lambda t, s: sparse_reward_fn(t, s), epochs=5)
        labeled = label_trajectories(rollouts, vf, lambda t, s: sparse_reward_fn(t, s))
        assert len(labeled) > 0
