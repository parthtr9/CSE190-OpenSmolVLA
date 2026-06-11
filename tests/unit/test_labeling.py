"""
Unit tests: label_trajectories — threshold logic and invariants.

Key properties:
1. Exactly (100 - threshold_pct)% of steps should be labeled positive
   (modulo rounding and correction overrides).
2. Human corrections are always labeled positive.
3. A higher threshold_pct → fewer positive labels.
4. Labeled data preserves all original step fields.
"""

import numpy as np
import pytest
import warnings

from recap_smolvla.advantage import (
    label_trajectories,
    advantage_distribution_stats,
    assert_advantage_label_invariants,
)
from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.rollout import make_dummy_rollouts, make_dummy_trajectory
from recap_smolvla.value_function import ValueFunction


OBS_DIM = 4
HIDDEN = 32


@pytest.fixture(scope="module")
def vf():
    return ValueFunction(obs_dim=OBS_DIM, hidden_dim=HIDDEN)


@pytest.fixture(scope="module")
def rollouts():
    return make_dummy_rollouts(n=10, obs_dim=OBS_DIM, length=8, success_rate=0.4, seed=7)


def rf(traj, suc):
    return sparse_reward_fn(traj, suc)


# ---------------------------------------------------------------------------
# Positive fraction matches threshold
# ---------------------------------------------------------------------------

class TestLabelingThreshold:
    def test_default_threshold_yields_approx_30pct_positive(self, vf, rollouts):
        labeled = label_trajectories(rollouts, vf, rf, threshold_pct=70.0)
        stats = advantage_distribution_stats(labeled)
        pct = stats["pct_positive"]
        # With threshold_pct=70, we label top 30% as positive.
        # Due to discrete rounding and ties, allow [20%, 40%] range.
        assert 0.20 <= pct <= 0.40, f"Expected ~30% positive, got {pct:.1%}"

    def test_threshold_0_all_positive(self, vf, rollouts):
        """threshold_pct=0 → everything above 0th percentile → all positive."""
        labeled = label_trajectories(rollouts, vf, rf, threshold_pct=0.0)
        pct = advantage_distribution_stats(labeled)["pct_positive"]
        assert pct > 0.90, f"threshold_pct=0 should label nearly all positive, got {pct:.1%}"

    def test_threshold_100_almost_none_positive(self, vf, rollouts):
        """threshold_pct=100 → threshold at maximum → almost no positives."""
        labeled = label_trajectories(rollouts, vf, rf, threshold_pct=100.0)
        pct = advantage_distribution_stats(labeled)["pct_positive"]
        assert pct < 0.05, f"threshold_pct=100 should label almost none positive, got {pct:.1%}"

    def test_higher_threshold_fewer_positives(self, vf, rollouts):
        labeled_30 = label_trajectories(rollouts, vf, rf, threshold_pct=30.0)
        labeled_70 = label_trajectories(rollouts, vf, rf, threshold_pct=70.0)
        pct_30 = advantage_distribution_stats(labeled_30)["pct_positive"]
        pct_70 = advantage_distribution_stats(labeled_70)["pct_positive"]
        assert pct_70 < pct_30, "Higher threshold_pct should yield fewer positive labels"


# ---------------------------------------------------------------------------
# Human corrections
# ---------------------------------------------------------------------------

class TestHumanCorrectionOverride:
    def test_correction_always_positive(self, vf):
        traj, _ = make_dummy_trajectory(length=5, obs_dim=OBS_DIM, success=False)
        # Mark every step as a correction
        for step in traj:
            step["is_correction"] = True
        labeled = label_trajectories([(traj, False)], vf, rf, threshold_pct=99.0)
        for step in labeled:
            assert step["advantage_positive"] is True, "All corrections must be labeled positive"

    def test_correction_mixed_with_non_correction(self, vf):
        traj, _ = make_dummy_trajectory(length=6, obs_dim=OBS_DIM, success=False)
        traj[0]["is_correction"] = True  # only first step is correction
        labeled = label_trajectories([(traj, False)], vf, rf, threshold_pct=99.0)
        assert labeled[0]["advantage_positive"] is True, "Step 0 (correction) must be positive"


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------

class TestLabeledDataStructure:
    def test_total_length_matches_all_steps(self, vf, rollouts):
        total = sum(len(t) for t, _ in rollouts)
        labeled = label_trajectories(rollouts, vf, rf)
        assert len(labeled) == total

    def test_original_keys_preserved(self, vf, rollouts):
        labeled = label_trajectories(rollouts, vf, rf)
        for step in labeled:
            assert "obs" in step
            assert "action" in step
            assert "t" in step
            assert "instruction" in step

    def test_new_keys_added(self, vf, rollouts):
        labeled = label_trajectories(rollouts, vf, rf)
        for step in labeled:
            assert "advantage" in step
            assert "return_" in step
            assert "advantage_positive" in step

    def test_advantages_finite(self, vf, rollouts):
        labeled = label_trajectories(rollouts, vf, rf)
        for step in labeled:
            assert np.isfinite(step["advantage"]), "All advantages must be finite"

    def test_assert_invariants_passes(self, vf, rollouts):
        labeled = label_trajectories(rollouts, vf, rf)
        assert_advantage_label_invariants(labeled)

    def test_empty_rollouts_returns_empty(self, vf):
        labeled = label_trajectories([], vf, rf)
        assert labeled == []
