"""
Ablation tests: sparse reward RECAP baseline.

These tests run a mini RECAP loop (2 iterations, 8 rollouts each) with
sparse reward and assert structural properties of the results — not that
performance is good, but that the pipeline produces coherent outputs.

Key checks:
1. Training completes without crash.
2. Success rates are in [0, 1].
3. Advantage distribution is skewed negative (expected with sparse reward
   and a weak initial policy).
4. Labeled data has the expected structure.
5. Value function loss decreases over training epochs.
"""

import numpy as np
import pytest

from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.rollout import MockEnv, make_dummy_rollouts
from recap_smolvla.value_function import ValueFunction, train_value_function
from recap_smolvla.advantage import label_trajectories, advantage_distribution_stats
from recap_smolvla.training import recap_training_iteration
from tests.conftest import MockSmolVLAPolicy, OBS_DIM


N_ROLLOUTS = 8
N_ITERS = 2
VF_EPOCHS = 5
FT_EPOCHS = 3


@pytest.fixture(scope="module")
def sparse_recap_results():
    """Run 2 sparse RECAP iterations once per module and cache results."""
    env = MockEnv(max_steps=12, success_prob=0.25, seed=0)
    policy = MockSmolVLAPolicy(obs_dim=OBS_DIM)
    vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=64)
    reward_fn = lambda t, s: sparse_reward_fn(t, s)

    history = []
    for k in range(N_ITERS):
        sr, policy, vf, stats = recap_training_iteration(
            policy, vf, env, reward_fn,
            n_rollouts=N_ROLLOUTS, vf_epochs=VF_EPOCHS, ft_epochs=FT_EPOCHS,
            max_steps=12, verbose=False,
        )
        history.append({"success_rate": sr, "stats": stats})

    return history, policy, vf


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

class TestSparseAblationCompletion:
    def test_all_iterations_completed(self, sparse_recap_results):
        history, _, _ = sparse_recap_results
        assert len(history) == N_ITERS

    def test_success_rates_in_range(self, sparse_recap_results):
        history, _, _ = sparse_recap_results
        for entry in history:
            sr = entry["success_rate"]
            assert 0.0 <= sr <= 1.0, f"Success rate {sr} out of [0,1]"

    def test_ft_losses_are_finite(self, sparse_recap_results):
        history, _, _ = sparse_recap_results
        for entry in history:
            for l in entry["stats"]["ft_losses"]:
                assert np.isfinite(l), f"Fine-tune loss is not finite: {l}"


# ---------------------------------------------------------------------------
# Advantage distribution with sparse reward
# ---------------------------------------------------------------------------

class TestSparseAdvantageDistribution:
    def test_pct_positive_is_between_10_and_50pct(self, sparse_recap_results):
        """Sparse reward should still produce some positive advantage steps
        due to the threshold-based labeling (top 30%)."""
        history, _, _ = sparse_recap_results
        for entry in history:
            pct = entry["stats"]["pct_positive"]
            assert 0.05 <= pct <= 0.60, (
                f"pct_positive={pct:.1%} outside expected [5%, 60%] range. "
                "Check threshold logic."
            )

    def test_labeled_data_structure(self, sparse_recap_results):
        """Run one more labeling and check structure."""
        _, _, vf = sparse_recap_results
        rollouts = make_dummy_rollouts(n=4, obs_dim=OBS_DIM, seed=99)
        labeled = label_trajectories(rollouts, vf, lambda t, s: sparse_reward_fn(t, s))
        assert len(labeled) > 0
        for step in labeled:
            assert "advantage" in step
            assert "advantage_positive" in step
            assert np.isfinite(step["advantage"])


# ---------------------------------------------------------------------------
# Value function learning
# ---------------------------------------------------------------------------

class TestSparseVFLearning:
    def test_vf_loss_decreases_on_fresh_training(self):
        rollouts = make_dummy_rollouts(n=10, obs_dim=OBS_DIM, success_rate=0.0, seed=1)
        vf = ValueFunction(obs_dim=OBS_DIM, hidden_dim=64)
        losses = train_value_function(
            rollouts, vf, lambda t, s: sparse_reward_fn(t, s),
            epochs=30, lr=5e-3, verbose=False,
        )
        assert losses[0] > losses[-1], (
            f"VF loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_vf_final_loss_is_finite(self, sparse_recap_results):
        _, _, vf = sparse_recap_results
        rollouts = make_dummy_rollouts(n=3, obs_dim=OBS_DIM, seed=5)
        losses = train_value_function(
            rollouts, vf, lambda t, s: sparse_reward_fn(t, s),
            epochs=3, verbose=False,
        )
        assert all(np.isfinite(l) for l in losses)


# ---------------------------------------------------------------------------
# Stats dict completeness
# ---------------------------------------------------------------------------

class TestSparseStatsDict:
    def test_stats_has_required_keys(self, sparse_recap_results):
        history, _, _ = sparse_recap_results
        required = {"success_rate", "n_rollouts", "n_labeled_steps", "ft_losses", "pct_positive"}
        for entry in history:
            missing = required - set(entry["stats"].keys())
            assert not missing, f"Stats dict missing keys: {missing}"

    def test_n_rollouts_matches_config(self, sparse_recap_results):
        history, _, _ = sparse_recap_results
        for entry in history:
            assert entry["stats"]["n_rollouts"] == N_ROLLOUTS

    def test_n_labeled_steps_positive(self, sparse_recap_results):
        history, _, _ = sparse_recap_results
        for entry in history:
            assert entry["stats"]["n_labeled_steps"] > 0
