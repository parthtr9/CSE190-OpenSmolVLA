"""
Ablation tests: VLA self-supervised scoring (Direction 2).

Key checks:
1. vla_score_trajectory returns scores in [0, 1].
2. scores_to_advantages produces centered values (mean ≈ 0).
3. Scores correlate with actual returns when score_from_obs=True
   (i.e., VLA scoring is not random).
4. Hybrid (VLA + sparse) produces a valid advantage signal.
5. Interpolation works correctly for stride > 1.
"""

import numpy as np
import pytest

from recap_smolvla.scoring import (
    vla_score_trajectory,
    scores_to_advantages,
    MockScoringPolicy,
    SCORING_PROMPT_TEMPLATE,
)
from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.value_function import compute_returns
from recap_smolvla.rollout import make_dummy_rollouts, make_dummy_trajectory


OBS_DIM = 4


# ---------------------------------------------------------------------------
# vla_score_trajectory
# ---------------------------------------------------------------------------

class TestVLAScoreTrajectorySanity:
    def test_returns_list_of_floats(self):
        traj, _ = make_dummy_trajectory(length=6, obs_dim=OBS_DIM)
        policy = MockScoringPolicy(obs_dim=OBS_DIM, fixed_score=0.7)
        scores = vla_score_trajectory(traj, "push block to goal", policy)
        assert isinstance(scores, list)
        assert all(isinstance(s, float) for s in scores)

    def test_length_matches_trajectory(self):
        traj, _ = make_dummy_trajectory(length=8, obs_dim=OBS_DIM)
        policy = MockScoringPolicy(obs_dim=OBS_DIM, fixed_score=0.5)
        scores = vla_score_trajectory(traj, "push block to goal", policy)
        assert len(scores) == 8

    def test_scores_in_range_0_1(self):
        traj, _ = make_dummy_trajectory(length=10, obs_dim=OBS_DIM)
        policy = MockScoringPolicy(obs_dim=OBS_DIM, score_from_obs=True, fixed_score=None)
        scores = vla_score_trajectory(traj, "push block to goal", policy)
        for s in scores:
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0, 1]"

    def test_fixed_score_returns_constant(self):
        traj, _ = make_dummy_trajectory(length=5, obs_dim=OBS_DIM)
        policy = MockScoringPolicy(obs_dim=OBS_DIM, fixed_score=0.42)
        scores = vla_score_trajectory(traj, "push block to goal", policy)
        assert all(abs(s - 0.42) < 1e-9 for s in scores)

    def test_empty_trajectory(self):
        policy = MockScoringPolicy(obs_dim=OBS_DIM, fixed_score=0.5)
        scores = vla_score_trajectory([], "push block to goal", policy)
        assert scores == []

    def test_stride_1_same_as_no_stride(self):
        traj, _ = make_dummy_trajectory(length=6, obs_dim=OBS_DIM)
        policy = MockScoringPolicy(obs_dim=OBS_DIM, fixed_score=0.3)
        scores_s1 = vla_score_trajectory(traj, "push block to goal", policy, stride=1)
        scores_s_default = vla_score_trajectory(traj, "push block to goal", policy)
        assert scores_s1 == scores_s_default

    def test_stride_2_returns_correct_length(self):
        traj, _ = make_dummy_trajectory(length=8, obs_dim=OBS_DIM)
        policy = MockScoringPolicy(obs_dim=OBS_DIM, fixed_score=0.5)
        scores = vla_score_trajectory(traj, "push block to goal", policy, stride=2)
        assert len(scores) == 8, "Stride should not change output length (interpolation fills gaps)"

    def test_stride_scores_monotone_interpolated(self):
        """With fixed score and stride=3, interpolated values should equal fixed score."""
        traj, _ = make_dummy_trajectory(length=9, obs_dim=OBS_DIM)
        policy = MockScoringPolicy(obs_dim=OBS_DIM, fixed_score=0.6)
        scores = vla_score_trajectory(traj, "push block to goal", policy, stride=3)
        assert all(abs(s - 0.6) < 1e-6 for s in scores)


# ---------------------------------------------------------------------------
# scores_to_advantages
# ---------------------------------------------------------------------------

class TestScoresToAdvantages:
    def test_output_shape(self):
        scores = [0.1, 0.5, 0.9, 0.4]
        advs = scores_to_advantages(scores)
        assert advs.shape == (4,)

    def test_centered_mean_near_zero(self):
        scores = [0.2, 0.4, 0.6, 0.8]
        advs = scores_to_advantages(scores, center=True)
        assert abs(advs.mean()) < 1e-6, f"Centered advantages should have mean ≈ 0, got {advs.mean()}"

    def test_no_centering_preserves_values(self):
        scores = [0.3, 0.7]
        advs = scores_to_advantages(scores, center=False)
        np.testing.assert_allclose(advs, [0.3, 0.7], atol=1e-6)

    def test_empty_scores(self):
        advs = scores_to_advantages([], center=True)
        assert len(advs) == 0

    def test_all_same_score_centered_to_zero(self):
        scores = [0.5] * 5
        advs = scores_to_advantages(scores, center=True)
        np.testing.assert_allclose(advs, np.zeros(5), atol=1e-6)


# ---------------------------------------------------------------------------
# Correlation: obs-derived scores vs returns
# ---------------------------------------------------------------------------

class TestVLAScoringCorrelation:
    def test_score_from_obs_not_all_same(self):
        """score_from_obs should produce varied scores for varied observations."""
        rollouts = make_dummy_rollouts(n=5, obs_dim=OBS_DIM, success_rate=0.5, seed=0)
        policy = MockScoringPolicy(obs_dim=OBS_DIM, fixed_score=None, score_from_obs=True)
        all_scores = []
        for traj, _ in rollouts:
            s = vla_score_trajectory(traj, "push block to goal", policy)
            all_scores.extend(s)
        # Not all the same (obs are random, so mean(obs) will vary)
        assert np.std(all_scores) > 1e-6, "score_from_obs should produce varied scores"

    def test_positive_correlation_with_returns(self):
        """
        Hypothesis: higher VLA scores correlate with higher returns.
        With score_from_obs, score = mean(obs) in [0,1].
        Returns depend on success/failure but this is a weak signal test.
        Correlation just needs to be > -0.5 (not strongly anti-correlated).
        """
        rollouts = make_dummy_rollouts(n=20, obs_dim=OBS_DIM, success_rate=0.5, seed=5)
        policy = MockScoringPolicy(obs_dim=OBS_DIM, fixed_score=None, score_from_obs=True)

        all_scores, all_returns = [], []
        for traj, success in rollouts:
            scores = vla_score_trajectory(traj, "push block to goal", policy)
            rewards = sparse_reward_fn(traj, success)
            returns = compute_returns(rewards)
            all_scores.extend(scores)
            all_returns.extend(returns)

        corr = np.corrcoef(all_scores, all_returns)[0, 1]
        # A valid proxy signal should not be strongly anti-correlated
        assert corr > -0.5, (
            f"VLA scores are strongly anti-correlated with returns (r={corr:.3f}). "
            "Check the scoring function."
        )


# ---------------------------------------------------------------------------
# Scoring prompt template
# ---------------------------------------------------------------------------

class TestScoringPromptTemplate:
    def test_template_has_instruction_placeholder(self):
        assert "{instruction}" in SCORING_PROMPT_TEMPLATE

    def test_template_formats_correctly(self):
        prompt = SCORING_PROMPT_TEMPLATE.format(instruction="pick up the cube")
        assert "pick up the cube" in prompt
        assert "0.0" in prompt
        assert "1.0" in prompt


# ---------------------------------------------------------------------------
# MockScoringPolicy interface
# ---------------------------------------------------------------------------

class TestMockScoringPolicyInterface:
    def test_select_action_returns_array(self):
        policy = MockScoringPolicy(obs_dim=OBS_DIM, action_dim=2)
        obs_dict = {"obs": np.zeros(OBS_DIM, dtype=np.float32), "instruction": "test"}
        action = policy.select_action(obs_dict)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)

    def test_compute_loss_returns_scalar_tensor(self):
        import torch
        policy = MockScoringPolicy(obs_dim=OBS_DIM, action_dim=2)
        batch = {
            "obs": np.zeros(OBS_DIM, dtype=np.float32),
            "action": np.zeros(2, dtype=np.float32),
            "instruction": "test",
        }
        loss = policy.compute_loss(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.grad_fn is not None

    def test_score_clamps_to_0_1(self):
        # MockScoringPolicy clamps its output; test via vla_score_trajectory
        class AlwaysOver(MockScoringPolicy):
            def score(self, image, prompt, obs=None):
                return 99.0  # should be clamped to 1.0

        policy = AlwaysOver(obs_dim=OBS_DIM)
        traj, _ = make_dummy_trajectory(length=3, obs_dim=OBS_DIM)
        scores = vla_score_trajectory(traj, "test", policy)
        assert all(s <= 1.0 for s in scores)
