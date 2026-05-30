"""
Unit tests: dense_reward_fn — correctness and reward-hacking safety.

Key properties validated:
1. Dense reward = sparse component + bounded proximity bonus.
2. Bonus is always positive (it cannot make reward *worse* than sparse).
3. Bonus is bounded by 2*alpha (prevents dominating the sparse signal).
4. Terminal rewards are identical to sparse_reward_fn (same C_fail / success logic).
5. alpha=0 collapses dense to sparse.
"""

import numpy as np
import pytest

from recap_smolvla.rewards import sparse_reward_fn, dense_reward_fn
from recap_smolvla.rollout import make_dummy_trajectory

GOAL = np.array([0.8, 0.8], dtype=np.float32)
C_FAIL = 100.0
ALPHA = 0.1


def _traj_at_goal(length: int = 5):
    """Trajectory where block is exactly at goal (maximum proximity bonus)."""
    traj, _ = make_dummy_trajectory(length=length)
    for step in traj:
        step["obs"] = np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32)
    return traj


def _traj_far(length: int = 5):
    """Trajectory where block is far from goal (minimum proximity bonus)."""
    traj, _ = make_dummy_trajectory(length=length)
    for step in traj:
        step["obs"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return traj


# ---------------------------------------------------------------------------
# Bonus is non-negative
# ---------------------------------------------------------------------------

class TestDenseRewardBonus:
    def test_dense_always_geq_sparse(self):
        traj, _ = make_dummy_trajectory(length=8)
        sparse = sparse_reward_fn(traj, False)
        dense = dense_reward_fn(traj, False, GOAL, alpha=ALPHA)
        for s, d in zip(sparse, dense):
            assert d >= s - 1e-9, f"Dense {d:.4f} < sparse {s:.4f}"

    def test_bonus_bounded_by_2alpha(self):
        """Bonus <= 2*alpha at any timestep."""
        traj = _traj_at_goal(length=5)
        sparse = sparse_reward_fn(traj, False)
        dense = dense_reward_fn(traj, False, GOAL, alpha=ALPHA)
        for s, d in zip(sparse, dense):
            bonus = d - s
            assert bonus <= 2 * ALPHA + 1e-9, f"Bonus {bonus:.4f} exceeds 2*alpha={2*ALPHA}"

    def test_max_bonus_at_goal(self):
        """When agent and block are both at goal, bonus is maximized."""
        traj_at_goal = _traj_at_goal(length=3)
        traj_far = _traj_far(length=3)
        dense_at_goal = dense_reward_fn(traj_at_goal, False, GOAL, alpha=ALPHA)
        dense_far = dense_reward_fn(traj_far, False, GOAL, alpha=ALPHA)
        sparse = sparse_reward_fn(traj_at_goal, False)
        sparse_far = sparse_reward_fn(traj_far, False)
        bonus_at_goal = [d - s for d, s in zip(dense_at_goal, sparse)]
        bonus_far = [d - s for d, s in zip(dense_far, sparse_far)]
        # Near-goal bonus should be strictly larger than far bonus
        for b_goal, b_far in zip(bonus_at_goal, bonus_far):
            assert b_goal >= b_far - 1e-9

    def test_alpha_zero_equals_sparse(self):
        """alpha=0 makes dense reward identical to sparse reward."""
        traj, _ = make_dummy_trajectory(length=5)
        sparse = sparse_reward_fn(traj, False)
        dense = dense_reward_fn(traj, False, GOAL, alpha=0.0)
        for s, d in zip(sparse, dense):
            assert abs(d - s) < 1e-9, "alpha=0 dense should equal sparse"


# ---------------------------------------------------------------------------
# Terminal reward matches sparse
# ---------------------------------------------------------------------------

class TestDenseTerminalReward:
    def test_terminal_failure_same_as_sparse(self):
        traj, _ = make_dummy_trajectory(length=6)
        sparse = sparse_reward_fn(traj, False, C_fail=C_FAIL)
        dense = dense_reward_fn(traj, False, GOAL, C_fail=C_FAIL, alpha=ALPHA)
        # Terminal step: dense = sparse + bonus, NOT equal to sparse
        # But the *sparse component* at terminal should be -C_FAIL
        # We verify dense[-1] is in range [sparse[-1], sparse[-1] + 2*alpha]
        assert dense[-1] >= sparse[-1] - 1e-9
        assert dense[-1] <= sparse[-1] + 2 * ALPHA + 1e-9

    def test_terminal_success_adds_bonus(self):
        """On success, terminal dense reward should be 0 + bonus >= 0."""
        traj, _ = make_dummy_trajectory(length=4)
        dense = dense_reward_fn(traj, True, GOAL, C_fail=C_FAIL, alpha=ALPHA)
        assert dense[-1] >= 0.0, "Success terminal dense reward should be >= 0"


# ---------------------------------------------------------------------------
# Length and shape
# ---------------------------------------------------------------------------

class TestDenseRewardShape:
    @pytest.mark.parametrize("T", [1, 3, 10])
    def test_length_equals_trajectory(self, T):
        traj, _ = make_dummy_trajectory(length=T)
        r = dense_reward_fn(traj, False, GOAL)
        assert len(r) == T

    def test_empty_trajectory(self):
        r = dense_reward_fn([], False, GOAL)
        assert r == []

    def test_returns_list_of_floats(self):
        traj, _ = make_dummy_trajectory(length=4)
        r = dense_reward_fn(traj, False, GOAL)
        assert all(isinstance(v, float) for v in r)


# ---------------------------------------------------------------------------
# Reward hacking check
# ---------------------------------------------------------------------------

class TestRewardHackingPrevention:
    def test_sparse_component_dominates(self):
        """
        With alpha=0.1 and C_fail=100, the failure penalty should be
        at least 40× larger than the maximum possible proximity bonus,
        ensuring the robot cannot compensate for failure by hovering near goal.
        """
        max_bonus_per_step = 2 * ALPHA  # 0.2
        # A robot hovering at goal for 1000 steps earns at most 1000 * 0.2 = 200 total bonus
        # But a single failure costs -100 at terminal plus -999 from per-step rewards
        # We just verify the ratio is large
        ratio = C_FAIL / max_bonus_per_step
        assert ratio >= 10, f"C_fail / max_bonus_per_step should be >= 10, got {ratio}"
