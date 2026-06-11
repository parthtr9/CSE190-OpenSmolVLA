"""
Unit tests: sparse_reward_fn — deterministic correctness.

Every test uses a fixed, hand-crafted trajectory so the expected output
can be computed analytically without any model inference.
"""

import pytest
from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.rollout import make_dummy_trajectory

C_FAIL = 100.0


def _traj(length: int):
    t, _ = make_dummy_trajectory(length=length)
    return t


# ---------------------------------------------------------------------------
# Terminal reward correctness
# ---------------------------------------------------------------------------

class TestSparseRewardTerminal:
    def test_failure_terminal_reward(self):
        r = sparse_reward_fn(_traj(5), False, C_fail=C_FAIL)
        assert r[-1] == -C_FAIL

    def test_success_terminal_reward(self):
        r = sparse_reward_fn(_traj(5), True, C_fail=C_FAIL)
        assert r[-1] == 0.0

    def test_custom_c_fail(self):
        r = sparse_reward_fn(_traj(3), False, C_fail=50.0)
        assert r[-1] == -50.0

    def test_single_step_failure(self):
        r = sparse_reward_fn(_traj(1), False)
        assert len(r) == 1
        assert r[0] == -C_FAIL

    def test_single_step_success(self):
        r = sparse_reward_fn(_traj(1), True)
        assert r[0] == 0.0


# ---------------------------------------------------------------------------
# Per-step -1 reward
# ---------------------------------------------------------------------------

class TestSparseRewardPerStep:
    def test_non_terminal_steps_are_minus_one(self):
        T = 6
        r = sparse_reward_fn(_traj(T), False)
        for i in range(T - 1):
            assert r[i] == -1.0, f"Step {i} should be -1.0, got {r[i]}"

    def test_non_terminal_steps_success_episode(self):
        T = 4
        r = sparse_reward_fn(_traj(T), True)
        for i in range(T - 1):
            assert r[i] == -1.0


# ---------------------------------------------------------------------------
# Length invariant
# ---------------------------------------------------------------------------

class TestSparseRewardLength:
    @pytest.mark.parametrize("T", [1, 2, 5, 10, 50, 100])
    def test_length_equals_trajectory_length(self, T):
        r = sparse_reward_fn(_traj(T), False)
        assert len(r) == T

    def test_empty_trajectory(self):
        r = sparse_reward_fn([], False)
        assert r == []


# ---------------------------------------------------------------------------
# Sum of rewards (analytical)
# ---------------------------------------------------------------------------

class TestSparseRewardSum:
    def test_failure_sum(self):
        T = 5
        r = sparse_reward_fn(_traj(T), False)
        # (T-1) steps of -1 plus one terminal of -C_FAIL
        expected = -(T - 1) - C_FAIL
        assert abs(sum(r) - expected) < 1e-8

    def test_success_sum(self):
        T = 5
        r = sparse_reward_fn(_traj(T), True)
        # (T-1) steps of -1 plus terminal 0
        expected = -(T - 1)
        assert abs(sum(r) - expected) < 1e-8
