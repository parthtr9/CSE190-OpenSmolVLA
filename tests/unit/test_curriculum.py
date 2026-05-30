"""
Unit tests: CurriculumEnvWrapper and MockCurriculumEnv.

Key properties:
1. Difficulty increases monotonically as set_difficulty(iteration) is called.
2. auto_advance() advances difficulty when success threshold is met.
3. The wrapper transparently passes Gymnasium API to the base env.
4. MockCurriculumEnv adjusts success_prob with distance.
"""

import numpy as np
import pytest

from recap_smolvla.curriculum import (
    CurriculumEnvWrapper,
    MockCurriculumEnv,
    DEFAULT_SCHEDULE,
)
from recap_smolvla.rollout import MockEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_env():
    return MockCurriculumEnv(max_steps=10, seed=0)


@pytest.fixture
def wrapped_env(base_env):
    return CurriculumEnvWrapper(
        base_env,
        schedule=[0.05, 0.15, "random"],
        success_threshold=0.50,
        window=4,
    )


# ---------------------------------------------------------------------------
# DEFAULT_SCHEDULE
# ---------------------------------------------------------------------------

def test_default_schedule_not_empty():
    assert len(DEFAULT_SCHEDULE) > 0


def test_default_schedule_increases_difficulty():
    """Each level should be >= previous or 'random' (hardest)."""
    for i in range(len(DEFAULT_SCHEDULE) - 1):
        curr = DEFAULT_SCHEDULE[i]
        nxt = DEFAULT_SCHEDULE[i + 1]
        if isinstance(curr, float) and isinstance(nxt, float):
            assert nxt >= curr, "Schedule should have non-decreasing distances"


# ---------------------------------------------------------------------------
# CurriculumEnvWrapper: set_difficulty
# ---------------------------------------------------------------------------

class TestSetDifficulty:
    def test_set_difficulty_iter_0(self, wrapped_env):
        wrapped_env.set_difficulty(0)
        assert wrapped_env.current_difficulty() == 0.05

    def test_set_difficulty_iter_1(self, wrapped_env):
        wrapped_env.set_difficulty(1)
        assert wrapped_env.current_difficulty() == 0.15

    def test_set_difficulty_iter_2(self, wrapped_env):
        wrapped_env.set_difficulty(2)
        assert wrapped_env.current_difficulty() == "random"

    def test_set_difficulty_beyond_schedule_clamps(self, wrapped_env):
        wrapped_env.set_difficulty(100)
        assert wrapped_env.current_difficulty() == "random"

    def test_difficulty_progression_is_monotone(self, wrapped_env):
        difficulties = []
        for i in range(3):
            wrapped_env.set_difficulty(i)
            d = wrapped_env.current_difficulty()
            difficulties.append(d)
        # Difficulty should not decrease
        numeric = [d if isinstance(d, float) else float("inf") for d in difficulties]
        for a, b in zip(numeric, numeric[1:]):
            assert b >= a, "Difficulty should not decrease across iterations"


# ---------------------------------------------------------------------------
# CurriculumEnvWrapper: Gymnasium API
# ---------------------------------------------------------------------------

class TestCurriculumEnvAPI:
    def test_reset_returns_obs(self, wrapped_env):
        obs, info = wrapped_env.reset()
        assert obs is not None

    def test_step_returns_5_tuple(self, wrapped_env):
        wrapped_env.reset()
        action = np.zeros(MockEnv.ACTION_DIM, dtype=np.float32)
        result = wrapped_env.step(action)
        assert len(result) == 5

    def test_info_has_curriculum_keys(self, wrapped_env):
        wrapped_env.reset()
        _, _, _, _, info = wrapped_env.step(np.zeros(MockEnv.ACTION_DIM))
        assert "curriculum_advanced" in info
        assert "curriculum_level" in info
        assert "curriculum_iteration" in info

    def test_obs_space_forwarded(self, wrapped_env):
        assert wrapped_env.observation_space is not None

    def test_action_space_forwarded(self, wrapped_env):
        assert wrapped_env.action_space is not None

    def test_render_forwarded(self, wrapped_env):
        wrapped_env.reset()
        img = wrapped_env.render()
        assert isinstance(img, np.ndarray)

    def test_close_does_not_raise(self, wrapped_env):
        wrapped_env.reset()
        wrapped_env.close()


# ---------------------------------------------------------------------------
# auto_advance
# ---------------------------------------------------------------------------

class TestAutoAdvance:
    def test_auto_advance_on_repeated_success(self, base_env):
        env = CurriculumEnvWrapper(
            base_env,
            schedule=[0.05, 0.15, "random"],
            success_threshold=0.50,
            window=4,
        )
        env.set_difficulty(0)
        initial = env.current_difficulty()
        # Feed 4 successes — should cross 50% threshold in window=4
        for _ in range(4):
            advanced = env.auto_advance(True)
        # After enough successes, difficulty should advance
        final = env.current_difficulty()
        # Either difficulty advanced or we're already at max
        assert final != initial or env._iteration >= len(env.schedule) - 1

    def test_auto_advance_on_repeated_failure(self, base_env):
        env = CurriculumEnvWrapper(
            base_env,
            schedule=[0.05, 0.15, "random"],
            success_threshold=0.80,
            window=10,
        )
        env.set_difficulty(0)
        initial_iter = env._iteration
        for _ in range(10):
            env.auto_advance(False)
        assert env._iteration == initial_iter, "Should not advance on failures"

    def test_auto_advance_returns_bool(self, base_env):
        env = CurriculumEnvWrapper(base_env, schedule=[0.05, "random"])
        result = env.auto_advance(True)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# MockCurriculumEnv
# ---------------------------------------------------------------------------

class TestMockCurriculumEnv:
    def test_set_easy_distance_increases_success_prob(self):
        env = MockCurriculumEnv(seed=0)
        env.set_initial_block_distance(0.05)
        easy_prob = env._base.success_prob
        env.set_initial_block_distance(0.25)
        hard_prob = env._base.success_prob
        assert easy_prob > hard_prob, "Closer distance should give higher success probability"

    def test_set_random_gives_low_prob(self):
        env = MockCurriculumEnv(seed=0)
        env.set_initial_block_distance("random")
        assert env._base.success_prob <= 0.3

    def test_reset_after_difficulty_change(self):
        env = MockCurriculumEnv(seed=0)
        env.set_initial_block_distance(0.05)
        obs, info = env.reset()
        assert obs is not None

    def test_step_after_difficulty_change(self):
        env = MockCurriculumEnv(seed=0)
        env.set_initial_block_distance(0.05)
        env.reset()
        obs, reward, term, trunc, info = env.step(np.zeros(MockEnv.ACTION_DIM))
        assert obs is not None
