"""
Setup tests: environment creation and basic Gymnasium API contract.

Tests here verify that MockEnv and (optionally) gym_pusht satisfy the
interface expected by collect_rollout and the RECAP training loop.
"""

import numpy as np
import pytest

from recap_smolvla.rollout import MockEnv


# ---------------------------------------------------------------------------
# MockEnv API
# ---------------------------------------------------------------------------

class TestMockEnvCreation:
    def test_instantiates(self):
        env = MockEnv()
        assert env is not None

    def test_obs_space_shape(self):
        env = MockEnv()
        assert env.observation_space.shape == (MockEnv.OBS_DIM,)

    def test_action_space_shape(self):
        env = MockEnv()
        assert env.action_space.shape == (MockEnv.ACTION_DIM,)

    def test_reset_returns_obs_and_info(self):
        env = MockEnv(seed=0)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray), "reset() obs should be ndarray"
        assert obs.shape == (MockEnv.OBS_DIM,)
        assert isinstance(info, dict)

    def test_reset_is_deterministic_with_same_seed(self):
        obs_a, _ = MockEnv(seed=5).reset()
        obs_b, _ = MockEnv(seed=5).reset()
        np.testing.assert_array_equal(obs_a, obs_b)

    def test_reset_differs_with_different_seeds(self):
        obs_a, _ = MockEnv(seed=1).reset()
        obs_b, _ = MockEnv(seed=2).reset()
        assert not np.allclose(obs_a, obs_b), "Different seeds should produce different initial obs"

    def test_step_returns_5_tuple(self):
        env = MockEnv(seed=0)
        env.reset()
        action = np.zeros(MockEnv.ACTION_DIM, dtype=np.float32)
        result = env.step(action)
        assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"

    def test_step_obs_shape(self):
        env = MockEnv(seed=0)
        env.reset()
        obs, *_ = env.step(np.zeros(MockEnv.ACTION_DIM))
        assert obs.shape == (MockEnv.OBS_DIM,)

    def test_step_reward_is_float(self):
        env = MockEnv(seed=0)
        env.reset()
        _, reward, *_ = env.step(np.zeros(MockEnv.ACTION_DIM))
        assert isinstance(reward, float)

    def test_step_terminated_or_truncated_is_bool(self):
        env = MockEnv(seed=0)
        env.reset()
        _, _, terminated, truncated, _ = env.step(np.zeros(MockEnv.ACTION_DIM))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_info_has_is_success(self):
        env = MockEnv(seed=0)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(MockEnv.ACTION_DIM))
        assert "is_success" in info

    def test_episode_terminates_within_max_steps(self):
        env = MockEnv(max_steps=8, success_prob=0.0, seed=0)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            _, _, terminated, truncated, _ = env.step(np.zeros(MockEnv.ACTION_DIM))
            done = terminated or truncated
            steps += 1
        assert done, "Episode should terminate within max_steps"
        assert steps <= 8

    def test_render_returns_array(self):
        env = MockEnv(seed=0)
        env.reset()
        img = env.render()
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3  # H × W × C

    def test_close_does_not_raise(self):
        env = MockEnv(seed=0)
        env.reset()
        env.close()  # should be a no-op, not raise

    def test_obs_values_in_range(self):
        """Observations should be in [0, 1] for the mock env."""
        env = MockEnv(seed=0)
        obs, _ = env.reset()
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_action_clipping(self):
        """Actions outside [-0.1, 0.1] should be silently clipped."""
        env = MockEnv(seed=0)
        env.reset()
        large_action = np.array([999.0, -999.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(large_action)
        assert obs.shape == (MockEnv.OBS_DIM,), "Should not crash on out-of-range action"


# ---------------------------------------------------------------------------
# gym_pusht (integration — skip if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPushTEnvCreation:
    @pytest.fixture(autouse=True)
    def require_pusht(self):
        pytest.importorskip("gym_pusht", reason="gym_pusht not installed")
        pytest.importorskip("gymnasium", reason="gymnasium not installed")

    def test_pusht_make(self):
        import gymnasium as gym
        import gym_pusht  # noqa: F401

        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
        assert env is not None

    def test_pusht_reset(self):
        import gymnasium as gym
        import gym_pusht  # noqa: F401

        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
        obs, info = env.reset(seed=0)
        assert obs is not None
        env.close()

    def test_pusht_step(self):
        import gymnasium as gym
        import gym_pusht  # noqa: F401

        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        env.close()
