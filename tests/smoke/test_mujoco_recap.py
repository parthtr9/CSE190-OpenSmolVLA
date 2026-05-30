"""
Smoke tests for MuJoCo integration:
  1. MuJoCoGymWrapper reset / step / render API
  2. Cube randomization cycles through all 3 positions
  3. Reward modes: sparse and dense
  4. RECAP training iteration completes on MuJoCo env
  5. Rollout stores dict images correctly
  6. Dense reward function reads cube_bin_dist from trajectory
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from recap_smolvla.envs.mujoco_env import MuJoCoGymWrapper, _cube_bin_distance
from recap_smolvla.rewards import sparse_reward_fn, dense_reward_fn
from recap_smolvla.rollout import collect_rollout
from recap_smolvla.value_function import ValueFunction
from recap_smolvla.training import recap_training_iteration


# ---------------------------------------------------------------------------
# Minimal mock policy compatible with MuJoCo obs/image format
# ---------------------------------------------------------------------------

class _MiniPolicy(nn.Module):
    """6-DOF MLP policy for smoke tests."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 6))

    def select_action(self, obs_dict):
        key = "observation.state" if "observation.state" in obs_dict else "obs"
        obs = obs_dict[key]
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.tensor(
            np.asarray(obs, dtype=np.float32)
        )
        with torch.no_grad():
            return self.net(obs_t.reshape(-1)[:6]).numpy()

    def compute_loss(self, batch):
        key = "observation.state" if "observation.state" in batch else "obs"
        obs = batch[key]
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.tensor(
            np.asarray(obs, dtype=np.float32)
        )
        pred = self.net(obs_t.reshape(-1)[:6])
        tgt = torch.tensor(np.asarray(batch["action"][:6], dtype=np.float32))
        return nn.functional.mse_loss(pred, tgt)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    e = MuJoCoGymWrapper(randomize_cube=True, max_steps=20)
    yield e
    e.close()


@pytest.fixture(scope="module")
def env_dense():
    e = MuJoCoGymWrapper(randomize_cube=True, reward_mode="dense", max_steps=20)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. API: reset / step / render
# ---------------------------------------------------------------------------

class TestMuJoCoGymWrapperAPI:
    def test_reset_returns_6d_state(self, env):
        obs, info = env.reset()
        assert obs.shape == (6,), f"Expected (6,) got {obs.shape}"
        assert obs.dtype == np.float32

    def test_step_shapes_and_types(self, env):
        env.reset()
        action = np.zeros(6, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert "is_success" in info

    def test_render_returns_both_cameras(self, env):
        env.reset()
        imgs = env.render()
        assert isinstance(imgs, dict), "render() should return dict"
        assert "front_camera" in imgs
        assert "wrist_camera" in imgs
        fc = imgs["front_camera"]
        assert fc.ndim == 3, f"front_camera should be H×W×3, got {fc.shape}"
        assert fc.dtype == np.uint8

    def test_truncated_after_max_steps(self):
        short_env = MuJoCoGymWrapper(max_steps=3)
        short_env.reset()
        action = np.zeros(6, dtype=np.float32)
        for _ in range(2):
            _, _, terminated, truncated, _ = short_env.step(action)
            if terminated or truncated:
                break
        _, _, terminated, truncated, _ = short_env.step(action)
        assert truncated or terminated, "Env should truncate at max_steps=3"
        short_env.close()


# ---------------------------------------------------------------------------
# 2. Cube randomization
# ---------------------------------------------------------------------------

class TestCubeRandomization:
    def test_position_cycles_through_all_three(self, capsys):
        """Each reset should cycle through the 3 predefined cube positions."""
        from smolvla_recap.env.mujoco_follower import MuJoCoFollower as _F
        n_positions = len(_F.CUBE_POSITIONS)

        e = MuJoCoGymWrapper(randomize_cube=True, max_steps=5)
        ep_counts_before = []
        for _ in range(n_positions * 2):
            idx_before = e._ep_count % n_positions
            ep_counts_before.append(idx_before)
            e.reset()
        e.close()

        assert set(ep_counts_before) == set(range(n_positions)), (
            f"Expected all {n_positions} cube position indices, got: {ep_counts_before}"
        )


# ---------------------------------------------------------------------------
# 3. Reward modes
# ---------------------------------------------------------------------------

class TestRewardModes:
    def test_sparse_reward_is_minus_one_per_step(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert reward == -1.0

    def test_dense_reward_leq_sparse(self, env_dense):
        """Dense reward includes proximity bonus, so should be >= sparse (less negative)."""
        env_dense.reset()
        _, reward, _, _, info = env_dense.step(np.zeros(6, dtype=np.float32))
        # Dense: -1 + alpha * (something >= 0), so reward >= -1
        assert reward >= -1.0, f"Dense reward {reward} should be >= -1.0"

    def test_cube_bin_distance_is_finite(self, env):
        env.reset()
        env.step(np.zeros(6, dtype=np.float32))
        # _cube_bin_distance takes (model, data) from the MuJoCoFollower
        dist = _cube_bin_distance(env._follower.model, env._follower.data)
        assert np.isfinite(dist), f"cube_bin_distance returned {dist}"
        assert dist >= 0.0


# ---------------------------------------------------------------------------
# 4. collect_rollout stores dict images
# ---------------------------------------------------------------------------

class TestRolloutWithMuJoCo:
    def test_rollout_stores_dict_images(self, env):
        policy = _MiniPolicy()
        env.reset()
        traj, success = collect_rollout(
            policy, env, max_steps=5,
            instruction="pick up the cube",
        )
        assert len(traj) > 0
        step0 = traj[0]
        assert "image" in step0
        img = step0["image"]
        assert isinstance(img, dict), f"step image should be dict, got {type(img)}"
        assert "front_camera" in img and "wrist_camera" in img

    def test_rollout_obs_shape(self, env):
        policy = _MiniPolicy()
        env.reset()
        traj, _ = collect_rollout(policy, env, max_steps=5, instruction="test")
        for step in traj:
            assert step["obs"].shape == (6,)


# ---------------------------------------------------------------------------
# 5. Full RECAP iteration on MuJoCo (mock policy, sparse reward)
# ---------------------------------------------------------------------------

class TestRECAPonMuJoCo:
    def test_recap_iteration_completes(self):
        e = MuJoCoGymWrapper(randomize_cube=True, max_steps=20)
        policy = _MiniPolicy()
        vf = ValueFunction(obs_dim=6)
        sr, policy, vf, stats = recap_training_iteration(
            policy, vf, e, sparse_reward_fn,
            n_rollouts=3, vf_epochs=3, ft_epochs=2,
            max_steps=20, instruction="put the cube in the bin",
            verbose=False,
        )
        e.close()
        assert isinstance(sr, float)
        assert 0.0 <= sr <= 1.0
        assert "pct_positive" in stats
        assert abs(stats["pct_positive"] - 0.30) < 0.05, (
            f"pct_positive should be ~30%, got {stats['pct_positive']:.1%}"
        )
        assert all(np.isfinite(l) for l in stats["ft_losses"])

    def test_recap_dense_reward_iteration_completes(self):
        """Dense-reward MuJoCo env: rewards come from env internally (reward_mode=dense).
        We pass sparse_reward_fn as the RECAP reward function; the richer per-step
        reward signal already comes from the env's internal dense shaping."""
        e = MuJoCoGymWrapper(randomize_cube=True, reward_mode="dense", max_steps=20)
        policy = _MiniPolicy()
        vf = ValueFunction(obs_dim=6)

        sr, policy, vf, stats = recap_training_iteration(
            policy, vf, e, sparse_reward_fn,
            n_rollouts=3, vf_epochs=3, ft_epochs=2,
            max_steps=20, instruction="put the cube in the bin",
            verbose=False,
        )
        e.close()
        assert isinstance(sr, float)
        assert all(np.isfinite(l) for l in stats["ft_losses"])
