"""Gym-compatible wrapper around MuJoCoFollower for RECAP rollout collection.

MuJoCoGymWrapper presents the SO-101 cube-bin task as a standard gymnasium
environment so it can be passed directly to collect_rollout and
recap_training_iteration without modification.

Key design choices
------------------
- reset() returns a flat (6,) state array (joint positions) — what the
  value function trains on.
- render() returns {"front_camera": img, "wrist_camera": img} — both cameras
  that the Gueso/hf_smolvla_recordpolicy0 checkpoint expects.
- step() caches the last observation so render() does not trigger a second
  renderer pass in the same control step.
- Cube randomization cycles through MuJoCoFollower.CUBE_POSITIONS using the
  episode counter (matching Gustavo's eval_sim.py setup).
- check_success is ported directly from eval_sim.py (3 cm horizontal,
  0–5 cm vertical threshold via site positions).
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from smolvla_recap.env.mujoco_follower import MuJoCoFollower, JOINT_NAMES


# ---------------------------------------------------------------------------
# Success detection (ported from eval_sim.py)
# ---------------------------------------------------------------------------

def _check_success(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Return True when the cube is inside the bin."""
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
    bin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bin_site")

    cube_pos = data.site_xpos[cube_id]
    bin_pos = data.site_xpos[bin_id]

    dx = abs(cube_pos[0] - bin_pos[0])
    dy = abs(cube_pos[1] - bin_pos[1])
    dz = cube_pos[2] - bin_pos[2]

    return dx < 0.03 and dy < 0.03 and 0.0 < dz < 0.05


def _cube_bin_distance(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Horizontal L2 distance between cube and bin sites (in metres)."""
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
    bin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bin_site")
    diff = data.site_xpos[cube_id][:2] - data.site_xpos[bin_id][:2]
    return float(np.linalg.norm(diff))


# Typical maximum horizontal distance across the table (for normalization)
_MAX_DIST = 0.30


# ---------------------------------------------------------------------------
# Gymnasium-compatible wrapper
# ---------------------------------------------------------------------------

class MuJoCoGymWrapper:
    """Wrap MuJoCoFollower as a gymnasium-compatible env for RECAP.

    obs  = flat joint-state vector, shape (6,), normalized to [-100, 100]
           (same scale as follower.get_observation()["state"]).
    act  = flat joint-command vector, shape (6,), same normalization.
    render() returns {"front_camera": H×W×3, "wrist_camera": H×W×3}.

    Parameters
    ----------
    randomize_cube:
        Cycle through the three training cube positions each episode.
    max_steps:
        Episode length cap (Gustavo's setting: 750 = 25 s at 30 Hz).
    reward_mode:
        "sparse"  – -1/step; 0 on success.
        "dense"   – -1/step + alpha*(1 - dist/max_dist); +1 on success.
    alpha:
        Weight of the proximity bonus in dense mode.
    headless:
        Open MuJoCo viewer window if False (requires a display).
    """

    OBS_DIM = 6
    ACTION_DIM = 6

    def __init__(
        self,
        randomize_cube: bool = True,
        max_steps: int = 750,
        reward_mode: str = "sparse",
        alpha: float = 0.1,
        headless: bool = True,
    ) -> None:
        self._randomize_cube = randomize_cube
        self._max_steps = max_steps
        self._reward_mode = reward_mode
        self._alpha = alpha

        self._follower = MuJoCoFollower()
        self._follower.connect(viewer=not headless)

        self._step_count = 0
        self._ep_count = 0
        self._last_full_obs: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        self._step_count = 0
        self._follower.reset(
            keyframe="ready",
            randomize_cube=self._randomize_cube,
            cube_position_idx=self._ep_count if self._randomize_cube else None,
        )
        self._ep_count += 1
        full_obs = self._follower.get_observation()
        self._last_full_obs = full_obs
        return full_obs["state"].copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        # Convert flat array → follower action dict
        action_dict = {
            f"{name}.pos": float(action[i])
            for i, name in enumerate(JOINT_NAMES)
        }
        self._follower.send_action(action_dict)
        self._step_count += 1

        full_obs = self._follower.get_observation()
        self._last_full_obs = full_obs

        success = _check_success(self._follower.model, self._follower.data)
        terminated = success
        truncated = self._step_count >= self._max_steps

        reward = self._compute_reward(success)

        return full_obs["state"].copy(), reward, terminated, truncated, {
            "is_success": success,
        }

    def render(self) -> dict[str, np.ndarray]:
        """Return both camera images for the current sim state.

        Returns {"front_camera": (H,W,3) uint8, "wrist_camera": (H,W,3) uint8}.
        Uses the cached observation from the last step/reset to avoid
        re-rendering.
        """
        if self._last_full_obs is None:
            full_obs = self._follower.get_observation()
            self._last_full_obs = full_obs
        return {
            "front_camera": self._last_full_obs["front_camera"],
            "wrist_camera": self._last_full_obs["wrist_camera"],
        }

    def close(self) -> None:
        self._follower.disconnect()
        self._last_full_obs = None

    # ------------------------------------------------------------------
    # Spaces (duck-typed to match gymnasium interface)
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> Any:
        class _Space:
            shape = (MuJoCoGymWrapper.OBS_DIM,)
            dtype = np.float32
        return _Space()

    @property
    def action_space(self) -> Any:
        class _Space:
            shape = (MuJoCoGymWrapper.ACTION_DIM,)
            dtype = np.float32

            def sample(self) -> np.ndarray:
                # Uniform in [-10, 10] — small random commands near zero
                return np.random.uniform(-10.0, 10.0, MuJoCoGymWrapper.ACTION_DIM).astype(
                    np.float32
                )
        return _Space()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, success: bool) -> float:
        if self._reward_mode == "dense":
            dist = _cube_bin_distance(self._follower.model, self._follower.data)
            proximity = self._alpha * (1.0 - min(dist, _MAX_DIST) / _MAX_DIST)
            base = 1.0 if success else -1.0 / self._max_steps
            return base + proximity
        # sparse (default)
        return 0.0 if success else -1.0
