"""
Trajectory collection utilities.

collect_rollout    Runs one episode with any policy + Gymnasium-compatible env.
MockEnv            Lightweight NumPy-only environment for testing without gym_pusht.
make_dummy_trajectory  Creates synthetic trajectory data for unit tests.
"""

from __future__ import annotations

import random
from typing import Any, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Protocol: policy interface
# ---------------------------------------------------------------------------

class Policy(Protocol):
    """Minimal interface expected from any policy (real or mock)."""

    def select_action(self, obs_dict: dict[str, Any]) -> np.ndarray:
        """Return an action array given an observation dict."""
        ...


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_rollout(
    policy: Policy,
    env: Any,
    *,
    max_steps: int = 300,
    instruction: str = "push the block to the goal",
    render_key: str = "observation.image",
    state_key: str = "observation.state",
    device: str = "cpu",
) -> tuple[list[dict], bool]:
    """Run one episode and return the trajectory + success flag.

    Parameters
    ----------
    policy:
        Object with a ``.select_action(obs_dict)`` method.
    env:
        Gymnasium-compatible environment.
    max_steps:
        Hard episode length cap.
    instruction:
        Language instruction prepended to the obs dict.
    render_key:
        Key under which the rendered image is stored in obs_dict.
    state_key:
        Key under which the flat observation is stored in obs_dict.
    device:
        Passed to policy (unused for MockPolicy, relevant for real SmolVLA).

    Returns
    -------
    (trajectory, success)
        trajectory is a list of step dicts; success is bool.
    """
    import torch

    obs, _info = env.reset()
    trajectory: list[dict] = []
    done = False
    success = False

    while not done and len(trajectory) < max_steps:
        # Render may return None on headless envs that don't support it
        try:
            image = env.render()
        except Exception:
            image = np.zeros((64, 64, 3), dtype=np.uint8)

        obs_dict: dict[str, Any] = {
            state_key: torch.tensor(
                np.asarray(obs, dtype=np.float32)
            ),
            "instruction": instruction,
        }
        if image is not None:
            if isinstance(image, dict):
                # Multi-camera env (e.g. MuJoCoGymWrapper) — pass dict through;
                # _SmolVLAWrapper._to_batch handles camera-key mapping.
                obs_dict[render_key] = image
            elif isinstance(image, np.ndarray) and image.ndim == 3:
                obs_dict[render_key] = torch.tensor(
                    image.astype(np.float32) / 255.0
                ).permute(2, 0, 1)
            else:
                obs_dict[render_key] = torch.tensor(image)

        action = policy.select_action(obs_dict)
        action_np = action.numpy() if hasattr(action, "numpy") else np.asarray(action)

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        success = bool(info.get("is_success", terminated and reward > 0))

        trajectory.append(
            {
                "t": len(trajectory),
                "obs": np.asarray(obs, dtype=np.float32),
                "action": action_np,
                "image": image,
                "reward": float(reward),
                "success": success,
                "instruction": instruction,
                "is_correction": False,
            }
        )
        obs = next_obs

    return trajectory, success


# ---------------------------------------------------------------------------
# MockEnv — no external dependencies
# ---------------------------------------------------------------------------

class MockEnv:
    """Contact-physics 2-D push environment for testing without gym_pusht.

    Designed to be genuinely hard for a random policy so that the sparse
    vs dense reward ablation produces an interesting advantage distribution.

    Physics
    -------
    - Agent moves by (dx, dy), clamped to [-0.15, 0.15].
    - Block only moves when the agent is within CONTACT_RADIUS (0.18).
      When in contact, the block is pushed in the agent→block direction
      proportional to (CONTACT_RADIUS - dist) * PUSH_STRENGTH.
    - No constant drift — the agent must physically reach and push the block.

    Difficulty
    ----------
    - Random policy baseline SR: ~5–15% over 50 steps (depends on luck).
    - A policy that learns to move toward the block then push toward goal
      can reach >60% SR, making RECAP improvement visible.

    Success
    -------
    Physics-only: dist(block, goal) < SUCCESS_RADIUS (0.12).
    The stochastic success_prob shortcut is kept only for unit tests that
    need a controllable non-zero SR without running real rollouts.
    Set success_prob=0.0 (default) for all ablation experiments.

    State
    -----
    [agent_x, agent_y, block_x, block_y]  — float32 in [0, 1]^4
    """

    OBS_DIM = 4
    ACTION_DIM = 2
    CONTACT_RADIUS = 0.18
    PUSH_STRENGTH = 0.25
    SUCCESS_RADIUS = 0.12
    ACTION_CLIP = 0.15

    def __init__(
        self,
        max_steps: int = 50,
        success_prob: float = 0.0,
        seed: int | None = 42,
    ) -> None:
        self.max_steps = max_steps
        self.success_prob = success_prob          # 0.0 for ablations; small value for unit tests
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._obs: np.ndarray = np.zeros(self.OBS_DIM, dtype=np.float32)
        self.goal_pos = np.array([0.8, 0.8], dtype=np.float32)

    # Gymnasium API
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        # Agent starts in lower-left quadrant; block starts near centre —
        # the agent must cross the arena to reach and push the block.
        self._obs = np.array([
            self._rng.uniform(0.0, 0.35),   # agent x
            self._rng.uniform(0.0, 0.35),   # agent y
            self._rng.uniform(0.35, 0.65),  # block x  (starts near centre)
            self._rng.uniform(0.35, 0.65),  # block y
        ], dtype=np.float32)
        return self._obs.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -self.ACTION_CLIP, self.ACTION_CLIP)

        # Move agent
        self._obs[:2] = np.clip(self._obs[:2] + action, 0.0, 1.0)

        # Contact-based block physics — block only moves when agent touches it
        agent_pos = self._obs[:2]
        block_pos = self._obs[2:4]
        dist_agent_block = float(np.linalg.norm(agent_pos - block_pos))

        if dist_agent_block < self.CONTACT_RADIUS and dist_agent_block > 1e-6:
            # Push direction: agent → block (agent pushes block away from itself)
            push_dir = (block_pos - agent_pos) / dist_agent_block
            push_mag = (self.CONTACT_RADIUS - dist_agent_block) * self.PUSH_STRENGTH
            self._obs[2:4] = np.clip(block_pos + push_dir * push_mag, 0.0, 1.0)

        dist_to_goal = float(np.linalg.norm(self._obs[2:4] - self.goal_pos))

        # Physics-only success; optional stochastic floor for unit tests
        physics_success = dist_to_goal < self.SUCCESS_RADIUS
        random_success = (self.success_prob > 0) and (self._rng.random() < self.success_prob)
        success = bool(physics_success or random_success)

        terminated = success
        truncated = self._step_count >= self.max_steps
        reward = 0.0 if (success and terminated) else -1.0

        return self._obs.copy(), reward, terminated, truncated, {
            "is_success": success,
            "dist_to_goal": dist_to_goal,
            "dist_agent_block": dist_agent_block,
            "in_contact": dist_agent_block < self.CONTACT_RADIUS,
        }

    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        pass

    @property
    def observation_space(self) -> Any:
        class _Space:
            shape = (MockEnv.OBS_DIM,)
            dtype = np.float32
        return _Space()

    @property
    def action_space(self) -> Any:
        class _Space:
            shape = (MockEnv.ACTION_DIM,)
            dtype = np.float32

            def sample(self) -> np.ndarray:
                return np.random.uniform(
                    -MockEnv.ACTION_CLIP, MockEnv.ACTION_CLIP, MockEnv.ACTION_DIM
                ).astype(np.float32)
        return _Space()


# ---------------------------------------------------------------------------
# ScriptedMockPolicy — noisy oracle for ablation experiments
# ---------------------------------------------------------------------------

class ScriptedMockPolicy:
    """Two-phase push controller with tunable noise for ablation experiments.

    Baseline success rate with default settings: ~20–40% depending on seed.
    This gives RECAP something to improve from rather than pure 0% SR.

    Phase 1 — Navigate: move agent toward block until in contact range.
    Phase 2 — Push:     move agent in the direction that pushes block to goal.

    Parameters
    ----------
    noise_std:
        Gaussian noise added to each action.  Higher = harder task.
        0.05 → ~60% SR.  0.12 → ~25% SR.  0.20 → ~10% SR.
    obs_keys:
        Indices into the flat obs vector:
        [agent_x, agent_y, block_x, block_y] by default.
    goal:
        Target position for the block.
    action_clip:
        Maximum action magnitude (should match MockEnv.ACTION_CLIP).
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        noise_std: float = 0.12,
        goal: np.ndarray | None = None,
        action_clip: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.noise_std = noise_std
        self.goal = goal if goal is not None else np.array([0.8, 0.8], dtype=np.float32)
        self.action_clip = action_clip
        self._rng = np.random.default_rng(seed)

    def select_action(self, obs_dict: dict) -> np.ndarray:
        """Compute scripted action from obs + add Gaussian noise."""
        for key in ("obs", "observation.state"):
            if key in obs_dict:
                v = obs_dict[key]
                if hasattr(v, "numpy"):
                    obs = v.numpy().flatten()
                else:
                    obs = np.asarray(v, dtype=np.float32).flatten()
                break
        else:
            obs = np.zeros(4, dtype=np.float32)

        agent = obs[:2]
        block = obs[2:4]

        dist_to_block = float(np.linalg.norm(agent - block))

        if dist_to_block > 0.14:
            # Phase 1: navigate toward block
            direction = (block - agent) / (dist_to_block + 1e-8)
            action = direction * self.action_clip
        else:
            # Phase 2: push block toward goal
            push_dir = self.goal - block
            push_norm = float(np.linalg.norm(push_dir))
            if push_norm > 1e-6:
                push_dir = push_dir / push_norm
            # Position agent behind block relative to goal direction, then push
            action = push_dir * self.action_clip

        # Add noise and clip
        noise = self._rng.normal(0, self.noise_std, size=2).astype(np.float32)
        action = np.clip(action + noise, -self.action_clip, self.action_clip)
        return action

    def compute_loss(self, batch: dict) -> "torch.Tensor":
        """Not used for scripted policy; included for interface compatibility."""
        import torch
        return torch.tensor(0.0, requires_grad=False)


# ---------------------------------------------------------------------------
# Synthetic trajectory factory for tests
# ---------------------------------------------------------------------------

def make_dummy_trajectory(
    length: int = 10,
    obs_dim: int = 4,
    action_dim: int = 2,
    success: bool = False,
    *,
    instruction: str = "push block to goal",
    rng: np.random.Generator | None = None,
) -> tuple[list[dict], bool]:
    """Create a synthetic trajectory for unit tests.

    All observations, actions, and images are random float arrays.
    No gym or torch dependency required.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    trajectory = []
    for t in range(length):
        step = {
            "t": t,
            "obs": rng.random(obs_dim).astype(np.float32),
            "action": rng.random(action_dim).astype(np.float32) * 0.2 - 0.1,
            "image": np.zeros((64, 64, 3), dtype=np.uint8),
            "reward": 0.0 if (t == length - 1 and success) else -1.0,
            "success": success and (t == length - 1),
            "instruction": instruction,
            "is_correction": False,
        }
        trajectory.append(step)
    return trajectory, success


def make_dummy_rollouts(
    n: int = 5,
    length: int = 10,
    obs_dim: int = 4,
    action_dim: int = 2,
    success_rate: float = 0.4,
    seed: int = 0,
) -> list[tuple[list[dict], bool]]:
    """Create multiple synthetic rollouts for testing.

    Approximately ``success_rate`` fraction of rollouts have success=True.
    """
    rng = np.random.default_rng(seed)
    rollouts = []
    for i in range(n):
        success = bool(rng.random() < success_rate)
        traj, suc = make_dummy_trajectory(
            length=length,
            obs_dim=obs_dim,
            action_dim=action_dim,
            success=success,
            rng=rng,
        )
        rollouts.append((traj, suc))
    return rollouts
