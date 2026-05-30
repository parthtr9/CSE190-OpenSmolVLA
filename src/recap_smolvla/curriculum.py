"""
Curriculum learning wrapper for RECAP (Direction 3).

CurriculumEnvWrapper progressively increases task difficulty based on the
current training iteration.  The motivation: when SmolVLA has near-zero
success rate on the full task, starting with easier configurations ensures
positive examples exist from iteration 0, bootstrapping the RECAP loop.

For PushT / cube-stacking tasks, difficulty is parameterized by the initial
distance between the block and its goal.  The wrapper exposes the same
Gymnasium-compatible interface as the base environment.

Usage
-----
    from recap_smolvla.curriculum import CurriculumEnvWrapper

    env = CurriculumEnvWrapper(base_env, schedule=[0.05, 0.15, "random"])
    env.set_difficulty(iteration=0)   # easy
    env.reset()
    env.set_difficulty(iteration=2)   # full task
    env.reset()
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Difficulty schedule
# ---------------------------------------------------------------------------

DEFAULT_SCHEDULE: list[float | str] = [0.05, 0.15, "random"]
"""
iter 0 → block starts 5 cm from goal    (easy win)
iter 1 → block starts 15 cm from goal   (moderate)
iter ≥ 2 → random initial position      (full task)
"""


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class CurriculumEnvWrapper:
    """Gymnasium-compatible wrapper that schedules task difficulty.

    Parameters
    ----------
    base_env:
        Underlying environment.  Must expose ``reset()`` / ``step()`` /
        ``render()`` and optionally ``set_initial_block_distance(dist)``.
    schedule:
        List of difficulty levels indexed by iteration.  Each entry is
        either a float (initial block-to-goal distance in env units) or
        "random" (full task).  If iteration exceeds the schedule length,
        the last entry is used.
    success_threshold:
        When the running success rate on the current difficulty level
        exceeds this threshold, ``auto_advance()`` moves to the next level.
    window:
        Number of recent episodes used to compute the running success rate.
    """

    def __init__(
        self,
        base_env: Any,
        schedule: list[float | str] | None = None,
        *,
        success_threshold: float = 0.70,
        window: int = 20,
    ) -> None:
        self.env = base_env
        self.schedule: list[float | str] = schedule or DEFAULT_SCHEDULE
        self.success_threshold = success_threshold
        self.window = window

        self._iteration: int = 0
        self._recent_successes: list[bool] = []

    # ------------------------------------------------------------------
    # Difficulty control
    # ------------------------------------------------------------------

    def set_difficulty(self, iteration: int) -> None:
        """Set difficulty to correspond to the given RECAP iteration."""
        self._iteration = iteration
        level = self._current_level()
        self._apply_difficulty(level)

    def auto_advance(self, success: bool) -> bool:
        """Record an episode outcome and auto-advance difficulty if ready.

        Returns True if difficulty was advanced.
        """
        self._recent_successes.append(success)
        if len(self._recent_successes) > self.window:
            self._recent_successes.pop(0)

        current_sr = float(np.mean(self._recent_successes)) if self._recent_successes else 0.0
        if (
            current_sr >= self.success_threshold
            and self._iteration < len(self.schedule) - 1
        ):
            self._iteration += 1
            self._apply_difficulty(self._current_level())
            self._recent_successes.clear()
            return True
        return False

    def current_difficulty(self) -> float | str:
        """Return the current difficulty level."""
        return self._current_level()

    # ------------------------------------------------------------------
    # Gymnasium API pass-through
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        success = info.get("is_success", False)
        advanced = self.auto_advance(bool(success))
        info["curriculum_advanced"] = advanced
        info["curriculum_level"] = self._current_level()
        info["curriculum_iteration"] = self._iteration
        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_level(self) -> float | str:
        idx = min(self._iteration, len(self.schedule) - 1)
        return self.schedule[idx]

    def _apply_difficulty(self, level: float | str) -> None:
        """Apply difficulty to the base env if it supports it."""
        if hasattr(self.env, "set_initial_block_distance"):
            self.env.set_initial_block_distance(level)
        elif hasattr(self.env, "unwrapped") and hasattr(
            self.env.unwrapped, "set_initial_block_distance"
        ):
            self.env.unwrapped.set_initial_block_distance(level)
        # If the env doesn't support difficulty control, we silently pass —
        # the wrapper still tracks success rate and can report iteration.


# ---------------------------------------------------------------------------
# MockCurriculumEnv — for testing without gym_pusht
# ---------------------------------------------------------------------------

class MockCurriculumEnv:
    """Extends MockEnv to support set_initial_block_distance.

    The distance is used to bias success_prob: shorter distances → higher
    probability of success, simulating easier configurations.
    """

    OBS_DIM = 4
    ACTION_DIM = 2

    def __init__(
        self,
        max_steps: int = 50,
        seed: int = 42,
    ) -> None:
        from recap_smolvla.rollout import MockEnv
        # Physics-only success; success_prob overridden per difficulty level
        self._base = MockEnv(max_steps=max_steps, success_prob=0.0, seed=seed)
        self._distance: float | str = "random"

    def set_initial_block_distance(self, distance: float | str) -> None:
        self._distance = distance
        if isinstance(distance, str) and distance == "random":
            # Full task — physics only, no stochastic help
            self._base.success_prob = 0.0
        else:
            # Closer block → small stochastic boost to simulate easier start
            d = float(distance)
            # 0.05 m → 15% boost;  0.15 m → 5% boost;  0.25+ m → 0%
            self._base.success_prob = float(np.clip(0.15 - d * 0.5, 0.0, 0.15))

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        return self._base.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        return self._base.step(action)

    def render(self) -> Any:
        return self._base.render()

    def close(self) -> None:
        self._base.close()

    @property
    def observation_space(self) -> Any:
        return self._base.observation_space

    @property
    def action_space(self) -> Any:
        return self._base.action_space
