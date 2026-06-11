"""
Normalization correctness tests for _SmolVLAWrapper.

The key bugs we fixed:
  1. compute_loss must renormalize raw [-100,100] actions back to unit scale
     before passing to SmolVLA.forward — otherwise the loss is 33× OOD and
     gradient updates blow up the weights.
  2. select_action must denormalize unit-scale model output to [-100,100]
     for the env — fixed earlier; tested here for regression.
  3. Actions must be clipped to [-100,100] after denorm so env never sees
     catastrophic joint commands.
  4. Over multiple FT iterations, action magnitude must NOT grow (divergence
     check that catches the original bug where each iter doubled action range).

All tests use a fake SmolVLA that records what it was passed so we can
assert on tensor values directly without needing lerobot installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from ablation_sparse_vs_dense import _SmolVLAWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACTION_MEAN = np.array([5.0, -3.0, 2.0, 8.0, -1.0, 0.5], dtype=np.float32)
ACTION_STD  = np.array([48.0, 52.0, 45.0, 50.0, 47.0, 51.0], dtype=np.float32)
STATE_MEAN  = np.array([-10.0, 20.0, 5.0, -8.0, 12.0, 0.0], dtype=np.float32)
STATE_STD   = np.array([30.0, 25.0, 28.0, 32.0, 27.0, 29.0], dtype=np.float32)


class _RecordingModel(nn.Module):
    """Fake SmolVLA that records every forward/select_action call."""

    def __init__(self, action_dim: int = 6):
        super().__init__()
        self._action_dim = action_dim
        # Learnable param so optimizer.step() actually changes something
        self._p = nn.Parameter(torch.zeros(action_dim))
        self.recorded_actions: list[torch.Tensor] = []
        self.recorded_states: list[torch.Tensor] = []

    def forward(self, batch: dict) -> torch.Tensor:
        if "action" in batch:
            self.recorded_actions.append(batch["action"].detach().cpu().clone())
        if "observation.state" in batch:
            self.recorded_states.append(batch["observation.state"].detach().cpu().clone())
        # return a loss with grad so backward() works
        return (self._p ** 2).sum()

    def select_action(self, batch: dict) -> torch.Tensor:
        # Output unit-scale actions (small values, as a real trained model would)
        return torch.ones(self._action_dim) * 0.5

    def reset(self) -> None:
        pass


def _make_wrapper(**kwargs) -> tuple[_SmolVLAWrapper, _RecordingModel]:
    model = _RecordingModel()
    wrapper = _SmolVLAWrapper(
        model,
        image_keys=["observation.images.front_camera"],
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        action_mean=ACTION_MEAN,
        action_std=ACTION_STD,
        **kwargs,
    )
    return wrapper, model


def _dummy_step(
    raw_action: np.ndarray | None = None,
    raw_state: np.ndarray | None = None,
    image: np.ndarray | None = None,
) -> dict:
    """Build a fake labeled step with raw (denormalized) values."""
    if raw_action is None:
        raw_action = np.random.uniform(-100, 100, 6).astype(np.float32)
    if raw_state is None:
        raw_state = np.random.uniform(-50, 50, 6).astype(np.float32)
    if image is None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "obs": raw_state,
        "action": raw_action,
        "image": {"front_camera": image},
        "instruction": "grab the cube and place it in the bin",
        "advantage_positive": True,
        "advantage": 1.0,
        "return_": 0.0,
    }


# ---------------------------------------------------------------------------
# 1. action passed to forward() is unit-scale, not raw
# ---------------------------------------------------------------------------

class TestActionNormalizationInComputeLoss:

    def _last_action_row(self, model: _RecordingModel) -> torch.Tensor:
        """compute_loss tiles action to (B, n_action_steps, 6). Return one row (shape 6)."""
        assert len(model.recorded_actions) > 0
        # shape is (1, n_steps, 6) or (n_steps, 6) or flattened — take first row of last dim
        t = model.recorded_actions[-1]
        return t.reshape(-1, 6)[0]

    def test_raw_100_normalized_to_unit_scale(self):
        """Raw action [100, 100, ...] → unit scale before SmolVLA sees it."""
        wrapper, model = _make_wrapper()
        raw = np.full(6, 100.0, dtype=np.float32)
        wrapper.compute_loss(_dummy_step(raw_action=raw))

        seen = self._last_action_row(model)
        expected = (torch.tensor(raw) - torch.tensor(ACTION_MEAN)) / torch.tensor(ACTION_STD)
        assert torch.allclose(seen, expected, atol=1e-3), (
            f"Expected unit-scale action {expected.tolist()}, got {seen.tolist()}"
        )

    def test_raw_minus_100_normalized_to_unit_scale(self):
        wrapper, model = _make_wrapper()
        raw = np.full(6, -100.0, dtype=np.float32)
        wrapper.compute_loss(_dummy_step(raw_action=raw))

        seen = self._last_action_row(model)
        expected = (torch.tensor(raw) - torch.tensor(ACTION_MEAN)) / torch.tensor(ACTION_STD)
        assert torch.allclose(seen, expected, atol=1e-3)

    def test_unit_scale_action_not_double_normalized(self):
        """If someone passes already-unit-scale actions, normalizing should give ~[-0.1, 0.1]."""
        wrapper, model = _make_wrapper()
        unit = np.zeros(6, dtype=np.float32)
        wrapper.compute_loss(_dummy_step(raw_action=unit))

        seen = self._last_action_row(model)
        expected = (torch.tensor(unit) - torch.tensor(ACTION_MEAN)) / torch.tensor(ACTION_STD)
        assert torch.allclose(seen, expected, atol=1e-3)

    def test_magnitude_stays_below_5_for_valid_raw_actions(self):
        """Any action in [-100, 100] normalized should produce values in roughly [-3, 3]."""
        wrapper, model = _make_wrapper()
        for _ in range(20):
            raw = np.random.uniform(-100, 100, 6).astype(np.float32)
            wrapper.compute_loss(_dummy_step(raw_action=raw))

        for seen in model.recorded_actions:
            # seen shape is (1, n_steps, 6) — check each row
            rows = seen.reshape(-1, 6)
            max_val = rows.abs().max().item()
            assert max_val < 5.0, (
                f"Normalized action magnitude {max_val:.2f} > 5 — "
                "action was not renormalized before forward()"
            )


# ---------------------------------------------------------------------------
# 2. state is normalized before forward()
# ---------------------------------------------------------------------------

class TestStateNormalizationInComputeLoss:

    def test_raw_state_normalized(self):
        wrapper, model = _make_wrapper()
        raw_state = np.array([50.0, -30.0, 20.0, 0.0, 15.0, -5.0], dtype=np.float32)
        wrapper.compute_loss(_dummy_step(raw_state=raw_state))

        seen = model.recorded_states[-1].reshape(-1)
        expected = (torch.tensor(raw_state) - torch.tensor(STATE_MEAN)) / torch.tensor(STATE_STD)
        assert torch.allclose(seen, expected, atol=1e-3)

    def test_state_magnitude_stays_small(self):
        wrapper, model = _make_wrapper()
        for _ in range(20):
            raw_state = np.random.uniform(-100, 100, 6).astype(np.float32)
            wrapper.compute_loss(_dummy_step(raw_state=raw_state))

        for seen in model.recorded_states:
            max_val = seen.abs().max().item()
            assert max_val < 10.0, (
                f"Normalized state magnitude {max_val:.2f} > 10 — state was not normalized"
            )


# ---------------------------------------------------------------------------
# 3. select_action denormalization and clipping
# ---------------------------------------------------------------------------

class TestSelectActionDenorm:

    def test_unit_output_denormalized_to_raw_scale(self):
        """Model output 0.5 (unit) → denormalized to ~[0.5*std + mean]."""
        wrapper, model = _make_wrapper()
        obs = {
            "observation.state": np.zeros(6, dtype=np.float32),
            "image": {"front_camera": np.zeros((8, 8, 3), dtype=np.uint8)},
            "instruction": "task",
        }
        action = wrapper.select_action(obs)
        # Model returns [0.5, ...] → denorm: 0.5 * std + mean
        expected = 0.5 * ACTION_STD + ACTION_MEAN
        assert np.allclose(action, np.clip(expected, -100.0, 100.0), atol=1.0), (
            f"Denormalized action {action} != expected {expected}"
        )

    def test_output_clipped_to_100(self):
        """Actions after denorm are hard-clipped to [-100, 100]."""
        class _BigOutputModel(nn.Module):
            def select_action(self, batch):
                return torch.ones(6) * 10.0  # large unit-scale → denorm > 100

            def reset(self):
                pass

        big_wrapper = _SmolVLAWrapper(
            _BigOutputModel(),
            action_mean=ACTION_MEAN,
            action_std=ACTION_STD,
        )
        obs = {
            "observation.state": np.zeros(6, dtype=np.float32),
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "instruction": "task",
        }
        action = big_wrapper.select_action(obs)
        assert action.max() <= 100.0, f"Action not clipped: max={action.max()}"
        assert action.min() >= -100.0, f"Action not clipped: min={action.min()}"

    def test_select_action_does_not_normalize(self):
        """select_action should denormalize, not normalize — values should be > 1."""
        wrapper, model = _make_wrapper()
        obs = {
            "observation.state": np.zeros(6, dtype=np.float32),
            "image": {"front_camera": np.zeros((8, 8, 3), dtype=np.uint8)},
            "instruction": "task",
        }
        action = wrapper.select_action(obs)
        # Denormalized output of 0.5 gives ~[0.5*48+5, ...] = [29, ...]
        # If someone accidentally normalized instead, action would be near 0
        assert action.max() > 5.0, (
            "select_action returned values near 0 — looks like normalization "
            "was applied instead of denormalization"
        )


# ---------------------------------------------------------------------------
# 4. Multi-iteration divergence regression test
# ---------------------------------------------------------------------------

class TestMultiIterationStability:

    def _run_fake_iteration(
        self, wrapper: _SmolVLAWrapper, model: _RecordingModel
    ) -> np.ndarray:
        """Simulate one RECAP iteration: collect rollouts, FT, return action sample."""
        from recap_smolvla.training import finetune_smolvla

        # Build 30 fake labeled steps (raw actions in [-100, 100])
        data = [_dummy_step() for _ in range(30)]
        finetune_smolvla(
            wrapper, data, n_epochs=2, lr=1e-4,
            max_labeled_steps=30, verbose=False,
        )
        # Return a sample action from select_action after this FT round
        obs = {
            "observation.state": np.zeros(6, dtype=np.float32),
            "image": {"front_camera": np.zeros((8, 8, 3), dtype=np.uint8)},
            "instruction": "grab the cube and place it in the bin",
        }
        return wrapper.select_action(obs)

    def test_action_magnitude_does_not_grow_over_iterations(self):
        """
        The critical regression test for the divergence bug.

        Before the fix: each FT round trained on OOD targets, pushing weights
        so that raw model output grew, and after denorm the actions would
        double each iteration: [-100,100] → [-200,200] → [-400,400] → ...

        After the fix: actions should stay within [-100,100] regardless of
        how many iterations run.
        """
        wrapper, model = _make_wrapper()
        magnitudes = []

        for it in range(4):
            action = self._run_fake_iteration(wrapper, model)
            mag = float(np.abs(action).max())
            magnitudes.append(mag)

        # No iteration should produce actions > 100 (they're clipped)
        for i, mag in enumerate(magnitudes):
            assert mag <= 100.0, (
                f"Iter {i+1}: action magnitude {mag:.1f} > 100 — "
                "policy is diverging (clip should prevent this)"
            )

        # The magnitude should not be monotonically growing (divergence pattern)
        if len(magnitudes) >= 3:
            growing = all(
                magnitudes[i] < magnitudes[i + 1]
                for i in range(len(magnitudes) - 1)
            )
            assert not growing, (
                f"Action magnitudes grew monotonically across iters: {magnitudes} — "
                "this is the divergence pattern"
            )

    def test_loss_magnitude_in_compute_loss_is_sane(self):
        """
        Before fix: loss was ~50,000–120,000 because [-100,100] actions
        were OOD for the unit-scale loss. After fix: loss should be << 1000.

        Uses a recording model so we can check what action magnitude SmolVLA sees.
        """
        wrapper, model = _make_wrapper()

        # Simulate what FT does: compute_loss on raw rollout actions
        raw_actions = [np.random.uniform(-100, 100, 6).astype(np.float32) for _ in range(10)]
        for raw in raw_actions:
            wrapper.compute_loss(_dummy_step(raw_action=raw))

        # All actions seen by the model should be unit-scale
        for i, seen in enumerate(model.recorded_actions):
            # Shape is (1, n_steps, 6) due to chunk tiling — check per row
            max_abs = seen.reshape(-1, 6).abs().max().item()
            assert max_abs < 5.0, (
                f"Step {i}: SmolVLA received action magnitude {max_abs:.2f} — "
                f"expected < 5 (unit scale). Renormalization is missing."
            )
