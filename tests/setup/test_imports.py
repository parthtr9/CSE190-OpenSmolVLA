"""
Setup tests: verify all package modules import cleanly.

These run first in CI and catch broken __init__.py, missing deps,
circular imports, and syntax errors before any logic is exercised.
"""

import importlib
import sys


# ---------------------------------------------------------------------------
# Core module imports
# ---------------------------------------------------------------------------

def test_import_recap_smolvla_top_level():
    """Top-level package imports without error."""
    import recap_smolvla  # noqa: F401


def test_import_rewards():
    import recap_smolvla.rewards as m
    assert hasattr(m, "sparse_reward_fn")
    assert hasattr(m, "dense_reward_fn")
    assert hasattr(m, "REWARD_REGISTRY")


def test_import_value_function():
    import recap_smolvla.value_function as m
    assert hasattr(m, "ValueFunction")
    assert hasattr(m, "DistributionalValueFunction")
    assert hasattr(m, "compute_returns")
    assert hasattr(m, "train_value_function")


def test_import_advantage():
    import recap_smolvla.advantage as m
    assert hasattr(m, "compute_advantages")
    assert hasattr(m, "label_trajectories")
    assert hasattr(m, "advantage_distribution_stats")
    assert hasattr(m, "assert_advantage_label_invariants")


def test_import_rollout():
    import recap_smolvla.rollout as m
    assert hasattr(m, "collect_rollout")
    assert hasattr(m, "MockEnv")
    assert hasattr(m, "make_dummy_trajectory")
    assert hasattr(m, "make_dummy_rollouts")


def test_import_training():
    import recap_smolvla.training as m
    assert hasattr(m, "finetune_smolvla")
    assert hasattr(m, "recap_training_iteration")
    assert hasattr(m, "ADVANTAGE_POSITIVE_TOKEN")
    assert hasattr(m, "ADVANTAGE_NEGATIVE_TOKEN")


def test_import_scoring():
    import recap_smolvla.scoring as m
    assert hasattr(m, "vla_score_trajectory")
    assert hasattr(m, "scores_to_advantages")
    assert hasattr(m, "MockScoringPolicy")
    assert hasattr(m, "SCORING_PROMPT_TEMPLATE")


def test_import_curriculum():
    import recap_smolvla.curriculum as m
    assert hasattr(m, "CurriculumEnvWrapper")
    assert hasattr(m, "MockCurriculumEnv")
    assert hasattr(m, "DEFAULT_SCHEDULE")


def test_import_data():
    import recap_smolvla.data as m
    assert hasattr(m, "dataset_to_rollouts")
    assert hasattr(m, "split_rollouts")
    assert hasattr(m, "offline_eval_metrics")
    assert hasattr(m, "MockLeRobotDataset")
    assert hasattr(m, "load_lerobot_dataset")
    assert hasattr(m, "dataset_info")


# ---------------------------------------------------------------------------
# Public API surface exported from __init__.py
# ---------------------------------------------------------------------------

def test_top_level_api_completeness():
    """All symbols listed in __all__ are importable from the top-level package."""
    import recap_smolvla
    expected = [
        "sparse_reward_fn",
        "dense_reward_fn",
        "ValueFunction",
        "compute_returns",
        "train_value_function",
        "compute_advantages",
        "label_trajectories",
        "collect_rollout",
        "recap_training_iteration",
        "finetune_smolvla",
        "dataset_to_rollouts",
        "split_rollouts",
        "offline_eval_metrics",
        "MockLeRobotDataset",
    ]
    for name in expected:
        assert hasattr(recap_smolvla, name), f"recap_smolvla.{name} missing from top-level"


# ---------------------------------------------------------------------------
# No circular imports
# ---------------------------------------------------------------------------

def test_no_circular_imports():
    """Import every module independently; none should pull in the full graph."""
    modules = [
        "recap_smolvla.rewards",
        "recap_smolvla.value_function",
        "recap_smolvla.advantage",
        "recap_smolvla.rollout",
        "recap_smolvla.training",
        "recap_smolvla.scoring",
        "recap_smolvla.curriculum",
        "recap_smolvla.data",
    ]
    for mod_name in modules:
        # Remove from cache to force a clean import
        to_remove = [k for k in sys.modules if k.startswith("recap_smolvla")]
        for k in to_remove:
            sys.modules.pop(k, None)
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import {mod_name}"


# ---------------------------------------------------------------------------
# Core deps available
# ---------------------------------------------------------------------------

def test_torch_available():
    import torch
    assert torch.__version__, "PyTorch not installed"


def test_numpy_available():
    import numpy as np
    assert np.__version__, "NumPy not installed"


# ---------------------------------------------------------------------------
# Optional deps (skip gracefully if absent)
# ---------------------------------------------------------------------------

def test_gymnasium_available():
    gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")
    assert gym.__version__


def test_gym_pusht_optional():
    """gym_pusht is optional; just report whether it is available."""
    try:
        import gym_pusht  # noqa: F401
        available = True
    except ImportError:
        available = False
    # Not a failure either way — just informational
    print(f"\n  gym_pusht available: {available}")


def test_lerobot_optional():
    """lerobot is optional; report availability."""
    try:
        import lerobot  # noqa: F401
        available = True
    except ImportError:
        available = False
    print(f"\n  lerobot available: {available}")


import pytest  # noqa: E402 (placed at end to allow the non-pytest tests above to run stand-alone)
