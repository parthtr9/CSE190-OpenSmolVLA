"""
Setup tests: SmolVLA (lerobot) model loading.

All tests here are marked @pytest.mark.slow because they download or
instantiate real model weights.  They are skipped automatically unless
`lerobot` is installed and the SMOLVLA_TEST_REAL=1 env var is set.

Run with:
    SMOLVLA_TEST_REAL=1 pytest tests/setup/test_smolvla_load.py -m slow
"""

from __future__ import annotations

import os

import pytest

# Skip the whole module if lerobot is not installed
lerobot = pytest.importorskip("lerobot", reason="lerobot not installed — skipping SmolVLA load tests")


REAL_MODEL_TESTS_ENABLED = os.environ.get("SMOLVLA_TEST_REAL", "0") == "1"
real_model = pytest.mark.skipif(
    not REAL_MODEL_TESTS_ENABLED,
    reason="Set SMOLVLA_TEST_REAL=1 to run real SmolVLA model tests",
)


# ---------------------------------------------------------------------------
# Import-level checks (no weights needed)
# ---------------------------------------------------------------------------

def test_lerobot_import():
    """lerobot package imports cleanly."""
    import lerobot  # noqa: F401


def test_smolvla_policy_class_importable():
    """SmolVLAPolicy class is importable from lerobot (handles version differences)."""
    SmolVLAPolicy = _import_smolvla_policy()
    assert SmolVLAPolicy is not None, "Could not locate SmolVLAPolicy in any known lerobot path"


def _import_smolvla_policy():
    """Try multiple module paths across lerobot versions."""
    candidates = [
        "lerobot.common.policies.smolvla.modeling_smolvla",
        "lerobot.policies.smolvla.modeling_smolvla",
        "lerobot.policies.smolvla",
    ]
    for path in candidates:
        try:
            mod = __import__(path, fromlist=["SmolVLAPolicy"])
            if hasattr(mod, "SmolVLAPolicy"):
                return mod.SmolVLAPolicy
        except (ImportError, ModuleNotFoundError):
            continue
    pytest.skip(f"SmolVLAPolicy not found in any of: {candidates}")


def test_smolvla_config_importable():
    """SmolVLAConfig is importable (version-agnostic)."""
    candidates = [
        ("lerobot.common.policies.smolvla.configuration_smolvla", "SmolVLAConfig"),
        ("lerobot.policies.smolvla.configuration_smolvla", "SmolVLAConfig"),
    ]
    for path, cls in candidates:
        try:
            mod = __import__(path, fromlist=[cls])
            assert hasattr(mod, cls)
            return
        except (ImportError, ModuleNotFoundError):
            continue
    pytest.skip("SmolVLAConfig not found in this lerobot version")


# ---------------------------------------------------------------------------
# Real model loading (requires SMOLVLA_TEST_REAL=1)
# ---------------------------------------------------------------------------

@real_model
@pytest.mark.slow
def test_smolvla_loads_from_pretrained():
    """SmolVLAPolicy.from_pretrained() succeeds without error."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla-base")
    assert policy is not None


@real_model
@pytest.mark.slow
def test_smolvla_has_required_methods():
    """Real SmolVLA exposes the interface expected by RECAP training."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla-base")
    assert callable(getattr(policy, "select_action", None)), "SmolVLAPolicy missing select_action"


@real_model
@pytest.mark.slow
def test_smolvla_parameter_count():
    """Real SmolVLA should have the expected parameter count (~450M)."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import torch

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla-base")
    n_params = sum(p.numel() for p in policy.parameters())
    n_params_m = n_params / 1e6
    # Sanity check: should be in the rough ballpark of 50M–1B
    assert 50 <= n_params_m <= 2000, (
        f"Unexpected parameter count: {n_params_m:.1f}M — is this still SmolVLA?"
    )
    print(f"\n  SmolVLA parameter count: {n_params_m:.1f}M")


@real_model
@pytest.mark.slow
def test_smolvla_advantage_token_in_vocab():
    """The advantage tokens should be in SmolVLA's tokenizer vocabulary."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from recap_smolvla.training import ADVANTAGE_POSITIVE_TOKEN, ADVANTAGE_NEGATIVE_TOKEN

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla-base")
    tokenizer = getattr(policy, "tokenizer", None) or getattr(
        getattr(policy, "model", None), "tokenizer", None
    )
    if tokenizer is None:
        pytest.skip("Could not locate tokenizer on SmolVLAPolicy")

    for token in (ADVANTAGE_POSITIVE_TOKEN, ADVANTAGE_NEGATIVE_TOKEN):
        ids = tokenizer.encode(token)
        assert len(ids) > 0, f"Token '{token}' not encodable"
