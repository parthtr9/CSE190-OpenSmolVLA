"""Unit tests for the frozen-encoder action-conditioned world model."""

import pytest
import torch

from recap_smolvla.rollout import make_dummy_rollouts
from recap_smolvla.value_function import ValueFunction, train_value_function
from recap_smolvla.world_model import (
    ActionConditionedLatentWorldModel,
    StateEncoder,
    augment_rollouts_with_latents,
    rollout_batches,
    train_world_model,
)


@pytest.fixture(autouse=True)
def _restore_torch_rng():
    """Keep model-initialization randomness from leaking into older tests."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


def _model() -> ActionConditionedLatentWorldModel:
    return ActionConditionedLatentWorldModel(
        StateEncoder(4, 12),
        latent_dim=12,
        action_dim=2,
        predictor_dim=24,
        num_layers=1,
        num_heads=4,
    )


def test_world_model_predicts_one_target_per_transition():
    model = _model()
    prediction, target = model(torch.randn(2, 5, 4), torch.randn(2, 5, 2))
    assert prediction.shape == target.shape == (2, 4, 12)
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_world_model_loss_trains_predictor_not_frozen_encoder():
    model = _model()
    before = [p.detach().clone() for p in model.encoder.parameters()]
    history = train_world_model(model, [(torch.randn(2, 5, 4), torch.randn(2, 5, 2))], epochs=2)
    assert len(history) == 2 and all(loss >= 0 for loss in history)
    assert all(torch.equal(old, new) for old, new in zip(before, model.encoder.parameters()))


def test_latents_can_drive_existing_value_function():
    model = _model()
    rollouts = make_dummy_rollouts(n=3, obs_dim=4, action_dim=2, length=5, seed=2)
    latent_rollouts = augment_rollouts_with_latents(rollouts, model)
    assert latent_rollouts[0][0][0]["latent"].shape == (12,)
    vf = ValueFunction(obs_dim=12, hidden_dim=16)
    losses = train_value_function(
        latent_rollouts,
        vf,
        lambda t, s: [-1.0] * len(t),
        epochs=2,
        feature_key="latent",
    )
    assert len(losses) == 2
    assert len(rollout_batches(rollouts)) == len(rollouts)
