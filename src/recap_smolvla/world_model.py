"""Action-conditioned JEPA-style latent dynamics for robot rollouts.

The module deliberately separates representation learning from robot-specific
dynamics: supply a pretrained, frozen encoder and only train
``ActionConditionedLatentWorldModel.predictor`` on robot trajectories.  The
included :class:`StateEncoder` is a small test/simulation fallback; it is not
intended to replace a released video-pretrained visual encoder.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """Projection fallback for state-only simulation and deterministic tests.

    For a V-JEPA-2-AC experiment replace this with a frozen video encoder
    whose ``forward`` maps ``(B, T, ...)`` observations to ``(B, T, D)``.
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.proj(observations.float())


class ActionConditionedLatentWorldModel(nn.Module):
    """Predict future frozen latents from latent history and robot actions.

    At each timestep the causal transformer receives ``z_t + W_a a_t`` and
    predicts ``z_(t+1)``.  The target encoder is evaluated under
    ``inference_mode`` by default, so the robot dataset only trains the small
    action-conditioned predictor with an L1 objective.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        latent_dim: int,
        action_dim: int,
        predictor_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        if predictor_dim % num_heads:
            raise ValueError("predictor_dim must be divisible by num_heads")
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.freeze_encoder = freeze_encoder
        self.latent_in = nn.Linear(latent_dim, predictor_dim)
        self.action_in = nn.Linear(action_dim, predictor_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=4 * predictor_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(predictor_dim), nn.Linear(predictor_dim, latent_dim)
        )
        if freeze_encoder:
            self.set_encoder_frozen(True)

    def set_encoder_frozen(self, frozen: bool = True) -> None:
        """Freeze/unfreeze the representation encoder explicitly."""
        self.freeze_encoder = frozen
        for param in self.encoder.parameters():
            param.requires_grad_(not frozen)
        if frozen:
            self.encoder.eval()

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Return ``(B, T, latent_dim)`` features from the supplied encoder."""
        if observations.ndim < 3:
            raise ValueError("observations must have shape (batch, time, ...)")
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.inference_mode():
                latents = self.encoder(observations)
            return latents.detach()
        return self.encoder(observations)

    def predict_next_latents(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict next latent at every timestep, preserving causal order."""
        if latents.ndim != 3 or actions.ndim != 3:
            raise ValueError("latents and actions must both be (batch, time, features)")
        if latents.shape[:2] != actions.shape[:2]:
            raise ValueError("latents and actions must have matching batch/time dimensions")
        if latents.shape[-1] != self.latent_dim or actions.shape[-1] != self.action_dim:
            raise ValueError("latent/action feature dimensions do not match model configuration")
        steps = latents.shape[1]
        causal_mask = torch.full((steps, steps), float("-inf"), device=latents.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        hidden = self.latent_in(latents) + self.action_in(actions.float())
        return self.output(self.predictor(hidden, mask=causal_mask))

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predictions and frozen targets for transitions in a sequence."""
        latents = self.encode(observations)
        if latents.shape[1] < 2:
            raise ValueError("at least two timesteps are required for dynamics training")
        predictions = self.predict_next_latents(latents[:, :-1], actions[:, :-1])
        return predictions, latents[:, 1:]

    def loss(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Mean L1 latent prediction loss, the V-JEPA-style dynamics objective."""
        prediction, target = self(observations, actions)
        return F.l1_loss(prediction, target)


def train_world_model(
    model: ActionConditionedLatentWorldModel,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int = 1,
    lr: float = 1e-4,
    max_grad_norm: float = 1.0,
) -> list[float]:
    """Train only the dynamics predictor on ``(observations, actions)`` batches."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("world model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    cached_batches = list(batches)
    if not cached_batches:
        return []
    history: list[float] = []
    model.train()
    if model.freeze_encoder:
        model.encoder.eval()
    for _ in range(epochs):
        losses = []
        for observations, actions in cached_batches:
            loss = model.loss(observations, actions)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


def rollout_batches(
    rollouts: list[tuple[list[dict[str, Any]], bool]],
    *,
    device: torch.device | str = "cpu",
    observation_key: str = "obs",
    action_key: str = "action",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Convert variable-length RECAP rollouts to independently trainable batches.

    Batching per episode avoids padding/masking bugs and lets callers compose
    visual encoders later without changing RECAP's rollout contract.
    """
    batches = []
    for trajectory, _ in rollouts:
        if len(trajectory) < 2:
            continue
        observations = np.stack([step[observation_key] for step in trajectory]).astype(np.float32)
        actions = np.stack([step[action_key] for step in trajectory]).astype(np.float32)
        batches.append((
            torch.as_tensor(observations, device=device).unsqueeze(0),
            torch.as_tensor(actions, device=device).unsqueeze(0),
        ))
    return batches


@torch.no_grad()
def augment_rollouts_with_latents(
    rollouts: list[tuple[list[dict[str, Any]], bool]],
    model: ActionConditionedLatentWorldModel,
    *,
    source_key: str = "obs",
    latent_key: str = "latent",
    device: torch.device | str = "cpu",
) -> list[tuple[list[dict[str, Any]], bool]]:
    """Copy rollouts and attach frozen encoder latents to every step.

    The resulting rollouts can be passed to RECAP with ``feature_key='latent'``
    so its value function is grounded in the predictive representation.
    """
    model.eval()
    output: list[tuple[list[dict[str, Any]], bool]] = []
    for trajectory, success in rollouts:
        if not trajectory:
            output.append(([], success))
            continue
        observations = torch.as_tensor(
            np.stack([step[source_key] for step in trajectory]).astype(np.float32), device=device
        ).unsqueeze(0)
        latents = model.encode(observations).squeeze(0).cpu().numpy().astype(np.float32)
        latent_trajectory = [
            {**step, latent_key: latent} for step, latent in zip(trajectory, latents)
        ]
        output.append((latent_trajectory, success))
    return output
