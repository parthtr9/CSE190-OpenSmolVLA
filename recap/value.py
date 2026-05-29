"""Value function V_phi: a tiny MLP that maps observation.state -> scalar return.

This is the actor-critic *baseline*, NOT a Q-network:
  * Input is the state only (no action), so it cannot be used to select actions.
  * It is trained by regression onto the observed discounted return R_t (Monte
    Carlo target), not a bootstrapped Bellman target. No target network, no max.

Its only job is to provide V(s_t) so the advantage A_t = R_t - V(s_t) tells us how
much better the observed return was than expected from that state. That subtraction
is what turns a noisy per-frame reward into a temporally-aware good/bad signal.

Kept deliberately small (state-only MLP) so it trains in seconds on CPU/MPS and is
easy to debug. A more faithful version (image features / distributional 201-bin head)
can replace this later behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ValueConfig:
    state_dim: int = 6
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    # Standardize the regression target (returns) for stable training. The mean/std
    # are filled in from data by `fit_target_normalization` before training.
    normalize_target: bool = True


class ValueMLP(nn.Module):
    """state -> scalar value. Optionally un-normalizes its output to return-space."""

    def __init__(self, config: ValueConfig | None = None):
        super().__init__()
        self.config = config or ValueConfig()

        dims = [self.config.state_dim, *self.config.hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*layers)

        # Target normalization buffers (persisted with the checkpoint).
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))

    def fit_target_normalization(self, returns: torch.Tensor) -> None:
        """Set target mean/std from the training returns (call before training)."""
        returns = returns.reshape(-1).float()
        self.target_mean = returns.mean().detach().reshape(1)
        std = returns.std().detach().reshape(1)
        # Guard against zero-variance targets.
        self.target_std = torch.clamp(std, min=1e-6)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return V(s) in *return-space*, shape (B,).

        The MLP predicts a normalized value; we un-normalize so callers always get
        values comparable to raw returns (so A = R - V is meaningful).
        """
        normed = self.net(state).squeeze(-1)
        if self.config.normalize_target:
            return normed * self.target_std + self.target_mean
        return normed

    def normalize_target(self, returns: torch.Tensor) -> torch.Tensor:
        """Map raw returns into the normalized space the MLP regresses on."""
        if not self.config.normalize_target:
            return returns
        return (returns - self.target_mean) / self.target_std

    def raw_forward(self, state: torch.Tensor) -> torch.Tensor:
        """The MLP's normalized prediction (used to compute the training loss)."""
        return self.net(state).squeeze(-1)
