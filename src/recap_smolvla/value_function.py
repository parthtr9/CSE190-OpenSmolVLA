"""
Value function for RECAP advantage estimation.

In the full π*0.6 system the value function is a 670M-parameter VLM.
For the simulation setting we use a lightweight MLP with LayerNorm that
takes the flat observation vector and predicts the scalar expected return.

The distributional variant described in the paper (B=201 value bins) is
provided as ``DistributionalValueFunction`` for the stretch goal.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def compute_returns(
    rewards: list[float],
    gamma: float = 1.0,
) -> list[float]:
    """Compute discounted cumulative returns G_t = Σ γ^k r_{t+k}.

    Parameters
    ----------
    rewards:
        Per-step reward list (length T).
    gamma:
        Discount factor.  Paper uses γ=1 (undiscounted).

    Returns
    -------
    List of returns, same length as rewards.
    """
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ---------------------------------------------------------------------------
# MLP value function
# ---------------------------------------------------------------------------

class ValueFunction(nn.Module):
    """Two-hidden-layer MLP value function V(s).

    Parameters
    ----------
    obs_dim:
        Dimensionality of the flat observation vector.
    hidden_dim:
        Width of each hidden layer (default 256, matching paper scale-down).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs:
            Float tensor of shape (B, obs_dim) or (obs_dim,).

        Returns
        -------
        Scalar value predictions of shape (B,) or scalar.
        """
        return self.net(obs).squeeze(-1)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Numpy convenience wrapper (no gradient)."""
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32)
            return self.forward(t).numpy()


# ---------------------------------------------------------------------------
# Distributional value function (stretch goal — B=201 bins)
# ---------------------------------------------------------------------------

class DistributionalValueFunction(nn.Module):
    """Distributional value function from the π*0.6 paper.

    Predicts a distribution over B discretized value bins rather than a
    point estimate, following the cross-entropy training objective:

        min_φ E[Σ H(R^B_t(τ), p_φ(V|o_t,l))]

    Parameters
    ----------
    obs_dim:
        Dimensionality of flat obs.
    n_bins:
        Number of value bins (paper uses B=201).
    v_min, v_max:
        Value range to discretize into bins.
    """

    def __init__(
        self,
        obs_dim: int,
        n_bins: int = 201,
        v_min: float = -1.0,
        v_max: float = 0.0,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_bins = n_bins
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_bins)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bins),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns log-softmax over bins, shape (B, n_bins)."""
        return torch.log_softmax(self.net(obs), dim=-1)

    def expected_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns E[V] = Σ p_i * z_i, shape (B,)."""
        log_probs = self.forward(obs)
        probs = log_probs.exp()
        support = self.support.to(obs.device)
        return (probs * support).sum(dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

Rollout = tuple[list[dict], bool]


def train_value_function(
    rollouts: list[Rollout],
    value_fn: ValueFunction,
    reward_fn: Callable,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    gamma: float = 1.0,
    verbose: bool = False,
) -> list[float]:
    """Train value_fn to predict cumulative returns via MSE loss.

    Parameters
    ----------
    rollouts:
        List of (trajectory, success) tuples.
    value_fn:
        ValueFunction instance to update in-place.
    reward_fn:
        Callable(trajectory, success) → list[float].
    epochs:
        Number of full passes over the rollout data.
    lr:
        AdamW learning rate.
    gamma:
        Discount factor passed to compute_returns.
    verbose:
        Print epoch losses when True.

    Returns
    -------
    List of per-epoch mean losses (length == epochs).
    """
    optimizer = optim.AdamW(value_fn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        total_loss = 0.0
        n_steps = 0

        for trajectory, success in rollouts:
            if len(trajectory) == 0:
                continue

            rewards = reward_fn(trajectory, success)
            returns = compute_returns(rewards, gamma=gamma)

            obs = torch.from_numpy(
                np.stack([step["obs"] for step in trajectory]).astype(np.float32)
            )
            targets = torch.tensor(returns, dtype=torch.float32)

            predicted = value_fn(obs)
            loss = loss_fn(predicted, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(value_fn.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(trajectory)
            n_steps += len(trajectory)

        mean_loss = total_loss / max(n_steps, 1)
        epoch_losses.append(mean_loss)

        if verbose and epoch % 10 == 0:
            print(f"  VF epoch {epoch:4d}/{epochs} | loss {mean_loss:.4f}")

    return epoch_losses
