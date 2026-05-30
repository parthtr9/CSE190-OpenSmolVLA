"""
RECAP training loop and SmolVLA fine-tuning.

recap_training_iteration
    One full RECAP loop: collect → train VF → label → fine-tune policy.

finetune_smolvla
    Fine-tune a policy on advantage-labeled data by prepending
    "Advantage: positive" or "Advantage: negative" to the instruction.

Both functions are policy-agnostic: they work with MockSmolVLAPolicy in
tests and with the real SmolVLA policy in production.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from recap_smolvla.advantage import label_trajectories, advantage_distribution_stats
from recap_smolvla.rollout import collect_rollout
from recap_smolvla.value_function import ValueFunction, train_value_function


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADVANTAGE_POSITIVE_TOKEN = "Advantage: positive"
ADVANTAGE_NEGATIVE_TOKEN = "Advantage: negative"

# Probability of dropping the advantage token during training.
# Enables classifier-free guidance (CFG) at inference — matches paper.
ADVANTAGE_DROPOUT_PROB = 0.30


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune_smolvla(
    policy: nn.Module,
    labeled_data: list[dict[str, Any]],
    *,
    n_epochs: int = 10,
    lr: float = 1e-5,
    advantage_dropout: float = ADVANTAGE_DROPOUT_PROB,
    batch_size: int = 32,
    verbose: bool = False,
) -> list[float]:
    """Fine-tune policy on advantage-labeled trajectory data.

    Each step's instruction is prepended with the advantage token:
        "Advantage: positive. <original instruction>"
        "Advantage: negative. <original instruction>"

    With probability ``advantage_dropout`` the token is omitted entirely,
    so the model learns both the conditional and unconditional policy
    (required for classifier-free guidance at inference time).

    Parameters
    ----------
    policy:
        Any nn.Module with a ``.compute_loss(batch_dict) → Tensor`` method.
    labeled_data:
        Output of label_trajectories; each dict must contain "obs",
        "action", "image", "instruction", and "advantage_positive".
    n_epochs:
        Number of passes over labeled_data.
    lr:
        AdamW learning rate.
    advantage_dropout:
        Fraction of steps where advantage token is randomly dropped.
    batch_size:
        Mini-batch size.  Steps are shuffled each epoch.
    verbose:
        Print per-epoch loss.

    Returns
    -------
    List of per-epoch mean losses (length == n_epochs).
    """
    if not labeled_data:
        return []

    optimizer = optim.AdamW(policy.parameters(), lr=lr)
    rng = np.random.default_rng(42)
    epoch_losses: list[float] = []

    for epoch in range(n_epochs):
        indices = rng.permutation(len(labeled_data)).tolist()
        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            batch_loss = torch.tensor(0.0, requires_grad=False)
            step_losses = []

            for i in batch_idx:
                step = labeled_data[i]

                # --- Build advantage-conditioned instruction ---
                if rng.random() < advantage_dropout:
                    conditioned = step["instruction"]
                else:
                    token = (
                        ADVANTAGE_POSITIVE_TOKEN
                        if step["advantage_positive"]
                        else ADVANTAGE_NEGATIVE_TOKEN
                    )
                    conditioned = f"{token}. {step['instruction']}"

                batch_dict = {
                    "obs": step["obs"],
                    "action": step["action"],
                    "image": step["image"],
                    "instruction": conditioned,
                    "advantage_positive": step["advantage_positive"],
                }

                loss = policy.compute_loss(batch_dict)
                step_losses.append(loss)

            # Accumulate and step
            if step_losses:
                batch_loss = torch.stack(step_losses).mean()
                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += batch_loss.item()
                n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        if verbose:
            print(f"  FT epoch {epoch+1:3d}/{n_epochs} | loss {mean_loss:.4f}")

    return epoch_losses


# ---------------------------------------------------------------------------
# Full RECAP iteration
# ---------------------------------------------------------------------------

def recap_training_iteration(
    policy: nn.Module,
    value_fn: ValueFunction,
    env: Any,
    reward_fn: Callable,
    *,
    n_rollouts: int = 50,
    vf_epochs: int = 50,
    ft_epochs: int = 10,
    ft_lr: float = 1e-5,
    advantage_threshold_pct: float = 70.0,
    gamma: float = 1.0,
    max_steps: int = 300,
    instruction: str = "push the block to the goal",
    verbose: bool = True,
) -> tuple[float, nn.Module, ValueFunction, dict[str, Any]]:
    """Execute one full RECAP training iteration.

    Steps:
    1. Collect n_rollouts episodes with the current policy.
    2. Train value_fn on the collected data.
    3. Compute advantages and assign binary labels.
    4. Fine-tune policy with advantage conditioning.

    Parameters
    ----------
    policy:
        Current policy (updated in-place).
    value_fn:
        Current value function (updated in-place).
    env:
        Gymnasium-compatible environment.
    reward_fn:
        Callable(trajectory, success) → list[float].
    n_rollouts:
        Number of episodes to collect.
    vf_epochs:
        Epochs for value function training.
    ft_epochs:
        Epochs for policy fine-tuning.
    ft_lr:
        Learning rate for policy fine-tuning.
    advantage_threshold_pct:
        Percentile for advantage binarization (paper default: 70 → ~30% positive).
    gamma:
        Discount factor.
    max_steps:
        Episode length cap.
    instruction:
        Language instruction for the task.
    verbose:
        Print progress.

    Returns
    -------
    (success_rate, policy, value_fn, stats)
        success_rate: fraction of collected episodes that succeeded.
        stats:        diagnostic dict with advantage distribution info.
    """
    if verbose:
        print("--- Collecting rollouts ---")

    rollouts = []
    successes = []
    for i in range(n_rollouts):
        traj, success = collect_rollout(
            policy, env, max_steps=max_steps, instruction=instruction
        )
        rollouts.append((traj, success))
        successes.append(success)
        if verbose and (i + 1) % max(1, n_rollouts // 5) == 0:
            sr = np.mean(successes)
            print(f"  [{i+1:3d}/{n_rollouts}] running success rate: {sr:.2%}")

    success_rate = float(np.mean(successes))

    if verbose:
        print(f"--- Training value function ({vf_epochs} epochs) ---")

    train_value_function(
        rollouts,
        value_fn,
        reward_fn,
        epochs=vf_epochs,
        gamma=gamma,
        verbose=verbose,
    )

    if verbose:
        print("--- Labeling trajectories ---")

    labeled_data = label_trajectories(
        rollouts,
        value_fn,
        reward_fn,
        threshold_pct=advantage_threshold_pct,
        gamma=gamma,
    )

    stats = advantage_distribution_stats(labeled_data)
    if verbose:
        pct = stats.get("pct_positive", 0.0)
        print(f"  {pct:.1%} of steps labeled advantage_positive")

    if verbose:
        print(f"--- Fine-tuning policy ({ft_epochs} epochs) ---")

    ft_losses = finetune_smolvla(
        policy,
        labeled_data,
        n_epochs=ft_epochs,
        lr=ft_lr,
        verbose=verbose,
    )

    stats["ft_losses"] = ft_losses
    stats["success_rate"] = success_rate
    stats["n_rollouts"] = n_rollouts
    stats["n_labeled_steps"] = len(labeled_data)

    return success_rate, policy, value_fn, stats
