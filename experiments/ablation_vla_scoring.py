"""
Ablation 2: VLA Self-Supervised Scoring (Direction 2).

Compares three advantage signal sources:
  (A) Sparse reward + trained value function  (baseline RECAP)
  (B) VLA self-scoring with fixed threshold   (no value function)
  (C) VLA self-scoring + sparse reward hybrid (combined)

Produces:
  - scoring_correlation.png    scatter: VLA score vs actual return
  - scoring_advantage_dist.png advantage distributions for A / B / C
  - results_vla_scoring.json

Usage
-----
    python experiments/ablation_vla_scoring.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.rollout import MockEnv, make_dummy_rollouts
from recap_smolvla.value_function import ValueFunction, compute_returns, train_value_function
from recap_smolvla.advantage import label_trajectories
from recap_smolvla.scoring import (
    MockScoringPolicy,
    vla_score_trajectory,
    scores_to_advantages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_vla_advantages(
    rollouts: list[tuple[list[dict], bool]],
    scoring_policy: MockScoringPolicy,
    instruction: str = "push block to goal",
    *,
    hybrid_alpha: float = 0.0,
    reward_fn=None,
) -> list[dict]:
    """Build labeled data using VLA scores as the advantage proxy.

    hybrid_alpha=0: pure VLA scoring
    hybrid_alpha>0: linear blend  adv = (1-alpha)*vla_adv + alpha*sparse_return
    """
    labeled: list[dict] = []
    all_advs: list[float] = []
    per_rollout_advs: list[np.ndarray] = []

    for trajectory, success in rollouts:
        scores = vla_score_trajectory(trajectory, instruction, scoring_policy)
        vla_adv = scores_to_advantages(scores)

        if hybrid_alpha > 0 and reward_fn is not None:
            rewards = reward_fn(trajectory, success)
            returns = np.array(compute_returns(rewards), dtype=np.float32)
            # Normalize returns to [0,1]
            r_min, r_max = returns.min(), returns.max()
            norm_returns = (returns - r_min) / max(r_max - r_min, 1e-6)
            hybrid = (1 - hybrid_alpha) * vla_adv + hybrid_alpha * (norm_returns - 0.5)
            per_rollout_advs.append(hybrid)
        else:
            per_rollout_advs.append(vla_adv)

        all_advs.extend(per_rollout_advs[-1].tolist())

    if not all_advs:
        return []

    threshold = float(np.percentile(all_advs, 70))

    for (trajectory, _), advs in zip(rollouts, per_rollout_advs):
        for step, adv in zip(trajectory, advs):
            labeled.append(
                {
                    **step,
                    "advantage": float(adv),
                    "return_": 0.0,
                    "advantage_positive": float(adv) > threshold,
                }
            )
    return labeled


def _pct_positive(labeled: list[dict]) -> float:
    if not labeled:
        return 0.0
    return float(np.mean([d["advantage_positive"] for d in labeled]))


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def _score_return_correlation(
    rollouts: list[tuple[list[dict], bool]],
    scoring_policy: MockScoringPolicy,
    reward_fn,
    instruction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vla_scores, actual_returns) for correlation analysis."""
    scores_flat, returns_flat = [], []
    for trajectory, success in rollouts:
        scores = vla_score_trajectory(trajectory, instruction, scoring_policy)
        rewards = reward_fn(trajectory, success)
        returns = compute_returns(rewards)
        scores_flat.extend(scores)
        returns_flat.extend(returns)
    return np.array(scores_flat), np.array(returns_flat)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_correlation(scores: np.ndarray, returns: np.ndarray, out_path: Path) -> None:
    corr = float(np.corrcoef(scores, returns)[0, 1]) if len(scores) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(scores, returns, alpha=0.4, s=10, color="#182B49")
    ax.set_xlabel("VLA Score")
    ax.set_ylabel("Actual Return G_t")
    ax.set_title(f"VLA Score vs Actual Return\nr = {corr:.3f}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_advantage_distributions(
    labeled_a: list[dict],
    labeled_b: list[dict],
    labeled_c: list[dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    configs = [
        (labeled_a, "A: Sparse + Value Fn", "#182B49"),
        (labeled_b, "B: VLA Scoring (pure)", "#1D9E75"),
        (labeled_c, "C: Hybrid (VLA + Sparse)", "#E07B39"),
    ]
    for ax, (labeled, title, color) in zip(axes, configs):
        advs = np.array([d["advantage"] for d in labeled])
        ax.hist(advs, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        pct = _pct_positive(labeled) * 100
        ax.set_title(f"{title}\n{pct:.1f}% positive", fontsize=9)
        ax.set_xlabel("Advantage")
    plt.suptitle("Advantage Distributions: Three Signal Sources", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    n_rollouts: int = 30,
    obs_dim: int = 4,
    out_dir: str = "results",
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rollouts = make_dummy_rollouts(
        n=n_rollouts, length=20, obs_dim=obs_dim, success_rate=0.3, seed=seed
    )
    instruction = "push block to goal"
    reward_fn = lambda traj, suc: sparse_reward_fn(traj, suc)

    # --- A: Sparse reward + trained value function ---
    print("Building advantage signal A (sparse + value function)...")
    vf = ValueFunction(obs_dim=obs_dim)
    train_value_function(rollouts, vf, reward_fn, epochs=40, verbose=False)
    labeled_a = label_trajectories(rollouts, vf, reward_fn)

    # --- B: Pure VLA self-scoring ---
    print("Building advantage signal B (VLA scoring)...")
    # score_from_obs=True means score = mean(obs), a non-trivial proxy
    scoring_policy = MockScoringPolicy(
        obs_dim=obs_dim, fixed_score=None, score_from_obs=True
    )
    labeled_b = _compute_vla_advantages(rollouts, scoring_policy, instruction)

    # --- C: Hybrid ---
    print("Building advantage signal C (hybrid)...")
    labeled_c = _compute_vla_advantages(
        rollouts, scoring_policy, instruction,
        hybrid_alpha=0.5, reward_fn=reward_fn,
    )

    # Correlation analysis
    vla_scores, actual_returns = _score_return_correlation(
        rollouts, scoring_policy, reward_fn, instruction
    )

    # Plots
    plot_correlation(vla_scores, actual_returns, out / "scoring_correlation.png")
    plot_advantage_distributions(labeled_a, labeled_b, labeled_c, out / "scoring_advantage_dist.png")

    summary = {
        "pct_positive_A_sparse_vf": _pct_positive(labeled_a),
        "pct_positive_B_vla_pure": _pct_positive(labeled_b),
        "pct_positive_C_hybrid": _pct_positive(labeled_c),
        "score_return_correlation": float(
            np.corrcoef(vla_scores, actual_returns)[0, 1]
        ) if len(vla_scores) > 1 else 0.0,
    }
    (out / "results_vla_scoring.json").write_text(json.dumps(summary, indent=2))
    print("\nVLA Scoring Ablation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Ablation: VLA self-supervised scoring")
    p.add_argument("--n_rollouts", type=int, default=30)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_experiment(n_rollouts=args.n_rollouts, out_dir=args.out_dir, seed=args.seed)
