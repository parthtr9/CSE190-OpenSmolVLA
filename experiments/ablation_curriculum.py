"""
Ablation 3: Curriculum Learning for RECAP (Direction 3).

Compares two training regimes:
  (A) Standard RECAP — full task from iteration 0
  (B) Curriculum RECAP — difficulty increases across iterations

Produces:
  - curriculum_comparison.png  success rate vs iteration for A and B
  - curriculum_difficulty.png  difficulty level progression for B
  - results_curriculum.json

Usage
-----
    python experiments/ablation_curriculum.py
    python experiments/ablation_curriculum.py --n_iters 4 --n_rollouts 30
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
from recap_smolvla.value_function import ValueFunction
from recap_smolvla.training import recap_training_iteration
from recap_smolvla.curriculum import CurriculumEnvWrapper, MockCurriculumEnv


# ---------------------------------------------------------------------------
# Mock policy (duplicate of sparse_vs_dense for self-contained script)
# ---------------------------------------------------------------------------

class _MockPolicy(nn.Module):
    def __init__(self, obs_dim: int = 4, action_dim: int = 2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def select_action(self, obs_dict: dict) -> np.ndarray:
        key = "observation.state" if "observation.state" in obs_dict else "obs"
        obs = obs_dict.get(key, np.zeros(self.obs_dim))
        if isinstance(obs, torch.Tensor):
            obs = obs.float()
        else:
            obs = torch.tensor(np.asarray(obs, dtype=np.float32))
        obs = obs.reshape(-1)[: self.obs_dim]
        with torch.no_grad():
            return self.head(obs).numpy()

    def compute_loss(self, batch: dict) -> torch.Tensor:
        key = "observation.state" if "observation.state" in batch else "obs"
        obs = batch.get(key, np.zeros(self.obs_dim))
        if isinstance(obs, torch.Tensor):
            obs = obs.float()
        else:
            obs = torch.tensor(np.asarray(obs, dtype=np.float32))
        obs = obs.reshape(-1)[: self.obs_dim]
        pred = self.head(obs)
        target = torch.tensor(
            np.asarray(batch["action"], dtype=np.float32)[: self.action_dim]
        )
        return nn.functional.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Evaluation on full task (held-out, not seen during curriculum training)
# ---------------------------------------------------------------------------

def _eval_on_full_task(
    policy: nn.Module,
    n_eval: int = 20,
    obs_dim: int = 4,
    seed: int = 99,
) -> float:
    """Evaluate policy success rate on the full (hardest) task."""
    env = MockEnv(max_steps=20, success_prob=0.2, seed=seed)
    from recap_smolvla.rollout import collect_rollout
    successes = []
    for i in range(n_eval):
        env._rng = np.random.default_rng(seed + i)
        _, success = collect_rollout(policy, env, max_steps=20)
        successes.append(success)
    return float(np.mean(successes))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(
    std_rates: list[float],
    curr_rates: list[float],
    curr_held_out: list[float],
    baseline: float,
    out_path: Path,
) -> None:
    iters = list(range(1, len(std_rates) + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(iters, std_rates, "o-", color="#182B49", label="Standard RECAP", lw=2, ms=7)
    ax.plot(iters, curr_rates, "o-", color="#1D9E75", label="Curriculum RECAP (train)", lw=2, ms=7)
    ax.plot(iters, curr_held_out, "s--", color="#1D9E75", alpha=0.6, label="Curriculum RECAP (held-out full task)", lw=1.5, ms=5)
    ax.axhline(baseline, color="gray", linestyle="--", lw=1.5, label="BC baseline")
    ax.set_xlabel("RECAP Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_title("Curriculum vs Standard RECAP\nSuccess Rate over Iterations")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_difficulty_progression(
    difficulties: list[float | str],
    success_rates: list[float],
    out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    iters = list(range(1, len(difficulties) + 1))

    diff_numeric = []
    for d in difficulties:
        if isinstance(d, str) and d == "random":
            diff_numeric.append(0.3)
        else:
            diff_numeric.append(float(d))

    ax1.bar(iters, diff_numeric, color="#D4A017", alpha=0.6, label="Block dist to goal (m)")
    ax1.set_ylabel("Initial Block Distance (m)", color="#D4A017")
    ax1.tick_params(axis="y", labelcolor="#D4A017")

    ax2 = ax1.twinx()
    ax2.plot(iters, success_rates, "o-", color="#1D9E75", lw=2, ms=7, label="Train Success Rate")
    ax2.set_ylabel("Success Rate", color="#1D9E75")
    ax2.tick_params(axis="y", labelcolor="#1D9E75")
    ax2.set_ylim(-0.05, 1.05)

    ax1.set_xlabel("RECAP Iteration")
    ax1.set_title("Curriculum Difficulty Progression")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    n_iters: int = 3,
    n_rollouts: int = 30,
    vf_epochs: int = 30,
    ft_epochs: int = 5,
    out_dir: str = "results",
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = MockEnv.OBS_DIM
    reward_fn = lambda traj, suc: sparse_reward_fn(traj, suc)
    _bl_env = MockEnv(max_steps=50, success_prob=0.0, seed=seed + 77)
    _bl_policy = _MockPolicy(obs_dim=obs_dim)
    from recap_smolvla.rollout import collect_rollout as _cr
    baseline = float(np.mean([_cr(_bl_policy, _bl_env, max_steps=50)[1] for _ in range(15)]))

    # --- A: Standard RECAP ---
    print("Running A: Standard RECAP...")
    std_env = MockEnv(max_steps=50, success_prob=0.0, seed=seed)
    std_policy = _MockPolicy(obs_dim=obs_dim)
    std_vf = ValueFunction(obs_dim=obs_dim)
    std_rates = []

    for k in range(n_iters):
        print(f"  Iteration {k+1}/{n_iters}")
        sr, std_policy, std_vf, _ = recap_training_iteration(
            std_policy, std_vf, std_env, reward_fn,
            n_rollouts=n_rollouts, vf_epochs=vf_epochs, ft_epochs=ft_epochs, verbose=False,
        )
        std_rates.append(sr)

    # --- B: Curriculum RECAP ---
    print("\nRunning B: Curriculum RECAP...")
    base_curr_env = MockCurriculumEnv(max_steps=20, seed=seed)
    curr_env = CurriculumEnvWrapper(
        base_curr_env,
        schedule=[0.05, 0.15, "random"],
        success_threshold=0.50,
        window=10,
    )
    curr_policy = _MockPolicy(obs_dim=obs_dim)
    curr_vf = ValueFunction(obs_dim=obs_dim)
    curr_rates: list[float] = []
    curr_held_out: list[float] = []
    difficulties: list[float | str] = []

    for k in range(n_iters):
        curr_env.set_difficulty(k)
        level = curr_env.current_difficulty()
        difficulties.append(level)
        print(f"  Iteration {k+1}/{n_iters}  difficulty={level}")
        sr, curr_policy, curr_vf, _ = recap_training_iteration(
            curr_policy, curr_vf, curr_env, reward_fn,
            n_rollouts=n_rollouts, vf_epochs=vf_epochs, ft_epochs=ft_epochs, verbose=False,
        )
        curr_rates.append(sr)
        held = _eval_on_full_task(curr_policy, n_eval=15, obs_dim=obs_dim, seed=seed + k * 7)
        curr_held_out.append(held)
        print(f"    train SR={sr:.2%}  held-out full-task SR={held:.2%}")

    # Plots
    plot_comparison(std_rates, curr_rates, curr_held_out, baseline, out / "curriculum_comparison.png")
    plot_difficulty_progression(difficulties, curr_rates, out / "curriculum_difficulty.png")

    summary = {
        "baseline_sr": baseline,
        "standard_recap_success_rates": std_rates,
        "curriculum_recap_train_success_rates": curr_rates,
        "curriculum_recap_held_out_success_rates": curr_held_out,
        "difficulty_schedule": [str(d) for d in difficulties],
    }
    (out / "results_curriculum.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out}/results_curriculum.json")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Ablation: Curriculum vs Standard RECAP")
    p.add_argument("--n_iters", type=int, default=3)
    p.add_argument("--n_rollouts", type=int, default=30)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_experiment(
        n_iters=args.n_iters,
        n_rollouts=args.n_rollouts,
        out_dir=args.out_dir,
        seed=args.seed,
    )
