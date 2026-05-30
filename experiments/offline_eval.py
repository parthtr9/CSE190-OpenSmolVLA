"""
Offline RECAP Evaluation — validate the full pipeline on pre-collected data.

This script loads a LeRobot dataset, runs the RECAP training pipeline
(value function + advantage labeling + policy fine-tuning) entirely offline,
and reports metrics that are predictive of live simulation performance —
without ever instantiating a gym environment.

Workflow
--------
1. Load a LeRobotDataset from the Hub (or use mock data for a dry run).
2. Split into train / eval episodes (stratified by success).
3. Train a value function on train episodes.
4. Label train episodes with advantages (sparse and dense variants).
5. Fine-tune the policy on labeled data.
6. Evaluate on held-out episodes:
     - VF MSE vs actual returns       (is the value function learning?)
     - % positive-advantage steps     (is the advantage signal meaningful?)
     - BC loss on positive vs neg steps (is conditioning making a difference?)
     - Pearson r(VF prediction, G_t)  (signal quality)
7. Produce plots and a JSON summary.

Usage
-----
# Dry run with mock data (no network, no lerobot required):
    python experiments/offline_eval.py --dataset mock

# PushT (requires lerobot + network):
    python experiments/offline_eval.py --dataset lerobot/pusht

# ALOHA insertion (stress-tests sparse reward):
    python experiments/offline_eval.py --dataset lerobot/aloha_sim_insertion_human

# Subset for faster iteration:
    python experiments/offline_eval.py --dataset lerobot/pusht --max_episodes 50

# Compare sparse vs dense reward:
    python experiments/offline_eval.py --dataset lerobot/pusht --reward both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recap_smolvla.data import (
    MockLeRobotDataset,
    dataset_info,
    dataset_to_rollouts,
    load_lerobot_dataset,
    offline_eval_metrics,
    split_rollouts,
)
from recap_smolvla.rewards import dense_reward_fn, sparse_reward_fn
from recap_smolvla.value_function import ValueFunction, train_value_function
from recap_smolvla.advantage import label_trajectories, advantage_distribution_stats
from recap_smolvla.training import finetune_smolvla


# ---------------------------------------------------------------------------
# Mock policy (same lightweight MLP used in ablation scripts)
# ---------------------------------------------------------------------------

class _MockPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.head = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.advantage_bias = nn.Parameter(torch.ones(128) * 0.1)

    def _encode(self, obs: torch.Tensor, instruction: str) -> torch.Tensor:
        h = self.head[:3](obs)
        if "Advantage: positive" in instruction:
            h = h + self.advantage_bias
        return self.head[3:](h)

    def select_action(self, obs_dict: dict) -> np.ndarray:
        obs = _to_tensor(obs_dict, self.obs_dim)
        with torch.no_grad():
            return self._encode(obs, obs_dict.get("instruction", "")).numpy()

    def compute_loss(self, batch: dict) -> torch.Tensor:
        obs = _to_tensor(batch, self.obs_dim)
        pred = self._encode(obs, batch.get("instruction", ""))
        target = torch.tensor(
            np.asarray(batch["action"], dtype=np.float32).flatten()[: self.action_dim]
        )
        return nn.functional.mse_loss(pred, target)


def _to_tensor(d: dict, obs_dim: int) -> torch.Tensor:
    for key in ("obs", "observation.state"):
        if key in d:
            v = d[key]
            t = v.float() if isinstance(v, torch.Tensor) else torch.tensor(
                np.asarray(v, dtype=np.float32)
            )
            return t.reshape(-1)[:obs_dim]
    return torch.zeros(obs_dim)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load(args: argparse.Namespace):
    """Return (rollouts, obs_dim, action_dim, goal_pos)."""
    if args.dataset == "mock":
        print("Using synthetic mock dataset (no network required)")
        ds = MockLeRobotDataset(
            n_episodes=args.max_episodes or 40,
            episode_length=12,
            obs_dim=5,
            action_dim=2,
            success_rate=0.35,
            seed=args.seed,
        )
    else:
        print(f"Loading {args.dataset} from HuggingFace Hub...")
        ds = load_lerobot_dataset(
            args.dataset,
            episodes=list(range(args.max_episodes)) if args.max_episodes else None,
        )

    info = dataset_info(ds)
    print(f"  Episodes: {info.get('n_episodes', '?')}")
    print(f"  Frames:   {info.get('n_frames', '?')}")
    print(f"  obs_dim:  {info.get('obs_dim', '?')}")
    print(f"  action_dim: {info.get('action_dim', '?')}")

    rollouts = dataset_to_rollouts(
        ds,
        instruction=args.instruction,
        max_episodes=args.max_episodes,
        verbose=True,
    )

    obs_dim = info.get("obs_dim") or len(rollouts[0][0][0]["obs"])
    action_dim = info.get("action_dim") or len(rollouts[0][0][0]["action"])

    # PushT goal is at (256, 256) in pixel space, normalized to (0.5, 0.5)
    # Generic fallback: mid-range goal
    goal_pos = np.array([0.5, 0.5], dtype=np.float32)
    if "pusht" in str(args.dataset).lower():
        goal_pos = np.array([0.6, 0.6], dtype=np.float32)

    return rollouts, int(obs_dim), int(action_dim), goal_pos


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_vf_calibration(
    preds: list[float],
    targets: list[float],
    label: str,
    out_path: Path,
) -> None:
    corr = float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, alpha=0.3, s=8, color="#182B49")
    lo = min(min(targets), min(preds))
    hi = max(max(targets), max(preds))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="perfect")
    ax.set_xlabel("Actual Return G_t")
    ax.set_ylabel("VF Prediction V(s)")
    ax.set_title(f"Value Function Calibration — {label}\nr = {corr:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def _plot_advantage_distributions(
    labeled_sparse: list[dict],
    labeled_dense: list[dict] | None,
    out_path: Path,
) -> None:
    n_cols = 2 if labeled_dense else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    configs = [(labeled_sparse, "Sparse RECAP", "#182B49")]
    if labeled_dense:
        configs.append((labeled_dense, "Dense RECAP", "#1D9E75"))

    for ax, (labeled, title, color) in zip(axes, configs):
        advs = np.array([d["advantage"] for d in labeled])
        ax.hist(advs, bins=40, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        pct = np.mean([d["advantage_positive"] for d in labeled]) * 100
        ax.set_title(f"{title}\n{pct:.1f}% positive advantage")
        ax.set_xlabel("Advantage A(s,a)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    plt.suptitle("Advantage Distributions on Offline Data", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def _plot_bc_loss_split(metrics_sparse: dict, metrics_dense: dict | None, out_path: Path) -> None:
    labels = ["Sparse"]
    pos_losses = [metrics_sparse.get("bc_loss_positive", float("nan"))]
    neg_losses = [metrics_sparse.get("bc_loss_negative", float("nan"))]

    if metrics_dense:
        labels.append("Dense")
        pos_losses.append(metrics_dense.get("bc_loss_positive", float("nan")))
        neg_losses.append(metrics_dense.get("bc_loss_negative", float("nan")))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, pos_losses, width, label="Adv positive steps", color="#1D9E75", alpha=0.85)
    ax.bar(x + width / 2, neg_losses, width, label="Adv negative steps", color="#D45F5F", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BC Loss (MSE)")
    ax.set_title("BC Loss: Positive vs Negative Advantage Steps\n"
                 "(positive < negative validates conditioning)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def _plot_training_curves(vf_losses: list[float], ft_losses: list[float], out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(vf_losses, color="#182B49", linewidth=2)
    ax1.set_title("Value Function Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(ft_losses, color="#1D9E75", linewidth=2)
    ax2.set_title("Policy Fine-tuning Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BC Loss")
    ax2.grid(alpha=0.3)

    plt.suptitle("Training Curves", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_offline_eval(args: argparse.Namespace) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load data
    print("\n=== 1. Loading dataset ===")
    rollouts, obs_dim, action_dim, goal_pos = _load(args)
    train_rollouts, eval_rollouts = split_rollouts(
        rollouts, eval_frac=args.eval_frac, seed=args.seed
    )
    print(f"  Train: {len(train_rollouts)} eps | Eval: {len(eval_rollouts)} eps")
    baseline_sr = float(np.mean([s for _, s in rollouts]))
    print(f"  Dataset success rate: {baseline_sr:.2%}")

    do_dense = args.reward in ("dense", "both")
    do_sparse = args.reward in ("sparse", "both")

    reward_fns: list[tuple[str, callable]] = []
    if do_sparse:
        reward_fns.append(("sparse", lambda t, s: sparse_reward_fn(t, s)))
    if do_dense:
        reward_fns.append(("dense", lambda t, s: dense_reward_fn(t, s, goal_pos)))

    all_results: dict[str, dict] = {}

    for reward_name, reward_fn in reward_fns:
        print(f"\n=== Running offline RECAP: {reward_name} reward ===")

        # 2. Train value function
        print(f"  Training value function ({args.vf_epochs} epochs)...")
        vf = ValueFunction(obs_dim=obs_dim, hidden_dim=args.hidden_dim)
        vf_losses = train_value_function(
            train_rollouts, vf, reward_fn,
            epochs=args.vf_epochs, lr=1e-3, verbose=False,
        )
        print(f"  VF loss: {vf_losses[0]:.4f} → {vf_losses[-1]:.4f}")

        # 3. Label training data
        print("  Labeling training data...")
        labeled_train = label_trajectories(
            train_rollouts, vf, reward_fn, threshold_pct=70.0
        )
        train_stats = advantage_distribution_stats(labeled_train)
        print(f"  Train pct_positive: {train_stats['pct_positive']:.1%}")

        # 4. Fine-tune policy
        print(f"  Fine-tuning policy ({args.ft_epochs} epochs)...")
        policy = _MockPolicy(obs_dim=obs_dim, action_dim=action_dim)
        ft_losses = finetune_smolvla(
            policy, labeled_train,
            n_epochs=args.ft_epochs, lr=1e-4, verbose=False,
        )
        print(f"  FT loss: {ft_losses[0]:.4f} → {ft_losses[-1]:.4f}")

        # 5. Offline eval metrics
        print("  Computing offline eval metrics...")
        metrics = offline_eval_metrics(policy, eval_rollouts, vf, reward_fn)
        print(f"  VF MSE:           {metrics['vf_mse']:.4f}")
        print(f"  Score-return r:   {metrics['score_return_corr']:.3f}")
        print(f"  BC loss mean:     {metrics['bc_loss_mean']:.4f}")
        print(f"  BC loss positive: {metrics['bc_loss_positive']:.4f}")
        print(f"  BC loss negative: {metrics['bc_loss_negative']:.4f}")
        print(f"  BC loss gap:      {metrics['bc_loss_gap']:.4f}  ← should be > 0")
        print(f"  pct_positive:     {metrics['pct_positive']:.1%}")

        # 6. Plots for this reward variant
        prefix = f"offline_{reward_name}"
        labeled_eval = label_trajectories(eval_rollouts, vf, reward_fn)
        all_preds, all_targets = [], []
        from recap_smolvla.value_function import compute_returns
        for traj, success in eval_rollouts:
            if traj:
                preds = vf.predict(np.stack([s["obs"] for s in traj]).astype(np.float32))
                targets = compute_returns(reward_fn(traj, success))
                all_preds.extend(preds.tolist())
                all_targets.extend(targets)

        _plot_vf_calibration(all_preds, all_targets, reward_name, out / f"{prefix}_vf_calibration.png")
        _plot_training_curves(vf_losses, ft_losses, out / f"{prefix}_training_curves.png")

        all_results[reward_name] = {
            "metrics": metrics,
            "vf_losses": vf_losses,
            "ft_losses": ft_losses,
            "train_stats": train_stats,
        }

    # 7. Comparison plots (only when both reward types run)
    if do_sparse and do_dense:
        _plot_advantage_distributions(
            labeled_train if do_sparse else [],
            label_trajectories(train_rollouts, vf, reward_fns[-1][1]) if do_dense else None,
            out / "offline_advantage_comparison.png",
        )
        _plot_bc_loss_split(
            all_results["sparse"]["metrics"],
            all_results.get("dense", {}).get("metrics"),
            out / "offline_bc_loss_split.png",
        )

    # 8. Save summary JSON
    summary = {
        "dataset": args.dataset,
        "baseline_success_rate": baseline_sr,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_train_episodes": len(train_rollouts),
        "n_eval_episodes": len(eval_rollouts),
        "results": {
            k: {
                "metrics": v["metrics"],
                "vf_loss_first": v["vf_losses"][0] if v["vf_losses"] else None,
                "vf_loss_last": v["vf_losses"][-1] if v["vf_losses"] else None,
                "ft_loss_first": v["ft_losses"][0] if v["ft_losses"] else None,
                "ft_loss_last": v["ft_losses"][-1] if v["ft_losses"] else None,
            }
            for k, v in all_results.items()
        },
    }
    out_json = out / "offline_eval_results.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults saved to {out_json}")

    # 9. Pass/fail summary
    print("\n=== Sanity Check Summary ===")
    for name, res in all_results.items():
        m = res["metrics"]
        checks = {
            "VF loss decreased": res["vf_losses"][0] > res["vf_losses"][-1] if len(res["vf_losses"]) > 1 else None,
            "FT loss decreased": res["ft_losses"][0] > res["ft_losses"][-1] if len(res["ft_losses"]) > 1 else None,
            "VF r > 0.3": m["score_return_corr"] > 0.3,
            "pct_positive in [10%, 50%]": 0.10 <= m["pct_positive"] <= 0.50,
            "BC gap > 0 (conditioning works)": m["bc_loss_gap"] > 0,
        }
        print(f"\n  {name.upper()} reward:")
        for check, result in checks.items():
            icon = "PASS" if result else ("WARN" if result is None else "FAIL")
            print(f"    [{icon}] {check}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline RECAP evaluation on LeRobot datasets")
    p.add_argument(
        "--dataset",
        default="mock",
        help="Dataset name: 'mock' or HF repo_id e.g. 'lerobot/pusht'",
    )
    p.add_argument("--reward", choices=["sparse", "dense", "both"], default="both")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Limit number of episodes (None=all)")
    p.add_argument("--eval_frac", type=float, default=0.2,
                   help="Fraction of episodes used for eval")
    p.add_argument("--vf_epochs", type=int, default=80)
    p.add_argument("--ft_epochs", type=int, default=15)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--out_dir", default="results/offline_eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--instruction",
        default="complete the task",
        help="Language instruction attached to all steps",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_offline_eval(args)
