"""
RECAP ablation on the SO-101 MuJoCo cube-bin task.

Loads Gueso/hf_smolvla_recordpolicy0 (BC-trained, ~50% SR with 3-position
cube randomization) as the starting policy and runs the full RECAP loop
comparing sparse vs dense rewards.

Usage (GPU cluster)
-------------------
    python experiments/ablation_mujoco_recap.py \
        --checkpoint Gueso/hf_smolvla_recordpolicy0 \
        --n_iters 5 --n_rollouts 30 --vf_epochs 50 --ft_epochs 10 \
        --out_dir runs/mujoco_recap

    # Smoke test (mock policy, no GPU needed)
    python experiments/ablation_mujoco_recap.py --policy mock --n_iters 1 --n_rollouts 5

Outputs
-------
    <out_dir>/mujoco_recap_results.json
    <out_dir>/success_curves.png
    <out_dir>/advantage_dist.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.rollout import collect_rollout
from recap_smolvla.value_function import ValueFunction
from recap_smolvla.advantage import label_trajectories, advantage_distribution_stats
from recap_smolvla.training import recap_training_iteration
from recap_smolvla.envs.mujoco_env import MuJoCoGymWrapper


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def _load_smolvla(checkpoint: str, device: str = "cuda"):
    """Load SmolVLA with normalization shim.

    Loads the checkpoint's preprocessing/postprocessing safetensors so that:
      - observation.state is MEAN_STD-normalized before the model
      - actions are MEAN_STD-denormalized after the model (raw [-100, 100])
    This matches the training pipeline exactly and is required for non-zero SR.
    """
    try:
        import types
        import sys as _sys
        _groot_stub = types.ModuleType("lerobot.policies.groot")
        _groot_stub.__path__ = []  # type: ignore[attr-defined]
        _sys.modules.setdefault("lerobot.policies.groot", _groot_stub)
        _groot_cfg_stub = types.ModuleType("lerobot.policies.groot.configuration_groot")
        _groot_cfg_stub.GrootConfig = type("GrootConfig", (), {})  # type: ignore[attr-defined]
        _sys.modules.setdefault("lerobot.policies.groot.configuration_groot", _groot_cfg_stub)
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _SV
    except ImportError:
        raise ImportError(
            "lerobot not installed. Run: pip install 'lerobot>=0.5.1'"
        )

    # Import shared wrapper from ablation_sparse_vs_dense
    sys.path.insert(0, str(Path(__file__).parent))
    from ablation_sparse_vs_dense import _SmolVLAWrapper

    print(f"Loading SmolVLA from {checkpoint}...")
    policy = _SV.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()

    image_keys: list[str] = []
    try:
        for k, v in policy.config.input_features.items():
            if "VISUAL" in str(getattr(v, "type", "")):
                image_keys.append(k)
    except Exception:
        pass
    if not image_keys:
        image_keys = [
            "observation.images.front_camera",
            "observation.images.wrist_camera",
        ]
    print(f"  Image keys: {image_keys}")

    # ------------------------------------------------------------------
    # Load normalization statistics from the checkpoint safetensors.
    # Without these the model sees out-of-distribution inputs and the
    # actions come back in unit scale (≈[-3, 3]) rather than [-100, 100].
    # ------------------------------------------------------------------
    state_mean = state_std = action_mean = action_std = None
    try:
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download

        pre_path = hf_hub_download(
            checkpoint,
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
        )
        with safe_open(pre_path, framework="pt") as f:
            state_mean = f.get_tensor("observation.state.mean").numpy()
            state_std  = f.get_tensor("observation.state.std").numpy()

        post_path = hf_hub_download(
            checkpoint,
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        )
        with safe_open(post_path, framework="pt") as f:
            action_mean = f.get_tensor("action.mean").numpy()
            action_std  = f.get_tensor("action.std").numpy()

        print(f"  State  mean: {state_mean.round(2)}, std: {state_std.round(2)}")
        print(f"  Action mean: {action_mean.round(2)}, std: {action_std.round(2)}")
    except Exception as e:
        print(f"  WARNING: could not load normalization stats ({e}). SR will be ~0%.")

    return _SmolVLAWrapper(
        policy,
        image_keys=image_keys,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        tokenizer_max_length=48,
    )


class _MockPolicy(torch.nn.Module):
    """Lightweight MLP that mimics a 6-DoF SO-101 policy (for smoke tests)."""
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.ReLU(), torch.nn.Linear(64, 6)
        )

    def select_action(self, obs_dict: dict) -> np.ndarray:
        key = "observation.state" if "observation.state" in obs_dict else "obs"
        obs = obs_dict.get(key, np.zeros(6))
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.tensor(
            np.asarray(obs, dtype=np.float32)
        )
        obs_t = obs_t.reshape(-1)[:6]
        with torch.no_grad():
            return self.head(obs_t).numpy()

    def compute_loss(self, batch: dict) -> torch.Tensor:
        key = "observation.state" if "observation.state" in batch else "obs"
        obs = batch.get(key, np.zeros(6))
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.tensor(
            np.asarray(obs, dtype=np.float32)
        )
        pred = self.head(obs_t.reshape(-1)[:6])
        tgt = torch.tensor(np.asarray(batch["action"], dtype=np.float32)[:6])
        return torch.nn.functional.mse_loss(pred, tgt)


# ---------------------------------------------------------------------------
# Dense reward for MuJoCo (uses cube-bin site distance each step)
# ---------------------------------------------------------------------------

def _mujoco_dense_reward_fn(trajectory, success, alpha=0.1, max_dist=0.30):
    """Dense reward: -1/step + alpha*(1 - dist/max_dist); +1 terminal on success."""
    rewards = []
    T = len(trajectory)
    for i, step in enumerate(trajectory):
        dist = step.get("cube_bin_dist", max_dist)
        prox = alpha * (1.0 - min(dist, max_dist) / max_dist)
        is_terminal = (i == T - 1)
        base = 1.0 if (is_terminal and success) else -1.0 / max(T, 1)
        rewards.append(base + prox)
    return rewards


# ---------------------------------------------------------------------------
# MuJoCo-aware rollout collector (stores cube-bin dist for dense reward)
# ---------------------------------------------------------------------------

def collect_mujoco_rollout(
    policy,
    env: MuJoCoGymWrapper,
    *,
    max_steps: int = 750,
    instruction: str = "pick up the cube and place it in the bin",
) -> tuple[list[dict], bool]:
    """Like collect_rollout but also stores cube-bin distance for dense reward."""
    import torch as _torch
    from recap_smolvla.envs.mujoco_env import _cube_bin_distance

    obs, _ = env.reset()
    # SmolVLA keeps an action queue that must be cleared at every episode
    # boundary, otherwise stale actions from the previous episode carry over.
    if hasattr(policy, "reset"):
        policy.reset()

    trajectory = []
    done = False
    success = False

    while not done and len(trajectory) < max_steps:
        image_dict = env.render()   # {"front_camera": img, "wrist_camera": img}

        obs_dict: dict = {
            "observation.state": _torch.tensor(obs.astype(np.float32)),
            "instruction": instruction,
            "image": image_dict,
        }
        action = policy.select_action(obs_dict)
        action_np = action if isinstance(action, np.ndarray) else np.asarray(action)

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        success = bool(info.get("is_success", False))

        # cube-bin distance at this step (for dense reward computation)
        dist = _cube_bin_distance(env._follower.model, env._follower.data)

        trajectory.append({
            "t": len(trajectory),
            "obs": obs.astype(np.float32),
            "action": action_np.astype(np.float32),
            "image": image_dict,
            "reward": float(reward),
            "success": success,
            "instruction": instruction,
            "is_correction": False,
            "cube_bin_dist": float(dist),
        })
        obs = next_obs

    return trajectory, success


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_success_curves(results: dict, baseline_sr: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    iters = list(range(1, len(results["sparse"]["success_rates"]) + 1))
    ax.plot(iters, results["sparse"]["success_rates"],
            "o-", color="#182B49", label="RECAP (sparse)", linewidth=2, markersize=7)
    ax.plot(iters, results["dense"]["success_rates"],
            "o-", color="#1D9E75", label="RECAP (dense)", linewidth=2, markersize=7)
    ax.axhline(baseline_sr, color="gray", linestyle="--", linewidth=1.5, label="BC baseline")
    ax.set_xlabel("RECAP Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_title("RECAP on SO-101 Cube-Bin (3-position randomization)\nSparse vs Dense Reward")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def _plot_advantage_dist(results: dict, out_path: Path) -> None:
    n_iters = max(len(results["sparse"]["advantages"]), 1)
    fig, axes = plt.subplots(2, n_iters, figsize=(5 * n_iters, 8), squeeze=False)
    for i in range(n_iters):
        for row, (key, name, color) in enumerate([
            ("sparse", "Sparse", "#182B49"), ("dense", "Dense", "#1D9E75")
        ]):
            ax = axes[row][i]
            advs = results[key]["advantages"][i] if i < len(results[key]["advantages"]) else []
            if advs:
                arr = np.array(advs)
                ax.hist(arr, bins=30, color=color, alpha=0.8, edgecolor="white")
                ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
                pct = float(np.mean(arr > 0)) * 100
                ax.set_title(f"{name} — Iter {i+1}\n{pct:.1f}% positive", fontsize=9)
            ax.set_xlabel("Advantage A(s,a)")
            if i == 0:
                ax.set_ylabel("Count")
    plt.suptitle("Advantage Distribution: Sparse vs Dense (MuJoCo SO-101)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    checkpoint: str = "Gueso/hf_smolvla_recordpolicy0",
    policy_name: str = "smolvla",
    n_iters: int = 5,
    n_rollouts: int = 30,
    vf_epochs: int = 50,
    ft_epochs: int = 10,
    out_dir: str = "runs/mujoco_recap",
    device: str = "cuda",
    seed: int = 0,
    max_ep_steps: int = 750,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Policy: {policy_name}  Checkpoint: {checkpoint}")
    print(f"Env: MuJoCo SO-101 cube-bin, 3-position cube randomization, max {max_ep_steps} steps")
    print(f"RECAP: {n_iters} iters × {n_rollouts} rollouts, VF={vf_epochs}ep, FT={ft_epochs}ep")

    env = MuJoCoGymWrapper(randomize_cube=True, max_steps=max_ep_steps)
    obs_dim = env.observation_space.shape[0]   # 6

    def load_policy():
        if policy_name == "smolvla":
            return _load_smolvla(checkpoint, device=device)
        return _MockPolicy()

    reward_configs = {
        "sparse": sparse_reward_fn,
        "dense": _mujoco_dense_reward_fn,
    }

    results: dict = {
        "sparse": {"success_rates": [], "pct_positive": [], "advantages": []},
        "dense": {"success_rates": [], "pct_positive": [], "advantages": []},
    }

    # --- Baseline ---
    print("\nMeasuring BC baseline success rate...")
    baseline_policy = load_policy()
    baseline_results = []
    for ep in range(min(n_rollouts, 20)):
        _, suc = collect_mujoco_rollout(baseline_policy, env, instruction=
                                        "pick up the cube and place it in the bin")
        baseline_results.append(suc)
    baseline_sr = float(np.mean(baseline_results))
    print(f"BC baseline: {baseline_sr:.2%}  (expect ~50% with 3-position randomization)")

    # --- RECAP loop ---
    for reward_name, reward_fn in reward_configs.items():
        print(f"\n{'=' * 55}")
        print(f"RECAP — {reward_name} reward")
        print("=" * 55)

        policy = load_policy()
        value_fn = ValueFunction(obs_dim=obs_dim)

        for it in range(n_iters):
            print(f"\n  Iteration {it + 1}/{n_iters}")

            # Collect rollouts with the MuJoCo-aware collector
            rollouts = []
            successes = []
            for ep in range(n_rollouts):
                traj, suc = collect_mujoco_rollout(
                    policy, env,
                    instruction="pick up the cube and place it in the bin",
                )
                rollouts.append((traj, suc))
                successes.append(suc)
                if (ep + 1) % max(1, n_rollouts // 5) == 0:
                    print(f"    [{ep+1:3d}/{n_rollouts}] running SR: {np.mean(successes):.2%}")

            sr = float(np.mean(successes))

            # Train value function
            from recap_smolvla.value_function import train_value_function
            train_value_function(rollouts, value_fn, reward_fn,
                                 epochs=vf_epochs, verbose=False)

            # Label and fine-tune
            labeled = label_trajectories(rollouts, value_fn, reward_fn)
            stats = advantage_distribution_stats(labeled)
            pct_pos = stats.get("pct_positive", 0.0)

            from recap_smolvla.training import finetune_smolvla
            finetune_smolvla(policy, labeled, n_epochs=ft_epochs, verbose=True)

            results[reward_name]["success_rates"].append(sr)
            results[reward_name]["pct_positive"].append(pct_pos)
            results[reward_name]["advantages"].append(
                [d["advantage"] for d in labeled]
            )
            print(f"  SR={sr:.2%}  pct_positive={pct_pos:.1%}")

    env.close()

    # --- Plots ---
    _plot_success_curves(results, baseline_sr, out / "success_curves.png")
    _plot_advantage_dist(results, out / "advantage_dist.png")

    summary = {
        "checkpoint": checkpoint,
        "baseline_success_rate": baseline_sr,
        "sparse_success_rates": results["sparse"]["success_rates"],
        "dense_success_rates": results["dense"]["success_rates"],
        "sparse_pct_positive": results["sparse"]["pct_positive"],
        "dense_pct_positive": results["dense"]["pct_positive"],
    }
    (out / "mujoco_recap_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out}/mujoco_recap_results.json")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RECAP ablation on MuJoCo SO-101 cube-bin task"
    )
    p.add_argument("--checkpoint", default="Gueso/hf_smolvla_recordpolicy0",
                   help="HF repo or local path for SmolVLA checkpoint")
    p.add_argument("--policy", choices=["smolvla", "mock"], default="smolvla",
                   help="Policy: 'smolvla' (real) or 'mock' (MLP, for smoke tests)")
    p.add_argument("--n_iters", type=int, default=5)
    p.add_argument("--n_rollouts", type=int, default=30)
    p.add_argument("--vf_epochs", type=int, default=50)
    p.add_argument("--ft_epochs", type=int, default=10)
    p.add_argument("--out_dir", default="runs/mujoco_recap")
    p.add_argument("--device", default="cuda",
                   help="Torch device for policy inference (cuda/cpu/mps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_ep_steps", type=int, default=750,
                   help="Max steps per episode (default 750 = 25s at 30Hz; use 200 for faster runs)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
        checkpoint=args.checkpoint,
        policy_name=args.policy,
        n_iters=args.n_iters,
        n_rollouts=args.n_rollouts,
        vf_epochs=args.vf_epochs,
        ft_epochs=args.ft_epochs,
        out_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
        max_ep_steps=args.max_ep_steps,
    )
