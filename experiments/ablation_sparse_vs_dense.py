"""
Ablation 1: Sparse vs Dense Reward in RECAP.

Runs the full RECAP loop for both reward variants and produces:
  - advantage_comparison.png  — advantage distribution at each iteration
  - recap_comparison.png      — success rate curves

Usage
-----
    python experiments/ablation_sparse_vs_dense.py
    python experiments/ablation_sparse_vs_dense.py --n_iters 5 --n_rollouts 100

With gym_pusht installed:
    python experiments/ablation_sparse_vs_dense.py --env pusht

With mock env (default, no extra deps):
    python experiments/ablation_sparse_vs_dense.py --env mock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make package importable from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recap_smolvla.rewards import sparse_reward_fn, dense_reward_fn
from recap_smolvla.rollout import MockEnv, make_dummy_rollouts, collect_rollout, ScriptedMockPolicy
from recap_smolvla.value_function import ValueFunction
from recap_smolvla.advantage import label_trajectories, advantage_distribution_stats
from recap_smolvla.training import recap_training_iteration, finetune_smolvla


# ---------------------------------------------------------------------------
# Mock policy (no lerobot required)
# ---------------------------------------------------------------------------

import torch.nn as nn


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
# Environment factory
# ---------------------------------------------------------------------------

def make_env(env_name: str, seed: int = 0):
    if env_name == "pusht":
        import gymnasium as gym
        import gym_pusht  # noqa: F401
        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
        env.reset(seed=seed)
        return env
    if env_name == "mujoco":
        from recap_smolvla.envs.mujoco_env import MuJoCoGymWrapper
        return MuJoCoGymWrapper(randomize_cube=True, max_steps=750)
    # success_prob=0.0: physics-only success so a random policy truly struggles
    return MockEnv(max_steps=50, success_prob=0.0, seed=seed)


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def make_policy(policy_name: str, obs_dim: int, action_dim: int, env_name: str,
                checkpoint: str = "lerobot/smolvla_base"):
    """Return a policy instance.  'smolvla' loads the real pretrained model."""
    if policy_name == "smolvla":
        return _load_smolvla(checkpoint)
    # Default: lightweight mock MLP
    return _MockPolicy(obs_dim=obs_dim, action_dim=action_dim)


def _load_smolvla(checkpoint: str = "lerobot/smolvla_base"):
    """Load SmolVLA from HuggingFace Hub with a compute_loss shim."""
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _SV
    except ImportError:
        try:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _SV
        except ImportError:
            raise ImportError(
                "Cannot import SmolVLAPolicy. "
                "Make sure lerobot 0.5+ is installed: pip install lerobot"
            )

    print(f"Loading SmolVLA from {checkpoint} (this may take a minute)...")
    policy = _SV.from_pretrained(checkpoint)
    policy.eval()

    # Discover what image keys this checkpoint actually expects
    image_keys: list[str] = []
    try:
        for k, v in policy.config.input_features.items():
            if "VISUAL" in str(getattr(v, "type", "")):
                image_keys.append(k)
    except Exception:
        pass
    # Gueso checkpoint uses front_camera/wrist_camera; base uses camera1/2/3
    if not image_keys:
        if "Gueso" in checkpoint or "recordpolicy" in checkpoint:
            image_keys = [
                "observation.images.front_camera",
                "observation.images.wrist_camera",
            ]
        else:
            image_keys = ["observation.images.top"]
    print(f"  Model expects image keys: {image_keys}")

    policy = _SmolVLAWrapper(policy, image_keys=image_keys)
    print("SmolVLA loaded.")
    return policy


class _SmolVLAWrapper(torch.nn.Module):
    """Thin wrapper giving SmolVLA the compute_loss / select_action interface.

    Handles the three normalization steps the checkpoint's preprocessing
    pipeline applies:
      1. State: MEAN_STD normalize before passing to the model.
      2. Action: MEAN_STD denormalize after select_action (model outputs
         unit-scale actions; send_action expects raw [-100, 100]).
      3. Images: passed as [0, 1] floats; SigLIP rescaling ([0,1]→[-1,1])
         is done internally by prepare_images.
    Pass state_mean/state_std/action_mean/action_std loaded from the
    checkpoint's safetensors to enable correct normalization.
    """

    def __init__(
        self,
        smolvla,
        image_keys: list[str] | None = None,
        state_mean: "np.ndarray | None" = None,
        state_std: "np.ndarray | None" = None,
        action_mean: "np.ndarray | None" = None,
        action_std: "np.ndarray | None" = None,
        tokenizer_max_length: int = 48,
    ):
        super().__init__()
        self._model = smolvla
        self._image_keys = image_keys or ["observation.images.top"]
        # Grab the tokenizer from inside the model for language conditioning
        try:
            self._tokenizer = smolvla.model.vlm_with_expert.processor.tokenizer
        except AttributeError:
            self._tokenizer = None
        self._tokenizer_max_length = tokenizer_max_length
        # Detect model device so we can move batch tensors to it
        try:
            self._device = next(smolvla.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")
        # Normalization stats (stored as CPU float32 tensors)
        def _to_t(arr):
            return torch.tensor(np.asarray(arr, dtype=np.float32)) if arr is not None else None
        self._state_mean = _to_t(state_mean)
        self._state_std  = _to_t(state_std)
        self._action_mean = _to_t(action_mean)
        self._action_std  = _to_t(action_std)

    def reset(self) -> None:
        """Clear the SmolVLA action queue — call at every episode reset."""
        if hasattr(self._model, "reset"):
            self._model.reset()

    def _tokenize(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, attention_mask) tensors for a task string."""
        if self._tokenizer is None:
            ids = torch.zeros((1, 1), dtype=torch.long)
            return ids, torch.ones_like(ids, dtype=torch.bool)
        enc = self._tokenizer(
            task,
            return_tensors="pt",
            padding="max_length",
            max_length=self._tokenizer_max_length,
            truncation=True,
        )
        return enc["input_ids"], enc["attention_mask"].bool()

    def select_action(self, obs_dict: dict) -> np.ndarray:
        with torch.no_grad():
            batch = self._to_batch(obs_dict)
            action = self._model.select_action(batch)
            if isinstance(action, torch.Tensor):
                action = action.cpu()
            else:
                action = torch.tensor(np.asarray(action, dtype=np.float32))
            # Denormalize: model outputs unit-scale, env expects raw [-100, 100]
            if self._action_mean is not None and self._action_std is not None:
                action = action * self._action_std + self._action_mean
            return action.numpy().flatten()

    def compute_loss(self, batch: dict) -> torch.Tensor:
        sv_batch = self._to_batch(batch)
        # SmolVLA.forward expects actions of shape (B, n_action_steps, action_dim).
        # Our per-step training gives (B, action_dim), so we tile to form a valid
        # chunk.  The gradient w.r.t. the single demonstrated action is still
        # correct because all chunk entries are identical.
        if "action" in sv_batch:
            act = sv_batch["action"]
            if act.ndim == 2:   # (B, action_dim)
                n_steps = getattr(
                    getattr(self._model, "config", None), "n_action_steps", 50
                )
                sv_batch["action"] = act.unsqueeze(1).expand(-1, n_steps, -1).contiguous()
        out = self._model.forward(sv_batch)
        if isinstance(out, tuple):
            # forward returns (loss_tensor, loss_dict)
            loss_tensor = out[0]
            return loss_tensor.mean() if loss_tensor.ndim > 0 else loss_tensor
        if isinstance(out, torch.Tensor):
            return out.mean() if out.ndim > 0 else out
        if hasattr(out, "loss"):
            return out.loss
        return torch.tensor(0.0, requires_grad=True)

    def _to_batch(self, obs_dict: dict) -> dict:
        batch = _to_smolvla_batch(obs_dict, image_keys=self._image_keys)
        # Normalize state: (raw - mean) / std  →  unit scale for the model
        if "observation.state" in batch and self._state_mean is not None:
            s = batch["observation.state"].float()
            m = self._state_mean.to(s.device)
            d = self._state_std.to(s.device).clamp(min=1e-8)
            batch["observation.state"] = (s - m) / d
        # Tokenize the task instruction and add language token keys
        task = obs_dict.get("instruction") or obs_dict.get("task", "grab the cube and place it in the bin")
        if isinstance(task, list):
            task = task[0]
        # SmolVLM tokenizer expects the prompt to end with "\n"
        task_str = str(task).rstrip() + "\n"
        ids, mask = self._tokenize(task_str)
        batch["observation.language.tokens"] = ids
        batch["observation.language.attention_mask"] = mask
        # Move every tensor to the model's device
        return {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def parameters(self, recurse=True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def zero_grad(self, set_to_none=True):
        self._model.zero_grad(set_to_none=set_to_none)


def _img_to_tensor(image) -> "torch.Tensor":
    """Convert a single HWC uint8/float image to a (1,C,H,W) float32 tensor."""
    import torch
    if isinstance(image, np.ndarray):
        img_t = torch.from_numpy(image.astype(np.float32))
        if img_t.ndim == 3:
            img_t = img_t.permute(2, 0, 1)  # HWC → CHW
        if img_t.max() > 1.0:
            img_t = img_t / 255.0
        # Do NOT resize here — SmolVLA's prepare_images calls resize_with_pad
        # internally to 512×512 while preserving aspect ratio (adding black
        # padding).  Pre-resizing to 256×256 (square) distorts the aspect
        # ratio so the model sees completely different visual features than
        # it was trained on (480×640 → 384×512 + 128px top pad at training).
        img_t = img_t.unsqueeze(0)  # (1, C, H, W) at native resolution
    else:
        img_t = image
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
    return img_t


def _to_smolvla_batch(obs_dict: dict, image_keys: list[str] | None = None) -> dict:
    """Convert our generic obs_dict format to the format SmolVLA expects."""
    import torch
    batch = {}

    # Image: render output lives under "observation.image" in collect_rollout.
    # MuJoCoGymWrapper.render() returns a dict {"front_camera": img, "wrist_camera": img}.
    # PushT/MockEnv render() returns a single numpy array.
    image = obs_dict.get("observation.image")
    if image is None:
        image = obs_dict.get("image")
    if image is not None:
        if isinstance(image, dict):
            # Multi-camera dict from MuJoCoGymWrapper — map each camera directly
            for cam_name, cam_img in image.items():
                obs_key = f"observation.images.{cam_name}"
                batch[obs_key] = _img_to_tensor(cam_img)
        else:
            img_t = _img_to_tensor(image)
            # Broadcast the single image to every camera key the model expects
            for key in (image_keys or ["observation.images.top"]):
                batch[key] = img_t

    # State
    for key in ("obs", "observation.state"):
        if key in obs_dict:
            v = obs_dict[key]
            state = v if isinstance(v, torch.Tensor) else torch.tensor(
                np.asarray(v, dtype=np.float32)
            )
            batch["observation.state"] = state.unsqueeze(0) if state.ndim == 1 else state
            break

    # Action (for compute_loss)
    if "action" in obs_dict:
        v = obs_dict["action"]
        act = v if isinstance(v, torch.Tensor) else torch.tensor(np.asarray(v, dtype=np.float32))
        batch["action"] = act.unsqueeze(0) if act.ndim == 1 else act

    return batch


# ---------------------------------------------------------------------------
# Per-iteration advantage plot
# ---------------------------------------------------------------------------

def plot_advantage_distribution(
    sparse_advs: list[list[float]],
    dense_advs: list[list[float]],
    out_path: Path,
) -> None:
    n_iters = len(sparse_advs)
    fig, axes = plt.subplots(2, n_iters, figsize=(5 * n_iters, 8), squeeze=False)

    for i in range(n_iters):
        for row, (advs, name, color) in enumerate(
            [(sparse_advs[i], "Sparse", "#182B49"), (dense_advs[i], "Dense", "#1D9E75")]
        ):
            ax = axes[row][i]
            arr = np.array(advs)
            ax.hist(arr, bins=30, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="zero")
            pct_pos = float(np.mean(arr > 0)) * 100
            ax.set_title(f"{name} — Iter {i+1}\n{pct_pos:.1f}% positive", fontsize=9)
            ax.set_xlabel("Advantage A(s,a)")
            if i == 0:
                ax.set_ylabel("Count")

    plt.suptitle("Advantage Distribution: Sparse vs Dense Reward", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Success rate plot
# ---------------------------------------------------------------------------

def plot_success_curves(
    results: dict,
    baseline: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    iters = list(range(1, len(results["sparse"]["success_rates"]) + 1))

    ax.plot(
        iters, results["sparse"]["success_rates"],
        "o-", color="#182B49", label="RECAP (sparse)", linewidth=2, markersize=7,
    )
    ax.plot(
        iters, results["dense"]["success_rates"],
        "o-", color="#1D9E75", label="RECAP (dense)", linewidth=2, markersize=7,
    )
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.5, label="BC baseline")

    ax.set_xlabel("RECAP Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_title("Dense vs Sparse Reward in RECAP\nSuccess Rate over Training Iterations")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    env_name: str = "mock",
    policy_name: str = "mock",
    n_iters: int = 3,
    n_rollouts: int = 30,
    vf_epochs: int = 30,
    ft_epochs: int = 5,
    out_dir: str = "results",
    seed: int = 0,
    checkpoint: str = "lerobot/smolvla_base",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(env_name, seed=seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # PushT goal position is accessible via the unwrapped env; MuJoCo uses site-based reward
    if env_name == "mujoco":
        from recap_smolvla.envs.mujoco_env import MuJoCoGymWrapper
        reward_configs = {
            "sparse": lambda traj, suc: sparse_reward_fn(traj, suc),
            "dense": lambda traj, suc: sparse_reward_fn(traj, suc),  # overridden by env reward_mode
        }
        max_ep_steps = 750
    else:
        unwrapped = getattr(env, "unwrapped", env)
        goal_pos = getattr(unwrapped, "goal_pos", np.array([256.0, 256.0]))
        if goal_pos.max() > 1.0:
            goal_pos = goal_pos / 512.0
        reward_configs = {
            "sparse": lambda traj, suc: sparse_reward_fn(traj, suc),
            "dense": lambda traj, suc: dense_reward_fn(traj, suc, goal_pos),
        }
        max_ep_steps = 300 if env_name == "pusht" else 50

    results: dict = {
        "sparse": {"success_rates": [], "pct_positive": [], "advantages": []},
        "dense": {"success_rates": [], "pct_positive": [], "advantages": []},
    }

    # Baseline success rate
    print(f"Measuring baseline success rate ({policy_name} policy)...")
    baseline_env = make_env(env_name, seed=seed + 99)
    baseline_policy = make_policy(policy_name, obs_dim, action_dim, env_name, checkpoint=checkpoint)
    baseline_rollouts = []
    for _ in range(min(n_rollouts, 20)):
        traj, suc = collect_rollout(baseline_policy, baseline_env, max_steps=max_ep_steps)
        baseline_rollouts.append((traj, suc))
    baseline_sr = float(np.mean([s for _, s in baseline_rollouts]))
    print(f"Baseline success rate: {baseline_sr:.2%}")

    for reward_name, reward_fn in reward_configs.items():
        print(f"\n{'=' * 50}")
        print(f"RECAP with {reward_name} reward  [{policy_name} policy]")
        print("=" * 50)

        policy = make_policy(policy_name, obs_dim, action_dim, env_name, checkpoint=checkpoint)
        value_fn = ValueFunction(obs_dim=obs_dim)

        for iteration in range(n_iters):
            print(f"\n  Iteration {iteration + 1}/{n_iters}")
            sr, policy, value_fn, stats = recap_training_iteration(
                policy,
                value_fn,
                env,
                reward_fn,
                n_rollouts=n_rollouts,
                vf_epochs=vf_epochs,
                ft_epochs=ft_epochs,
                verbose=True,
                max_steps=max_ep_steps,
            )
            results[reward_name]["success_rates"].append(sr)
            pct_pos = stats.get("pct_positive", 0.0)
            results[reward_name]["pct_positive"].append(pct_pos)
            # Collect advantages for distribution plot
            from recap_smolvla.rollout import make_dummy_rollouts as mdr
            sample_rollouts = mdr(n=10, obs_dim=obs_dim, seed=iteration)
            labeled = label_trajectories(sample_rollouts, value_fn, reward_fn)
            results[reward_name]["advantages"].append(
                [d["advantage"] for d in labeled]
            )
            print(f"  SR={sr:.2%}  pct_positive={pct_pos:.1%}")

    # Plots
    plot_advantage_distribution(
        results["sparse"]["advantages"],
        results["dense"]["advantages"],
        out / "advantage_comparison.png",
    )
    plot_success_curves(results, baseline_sr, out / "recap_comparison.png")

    # Save results JSON
    summary = {
        "baseline_success_rate": baseline_sr,
        "sparse_success_rates": results["sparse"]["success_rates"],
        "dense_success_rates": results["dense"]["success_rates"],
        "sparse_pct_positive": results["sparse"]["pct_positive"],
        "dense_pct_positive": results["dense"]["pct_positive"],
    }
    (out / "results_sparse_vs_dense.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out}/results_sparse_vs_dense.json")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation: Sparse vs Dense RECAP reward")
    p.add_argument("--env", choices=["mock", "pusht", "mujoco"], default="mock",
                   help="Environment: 'mock', 'pusht', or 'mujoco' (SO-101 cube-bin)")
    p.add_argument("--policy", choices=["mock", "smolvla"], default="mock",
                   help="Policy backbone: 'mock' (MLP) or 'smolvla' (real pretrained model)")
    p.add_argument("--checkpoint", default="lerobot/smolvla_base",
                   help="HuggingFace repo or local path for SmolVLA checkpoint "
                        "(default: lerobot/smolvla_base; use Gueso/hf_smolvla_recordpolicy0 "
                        "for the BC-trained SO-101 policy)")
    p.add_argument("--n_iters", type=int, default=3)
    p.add_argument("--n_rollouts", type=int, default=30)
    p.add_argument("--vf_epochs", type=int, default=30)
    p.add_argument("--ft_epochs", type=int, default=5)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
        env_name=args.env,
        policy_name=args.policy,
        n_iters=args.n_iters,
        n_rollouts=args.n_rollouts,
        vf_epochs=args.vf_epochs,
        ft_epochs=args.ft_epochs,
        out_dir=args.out_dir,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
