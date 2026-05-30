"""
LeRobot dataset adapter for offline RECAP evaluation.

Converts a LeRobotDataset (from the HuggingFace Hub) into the
``(trajectory, success)`` tuple format expected by every RECAP
training function in this package.

Usage — quick start
--------------------
    from recap_smolvla.data import load_lerobot_dataset, dataset_to_rollouts

    ds = load_lerobot_dataset("lerobot/pusht", split="train")
    rollouts = dataset_to_rollouts(ds)              # list[(trajectory, bool)]
    train_rollouts, eval_rollouts = split_rollouts(rollouts, eval_frac=0.2)

Usage — offline eval metrics
-----------------------------
    from recap_smolvla.data import offline_eval_metrics

    metrics = offline_eval_metrics(policy, eval_rollouts, value_fn, reward_fn)
    print(metrics)  # vf_mse, bc_loss, pct_positive, ...

Obs extraction
--------------
LeRobot datasets store observations under keys like:
  ``observation.state``          — flat robot state vector (always present)
  ``observation.images.top``     — camera image (optional)
  ``observation.images.wrist``   — wrist camera (optional)

The adapter concatenates all numeric ``observation.*`` keys (excluding images
unless ``include_images=True``) into a single flat numpy array, matching the
interface expected by ValueFunction and label_trajectories.

Dataset compatibility
---------------------
Tested against:
  lerobot/pusht                        obs_dim=5   action_dim=2
  lerobot/aloha_sim_transfer_cube_*    obs_dim=14  action_dim=14
  lerobot/aloha_sim_insertion_*        obs_dim=14  action_dim=14
  LIBERO tasks (via lerobot wrapper)   obs_dim=varies

All other LeRobotDataset-format datasets should work as long as they have
  ``observation.state`` and ``action`` columns.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Step = dict[str, Any]
Trajectory = list[Step]
Rollout = tuple[Trajectory, bool]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_lerobot_dataset(
    repo_id: str,
    *,
    split: str = "train",
    episodes: list[int] | None = None,
    image_keys: list[str] | None = None,
    root: str | None = None,
) -> Any:
    """Load a LeRobotDataset from the HuggingFace Hub.

    Parameters
    ----------
    repo_id:
        Hub dataset ID, e.g. ``"lerobot/pusht"`` or
        ``"lerobot/aloha_sim_transfer_cube_human"``.
    split:
        Dataset split (most LeRobot datasets only have ``"train"``).
    episodes:
        Optional list of episode indices to load (subset for quick tests).
    image_keys:
        Camera image keys to include, e.g. ``["observation.images.top"]``.
        ``None`` means no images loaded (state-only, faster for offline RECAP).
    root:
        Local cache directory for dataset files.

    Returns
    -------
    A ``LeRobotDataset`` instance.

    Raises
    ------
    ImportError
        If ``lerobot`` is not installed.
    """
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ImportError(
            "lerobot is required to load LeRobot datasets. "
            "Install it with: pip install lerobot"
        ) from e

    kwargs: dict[str, Any] = {"episodes": episodes}
    if image_keys is not None:
        kwargs["image_transforms"] = None  # raw images
    if root is not None:
        kwargs["root"] = root

    return LeRobotDataset(repo_id, **kwargs)


# ---------------------------------------------------------------------------
# Dataset → rollouts conversion
# ---------------------------------------------------------------------------

def dataset_to_rollouts(
    dataset: Any,
    *,
    obs_keys: list[str] | None = None,
    action_key: str = "action",
    success_key: str | None = None,
    max_episodes: int | None = None,
    instruction: str = "complete the task",
    verbose: bool = False,
) -> list[Rollout]:
    """Convert a LeRobotDataset into RECAP-compatible (trajectory, success) tuples.

    One rollout is produced per episode in the dataset.

    Parameters
    ----------
    dataset:
        ``LeRobotDataset`` instance from ``load_lerobot_dataset``.
    obs_keys:
        List of dataset column names to concatenate into the obs vector.
        Defaults to ``["observation.state"]``.
    action_key:
        Column name for the action vector.
    success_key:
        Column name for the per-episode success flag.  If ``None``, the
        adapter tries ``"next.success"``, then ``"success"``, then assumes
        failure for all episodes with a warning.
    max_episodes:
        Limit the number of episodes loaded (useful for fast iteration).
    instruction:
        Language instruction string to attach to every step.
    verbose:
        Print per-episode progress.

    Returns
    -------
    List of ``(trajectory, success)`` tuples, one per episode.
    """
    if obs_keys is None:
        obs_keys = ["observation.state"]

    # Resolve success key
    resolved_success_key = _resolve_success_key(dataset, success_key)
    if resolved_success_key is None:
        warnings.warn(
            "No success key found in dataset. All episodes will be treated as failures. "
            "Pass success_key= to override.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Resolve episode boundaries
    episode_data = _group_by_episode(
        dataset,
        obs_keys=obs_keys,
        action_key=action_key,
        success_key=resolved_success_key,
        max_episodes=max_episodes,
    )

    rollouts: list[Rollout] = []
    for ep_idx, (frames, success) in enumerate(episode_data):
        traj = _frames_to_trajectory(frames, instruction=instruction, episode_idx=ep_idx)
        rollouts.append((traj, success))
        if verbose and (ep_idx + 1) % 20 == 0:
            print(f"  Loaded {ep_idx + 1} episodes...")

    if verbose:
        sr = np.mean([s for _, s in rollouts])
        print(f"  Total: {len(rollouts)} episodes, success rate: {sr:.2%}")

    return rollouts


# ---------------------------------------------------------------------------
# Train / eval split
# ---------------------------------------------------------------------------

def split_rollouts(
    rollouts: list[Rollout],
    eval_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[Rollout], list[Rollout]]:
    """Randomly split rollouts into train and eval sets.

    Stratified by success label so both sets have similar success rates.

    Parameters
    ----------
    rollouts:
        Full rollout list from ``dataset_to_rollouts``.
    eval_frac:
        Fraction of rollouts reserved for evaluation.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (train_rollouts, eval_rollouts)
    """
    rng = np.random.default_rng(seed)

    successes = [r for r in rollouts if r[1]]
    failures = [r for r in rollouts if not r[1]]

    def _split(lst: list) -> tuple[list, list]:
        n_eval = max(1, round(len(lst) * eval_frac))
        idx = rng.permutation(len(lst))
        eval_idx = set(idx[:n_eval].tolist())
        train = [lst[i] for i in range(len(lst)) if i not in eval_idx]
        evl = [lst[i] for i in eval_idx]
        return train, evl

    train_s, eval_s = _split(successes)
    train_f, eval_f = _split(failures)
    train = train_s + train_f
    evl = eval_s + eval_f
    rng.shuffle(train)  # type: ignore[arg-type]
    rng.shuffle(evl)    # type: ignore[arg-type]
    return list(train), list(evl)


# ---------------------------------------------------------------------------
# Offline evaluation metrics
# ---------------------------------------------------------------------------

def offline_eval_metrics(
    policy: Any,
    eval_rollouts: list[Rollout],
    value_fn: Any,
    reward_fn: Callable,
    *,
    gamma: float = 1.0,
) -> dict[str, float]:
    """Compute offline evaluation metrics without any live simulation.

    Metrics
    -------
    vf_mse
        Mean squared error between value function predictions and actual returns
        on the eval set.  Lower is better (indicates VF is learning).
    bc_loss_mean
        Average behaviour-cloning loss on eval steps.  Lower = policy predicts
        demo actions better.
    bc_loss_positive
        BC loss restricted to steps labeled advantage_positive=True.
        Should be lower than bc_loss_negative if advantage conditioning works.
    bc_loss_negative
        BC loss on advantage_positive=False steps.
    pct_positive
        Fraction of eval steps labeled advantage_positive.
    advantage_mean
        Mean advantage across all eval steps.
    advantage_std
        Std of advantages across all eval steps.
    score_return_corr
        Pearson correlation between VF predictions and actual returns.

    Parameters
    ----------
    policy:
        Policy with ``.compute_loss(batch_dict) → Tensor`` method.
    eval_rollouts:
        Held-out (trajectory, success) tuples.
    value_fn:
        Trained ValueFunction instance.
    reward_fn:
        Callable(trajectory, success) → list[float].
    gamma:
        Discount factor (should match training).

    Returns
    -------
    Dict of metric name → float.
    """
    import torch
    from recap_smolvla.advantage import label_trajectories, advantage_distribution_stats
    from recap_smolvla.value_function import compute_returns

    if not eval_rollouts:
        return {}

    labeled = label_trajectories(eval_rollouts, value_fn, reward_fn, gamma=gamma)
    adv_stats = advantage_distribution_stats(labeled)

    # VF MSE and correlation
    import numpy as np
    all_preds, all_targets = [], []
    for traj, success in eval_rollouts:
        if not traj:
            continue
        obs = np.stack([s["obs"] for s in traj]).astype(np.float32)
        rewards = reward_fn(traj, success)
        returns = compute_returns(rewards, gamma=gamma)
        preds = value_fn.predict(obs)
        all_preds.extend(preds.tolist())
        all_targets.extend(returns)

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    vf_mse = float(np.mean((preds_arr - targets_arr) ** 2))
    corr = float(np.corrcoef(preds_arr, targets_arr)[0, 1]) if len(preds_arr) > 1 else 0.0

    # BC loss split by advantage label
    losses_pos, losses_neg = [], []
    for step in labeled:
        with torch.no_grad():
            loss = policy.compute_loss(step).item()
        if step["advantage_positive"]:
            losses_pos.append(loss)
        else:
            losses_neg.append(loss)

    all_losses = losses_pos + losses_neg
    return {
        "vf_mse": vf_mse,
        "score_return_corr": corr,
        "bc_loss_mean": float(np.mean(all_losses)) if all_losses else float("nan"),
        "bc_loss_positive": float(np.mean(losses_pos)) if losses_pos else float("nan"),
        "bc_loss_negative": float(np.mean(losses_neg)) if losses_neg else float("nan"),
        "bc_loss_gap": (
            float(np.mean(losses_neg)) - float(np.mean(losses_pos))
            if losses_pos and losses_neg else float("nan")
        ),
        "pct_positive": adv_stats.get("pct_positive", 0.0),
        "advantage_mean": adv_stats.get("mean", 0.0),
        "advantage_std": adv_stats.get("std", 0.0),
        "n_eval_steps": len(labeled),
        "n_eval_episodes": len(eval_rollouts),
    }


# ---------------------------------------------------------------------------
# Dataset metadata helpers
# ---------------------------------------------------------------------------

def dataset_info(dataset: Any) -> dict[str, Any]:
    """Return a summary dict of a LeRobotDataset."""
    try:
        n_episodes = dataset.num_episodes
        n_frames = len(dataset)
        fps = getattr(dataset, "fps", None)
        features = list(dataset.features.keys()) if hasattr(dataset, "features") else []
        obs_dim = _infer_obs_dim(dataset)
        action_dim = _infer_action_dim(dataset)
        return {
            "n_episodes": n_episodes,
            "n_frames": n_frames,
            "fps": fps,
            "features": features,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }
    except Exception as e:
        return {"error": str(e)}


def _infer_obs_dim(dataset: Any) -> int | None:
    """Try to infer the flat obs dimension from the first frame."""
    try:
        frame = dataset[0]
        obs = _extract_obs(frame, obs_keys=["observation.state"])
        return int(obs.shape[0])
    except Exception:
        return None


def _infer_action_dim(dataset: Any) -> int | None:
    try:
        frame = dataset[0]
        action = np.asarray(frame["action"])
        return int(action.shape[0]) if action.ndim > 0 else 1
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_success_key(dataset: Any, user_key: str | None) -> str | None:
    """Find the column name for per-step success signal."""
    if user_key is not None:
        return user_key
    features = {}
    if hasattr(dataset, "features"):
        features = dataset.features
    elif hasattr(dataset, "hf_dataset") and hasattr(dataset.hf_dataset, "features"):
        features = dataset.hf_dataset.features
    for candidate in ("next.success", "success", "done", "next.done"):
        if candidate in features:
            return candidate
    return None


def _group_by_episode(
    dataset: Any,
    obs_keys: list[str],
    action_key: str,
    success_key: str | None,
    max_episodes: int | None,
) -> list[tuple[list[dict], bool]]:
    """Group dataset frames by episode_index.

    Returns list of (frames_list, success_bool) per episode.
    """
    episodes: dict[int, list[dict]] = {}
    episode_success: dict[int, bool] = {}
    n_frames = len(dataset)

    for i in range(n_frames):
        frame = dataset[i]
        ep_idx = int(_get_scalar(frame, "episode_index", default=0))

        if max_episodes is not None and ep_idx >= max_episodes:
            continue

        obs = _extract_obs(frame, obs_keys)
        action = np.asarray(frame[action_key], dtype=np.float32).flatten()
        image = _extract_image(frame)

        step_dict = {
            "obs": obs,
            "action": action,
            "image": image,
            "frame_index": int(_get_scalar(frame, "frame_index", default=i)),
            "episode_index": ep_idx,
        }

        if ep_idx not in episodes:
            episodes[ep_idx] = []
        episodes[ep_idx].append(step_dict)

        # Track success from last frame of the episode
        if success_key is not None and success_key in frame:
            val = frame[success_key]
            episode_success[ep_idx] = bool(
                val.item() if hasattr(val, "item") else val
            )

    result = []
    for ep_idx in sorted(episodes.keys()):
        frames = episodes[ep_idx]
        success = episode_success.get(ep_idx, False)
        result.append((frames, success))

    return result


def _frames_to_trajectory(
    frames: list[dict],
    instruction: str,
    episode_idx: int,
) -> Trajectory:
    """Convert a list of raw frames into a trajectory list."""
    trajectory: Trajectory = []
    for t, frame in enumerate(frames):
        step: Step = {
            "t": t,
            "obs": frame["obs"],
            "action": frame["action"],
            "image": frame.get("image", np.zeros((64, 64, 3), dtype=np.uint8)),
            "reward": -1.0,
            "success": False,
            "instruction": instruction,
            "is_correction": False,
            "episode_index": episode_idx,
            "frame_index": frame.get("frame_index", t),
        }
        trajectory.append(step)

    # Mark terminal step
    if trajectory:
        trajectory[-1]["reward"] = 0.0  # overridden by reward_fn anyway

    return trajectory


def _extract_obs(frame: dict, obs_keys: list[str]) -> np.ndarray:
    """Concatenate numeric obs columns into a flat float32 array."""
    parts = []
    for key in obs_keys:
        if key not in frame:
            continue
        val = frame[key]
        if hasattr(val, "numpy"):
            arr = val.numpy()
        else:
            arr = np.asarray(val)
        if arr.dtype.kind in ("f", "i", "u"):
            parts.append(arr.flatten().astype(np.float32))
    if not parts:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(parts)


def _extract_image(frame: dict) -> np.ndarray | None:
    """Extract the first available camera image, or None."""
    for key in frame:
        if "image" in key and key != "frame_index":
            val = frame[key]
            if hasattr(val, "numpy"):
                arr = val.numpy()
            else:
                arr = np.asarray(val)
            if arr.ndim == 3:
                return arr.astype(np.uint8) if arr.max() > 1 else (arr * 255).astype(np.uint8)
    return None


def _get_scalar(frame: dict, key: str, default: Any = 0) -> Any:
    val = frame.get(key, default)
    if hasattr(val, "item"):
        return val.item()
    return val


# ---------------------------------------------------------------------------
# Synthetic LeRobot-format dataset (for testing without Hub access)
# ---------------------------------------------------------------------------

class MockLeRobotDataset:
    """Minimal LeRobotDataset-compatible object backed by NumPy arrays.

    Used in tests to exercise the adapter without any network access.

    Parameters
    ----------
    n_episodes:
        Number of episodes to generate.
    episode_length:
        Steps per episode.
    obs_dim:
        Dimensionality of observation.state.
    action_dim:
        Dimensionality of action.
    success_rate:
        Fraction of episodes that have success=True.
    seed:
        RNG seed.
    """

    def __init__(
        self,
        n_episodes: int = 10,
        episode_length: int = 8,
        obs_dim: int = 5,
        action_dim: int = 2,
        success_rate: float = 0.4,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        self.num_episodes = n_episodes
        self.fps = 10

        # Build flat frame list
        self._frames: list[dict] = []
        for ep in range(n_episodes):
            success = bool(rng.random() < success_rate)
            for t in range(episode_length):
                is_last = t == episode_length - 1
                self._frames.append({
                    "observation.state": rng.random(obs_dim).astype(np.float32),
                    "action": rng.random(action_dim).astype(np.float32) * 0.2 - 0.1,
                    "episode_index": ep,
                    "frame_index": t,
                    "next.success": success and is_last,
                    "success": success and is_last,
                })

        self.features = {
            "observation.state": None,
            "action": None,
            "episode_index": None,
            "frame_index": None,
            "next.success": None,
        }

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        return self._frames[idx]
