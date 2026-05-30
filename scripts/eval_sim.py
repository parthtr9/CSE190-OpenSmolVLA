#!/usr/bin/env python
"""Evaluate a trained SmolVLA policy in the MuJoCo simulation.

Loads a trained checkpoint (local directory or Hub repo), connects to the
MuJoCo simulation, and runs autonomous episodes. Supports both interactive
viewer mode (with display) and headless video-saving mode (over SSH).

Usage
-----
    # Interactive (with display):
    python scripts/eval_sim.py \
        --checkpoint Gueso/hf_smolvla_recordpolicy0 \
        --device cuda

    # Headless with video output (over SSH):
    python scripts/eval_sim.py \
        --checkpoint Gueso/hf_smolvla_recordpolicy0 \
        --save-video \
        --num-episodes 20 \
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import types

import cv2
import einops
import mujoco
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Work around a broken dataclass in the installed lerobot groot_n1 module.
# Importing anything from lerobot.policies triggers lerobot/policies/__init__.py
# which imports from the groot subpackage, whose __init__.py eagerly imports
# groot_n1.py containing a dataclass with non-default args after default args.
# We pre-register a stub so Python never executes the real groot __init__.py.
# ---------------------------------------------------------------------------
_groot_stub = types.ModuleType("lerobot.policies.groot")
_groot_stub.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("lerobot.policies.groot", _groot_stub)

_groot_cfg_stub = types.ModuleType("lerobot.policies.groot.configuration_groot")
_groot_cfg_stub.GrootConfig = type("GrootConfig", (), {})  # type: ignore[attr-defined]
sys.modules.setdefault("lerobot.policies.groot.configuration_groot", _groot_cfg_stub)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.relative_action_processor import (  # noqa: E402
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import (  # noqa: E402
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from smolvla_recap.env.mujoco_follower import MuJoCoFollower, JOINT_NAMES  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_processors(
    pretrained_path: str,
    device: str,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load pre- and post-processor pipelines from a checkpoint.

    Replicates the relevant portion of ``make_pre_post_processors`` from
    ``lerobot.policies.factory`` without importing that module (which
    triggers the broken groot import chain).
    """
    preprocessor_overrides = {"device_processor": {"device": device}}

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides=preprocessor_overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        overrides={},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # Re-establish the live reference between relative/absolute action steps
    # (not serializable, so must be wired up after loading).
    relative_step = next(
        (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)),
        None,
    )
    if relative_step is not None:
        for step in postprocessor.steps:
            if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
                step.relative_step = relative_step

    return preprocessor, postprocessor


def obs_to_policy_format(
    obs: dict[str, np.ndarray],
    task: str,
) -> dict[str, torch.Tensor | list[str]]:
    """Convert a MuJoCoFollower observation dict to the format expected by
    the LeRobot preprocessor pipeline.

    Images: uint8 (H,W,C) -> float32 (1,C,H,W) in [0,1]
    State: float32 (6,) -> float32 (1,6)
    Task: str -> list[str] (batch of 1)
    """
    result: dict[str, torch.Tensor | list[str]] = {}

    for cam_name in ("front_camera", "wrist_camera"):
        img = torch.from_numpy(obs[cam_name]).unsqueeze(0)        # (1,H,W,C)
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.float() / 255.0                                 # (1,C,H,W)
        result[f"observation.images.{cam_name}"] = img

    state = torch.from_numpy(obs["state"]).unsqueeze(0)            # (1,6)
    result["observation.state"] = state

    result["task"] = [task]

    return result


def check_success(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Check if the cube is inside the bin by comparing site positions."""
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
    bin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bin_site")

    cube_pos = data.site_xpos[cube_id]
    bin_pos = data.site_xpos[bin_id]

    # Cube is "in the bin" if it's within 3cm horizontally and above the bin floor
    dx = abs(cube_pos[0] - bin_pos[0])
    dy = abs(cube_pos[1] - bin_pos[1])
    dz = cube_pos[2] - bin_pos[2]

    return dx < 0.03 and dy < 0.03 and 0 < dz < 0.05


def make_side_by_side(front: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Combine front and wrist camera images side-by-side for video output."""
    return np.concatenate([front, wrist], axis=1)  # (H, 2*W, 3)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint: str,
    task: str,
    num_episodes: int,
    max_steps: int,
    fps: int,
    device: str,
    save_video: bool,
    video_dir: str,
) -> None:
    # --- Load policy ---
    print(f"Loading policy from: {checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path=checkpoint)
    policy.to(device)
    policy.eval()
    print(f"  Policy loaded on {device}")

    # --- Load pre/post processors ---
    preprocessor, postprocessor = _load_processors(checkpoint, device)
    print("  Preprocessor and postprocessor loaded")

    # --- Connect to MuJoCo sim ---
    use_viewer = not save_video
    follower = MuJoCoFollower()
    follower.connect(viewer=use_viewer)
    mode_str = "viewer" if use_viewer else "headless (saving videos)"
    print(f"  MuJoCo follower connected ({mode_str})")

    # --- Video output directory ---
    if save_video:
        out_dir = Path(video_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Videos will be saved to: {out_dir}")

    step_duration = 1.0 / fps
    results = []

    try:
        for ep in range(num_episodes):
            print(f"\n=== Episode {ep + 1}/{num_episodes} ===")

            # Reset scene and policy
            follower.reset(keyframe="ready")
            policy.reset()

            video_writer = None
            if save_video:
                video_path = out_dir / f"episode_{ep + 1:03d}.mp4"
                # Side-by-side: 2 cameras at 640px each = 1280 wide
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (1280, 480))

            ep_start = time.perf_counter()
            success_detected = False

            for step in range(max_steps):
                loop_start = time.perf_counter()

                # 1. Get observation from the sim
                obs = follower.get_observation()

                # 2. Save frame to video (before policy runs, to capture initial state too)
                if video_writer is not None:
                    frame = make_side_by_side(obs["front_camera"], obs["wrist_camera"])
                    # OpenCV expects BGR
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # 3. Convert to policy format
                obs_dict = obs_to_policy_format(obs, task)

                # 4. Preprocess (normalization, tokenization, device transfer)
                obs_dict = preprocessor(obs_dict)

                # 5. Run policy inference
                with torch.inference_mode():
                    action = policy.select_action(obs_dict)

                # 6. Postprocess (unnormalize)
                action = postprocessor(action)

                # 7. Convert action to follower format
                action_np = action.squeeze(0).cpu().numpy()  # (6,)
                action_dict = {
                    f"{name}.pos": float(action_np[i])
                    for i, name in enumerate(JOINT_NAMES)
                }

                # 8. Send action to follower
                follower.send_action(action_dict)

                # 9. Check success
                if not success_detected and check_success(follower.model, follower.data):
                    success_detected = True
                    print(f"  Success detected at step {step + 1}!")

                # 10. Maintain target FPS (skip in headless for speed)
                if use_viewer:
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = step_duration - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            if video_writer is not None:
                video_writer.release()

            ep_elapsed = time.perf_counter() - ep_start
            status = "SUCCESS" if success_detected else "FAIL"
            results.append(success_detected)
            print(f"  [{status}] {max_steps} steps in {ep_elapsed:.1f}s")

            if save_video and success_detected:
                print(f"  Video saved: {video_path}")
            elif save_video:
                print(f"  Video saved: {video_path}")

            # In viewer mode, wait for user between episodes
            if use_viewer and ep < num_episodes - 1:
                input("  Press Enter to start next episode...")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        follower.disconnect()

    # --- Print summary ---
    if results:
        successes = sum(results)
        total = len(results)
        rate = successes / total * 100
        print(f"\n{'=' * 40}")
        print(f"Results: {successes}/{total} episodes succeeded ({rate:.0f}%)")
        print(f"{'=' * 40}")

        if save_video:
            print(f"Videos saved to: {Path(video_dir).resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SmolVLA policy in MuJoCo simulation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory or Hub repo ID.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the cube and place it in the bin",
        help="Task description string for the policy.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Max steps per episode (default: 300, ~10s at 30fps).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control frequency in Hz (default: 30).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference: cuda or cpu (default: cuda).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Run headless and save episode videos instead of opening viewer.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="outputs/eval_videos",
        help="Directory to save videos (default: outputs/eval_videos).",
    )
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        task=args.task,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        device=args.device,
        save_video=args.save_video,
        video_dir=args.video_dir,
    )


if __name__ == "__main__":
    main()
