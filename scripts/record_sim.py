#!/usr/bin/env python
"""Record teleoperation data: real leader arm -> MuJoCo follower -> LeRobot dataset.

Prerequisites
-------------
    pip install -e /path/to/lerobot

Usage
-----
    python scripts/record_sim.py \
        --leader-port /dev/ttyACM0 \
        --repo-id user/my_sim_dataset \
        --task "pick up the cube" \
        --num-episodes 10 \
        --fps 30 \
        --viewer

Controls
--------
    Enter   – end the current episode and save it
    d       – discard the current episode (bad attempt, start over)
    q       – finish recording (saves remaining data and exits)
    Ctrl-C  – same as q
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# LeRobot imports (the lerobot package must be importable, e.g. via
# ``pip install -e /path/to/lerobot`` or by adding it to PYTHONPATH).
# ---------------------------------------------------------------------------
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

from smolvla_recap.env.mujoco_follower import MuJoCoFollower, JOINT_NAMES

# ---------------------------------------------------------------------------
# Dataset feature specification
# ---------------------------------------------------------------------------
FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [f"{j}.pos" for j in JOINT_NAMES],
    },
    "observation.images.front_camera": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist_camera": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [f"{j}.pos" for j in JOINT_NAMES],
    },
}


# ---------------------------------------------------------------------------
# Non-blocking key detection (POSIX)
# ---------------------------------------------------------------------------
def _setup_terminal():
    """Put stdin in non-blocking cbreak mode so we can poll for keypresses."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old_settings


def _restore_terminal(old_settings):
    import termios

    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _key_pressed() -> str | None:
    """Return the key if one is available, else ``None`` (non-blocking)."""
    import select

    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def make_leader(port: str, calibration_dir: str | None) -> SOLeader:
    """Create and return an SOLeader configured for the given USB port."""
    config = SOLeaderTeleopConfig(port=port)
    if calibration_dir is not None:
        config.calibration_dir = calibration_dir
    return SOLeader(config)


def record(
    leader_port: str,
    repo_id: str,
    task: str,
    num_episodes: int,
    fps: int,
    root: str | None,
    calibration_dir: str | None,
    overwrite: bool,
    viewer: bool = False,
) -> None:
    # --- Setup leader & follower ---
    leader = make_leader(leader_port, calibration_dir)
    follower = MuJoCoFollower()

    print("Connecting leader arm...")
    leader.connect()
    print("Connecting MuJoCo follower...")
    follower.connect(viewer=viewer)

    # --- Setup dataset ---
    dataset_root = Path(root) if root else None
    if overwrite and dataset_root and dataset_root.exists():
        shutil.rmtree(dataset_root)

    print(f"Creating dataset: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=FEATURES,
        root=dataset_root,
        robot_type="so101_mujoco",
        use_videos=True,
    )

    # --- Terminal setup for non-blocking input ---
    old_term = _setup_terminal()

    try:
        ep_idx = 0
        saved_count = 0
        while saved_count < num_episodes:
            print(f"\n=== Episode {saved_count + 1}/{num_episodes} ===")
            print("Recording... Enter=save, d=discard, q=quit.")

            episode_done = False
            discard = False
            quit_all = False
            frame_count = 0

            while not episode_done:
                loop_start = time.perf_counter()

                # 1. Read leader action (normalized joint positions)
                action = leader.get_action()

                # 2. Send to MuJoCo follower
                follower.send_action(action)

                # 3. Get observation from follower (state + images)
                obs = follower.get_observation()

                # 4. Build the dataset frame
                action_array = np.array(
                    [action[f"{j}.pos"] for j in JOINT_NAMES], dtype=np.float32
                )
                frame = {
                    "observation.state": obs["state"],
                    "observation.images.front_camera": obs["front_camera"],
                    "observation.images.wrist_camera": obs["wrist_camera"],
                    "action": action_array,
                    "task": task,
                }
                dataset.add_frame(frame)
                frame_count += 1

                # 5. Check for keypress
                key = _key_pressed()
                if key == "\n" or key == "\r":
                    episode_done = True
                elif key == "d":
                    episode_done = True
                    discard = True
                elif key == "q":
                    episode_done = True
                    quit_all = True

                # 6. Maintain target FPS
                elapsed = time.perf_counter() - loop_start
                sleep_time = (1.0 / fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # --- Save or discard episode ---
            if discard:
                dataset.clear_episode_buffer()
                print(f"  DISCARDED episode ({frame_count} frames). Resetting scene...")
                follower.reset()
                continue
            else:
                dataset.save_episode()
                saved_count += 1
                print(f"  Saved episode {saved_count}/{num_episodes}: {frame_count} frames")

            if quit_all:
                break

            # Auto-reset scene between episodes
            if saved_count < num_episodes:
                follower.reset()
                print("Scene reset. Enter=next episode, d=redo last, q=quit.")
                while True:
                    k = _key_pressed()
                    if k == "\n" or k == "\r":
                        break
                    if k == "q":
                        quit_all = True
                        break
                    time.sleep(0.05)
                if quit_all:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")
        # Save any buffered episode data
        if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
            dataset.save_episode()

    finally:
        _restore_terminal(old_term)
        dataset.finalize()
        print(f"\nDataset saved to {dataset.root}")
        print(f"  Episodes: {dataset.meta.total_episodes}")
        print(f"  Frames:   {dataset.meta.total_frames}")

        follower.disconnect()
        leader.disconnect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Record teleoperation data with real leader arm and MuJoCo follower."
    )
    parser.add_argument(
        "--leader-port",
        type=str,
        required=True,
        help="Serial port for the leader arm (e.g. /dev/ttyACM0).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/sim_record",
        help="Dataset repository ID (default: local/sim_record).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the cube and place it in the bin",
        help="Task description for all episodes.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to record (default: 10).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording framerate (default: 30).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for the dataset. Defaults to HF_LEROBOT_HOME/repo_id.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Directory containing leader arm calibration files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset at --root if it exists.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open an interactive viewer window to see the simulation while recording.",
    )
    args = parser.parse_args()

    record(
        leader_port=args.leader_port,
        repo_id=args.repo_id,
        task=args.task,
        num_episodes=args.num_episodes,
        fps=args.fps,
        root=args.root,
        calibration_dir=args.calibration_dir,
        overwrite=args.overwrite,
        viewer=args.viewer,
    )


if __name__ == "__main__":
    main()
