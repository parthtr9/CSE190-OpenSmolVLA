# SmolVLA + RECAP

SmolVLA fine-tuned with RECAP (Reinforcement Learning with Advantage-Conditioned Policy optimization) for robotic manipulation in MuJoCo.

## Quick Start

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- A display server (X11/Wayland) for the MuJoCo viewer, or use headless rendering
- CUDA GPU recommended for training and inference (CPU works but is slow)

### Installation

```bash
git clone https://github.com/parthtr9/CSE190-OpenSmolVLA.git
cd CSE190-OpenSmolVLA
cp .env.example .env          # Fill in your HF_TOKEN, WANDB_API_KEY
uv sync                       # Installs all dependencies into .venv/
```

### Verify MuJoCo loads correctly

```bash
uv run python -c "
import mujoco
model = mujoco.MjModel.from_xml_path('src/smolvla_recap/env/assets/so101_tabletop.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print(f'OK: {model.nbody} bodies, {model.ngeom} geoms, {model.nu} actuators')
"
```

---

## Project Structure

```
CSE190-OpenSmolVLA/
├── scripts/
│   ├── viewer.py              # View the MuJoCo scene interactively
│   ├── record_sim.py          # Record teleoperation demos (leader arm -> sim)
│   └── eval_sim.py            # Run a trained policy in sim and watch it
│
├── src/smolvla_recap/
│   └── env/
│       ├── mujoco_follower.py # Core sim interface (observation, action, reset)
│       └── assets/
│           ├── so101_tabletop.xml  # MuJoCo scene: arm + table + cube + bin
│           └── *.stl              # Robot mesh files
│
├── pyproject.toml             # Dependencies (managed by uv)
├── .env.example               # Template for API keys
└── uv.lock                    # Locked dependency versions
```

---

## What Each File Does

### Scripts (things you run)

#### `scripts/viewer.py` -- View the Scene

Opens an interactive 3D viewer showing the SO-101 arm, table, cube, and bin. Use this to visually inspect the environment. Mouse to orbit/zoom.

```bash
uv run python scripts/viewer.py
```

> Requires a display. If SSH'd in, use X-forwarding (`ssh -X`).

---

#### `scripts/record_sim.py` -- Record Demonstrations

Records human teleoperation data. You need a physical SO-101 leader arm connected via USB. Your hand movements on the leader arm are mirrored into the MuJoCo simulation, and observations + actions are saved as a LeRobot dataset.

```bash
uv run python scripts/record_sim.py \
    --leader-port /dev/ttyACM0 \
    --repo-id user/my_dataset \
    --task "pick up the cube and place it in the bin" \
    --num-episodes 50 \
    --fps 30 \
    --viewer
```

**Controls during recording:**
| Key     | Action                              |
|---------|-------------------------------------|
| `Enter` | Save current episode, start next    |
| `d`     | Discard current episode (redo)      |
| `q`     | Quit recording                      |

**Output:** A LeRobot-format dataset with:
- `observation.state` -- 6 joint angles (float32)
- `observation.images.front_camera` -- 480x640 RGB from fixed camera
- `observation.images.wrist_camera` -- 480x640 RGB from gripper camera
- `action` -- 6 joint commands (float32)

---

#### `scripts/eval_sim.py` -- Evaluate a Trained Policy

Loads a trained SmolVLA checkpoint and runs it autonomously in the MuJoCo simulation. Two modes:

- **Viewer mode** (default): Opens a MuJoCo window to watch live. Requires a display.
- **Video mode** (`--save-video`): Runs headless, saves MP4 videos of each episode. Works over SSH.

The script also automatically detects success (cube lands in bin) and prints a success rate summary.

```bash
# Interactive with viewer (needs display):
uv run python scripts/eval_sim.py \
    --checkpoint Gueso/hf_smolvla_recordpolicy0 \
    --device cuda

# Headless with video output (works over SSH):
uv run python scripts/eval_sim.py \
    --checkpoint Gueso/hf_smolvla_recordpolicy0 \
    --save-video \
    --num-episodes 20 \
    --device cuda
```

Videos are saved as side-by-side front + wrist camera views to `outputs/eval_videos/`.

**All options:**

| Flag             | Default | Description |
|------------------|---------|-------------|
| `--checkpoint`   | (required) | Path to checkpoint dir or HuggingFace repo ID |
| `--task`         | `"pick up the cube and place it in the bin"` | Task description for the policy |
| `--num-episodes` | `5` | Number of episodes to run |
| `--max-steps`    | `300` | Max steps per episode (~10s at 30fps) |
| `--fps`          | `30` | Control frequency |
| `--device`       | `cuda` | `cuda` or `cpu` |
| `--save-video`   | off | Run headless, save MP4 videos |
| `--video-dir`    | `outputs/eval_videos` | Where to save videos |

In viewer mode, press `Enter` between episodes. `Ctrl-C` to quit early.

**Example output:**
```
=== Episode 1/20 ===
  Success detected at step 187!
  [SUCCESS] 300 steps in 4.2s
...
========================================
Results: 18/20 episodes succeeded (90%)
========================================
Videos saved to: /home/user/Projects/CSE190-OpenSmolVLA/outputs/eval_videos
```

---

### Source Code (library)

#### `src/smolvla_recap/env/mujoco_follower.py` -- Simulation Interface

The core class `MuJoCoFollower` that wraps MuJoCo. It provides the same interface as a physical SO-101 robot arm:

| Method | What it does |
|--------|-------------|
| `connect(viewer=True)` | Load the MuJoCo model, set up cameras, optionally open viewer |
| `get_observation()` | Returns `{"state": (6,), "front_camera": (480,640,3), "wrist_camera": (480,640,3)}` |
| `send_action({"joint.pos": value})` | Send normalized joint commands, step physics |
| `reset(keyframe="ready")` | Reset scene to a known pose (arm + cube positions) |
| `disconnect()` | Clean up resources |

Joint values are **normalized**: [-100, 100] for body joints, [0, 100] for gripper.

---

## Training a Policy

Training uses LeRobot's training script with a SmolVLA config. Here's the command that produced the current checkpoint:

```bash
uv run python -m lerobot.scripts.train \
    --policy.type=smolvla \
    --dataset.repo_id=Gueso/so101_cube_bin_v2 \
    --output_dir=outputs/train/my_run \
    --batch_size=64 \
    --steps=20000 \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.push_to_hub=true \
    --policy.repo_id=YourUsername/your_model_name
```

**Key training parameters from the current run:**

| Parameter | Value |
|-----------|-------|
| Dataset | `Gueso/so101_cube_bin_v2` (200 episodes, 70,943 frames) |
| Model | SmolVLA (450M total params, 100M learnable) |
| Steps | 20,000 (~18 epochs) |
| Batch size | 64 |
| LR schedule | Cosine decay, peak 1e-4, warmup 1000 steps |
| Normalization | MEAN_STD for state/action, IDENTITY for images |
| Images | Resized to 512x512 with padding |
| Action chunk | 50 steps |
| VLM backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (frozen) |

Final training loss: **0.091** (from 1.549 initial).

### Training with Fewer Episodes

To train on a subset of episodes (e.g., for creating a weaker baseline policy for RECAP experiments), use the `--dataset.episodes` flag:

```bash
# Train on only the first 50 episodes (out of 200)
uv run python -m lerobot.scripts.train \
    --policy.type=smolvla \
    --dataset.repo_id=Gueso/so101_cube_bin_v2 \
    '--dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]' \
    --output_dir=outputs/train/50ep_baseline \
    --batch_size=64 \
    --steps=20000 \
    --policy.device=cuda
```

Or generate the episode list in Python:

```bash
# Generate the episode indices and pass them
EPISODES=$(python -c "print('[' + ','.join(str(i) for i in range(50)) + ']')")
uv run python -m lerobot.scripts.train \
    --policy.type=smolvla \
    --dataset.repo_id=Gueso/so101_cube_bin_v2 \
    "--dataset.episodes=$EPISODES" \
    --output_dir=outputs/train/50ep_baseline \
    --batch_size=64 \
    --steps=20000 \
    --policy.device=cuda
```

---

## End-to-End Workflow

```
1. Record demos        -->  scripts/record_sim.py
   (leader arm + sim)       Saves LeRobot dataset to HF Hub

2. Train policy         -->  lerobot.scripts.train
   (BC on demos)             Saves checkpoint to HF Hub

3. Evaluate policy      -->  scripts/eval_sim.py
   (watch it in sim)         Loads checkpoint, runs in MuJoCo viewer
```

---

## Environment Details

### Scene: `so101_tabletop.xml`

| Component | Description |
|-----------|-------------|
| **Robot** | SO-101 5-DOF arm + parallel gripper |
| **Cube** | 4cm red cube, free-floating |
| **Bin** | Open-top container for placing the cube |
| **Table** | 80cm x 60cm surface at 40cm height |
| **Cameras** | `front_camera` (fixed, 50° FOV), `wrist_camera` (gripper-mounted, 70.5° FOV) |

### Robot Joints

| Joint | Range (normalized) | Description |
|-------|-------------------|-------------|
| `shoulder_pan` | [-100, 100] | Base rotation |
| `shoulder_lift` | [-100, 100] | Shoulder pitch |
| `elbow_flex` | [-100, 100] | Elbow pitch |
| `wrist_flex` | [-100, 100] | Wrist pitch |
| `wrist_roll` | [-100, 100] | Wrist roll |
| `gripper` | [0, 100] | Gripper open/close |

### Keyframes

- **`home`**: Arm folded, gripper closed
- **`ready`**: Arm extended forward, gripper open (pre-grasp position)

---

## Project Roadmap

### Phase 1: BC Pre-training (current)

1. Collect teleoperated demos in MuJoCo sim
2. Fine-tune SmolVLA with behavioral cloning
3. Evaluate BC baseline success rate

### Phase 2: RECAP Training

1. Roll out BC policy to collect autonomous trajectories
2. Train value function V(s) on rollout returns
3. Compute advantages A = return - V(s), select top positive-advantage transitions
4. Re-fine-tune SmolVLA conditioned on advantage signal
5. Compare RECAP vs BC baseline

### Reward Modeling Ablation

| Reward Type | Description |
|-------------|-------------|
| **State-based** | Ground-truth from simulator (cube position, contact sensors) |
| **VLM-based** | Score frames with a VLM ("is the cube in the bin?") |
| **Learned** | Small classifier trained on success/fail labels |
