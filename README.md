# SmolVLA + RECAP

SmolVLA fine-tuned with RECAP (Reinforcement Learning with Advantage-Conditioned Policy optimization) for robotic manipulation in MuJoCo.

Starting from the pre-trained BC checkpoint [`Gueso/hf_smolvla_recordpolicy0`](https://huggingface.co/Gueso/hf_smolvla_recordpolicy0) (~50% success rate with 3-position cube randomization), RECAP iteratively fine-tunes the policy using advantage-weighted supervision.

---

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

### Verify the RECAP module

```bash
uv run python -c "from recap_smolvla import MuJoCoGymWrapper; print('OK')"
uv run python -c "from smolvla_recap.env.mujoco_follower import MuJoCoFollower; print('OK')"
```

> **MuJoCo note:** MuJoCo 3.x is a pure-Python wheel — no license file or system install needed.

---

## Project Structure

```
CSE190-OpenSmolVLA/
├── scripts/
│   ├── viewer.py              # View the MuJoCo scene interactively
│   ├── record_sim.py          # Record teleoperation demos (leader arm -> sim)
│   └── eval_sim.py            # Run a trained policy in sim and watch it
│
├── src/
│   ├── smolvla_recap/         # Simulation package
│   │   └── env/
│   │       ├── mujoco_follower.py  # Core sim interface (observation, action, reset)
│   │       └── assets/
│   │           ├── so101_tabletop.xml  # MuJoCo scene: arm + table + cube + bin
│   │           └── *.stl              # Robot mesh files
│   │
│   └── recap_smolvla/         # RECAP algorithm package
│       ├── envs/
│       │   └── mujoco_env.py       # MuJoCoGymWrapper — gym interface
│       ├── rewards.py              # sparse_reward_fn, dense_reward_fn
│       ├── value_function.py       # MLP value function + return computation
│       ├── advantage.py            # Advantage calculation + binary labeling (~30% positive)
│       ├── rollout.py              # Gymnasium-compatible rollout collector
│       ├── scoring.py              # VLA self-scoring
│       ├── curriculum.py           # Curriculum learning
│       ├── data.py                 # Data adapter
│       └── training.py            # recap_training_iteration — full RECAP loop
│
├── experiments/
│   ├── ablation_mujoco_recap.py     # Main experiment: sparse vs dense on SO-101
│   ├── ablation_sparse_vs_dense.py  # General ablation (also supports --env mujoco)
│   ├── ablation_vla_scoring.py      # Direction 2: VLA self-scoring
│   ├── ablation_curriculum.py       # Direction 3: curriculum learning
│   └── run_all_ablations.py         # Run all ablation experiments
│
├── tests/                     # Test suite (smoke, unit, gradient, integration)
├── results/                   # Ablation results and plots
├── pyproject.toml             # Dependencies (managed by uv)
├── .env.example               # Template for API keys
└── uv.lock                    # Locked dependency versions
```

---

## Scripts

### `scripts/viewer.py` — View the Scene

Opens an interactive 3D viewer showing the SO-101 arm, table, cube, and bin. Mouse to orbit/zoom.

```bash
uv run python scripts/viewer.py
```

> Requires a display. If SSH'd in, use X-forwarding (`ssh -X`).

### `scripts/record_sim.py` — Record Demonstrations

Records human teleoperation data. You need a physical SO-101 leader arm connected via USB. Movements on the leader arm are mirrored into MuJoCo, and observations + actions are saved as a LeRobot dataset.

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

### `scripts/eval_sim.py` — Evaluate a Trained Policy

Loads a trained SmolVLA checkpoint and runs it autonomously in MuJoCo. Two modes:

- **Viewer mode** (default): Opens a MuJoCo window to watch live.
- **Video mode** (`--save-video`): Runs headless, saves MP4s. Works over SSH.

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

---

## How RECAP Works

```
For each iteration:
  1. Collect N rollouts with the current policy in MuJoCo
     (cube randomized across 3 positions each episode)

  2. Compute discounted returns  R_t = Σ_{t'>=t} γ^(t'-t) r_{t'}

  3. Train an MLP value function  V_φ(s) to predict R_t via regression

  4. Compute advantages  A(s,a) = R_t − V_φ(s)

  5. Label top-30% advantage steps as "positive" → binary labels b_t ∈ {0,1}

  6. Fine-tune SmolVLA with advantage-weighted cross-entropy loss:
       L = −Σ_t b_t · log π_θ(a_t | s_t)

Repeat → policy learns to imitate its own best behaviors.
```

**Sparse reward:** `r_t = -1` per step, `0` on success
**Dense reward:** `r_t = -1/T + α·(1 − dist(cube, bin)/max_dist)` per step

The dense variant gives the value function a smoother signal to learn from, which is the core hypothesis of the ablation.

---

## Training

### Phase 1: Behavioral Cloning

Training uses LeRobot's training script with a SmolVLA config:

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

| Parameter | Value |
|-----------|-------|
| Dataset | `Gueso/so101_cube_bin_v2` (200 episodes, 70,943 frames) |
| Model | SmolVLA (450M total params, 100M learnable) |
| Steps | 20,000 (~18 epochs) |
| Batch size | 64 |
| LR schedule | Cosine decay, peak 1e-4, warmup 1000 steps |
| VLM backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (frozen) |

### Phase 2: RECAP Fine-tuning

Run the RECAP ablation with the BC checkpoint:

```bash
# Local (CPU, for verification)
uv run python experiments/ablation_mujoco_recap.py \
  --checkpoint Gueso/hf_smolvla_recordpolicy0 \
  --n_iters 2 --n_rollouts 5 --vf_epochs 10 --ft_epochs 3 \
  --device cpu \
  --out_dir runs/mujoco_recap_local

# GPU (full experiment)
uv run python experiments/ablation_mujoco_recap.py \
  --checkpoint Gueso/hf_smolvla_recordpolicy0 \
  --n_iters 5 --n_rollouts 30 --vf_epochs 50 --ft_epochs 10 \
  --out_dir runs/mujoco_recap
```

This will:
1. Download the BC checkpoint from HuggingFace
2. Measure the BC baseline success rate (~50% with cube randomization)
3. Run RECAP iterations for **sparse** and **dense** reward
4. Save plots and a JSON results file

---

## Running Tests

```bash
# Fast tests (no GPU, no HuggingFace download)
uv run python -m pytest tests/ -m "not slow and not integration" -q

# MuJoCo-specific smoke tests only
uv run python -m pytest tests/smoke/test_mujoco_recap.py -v
```

---

## End-to-End Workflow

```
1. Record demos        -->  scripts/record_sim.py
   (leader arm + sim)       Saves LeRobot dataset to HF Hub

2. Train BC policy     -->  lerobot.scripts.train
   (supervised on demos)    Saves checkpoint to HF Hub

3. Evaluate BC         -->  scripts/eval_sim.py
   (baseline SR)            Runs in MuJoCo viewer or saves video

4. RECAP fine-tuning   -->  experiments/ablation_mujoco_recap.py
   (advantage-weighted)     Iterative improvement over BC baseline
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

| Setting | Value |
|---|---|
| Observation | Joint positions, 6-D float32 |
| Action | Target joint positions, 6-D float32 |
| Cameras | `front_camera` 480x640x3, `wrist_camera` 480x640x3 |
| Max steps | 750 (25 s at 30 Hz) |
| Success threshold | Cube within 3 cm horizontal + 0-5 cm vertical of bin site |
| BC baseline SR | ~50% with 3-position randomization |

### Robot Joints

| Joint | Range (normalized) | Description |
|-------|-------------------|-------------|
| `shoulder_pan` | [-100, 100] | Base rotation |
| `shoulder_lift` | [-100, 100] | Shoulder pitch |
| `elbow_flex` | [-100, 100] | Elbow pitch |
| `wrist_flex` | [-100, 100] | Wrist pitch |
| `wrist_roll` | [-100, 100] | Wrist roll |
| `gripper` | [0, 100] | Gripper open/close |

### Cube Randomization

The environment cycles through 3 positions every episode to test generalization:

| Index | X (m) | Y (m) |
|---|---|---|
| 0 | -0.10 | -0.05 |
| 1 | -0.13 | -0.05 |
| 2 | -0.10 | -0.02 |

---

## Troubleshooting

**`ImportError: No module named 'smolvla_recap'`**
- Run `uv sync` or `pip install -e .` from the repo root.

**`ModuleNotFoundError: lerobot`**
- Run `pip install "lerobot>=0.5.1"`.

**MuJoCo segfault on headless server**
- Set `headless=True` (default) in `MuJoCoGymWrapper`. MuJoCo 3.x renders off-screen without a display.

**HuggingFace download slow on cluster**
- Set `export HF_HOME=/tmp/$USER/hf` to use fast local storage.

**Out of GPU memory during fine-tuning**
- Reduce `--n_rollouts` or `--ft_epochs`. SmolVLA is ~450M params; 16 GB VRAM is comfortable with batch size 1.
