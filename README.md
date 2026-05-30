# SmolVLA + RECAP — SO-101 Cube-Bin Ablations

This repo applies **RECAP** (advantage-conditioned offline RL) to the SmolVLA VLA policy on Gustavo's SO-101 MuJoCo cube-bin task. Starting from the pre-trained BC checkpoint [`Gueso/hf_smolvla_recordpolicy0`](https://huggingface.co/Gueso/hf_smolvla_recordpolicy0) (~50% success rate with 3-position cube randomization), RECAP iteratively fine-tunes the policy using advantage-weighted supervision — no environment reward classifier needed.

---

## Branch map

| Branch | What's in it | Touch? |
|---|---|---|
| `main` | Gustavo's original sim + eval scripts | Never |
| `recap-scaffold` | Core RECAP algorithm (value fn, advantage, rollout, training) | Never |
| `mujoco-recap` | **Integration** — MuJoCo gym wrapper + ablation scripts + tests | Work here |

---

## Repo layout

```
src/
  recap_smolvla/          # RECAP algorithm package
    envs/
      mujoco_env.py       # MuJoCoGymWrapper — gym interface around MuJoCoFollower
    rewards.py            # sparse_reward_fn, dense_reward_fn
    value_function.py     # MLP value function + return computation
    advantage.py          # advantage calculation + binary labeling (~30% positive)
    rollout.py            # collect_rollout — gymnasium-compatible rollout collector
    training.py           # recap_training_iteration — full RECAP loop
  smolvla_recap/          # Gustavo's sim package
    env/
      mujoco_follower.py  # MuJoCoFollower — low-level SO-101 physics interface
      assets/             # so101_tabletop.xml + .stl meshes

experiments/
  ablation_mujoco_recap.py     # Main experiment: sparse vs dense on SO-101
  ablation_sparse_vs_dense.py  # General ablation (also supports --env mujoco)
  ablation_vla_scoring.py      # Direction 2: VLA self-scoring
  ablation_curriculum.py       # Direction 3: curriculum learning

tests/
  smoke/test_mujoco_recap.py   # MuJoCo env + RECAP loop smoke tests (12 tests)
  unit/                        # Advantage math, labeling, value function unit tests
  smoke/                       # Full RECAP loop smoke tests
```

---

## 1 — Installation

### Local (Gustavo's machine or your laptop)

```bash
git clone https://github.com/<org>/CSE190-OpenSmolVLA
cd CSE190-OpenSmolVLA
git checkout mujoco-recap

# Install everything (mujoco, lerobot, the recap_smolvla package itself)
pip install -e .
pip install "lerobot>=0.5.1" num2words torchvision

# Verify
python -c "from recap_smolvla import MuJoCoGymWrapper; print('OK')"
python -c "from smolvla_recap.env.mujoco_follower import MuJoCoFollower; print('OK')"
```

> **MuJoCo note:** MuJoCo 3.x is a pure-Python wheel — no license file or system install needed. `pip install mujoco` is all that's required.

### UCSD DSMLp cluster

```bash
ssh prs007@dsmlp-login.ucsd.edu
cd ~/CSE190-OpenSmolVLA
git fetch origin && git checkout mujoco-recap && git pull

conda activate cse190_final   # or whatever your env is called
pip install -e .
pip install "lerobot>=0.5.1" num2words torchvision
```

---

## 2 — Verify the environment works (no GPU needed)

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from recap_smolvla.envs.mujoco_env import MuJoCoGymWrapper
import numpy as np

env = MuJoCoGymWrapper(randomize_cube=True, max_steps=20)
obs, _ = env.reset()
print("obs shape :", obs.shape)          # (6,)

imgs = env.render()
print("cameras   :", list(imgs.keys()))  # ['front_camera', 'wrist_camera']
print("img shape :", imgs["front_camera"].shape)  # (480, 640, 3)

obs2, reward, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
print("reward    :", reward)             # -1.0 (sparse, no success)
print("success   :", info["is_success"])

env.close()
print("ALL GOOD")
EOF
```

---

## 3 — Run the test suite

```bash
# Fast tests (~30 s, no GPU, no HuggingFace download)
python -m pytest tests/ -m "not slow and not integration" -q

# MuJoCo-specific smoke tests only
python -m pytest tests/smoke/test_mujoco_recap.py -v

# Expected: 296 passed, 2 skipped
```

---

## 4 — Smoke-test the RECAP loop locally (mock policy, no GPU)

This verifies the entire pipeline — rollout → value function → advantage → labeling → fine-tune — runs end-to-end with MuJoCo in ~10 seconds:

```bash
python - <<'EOF'
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "src")

from recap_smolvla.envs.mujoco_env import MuJoCoGymWrapper
from recap_smolvla.rewards import sparse_reward_fn
from recap_smolvla.value_function import ValueFunction
from recap_smolvla.training import recap_training_iteration

class MockPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 6))
    def select_action(self, obs):
        x = obs.get("observation.state", obs.get("obs", torch.zeros(6)))
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x.astype("float32"))
        with torch.no_grad():
            return self.net(x.reshape(-1)[:6]).numpy()
    def compute_loss(self, batch):
        x = batch.get("observation.state", batch.get("obs", np.zeros(6)))
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x.astype("float32"))
        pred = self.net(x.reshape(-1)[:6])
        tgt  = torch.tensor(np.asarray(batch["action"][:6], dtype="float32"))
        return nn.functional.mse_loss(pred, tgt)

env    = MuJoCoGymWrapper(randomize_cube=True, max_steps=30)
policy = MockPolicy()
vf     = ValueFunction(obs_dim=6)

sr, policy, vf, stats = recap_training_iteration(
    policy, vf, env, sparse_reward_fn,
    n_rollouts=3, vf_epochs=5, ft_epochs=2,
    max_steps=30,
    instruction="pick up the cube and place it in the bin",
    verbose=True,
)
env.close()
print(f"\nSR={sr:.2%}  pct_positive={stats['pct_positive']:.1%}")
# Expect: pct_positive ~30% (RECAP targets top-30% advantage labeling)
EOF
```

You should see:
```
--- Collecting rollouts ---
  Cube position 0: (-0.100, -0.050)
  Cube position 1: (-0.130, -0.050)
  Cube position 2: (-0.100, -0.020)
--- Training value function (5 epochs) ---
--- Labeling trajectories ---
  30.0% of steps labeled advantage_positive
--- Fine-tuning policy (2 epochs) ---
SR=0.00%  pct_positive=30.0%
```

SR=0% is expected — the random mock policy never succeeds. The cube cycling and labeling fraction confirm RECAP is working.

---

## 5 — Run the full ablation with the HuggingFace BC checkpoint

### On your laptop (CPU, slower — good for checking it runs)

```bash
python experiments/ablation_mujoco_recap.py \
  --checkpoint Gueso/hf_smolvla_recordpolicy0 \
  --n_iters 2 --n_rollouts 5 --vf_epochs 10 --ft_epochs 3 \
  --device cpu \
  --out_dir runs/mujoco_recap_local
```

This will:
1. Download `Gueso/hf_smolvla_recordpolicy0` from HuggingFace (~first run only, cached after)
2. Measure the BC baseline success rate (~50% with cube randomization)
3. Run 2 RECAP iterations each for **sparse** and **dense** reward
4. Save plots and a JSON results file to `runs/mujoco_recap_local/`

### On the UCSD cluster (GPU — full experiment)

```bash
# From the login node:
KBS_TIMEOUT=2419200 launch.sh -m 64 -c 10 -g 1 -b -p low -- bash -lc '
  export HF_HOME=/tmp/$USER/hf
  cd ~/CSE190-OpenSmolVLA
  conda activate cse190_final
  python -u experiments/ablation_mujoco_recap.py \
    --checkpoint Gueso/hf_smolvla_recordpolicy0 \
    --n_iters 5 --n_rollouts 30 --vf_epochs 50 --ft_epochs 10 \
    --out_dir runs/mujoco_recap
'
```

`HF_HOME=/tmp/$USER/hf` puts the model cache on fast local storage instead of NFS.

### Output files

```
runs/mujoco_recap/
  mujoco_recap_results.json   # SR per iteration for sparse + dense, baseline SR
  success_curves.png          # Success rate vs RECAP iteration (sparse vs dense)
  advantage_dist.png          # Advantage histograms for each iteration
```

---

## 6 — What RECAP is doing (algorithm summary)

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

## 7 — Run the general ablation (also supports PushT / other envs)

```bash
# MuJoCo + Gueso checkpoint
python experiments/ablation_sparse_vs_dense.py \
  --env mujoco \
  --checkpoint Gueso/hf_smolvla_recordpolicy0 \
  --n_iters 3 --n_rollouts 20 \
  --out_dir runs/sparse_vs_dense_mujoco

# PushT (if gym_pusht installed)
python experiments/ablation_sparse_vs_dense.py \
  --env pusht \
  --checkpoint lerobot/smolvla_base \
  --n_iters 3 --n_rollouts 20 \
  --out_dir runs/sparse_vs_dense_pusht
```

---

## 8 — Cube randomization details

Gustavo's setup cycles through 3 positions every episode to test generalization without needing new data:

| Index | X (m) | Y (m) |
|---|---|---|
| 0 | −0.10 | −0.05 |
| 1 | −0.13 | −0.05 |
| 2 | −0.10 | −0.02 |

The `MuJoCoGymWrapper` uses episode count mod 3 to cycle deterministically — same behavior as Gustavo's `eval_sim.py --randomize-cube` with `--max-steps 750`.

---

## 9 — Environment quick reference

| Setting | Value |
|---|---|
| Observation | Joint positions, 6-D float32 |
| Action | Target joint positions, 6-D float32 |
| Cameras | `front_camera` 480×640×3, `wrist_camera` 480×640×3 |
| Max steps | 750 (25 s at 30 Hz) |
| Success threshold | Cube within 3 cm horizontal + 0–5 cm vertical of bin site |
| BC baseline SR | ~50% with 3-position randomization |

```python
from recap_smolvla import MuJoCoGymWrapper

env = MuJoCoGymWrapper(
    randomize_cube=True,   # cycle through 3 cube positions
    max_steps=750,
    reward_mode="sparse",  # or "dense"
    alpha=0.1,             # proximity bonus weight (dense only)
)
obs, info = env.reset()    # obs: (6,) float32
imgs = env.render()        # {"front_camera": (480,640,3), "wrist_camera": (480,640,3)}
obs, r, done, trunc, info = env.step(action)  # action: (6,) float32
```

---

## Troubleshooting

**`ImportError: No module named 'smolvla_recap'`**  
→ Run `pip install -e .` from the repo root on the `mujoco-recap` branch.

**`ModuleNotFoundError: lerobot`**  
→ Run `pip install "lerobot>=0.5.1"`.

**MuJoCo segfault on headless server**  
→ Set `headless=True` (default) in `MuJoCoGymWrapper`. MuJoCo 3.x renders off-screen without a display.

**HuggingFace download slow on cluster**  
→ Set `export HF_HOME=/tmp/$USER/hf` before running; this writes to fast local NVMe instead of the shared NFS home directory.

**Out of GPU memory during fine-tuning**  
→ Reduce `--n_rollouts` or `--ft_epochs`. SmolVLA is ~450M params; 16 GB VRAM is comfortable with batch size 1.
