# SmolVLA + RECAP

Policy-training side of a RECAP implementation on top of LeRobot's SmolVLA.

RECAP = *RL with Experience and Corrections via Advantage-conditioned Policies*
([arXiv:2511.14759](https://arxiv.org/abs/2511.14759)). This repo owns the
**SmolVLA training loop**: turning per-frame rewards into a good/bad signal and
training SmolVLA to be conditioned on it. The reward functions themselves
(VLM / state-based / learned — the research variable) are provided by teammates
behind a small interface.

## The pipeline

```
classifier(frame) -> reward r_t          # teammate fills this in (recap/classifier.py)
        |
        v
R_t = sum_k gamma^k r_{t+k}              # discounted return, per episode
        |
        v
V_phi(state) regresses R_t              # value function = actor-critic baseline
        |
        v
A_t = R_t - V(state)                    # advantage
        |
        v
label I_t = 1[A_t > epsilon]           # epsilon set so ~30% of frames are positive
        |
        v
SmolVLA trained with "Advantage: positive/negative"
prepended to the task prompt, dropped 30% of the time (classifier-free guidance)
```

The value function `V` is kept (rather than thresholding the reward directly)
because a per-frame classifier judges actions in isolation. `V` propagates the
eventual outcome backward through `R_t`, so an action that looks unremarkable but
sets up a later success still gets credit. That temporal credit assignment is the
one thing `V` buys.

## Setup

Requires Python 3.12.

```bash
cd CSE190-OpenSmolVLA
python3.12 -m venv .venv
source .venv/bin/activate
pip install "lerobot[smolvla]"          # pulls torch, transformers, smolvla deps (~several GB)
```

To pull `smolvla_base` weights and any LeRobot dataset from the Hub, log in
(the CLI is `hf`, not the deprecated `huggingface-cli`):

```bash
hf auth login                            # paste a READ token at the prompt; say "no" to git credential
```

> Tokens are secrets. Paste them only into the `hf auth login` prompt — never on a
> command line (`--token ...` leaks into shell history) and never into chat.

## Current status

| Component | File | Status |
|---|---|---|
| Reward classifier interface + stub | `recap/classifier.py` | done |
| Reward -> return -> advantage -> label | `recap/advantages.py` | done |
| Value function (state MLP) | `recap/value.py` | done |
| Unit tests | `tests/` | done (12 passing) |
| Synthetic dataset fixture | `recap/make_fixture.py` | planned |
| Value-function training script | `train_value.py` | planned |
| Advantage-token injection + CFG dropout | (SmolVLA wrapper) | planned |
| End-to-end RECAP training script | `train_recap.py` | planned |

The done modules are framework-light (NumPy / a small torch MLP) and run on CPU.
The planned scripts are the ones that load SmolVLA and tie everything together.

## The data contract (for the teammate generating data)

The training loop consumes a standard **LeRobot v3 demonstration dataset**. Each
frame has at least:

| Field | Type | Notes |
|---|---|---|
| `observation.state` | float array `(state_dim,)` | proprioception (e.g. 6 joint values) |
| `observation.images.<camera>` | video / image | one or more camera views |
| `action` | float array `(action_dim,)` | the action taken at this frame |
| `task` | string | natural-language instruction |
| `episode_index`, `frame_index`, `index` | int | bookkeeping |

Plain demos carry **no reward and no success flag** — that is expected. The reward
signal is manufactured by the classifier (below); the training loop derives
returns, advantages, and labels from it. You do **not** need to add any columns.

## Plugging in a real reward classifier

`recap/classifier.py` defines the only seam between this training loop and the
reward research. Subclass `RewardClassifier` and implement one method:

```python
from recap.classifier import RewardClassifier
import numpy as np

class MyVLMReward(RewardClassifier):
    def predict_rewards(self, frames):
        # frames: list of per-frame dicts (see data contract above)
        # return: float32 array, shape (len(frames),). Higher = better.
        return np.array([self.score(f) for f in frames], dtype=np.float32)
```

Notes:
- Absolute scale does not matter. Advantages are relative to a learned baseline
  and the good/bad cutoff is a percentile — only the *ordering* of rewards matters.
- A `StubRewardClassifier` (deterministic, no real model) is provided so the rest
  of the pipeline runs before a real reward exists.

The same training loop runs unchanged for any classifier, which is what keeps the
three-way VLM / state / learned comparison fair.

## Using the modules today

```python
import numpy as np
from recap.classifier import StubRewardClassifier
from recap.value import ValueMLP, ValueConfig
from recap.advantages import compute_advantage_labels, AdvantageConfig

# 1. rewards from a (stub) classifier
clf = StubRewardClassifier()
rewards = clf(frames)                      # frames: list of per-frame dicts

# 2. fit the value baseline on discounted returns (see train_value.py, planned)
#    ... train ValueMLP to regress returns ...
values = ...                               # V(state) per frame, shape (N,)

# 3. rewards + values -> advantages -> binary labels
out = compute_advantage_labels(
    rewards, values, episode_ids,
    AdvantageConfig(gamma=0.99, positive_fraction=0.30),
)
out["advantage_label"]                     # int8 array in {0, 1}, one per frame
```

## Running the tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

## Vanilla SmolVLA smoke test (sanity check)

This trains stock SmolVLA (no RECAP) for a few steps to confirm the environment
works. Useful on a fresh machine before touching the RECAP code.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=mps \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=2 --steps=50 --num_workers=0 \
  --output_dir=outputs/smoke_test --job_name=smolvla_smoke \
  --wandb.enable=false
```

Why these flags:
- `PYTORCH_ENABLE_MPS_FALLBACK=1` — lets ops MPS doesn't implement fall back to CPU.
- `--policy.push_to_hub=false` — required, or training aborts asking for a Hub repo.
- `--rename_map=...` — `smolvla_base` expects cameras named `camera1/2/3`; this
  maps the dataset's `top`/`wrist` onto them.
- `--num_workers=0` — avoids MPS + dataloader-multiprocessing issues on macOS.

## Hardware

A Mac (MPS) is fine for **smoke tests and unit tests** but not real training:
SmolVLA is 450M params and the published run assumes an A100. Run actual training
on a GPU. Local development here is for getting the pipeline correct, not for
producing trained policies.
```
