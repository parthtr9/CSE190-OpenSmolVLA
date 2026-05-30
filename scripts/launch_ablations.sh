#!/usr/bin/env bash
# -----------------------------------------------------------------------
# launch_ablations.sh
#
# Submit all RECAP + SmolVLA ablation experiments to the cluster.
# Runs each ablation as an independent job so they can execute in parallel.
#
# Usage:
#   # Parallel jobs (default — fastest on cluster):
#   bash scripts/launch_ablations.sh
#
#   # Or run everything sequentially in one job:
#   bash scripts/launch_ablations.sh --sequential
#
# Adjust the variables in the CONFIG section below for your environment.
# -----------------------------------------------------------------------

set -euo pipefail

# -----------------------------------------------------------------------
# CONFIG — edit these for your server
# -----------------------------------------------------------------------
REPO="/home/prs007/CSE190-OpenSmolVLA"
CONDA_ENV="/home/prs007/.conda/envs/cse190_final"
PYTHON="${CONDA_ENV}/bin/python"
RUN_ROOT="${REPO}/runs"

ENV="pusht"           # mock | pusht
POLICY="smolvla"      # mock | smolvla
N_ITERS=5
N_ROLLOUTS=30
VF_EPOCHS=50
FT_EPOCHS=10
SEED=0

# launch.sh resource flags
MEM=64              # GB RAM
CPUS=10
GPUS=1
PRIORITY="low"      # low | normal | high
TIMEOUT=2419200     # 28 days in seconds
# -----------------------------------------------------------------------

SEQUENTIAL=false
if [[ "${1:-}" == "--sequential" ]]; then
    SEQUENTIAL=true
fi

SHARED_ARGS="--env ${ENV} --n_iters ${N_ITERS} --n_rollouts ${N_ROLLOUTS} \
  --vf_epochs ${VF_EPOCHS} --ft_epochs ${FT_EPOCHS} --seed ${SEED}"

LAUNCH="KBS_TIMEOUT=${TIMEOUT} launch.sh -m ${MEM} -c ${CPUS} -g ${GPUS} -b -p ${PRIORITY}"

# -----------------------------------------------------------------------
# Helper: build the bash -lc command for a single ablation
# -----------------------------------------------------------------------
make_cmd() {
    local script="$1"
    local out_dir="$2"
    local extra_args="${3:-}"
    echo "cd ${REPO} && ${PYTHON} -u ${script} \
  ${SHARED_ARGS} ${extra_args} \
  --out_dir ${out_dir}"
}

# -----------------------------------------------------------------------
# Ablation 1: Sparse vs Dense reward
# -----------------------------------------------------------------------
ABL1_DIR="${RUN_ROOT}/ablation_sparse_dense_${ENV}"
ABL1_CMD=$(make_cmd \
    "experiments/ablation_sparse_vs_dense.py" \
    "${ABL1_DIR}" \
    "--policy ${POLICY}")

# -----------------------------------------------------------------------
# Ablation 2: VLA self-scoring
# -----------------------------------------------------------------------
ABL2_DIR="${RUN_ROOT}/ablation_vla_scoring_${ENV}"
ABL2_CMD=$(make_cmd \
    "experiments/ablation_vla_scoring.py" \
    "${ABL2_DIR}")

# -----------------------------------------------------------------------
# Ablation 3: Curriculum learning
# -----------------------------------------------------------------------
ABL3_DIR="${RUN_ROOT}/ablation_curriculum_${ENV}"
ABL3_CMD=$(make_cmd \
    "experiments/ablation_curriculum.py" \
    "${ABL3_DIR}")

# -----------------------------------------------------------------------
# Offline evaluation (lighter — fewer resources needed)
# -----------------------------------------------------------------------
OFFLINE_DIR="${RUN_ROOT}/offline_eval"
OFFLINE_CMD="cd ${REPO} && ${PYTHON} -u experiments/offline_eval.py \
  --seed ${SEED} --out_dir ${OFFLINE_DIR}"
LAUNCH_OFFLINE="KBS_TIMEOUT=${TIMEOUT} launch.sh -m 32 -c 8 -g 1 -b -p ${PRIORITY}"

# -----------------------------------------------------------------------
# Submit
# -----------------------------------------------------------------------
if [[ "${SEQUENTIAL}" == "true" ]]; then
    echo "==> Running all ablations sequentially in one job..."
    ${LAUNCH} -- bash -lc "
        ${ABL1_CMD}
        ${ABL2_CMD}
        ${ABL3_CMD}
        ${OFFLINE_CMD}
    "
else
    echo "==> Submitting ablation jobs in parallel..."
    echo ""
    echo "[1/4] Sparse vs Dense  →  ${ABL1_DIR}"
    ${LAUNCH} -- bash -lc "${ABL1_CMD}"

    echo "[2/4] VLA Scoring      →  ${ABL2_DIR}"
    ${LAUNCH} -- bash -lc "${ABL2_CMD}"

    echo "[3/4] Curriculum       →  ${ABL3_DIR}"
    ${LAUNCH} -- bash -lc "${ABL3_CMD}"

    echo "[4/4] Offline Eval     →  ${OFFLINE_DIR}"
    ${LAUNCH_OFFLINE} -- bash -lc "${OFFLINE_CMD}"

    echo ""
    echo "==> All jobs submitted. Monitor with: watch -n 10 qstat"
fi
