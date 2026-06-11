"""
Run all RECAP + SmolVLA ablation experiments sequentially and write a
combined summary to --out_dir/all_results.json.

Usage (local, mock env):
    python experiments/run_all_ablations.py --env mock --policy mock

Usage (server, real SmolVLA + PushT):
    python experiments/run_all_ablations.py \
        --env pusht --policy smolvla \
        --n_iters 5 --n_rollouts 30 --vf_epochs 50 --ft_epochs 10 \
        --out_dir runs/all_ablations_pusht
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ABLATION_SCRIPTS = [
    ("sparse_vs_dense",  "experiments/ablation_sparse_vs_dense.py"),
    ("vla_scoring",      "experiments/ablation_vla_scoring.py"),
    ("curriculum",       "experiments/ablation_curriculum.py"),
]

OFFLINE_EVAL_SCRIPT = "experiments/offline_eval.py"


def _run_script(
    script: str,
    out_dir: Path,
    extra_args: list[str],
    label: str,
) -> dict:
    """Run a single ablation script as a subprocess, return its JSON results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", script, "--out_dir", str(out_dir)] + extra_args
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 60)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{label}] {status} — {elapsed:.1f}s")

    # Try to load the JSON the script wrote
    for json_file in out_dir.glob("*.json"):
        try:
            return {"label": label, "status": status, "elapsed_s": elapsed,
                    "results": json.loads(json_file.read_text())}
        except Exception:
            pass
    return {"label": label, "status": status, "elapsed_s": elapsed, "results": {}}


def main() -> None:
    p = argparse.ArgumentParser(description="Run all RECAP ablations")
    p.add_argument("--env",        choices=["mock", "pusht"], default="mock")
    p.add_argument("--policy",     choices=["mock", "smolvla"], default="mock")
    p.add_argument("--n_iters",    type=int, default=3)
    p.add_argument("--n_rollouts", type=int, default=10)
    p.add_argument("--vf_epochs",  type=int, default=20)
    p.add_argument("--ft_epochs",  type=int, default=5)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--out_dir",    default="runs/all_ablations")
    p.add_argument("--skip_offline_eval", action="store_true",
                   help="Skip the offline LeRobot dataset evaluation")
    p.add_argument("--ablations", nargs="+",
                   choices=["sparse_vs_dense", "vla_scoring", "curriculum", "all"],
                   default=["all"],
                   help="Which ablations to run (default: all)")
    args = p.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    run_all = "all" in args.ablations
    selected = set(args.ablations)

    # Args forwarded to every ablation script
    shared_args = [
        "--env",        args.env,
        "--n_iters",    str(args.n_iters),
        "--n_rollouts", str(args.n_rollouts),
        "--vf_epochs",  str(args.vf_epochs),
        "--ft_epochs",  str(args.ft_epochs),
        "--seed",       str(args.seed),
    ]

    # ablation_sparse_vs_dense also accepts --policy
    sparse_dense_args = shared_args + ["--policy", args.policy]

    all_results = []
    wall_t0 = time.perf_counter()

    for name, script in ABLATION_SCRIPTS:
        if not run_all and name not in selected:
            print(f"  Skipping {name}")
            continue
        extra = sparse_dense_args if name == "sparse_vs_dense" else shared_args
        res = _run_script(script, root / name, extra, label=name)
        all_results.append(res)

    if not args.skip_offline_eval and (run_all or "all" in selected):
        res = _run_script(
            OFFLINE_EVAL_SCRIPT,
            root / "offline_eval",
            ["--seed", str(args.seed)],
            label="offline_eval",
        )
        all_results.append(res)

    total_elapsed = time.perf_counter() - wall_t0
    summary = {
        "env":        args.env,
        "policy":     args.policy,
        "n_iters":    args.n_iters,
        "n_rollouts": args.n_rollouts,
        "total_elapsed_s": round(total_elapsed, 1),
        "ablations":  all_results,
    }

    out_path = root / "all_results.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  All ablations done in {total_elapsed / 60:.1f} min")
    print(f"  Summary → {out_path}")
    for r in all_results:
        print(f"    {r['label']:20s}  {r['status']:30s}  {r['elapsed_s']:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
