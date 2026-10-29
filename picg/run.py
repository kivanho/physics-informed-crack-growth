"""End-to-end demo: synthetic specimens -> nested CV -> metrics.

    python -m picg.run            # quick search (a few minutes on a laptop CPU)
    python -m picg.run --full     # thesis protocol: 8 candidates x 3 inner splits
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd  # noqa: E402

from .data import build_dataset, generate_specimen_table  # noqa: E402
from .metrics import cycle_metrics  # noqa: E402
from .model import NestedCVConfig, TrainConfig, run_nested_cv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, help="specimen CSV (default: generate a synthetic table)")
    parser.add_argument("--n-specimens", type=int, default=56)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--full", action="store_true", help="run the full inner search")
    parser.add_argument("--out", type=Path, help="write blind predictions to this CSV")
    args = parser.parse_args()

    start = time.time()
    table = pd.read_csv(args.table) if args.table else generate_specimen_table(args.n_specimens, args.seed)
    dataset = build_dataset(table)
    if args.full:
        cfg = NestedCVConfig(seed=args.seed)
    else:
        cfg = NestedCVConfig(
            n_inner=2,
            n_loss_profiles=2,
            inner_epochs=40,
            inner_patience=8,
            seed=args.seed,
            base=TrainConfig(max_epochs=100, patience=15),
        )
    n_candidates = 2 * cfg.n_loss_profiles
    print(
        f"{len(dataset.cycles)} synthetic specimens | {cfg.n_outer} outer folds | "
        f"{n_candidates} candidates x {cfg.n_inner} inner splits per fold"
    )
    predictions, _ = run_nested_cv(dataset, cfg)

    physics = cycle_metrics(predictions["observed_cycles"], predictions["physics_cycles"])
    hybrid = cycle_metrics(predictions["observed_cycles"], predictions["hybrid_cycles"])
    n = physics["n"]
    change = 100.0 * (hybrid["mae_log"] - physics["mae_log"]) / physics["mae_log"]
    print()
    print(f"{'model':<22}{'MAE (log)':>11}{'RMSE (log)':>12}{'late':>9}")
    print(f"{'physics baseline':<22}{physics['mae_log']:>11.3f}{physics['rmse_log']:>12.3f}{physics['late']:>6}/{n}")
    print(f"{'hybrid decoder':<22}{hybrid['mae_log']:>11.3f}{hybrid['rmse_log']:>12.3f}{hybrid['late']:>6}/{n}")
    print(f"MAE change vs physics: {change:+.1f} %   ({time.time() - start:.0f} s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.out, index=False)
        print(f"Blind predictions written to {args.out}")


if __name__ == "__main__":
    main()
