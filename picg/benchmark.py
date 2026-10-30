"""Benchmark: all model variants on identical outer folds, with artifacts.

    python -m picg.benchmark --out runs/main                       # full benchmark, one outer split seed
    python -m picg.benchmark --out runs/seed2027 --outer-seed 2027 # repeat with other outer folds
    python -m picg.benchmark --smoke --out runs/smoke               # tiny end-to-end check (CI)
    python -m picg.benchmark --summarize runs/main runs/seed2027 --out examples/benchmark_summary

Every variant (physics, constant correction, ridge correction, warp-only
network, full decoder, decoder ablations) sees the same outer folds and the
same inner splits. All fitting and all hyper-parameter / epoch selection
happen inside the outer training set.

The run directory receives: manifest.json (commit, environment, data
fingerprint, seed, full configuration, validity), folds.json (specimen ids per
fold and per inner split), predictions.csv (one column per variant, one row
per blind specimen), metrics.json, inner_search.csv, histories.json,
init_seeds.json (if requested) and summary.md.

Exit status: 0 when every variant produced a complete, valid prediction set;
2 when any variant is INVALID; 1 on a numerical failure (NumericalError).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")  # one tf.function per fitted network is intended; hide retracing warnings

from . import losses  # noqa: E402
from .baselines import (  # noqa: E402
    ConstantCorrectionVariant,
    DecoderVariant,
    PhysicsVariant,
    RidgeCorrectionVariant,
    Variant,
    WarpOnlyVariant,
    ablation_variants,
)
from .cv import inner_splits, outer_splits  # noqa: E402
from .data import TARGET, build_dataset, generate_specimen_table  # noqa: E402
from .decoder import threshold_from_extension_tf  # noqa: E402
from .manifest import data_fingerprint, environment, git_state, timestamp, write_json  # noqa: E402
from .metrics import STATUS_VALID, cycle_metrics, paired_bootstrap_mae_difference  # noqa: E402
from .model import NestedCVConfig, NumericalError, TrainConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALL_VARIANTS = ("physics", "constant", "ridge", "warp_net", "decoder")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="run directory (created); with --summarize, output prefix")
    parser.add_argument("--table", type=Path, help="specimen CSV (default: generate a synthetic table)")
    parser.add_argument("--n-specimens", type=int, default=56)
    parser.add_argument("--seed", type=int, default=2026, help="data generation seed and training seed base")
    parser.add_argument("--outer-seed", type=int, default=None, help="outer split seed (default: --seed)")
    parser.add_argument("--n-outer", type=int, default=5)
    parser.add_argument("--n-inner", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100, help="fixed training budget; epochs chosen on inner splits")
    parser.add_argument("--n-loss-profiles", type=int, default=4, help="decoder search: loss-weight profiles (x 2 backbones)")
    parser.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS), choices=ALL_VARIANTS)
    parser.add_argument("--ablations", dest="ablations", action="store_true", default=True)
    parser.add_argument("--no-ablations", dest="ablations", action="store_false")
    parser.add_argument("--init-seeds", type=int, nargs="*", default=[], help="extra init-seed offsets for the decoder final fits")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end configuration for CI")
    parser.add_argument("--summarize", nargs="+", type=Path, help="summarise finished run directories instead of running")
    args = parser.parse_args(argv)
    if args.smoke:
        args.n_specimens = min(args.n_specimens, 20)
        args.n_outer, args.n_inner, args.epochs, args.n_loss_profiles = 2, 1, 3, 1
        args.init_seeds = args.init_seeds or [1]
        args.n_boot = 200
    if args.outer_seed is None:
        args.outer_seed = args.seed
    return args


def build_variants(args) -> list[Variant]:
    base = TrainConfig(max_epochs=args.epochs)
    cv = NestedCVConfig(
        n_outer=args.n_outer, n_inner=args.n_inner, n_loss_profiles=args.n_loss_profiles,
        include_small_penalties=args.n_loss_profiles >= 2, epochs=args.epochs, seed=args.seed, base=base,
    )
    registry = {
        "physics": lambda: PhysicsVariant(),
        "constant": lambda: ConstantCorrectionVariant(),
        "ridge": lambda: RidgeCorrectionVariant(),
        "warp_net": lambda: WarpOnlyVariant(epochs=args.epochs),
        "decoder": lambda: DecoderVariant(cv),
    }
    variants = [registry[name]() for name in ALL_VARIANTS if name in args.variants]
    if args.ablations:
        ablations = ablation_variants(base, cv)
        if args.smoke:
            ablations = [v for v in ablations if v.name in ("ablation_no_penalties",)]
        variants.extend(ablations)
    return variants


def curve_fidelity(dataset, test_indices: np.ndarray, delta_a: np.ndarray, delta_c: np.ndarray) -> dict:
    """Fidelity of decoded curves to the *synthetic physics reference* curves.

    The references are outputs of the stand-in physics model, not measured
    crack-growth curves, so these numbers measure agreement with the physics
    prior, not accuracy against truth.
    """

    grid = tf.constant(dataset.event_grid)
    ref_a, ref_c = tf.constant(dataset.delta_a[test_indices]), tf.constant(dataset.delta_c[test_indices])
    pa, pc = tf.constant(delta_a, tf.float32), tf.constant(delta_c, tf.float32)
    th = [threshold_from_extension_tf(t) for t in (pa, pc, ref_a, ref_c)]
    i1 = int(np.argmin(np.abs(dataset.event_grid - 1.0)))
    ratio_pred = delta_c[:, i1] / np.maximum(delta_a[:, i1], 1e-12)
    ratio_ref = dataset.delta_c[test_indices, i1] / np.maximum(dataset.delta_a[test_indices, i1], 1e-12)
    return {
        "reference": "synthetic physics model curves (not measured truth)",
        "w1_shape_distance_mean": float(losses.shape_loss(pa, pc, ref_a, ref_c, grid)),
        "excess_convexity_penalty_mean": float(losses.threshold_concavity_loss(th[0], th[1], th[2], th[3], grid)),
        "terminal_ratio_mae": float(np.mean(np.abs(ratio_pred - ratio_ref))),
        "monotone_fraction": float(np.mean(np.all(np.diff(delta_a, axis=1) >= -1e-9, axis=1) & np.all(np.diff(delta_c, axis=1) >= -1e-9, axis=1))),
        "n_curves": int(len(test_indices)),
    }


def run_benchmark(args) -> int:
    start = time.perf_counter()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    raw_table = pd.read_csv(args.table) if args.table else generate_specimen_table(args.n_specimens, args.seed)
    dataset = build_dataset(raw_table)
    kept = set(dataset.table["specimen"])
    excluded = [
        {"specimen": s, "reason": "physics model found no transition within the cycle limit (non-propagating defect)"}
        for s in raw_table["specimen"] if s not in kept
    ]
    n = len(dataset.cycles)
    folds = outer_splits(n, args.n_outer, args.outer_seed)
    variants = build_variants(args)
    print(
        f"{n} specimens ({len(excluded)} excluded before evaluation) | {args.n_outer} outer folds (seed {args.outer_seed}) | "
        f"{args.n_inner} inner splits | budget {args.epochs} epochs | variants: {', '.join(v.name for v in variants)}"
    )

    specimen_ids = dataset.table["specimen"].to_numpy()
    pred_cols = {v.name: np.full(n, np.nan) for v in variants}
    fold_of = np.zeros(n, dtype=int)
    selected, inner_tables, histories, fold_seconds, param_counts = {}, [], [], {}, {}
    curves = {v.name: {"delta_a": np.zeros_like(dataset.delta_a), "delta_c": np.zeros_like(dataset.delta_c)} for v in variants if v.decodes_curves}
    searches = {}  # (variant, fold) -> inner search result, for init-seed reruns
    fold_records = []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        fold_of[test_idx] = fold
        splits = inner_splits(train_idx, args.n_inner, seed=args.outer_seed + fold)
        fold_records.append(
            {
                "outer_fold": fold,
                "train_specimens": specimen_ids[train_idx].tolist(),
                "test_specimens": specimen_ids[test_idx].tolist(),
                "inner_splits": [{"train": specimen_ids[a].tolist(), "validation": specimen_ids[b].tolist()} for a, b in splits],
            }
        )
        for variant in variants:
            result = variant.fit_predict(dataset, train_idx, test_idx, splits, fold, args.seed)
            pred_cols[variant.name][test_idx] = result.pred
            selected.setdefault(variant.name, {})[fold] = {
                "config": result.selected, "chosen_epochs": result.chosen_epochs, "parameter_count": result.parameter_count,
            }
            param_counts.setdefault(variant.name, []).append(result.parameter_count)
            fold_seconds.setdefault(variant.name, []).append(result.seconds)
            if result.inner_scores is not None:
                inner_tables.append(result.inner_scores.assign(variant=variant.name))
            for h in result.histories:
                histories.append({"variant": variant.name, **h})
            if result.curves is not None:
                curves[variant.name]["delta_a"][test_idx] = result.curves["delta_a"]
                curves[variant.name]["delta_c"][test_idx] = result.curves["delta_c"]
            if result.search is not None:
                searches[(variant.name, fold)] = result.search
            fold_metrics = cycle_metrics(dataset.cycles[test_idx], result.pred, specimen_ids[test_idx], on_invalid="mark")
            mae = fold_metrics["mae_log"]
            print(f"  fold {fold}/{args.n_outer} {variant.name:<26} MAE(log) {mae:7.3f}  [{fold_metrics['status']}]  {result.seconds:6.1f} s", flush=True)

    predictions = pd.DataFrame({"specimen": specimen_ids, "outer_fold": fold_of, "observed_cycles": dataset.cycles})
    for name, values in pred_cols.items():
        predictions[f"{name}_cycles"] = values
    predictions.to_csv(out / "predictions.csv", index=False)

    metrics = {}
    for variant in variants:
        m = cycle_metrics(dataset.cycles, pred_cols[variant.name], specimen_ids, on_invalid="mark")
        m.update(
            {
                "description": variant.description,
                "parameter_count": {"per_fold": param_counts[variant.name], "max": int(max(param_counts[variant.name]))},
                "fit_seconds_total": float(sum(fold_seconds[variant.name])),
                "selected_per_fold": selected[variant.name],
            }
        )
        if variant.decodes_curves:
            m["curve_fidelity"] = curve_fidelity(dataset, np.arange(n), curves[variant.name]["delta_a"], curves[variant.name]["delta_c"])
        else:
            m["curve_fidelity"] = {"reference": "uses the physics reference curves unchanged", "w1_shape_distance_mean": 0.0}
        if variant.name != "physics" and m["status"] == STATUS_VALID and "physics" in pred_cols:
            m["mae_difference_vs_physics"] = paired_bootstrap_mae_difference(
                dataset.cycles, pred_cols[variant.name], pred_cols["physics"], n_boot=args.n_boot, seed=args.seed
            )
        metrics[variant.name] = m

    init_seed_results = []
    if args.init_seeds and "decoder" in pred_cols:
        decoder = next(v for v in variants if v.name == "decoder")
        for offset in args.init_seeds:
            pred = np.full(n, np.nan)
            for fold, (train_idx, test_idx) in enumerate(folds, start=1):
                result = decoder.fit_predict(dataset, train_idx, test_idx, None, fold, args.seed, init_seed_offset=offset, search=searches[("decoder", fold)])
                pred[test_idx] = result.pred
            m = cycle_metrics(dataset.cycles, pred, specimen_ids, on_invalid="mark")
            init_seed_results.append({"init_seed_offset": offset, **{k: m[k] for k in ("status", "mae_log", "rmse_log", "bias_log", "late", "n_expected", "n_valid", "invalid_specimens")}})
            predictions[f"decoder_initseed{offset}_cycles"] = pred
            print(f"  init-seed offset {offset}: decoder MAE(log) {m['mae_log']:.3f} [{m['status']}]", flush=True)
        predictions.to_csv(out / "predictions.csv", index=False)

    overall_status = STATUS_VALID if all(m["status"] == STATUS_VALID for m in metrics.values()) else "INVALID"
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "summarize"}
    manifest = {
        "created": timestamp(),
        "git": git_state(ROOT),
        "environment": environment(),
        "data": {
            "source": str(args.table) if args.table else f"generate_specimen_table(n={args.n_specimens}, seed={args.seed})",
            "fingerprint_sha256": data_fingerprint(raw_table),
            "n_rows": int(len(raw_table)),
            "n_evaluated": n,
            "excluded_before_evaluation": excluded,
            "target": TARGET,
        },
        "config": config,
        "budget_strategy": "fixed epoch budget; epoch count chosen per candidate by mean inner-validation MAE, final fit uses that count",
        "variants": {v.name: v.description for v in variants},
        "status": overall_status,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "folds.json", fold_records)
    write_json(out / "metrics.json", metrics)
    write_json(out / "histories.json", histories)
    if init_seed_results:
        write_json(out / "init_seeds.json", init_seed_results)
    if inner_tables:
        pd.concat(inner_tables, ignore_index=True).to_csv(out / "inner_search.csv", index=False)
    (out / "summary.md").write_text(render_summary(manifest, metrics, init_seed_results))
    print()
    print((out / "summary.md").read_text())
    print(f"status {overall_status}; artifacts in {out} ({manifest['elapsed_seconds']:.0f} s)")
    return 0 if overall_status == STATUS_VALID else 2


def render_summary(manifest: dict, metrics: dict, init_seed_results: list[dict]) -> str:
    n = next(iter(metrics.values()))["n_expected"]
    lines = [
        f"# Benchmark summary ({manifest['status']})",
        "",
        f"{manifest['data']['n_evaluated']} evaluated specimens, {manifest['config']['n_outer']} outer folds (seed {manifest['config']['outer_seed']}), "
        f"{manifest['config']['n_inner']} inner splits, budget {manifest['config']['epochs']} epochs. Commit {manifest['git']['commit']}.",
        "",
        f"| variant | params | MAE (log) | RMSE (log) | bias (log) | late / {n} | W1 to physics curves | fit time [s] | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        late = f"{m['late']}" if m["late"] is not None else "n/a"
        w1 = m["curve_fidelity"].get("w1_shape_distance_mean")
        w1_text = "physics curves" if "unchanged" in m["curve_fidelity"]["reference"] else f"{w1:.4f}"
        lines.append(
            f"| {name} | {m['parameter_count']['max']} | {m['mae_log']:.3f} | {m['rmse_log']:.3f} | {m['bias_log']:+.3f} | {late} | {w1_text} | {m['fit_seconds_total']:.0f} | {m['status']} |"
        )
    if init_seed_results:
        lines += ["", "Initialisation sensitivity (decoder final fits, same folds and selections):", ""]
        for r in init_seed_results:
            lines.append(f"- init-seed offset {r['init_seed_offset']}: MAE {r['mae_log']:.3f}, late {r['late']}/{r['n_expected']} [{r['status']}]")
    return "\n".join(lines) + "\n"


def summarize(run_dirs: list[Path], out_prefix: Path) -> None:
    """Aggregate several finished runs (for example different outer-split seeds)."""

    runs = []
    for d in run_dirs:
        manifest = json.loads((d / "manifest.json").read_text())
        metrics = json.loads((d / "metrics.json").read_text())
        init = json.loads((d / "init_seeds.json").read_text()) if (d / "init_seeds.json").exists() else []
        runs.append({"dir": str(d), "manifest": manifest, "metrics": metrics, "init_seeds": init})
    variants = list(runs[0]["metrics"])
    rows = []
    for name in variants:
        maes = [r["metrics"][name]["mae_log"] for r in runs if name in r["metrics"]]
        lates = [r["metrics"][name]["late"] for r in runs if name in r["metrics"]]
        statuses = [r["metrics"][name]["status"] for r in runs if name in r["metrics"]]
        first = runs[0]["metrics"][name]
        rows.append(
            {
                "variant": name,
                "parameter_count": first["parameter_count"]["max"],
                "n_runs": len(maes),
                "mae_log_mean": float(np.mean(maes)),
                "mae_log_min": float(np.min(maes)),
                "mae_log_max": float(np.max(maes)),
                "rmse_log_mean": float(np.mean([r["metrics"][name]["rmse_log"] for r in runs if name in r["metrics"]])),
                "bias_log_mean": float(np.mean([r["metrics"][name]["bias_log"] for r in runs if name in r["metrics"]])),
                "late_mean": float(np.mean(lates)),
                "late_min": int(np.min(lates)),
                "late_max": int(np.max(lates)),
                "n_expected": first["n_expected"],
                "w1_shape_distance_mean": first["curve_fidelity"].get("w1_shape_distance_mean"),
                "fit_seconds_mean": float(np.mean([r["metrics"][name]["fit_seconds_total"] for r in runs if name in r["metrics"]])),
                "status": STATUS_VALID if all(s == STATUS_VALID for s in statuses) else "INVALID",
                "bootstrap_vs_physics_first_run": first.get("mae_difference_vs_physics"),
            }
        )
    init_rows = [{"run": r["dir"], **row} for r in runs for row in r["init_seeds"]]
    payload = {
        "created": timestamp(),
        "runs": [{"dir": r["dir"], "outer_seed": r["manifest"]["config"]["outer_seed"], "commit": r["manifest"]["git"]["commit"], "status": r["manifest"]["status"], "elapsed_seconds": r["manifest"]["elapsed_seconds"], "environment": r["manifest"]["environment"]} for r in runs],
        "config": {k: runs[0]["manifest"]["config"][k] for k in ("n_specimens", "n_outer", "n_inner", "epochs", "n_loss_profiles", "seed")},
        "variants": rows,
        "init_seed_runs": init_rows,
        "notes": [
            "Across-run spread mixes outer-split variability with optimisation variability (different folds change training sets and seeds).",
            "init_seed_runs vary only the final-fit initialisation and batch order on fixed folds and fixed selected configurations.",
            "bootstrap_vs_physics_first_run resamples blind specimens with the models fixed (statistical uncertainty of the metric), first run only.",
            "W1 shape distance is measured against synthetic physics reference curves, not measured crack-growth curves.",
        ],
    }
    write_json(out_prefix.with_suffix(".json"), payload)
    seeds = ", ".join(str(r["outer_seed"]) for r in payload["runs"])
    n = rows[0]["n_expected"]
    lines = [
        f"| variant | params | MAE (log) mean [min, max] over {len(runs)} outer-split seeds ({seeds}) | bias (log) | late / {n} mean [min, max] | fit time [s] | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['parameter_count']} | {r['mae_log_mean']:.3f} [{r['mae_log_min']:.3f}, {r['mae_log_max']:.3f}] | {r['bias_log_mean']:+.3f} | "
            f"{r['late_mean']:.1f} [{r['late_min']}, {r['late_max']}] | {r['fit_seconds_mean']:.0f} | {r['status']} |"
        )
    if init_rows:
        lines += ["", "Initialisation sensitivity (decoder, fixed folds and selections):", ""]
        for row in init_rows:
            lines.append(f"- {Path(row['run']).name}, init-seed offset {row['init_seed_offset']}: MAE {row['mae_log']:.3f}, late {row['late']}/{row['n_expected']} [{row['status']}]")
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.summarize:
        summarize(args.summarize, args.out)
        return 0
    try:
        return run_benchmark(args)
    except NumericalError as exc:
        print(f"NUMERICAL FAILURE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
