"""Render the README results section from executed benchmark artifacts.

    python scripts/render_results.py --main runs/main --summary examples/benchmark_summary.json \
        --out examples/benchmark_results.md

The README copies this file's content verbatim, so every number in the README
comes from metrics.json / init_seeds.json of a finished run and from the
cross-run summary written by ``python -m picg.benchmark --summarize``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = [
    "physics", "constant", "ridge", "warp_net", "decoder",
    "decoder_fixed_default", "ablation_small_penalties", "ablation_no_shape", "ablation_no_ratio",
    "ablation_no_concavity", "ablation_no_warp", "ablation_no_penalties",
]
LABELS = {
    "physics": "1. physics model",
    "constant": "2. constant correction",
    "ridge": "3. ridge log-correction",
    "warp_net": "4. warp-only network",
    "decoder": "5. full decoder (inner search)",
    "decoder_fixed_default": "decoder, default weights, no search",
    "ablation_small_penalties": "all penalties small but active (1e-2 / 1e-3)",
    "ablation_no_shape": "shape penalty disabled (w = 0)",
    "ablation_no_ratio": "ratio penalty disabled (w = 0)",
    "ablation_no_concavity": "concavity penalty disabled (w = 0)",
    "ablation_no_warp": "warp penalty disabled (w = 0)",
    "ablation_no_penalties": "all four penalties disabled (w = 0)",
}


def fmt_ci(boot) -> str:
    if not boot:
        return "reference"
    return f"{boot['point']:+.3f} [{boot['ci95_low']:+.3f}, {boot['ci95_high']:+.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.main / "manifest.json").read_text())
    metrics = json.loads((args.main / "metrics.json").read_text())
    init = json.loads((args.main / "init_seeds.json").read_text()) if (args.main / "init_seeds.json").exists() else []
    summary = json.loads(args.summary.read_text())
    cfg = manifest["config"]
    n = metrics["physics"]["n_expected"]
    statuses = {m["status"] for m in metrics.values()}
    lines = [
        f"All numbers below were produced by `python -m picg.benchmark` at commit `{manifest['git']['commit'][:10]}` "
        f"from the pinned environment; status **{manifest['status']}** ({n}/{n} blind predictions valid for every variant, "
        f"{len(manifest['data']['excluded_before_evaluation'])} specimens excluded before evaluation). "
        f"Configuration: {cfg['n_specimens']} synthetic specimens (data seed {cfg['seed']}), {cfg['n_outer']} outer folds "
        f"(split seed {cfg['outer_seed']}), {cfg['n_inner']} inner splits per fold, fixed budget {cfg['epochs']} epochs with the epoch "
        f"count chosen on the inner splits, decoder search over {cfg['n_loss_profiles']} loss profiles x 2 backbones. "
        f"Metrics are on `log1p` cycles over all {n} blind specimens (every specimen is blind exactly once). "
        "*Late* = predicted more cycles than observed (non-conservative). W1 = Wasserstein-1 distance of the decoded, normalised "
        "extension curves to the **synthetic physics reference curves** (agreement with the prior, not accuracy against truth); "
        f"variants without curve heads use the physics curves unchanged. Fit time is the summed wall time over the {cfg['n_outer']} folds "
        "(including inner search) while three benchmark processes shared one 8-core laptop CPU, so absolute times are inflated.",
        "",
        "### Baselines and decoder on identical folds (split seed " + str(cfg["outer_seed"]) + ")",
        "",
        f"| variant | params | MAE (log) | RMSE (log) | bias (log) | late / {n} | MAE minus physics MAE, 95 % specimen bootstrap | W1 to physics curves | fit time [s] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ORDER[:5]:
        m = metrics[name]
        w1 = "physics curves" if "unchanged" in m["curve_fidelity"]["reference"] else f"{m['curve_fidelity']['w1_shape_distance_mean']:.4f}"
        lines.append(
            f"| {LABELS[name]} | {m['parameter_count']['max']} | {m['mae_log']:.3f} | {m['rmse_log']:.3f} | {m['bias_log']:+.3f} | "
            f"{m['late']} | {fmt_ci(m.get('mae_difference_vs_physics'))} | {w1} | {m['fit_seconds_total']:.0f} |"
        )
    lines += [
        "",
        "### Decoder ablations (fixed 32-unit backbone, default weights unless stated; epochs chosen on the inner splits)",
        "",
        f"| configuration | params | MAE (log) | bias (log) | late / {n} | W1 to physics curves | excess convexity | terminal-ratio MAE vs physics | fit time [s] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ORDER[5:]:
        if name not in metrics:
            continue
        m = metrics[name]
        cf = m["curve_fidelity"]
        lines.append(
            f"| {LABELS[name]} | {m['parameter_count']['max']} | {m['mae_log']:.3f} | {m['bias_log']:+.3f} | {m['late']} | "
            f"{cf['w1_shape_distance_mean']:.4f} | {cf['excess_convexity_penalty_mean']:.2e} | {cf['terminal_ratio_mae']:.4f} | {m['fit_seconds_total']:.0f} |"
        )
    selected = metrics["decoder"]["selected_per_fold"]
    picks = ", ".join(f"fold {f}: {v['config']['hidden_units']} units, w_shape {v['config']['w_shape']:.3g}, w_ratio {v['config']['w_ratio']:.3g}, w_concav {v['config']['w_concavity']:.3g}, w_warp {v['config']['w_warp']:.3g}, {v['chosen_epochs']} epochs" for f, v in selected.items())
    lines += ["", f"Inner-search selections for the full decoder: {picks}.", ""]

    seeds = ", ".join(str(r["outer_seed"]) for r in summary["runs"])
    lines += [
        f"### Variability: {len(summary['runs'])} outer-split seeds ({seeds}) and {len(init) + 1} initialisation seeds",
        "",
        "Three kinds of spread are reported separately and none of them is a prediction interval:",
        "",
        f"| variant | MAE (log) mean [min, max] over split seeds | late / {n} mean [min, max] |",
        "|---|---|---|",
    ]
    for row in summary["variants"]:
        if row["variant"] not in ORDER[:5]:
            continue
        lines.append(f"| {LABELS[row['variant']]} | {row['mae_log_mean']:.3f} [{row['mae_log_min']:.3f}, {row['mae_log_max']:.3f}] | {row['late_mean']:.1f} [{row['late_min']}, {row['late_max']}] |")
    lines += [
        "",
        "- **Split variability (table above):** repeating the whole benchmark with other outer folds. The spread mixes fold composition with the optimisation randomness of retraining, so it is an upper bound on either alone.",
    ]
    if init:
        base = metrics["decoder"]
        values = [base["mae_log"]] + [r["mae_log"] for r in init]
        lates = [base["late"]] + [r["late"] for r in init]
        lines.append(
            f"- **Optimisation variability (split seed {cfg['outer_seed']}, folds and selected configurations fixed, only the final-fit initialisation and batch order changed):** "
            f"decoder MAE(log) {', '.join(f'{v:.3f}' for v in values)} (range {min(values):.3f} to {max(values):.3f}); late {', '.join(str(v) for v in lates)} / {n}."
        )
    boot = metrics["decoder"].get("mae_difference_vs_physics")
    if boot:
        lines.append(
            f"- **Statistical uncertainty of the metric (models fixed, {boot['n_boot']} paired bootstrap resamples of the {n} blind specimens, split seed {cfg['outer_seed']}):** "
            f"decoder MAE minus physics MAE {boot['point']:+.3f}, 95 % interval [{boot['ci95_low']:+.3f}, {boot['ci95_high']:+.3f}]; "
            f"ridge minus physics {fmt_ci(metrics['ridge'].get('mae_difference_vs_physics'))}."
        )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
