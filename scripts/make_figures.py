"""Render the README hero figure (light and dark variants) from synthetic data.

    python -m picg.benchmark --out runs/main   # optional; then pass --predictions runs/main/predictions.csv
    python scripts/make_figures.py

Panels:
  (a) the depth-front density knots on s <= 1 and the PCHIP density through them,
  (b) the integrated, monotone crack-extension curve against the physics reference,
  (c) blind outer-fold parity plot, physics model vs hybrid decoder.

(a) and (b) come from one decoder network trained on the training part of outer
fold 1 and evaluated on a blind specimen of that fold. Everything is synthetic.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from picg.cv import fit_transform_features, outer_splits  # noqa: E402
from picg.data import build_dataset, feature_columns, generate_specimen_table  # noqa: E402
from picg.decoder import decode_extension_curves_tf, monotone_cubic_interpolate_tf  # noqa: E402
from picg.model import TrainConfig, _batch, train_model  # noqa: E402
from picg.physics import transition_extension  # noqa: E402

THEMES = {
    "light": dict(bg="#FFFFFF", text="#142E46", grid="#E3E8EE", accent="#087E85", second="#4F6F94", warn="#9A5B10"),
    "dark": dict(bg="#0D1117", text="#E6EDF3", grid="#2A313C", accent="#43BEC4", second="#8FA8C8", warn="#D9A35B"),
}


def decoder_example(seed: int = 2026) -> dict[str, np.ndarray]:
    """Train on outer fold 1 and decode the depth front of one blind specimen."""

    dataset = build_dataset(generate_specimen_table(56, seed))
    numeric, categorical = feature_columns()
    train_idx, test_idx = outer_splits(len(dataset.cycles), 5, seed)[0]
    x_tr, x_te = fit_transform_features(dataset.table, train_idx, test_idx, numeric, categorical)
    config = TrainConfig(max_epochs=60)  # fixed budget; the benchmark chooses epochs on inner splits
    model = train_model(_batch(dataset, x_tr, train_idx), dataset.event_grid, config, seed=seed + 1)

    # Blind specimen whose physics crossing is closest to the fold median.
    crossings = dataset.physics_crossing[test_idx]
    pick = int(np.argmin(np.abs(crossings - np.median(crossings))))
    specimen = test_idx[pick]

    grid = dataset.event_grid
    knot_grid = np.asarray(config.decoder.knots, dtype=np.float32)
    density_a, density_c, ratio_raw, _ = model(x_te[pick : pick + 1], training=False)
    ratio = config.decoder.min_terminal_ratio + (
        config.decoder.max_terminal_ratio - config.decoder.min_terminal_ratio
    ) * tf.nn.sigmoid(tf.reshape(ratio_raw, [-1]))
    delta_a, _ = decode_extension_curves_tf(density_a, density_c, ratio, knot_grid, grid, transition_extension())
    knots = tf.nn.softplus(density_a) + 1e-8
    density = monotone_cubic_interpolate_tf(knot_grid, grid, knots).numpy()[0]
    i1 = int(np.argmin(np.abs(grid - 1.0)))
    d, g = density[: i1 + 1], grid[: i1 + 1]
    mass_at_1 = float(np.sum(0.5 * (d[1:] + d[:-1]) * np.diff(g)))

    # Only s <= 1 is shown: the short post-transition tail is not seen by any
    # loss term (see picg/decoder.py), so it carries no information.
    window = slice(0, i1 + 1)
    reference = dataset.delta_a[specimen, window] / dataset.delta_a[specimen, i1]
    in_window = knot_grid <= 1.0 + 1e-6
    return {
        "grid": grid[window],
        "knot_grid": knot_grid[in_window],
        "knots": knots.numpy()[0][in_window] / mass_at_1,
        "density": density[window] / mass_at_1,
        "reference_density": np.gradient(reference, grid[window]),
        "decoded": delta_a.numpy()[0][window] / transition_extension(),
        "reference": reference,
        "name": dataset.table.iloc[specimen]["specimen"],
    }


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["text"])
    ax.tick_params(colors=theme["text"], labelsize=10)
    ax.grid(True, color=theme["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(theme["text"])
    ax.yaxis.label.set_color(theme["text"])


def render(example: dict, predictions: pd.DataFrame, theme: dict, path: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), facecolor=theme["bg"])
    for ax in axes:
        style_axes(ax, theme)
    legend_kw = dict(frameon=False, labelcolor=theme["text"])

    ax = axes[0]
    ax.plot(example["grid"], example["reference_density"], color=theme["second"], lw=1.6, ls="--", label="physics reference")
    ax.plot(example["grid"], example["density"], color=theme["accent"], lw=2.0, label="decoded density (PCHIP)")
    ax.plot(example["knot_grid"], example["knots"], "o", color=theme["accent"], ms=6, mec=theme["bg"], mew=1.2, label="density knots")
    ax.set_xlabel("event coordinate s  (s = 1: physics crossing)")
    ax.set_ylabel("growth density  d(Δa / Δa$_{tr}$) / ds")
    ax.set_title("(a) density knots → PCHIP", loc="left", color=theme["text"])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", **legend_kw)

    ax = axes[1]
    ax.plot(example["grid"], example["reference"], color=theme["second"], lw=1.6, ls="--", label="physics reference")
    ax.plot(example["grid"], example["decoded"], color=theme["accent"], lw=2.0, label="decoded (monotone)")
    ax.set_xlabel("event coordinate s")
    ax.set_ylabel("crack extension  Δa / Δa$_{tr}$  (depth)")
    ax.set_title(f"(b) integrated curve, blind specimen {example['name']}", loc="left", color=theme["text"])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", **legend_kw)

    ax = axes[2]
    observed = predictions["observed_cycles"].to_numpy()
    lo = 0.6 * min(observed.min(), predictions[["physics_cycles", "hybrid_cycles"]].min().min())
    hi = 1.6 * max(observed.max(), predictions[["physics_cycles", "hybrid_cycles"]].max().max())
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=theme["warn"], alpha=0.10, lw=0)
    ax.text(lo * 1.4, hi / 1.6, "late (non-conservative)", color=theme["warn"], fontsize=9, va="top")
    ax.plot([lo, hi], [lo, hi], color=theme["text"], lw=0.9, ls="--", alpha=0.7)
    ax.scatter(observed, predictions["physics_cycles"], s=26, facecolors="none", edgecolors=theme["second"], lw=1.2, label="physics model")
    ax.scatter(observed, predictions["hybrid_cycles"], s=24, color=theme["accent"], edgecolors=theme["bg"], lw=0.5, label="hybrid decoder")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("observed cycles to crack nucleation")
    ax.set_ylabel("predicted cycles")
    ax.set_title("(c) blind outer folds, synthetic data", loc="left", color=theme["text"])
    ax.legend(loc="lower right", **legend_kw)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, dpi=150, facecolor=theme["bg"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=ROOT / "examples" / "synthetic_blind_predictions.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    if not args.predictions.exists():
        raise SystemExit(f"{args.predictions} not found; run `python -m picg.run --out {args.predictions}` first.")
    predictions = pd.read_csv(args.predictions)
    example = decoder_example()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        path = args.out_dir / f"hero_{name}.png"
        render(example, predictions, theme, path)
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
