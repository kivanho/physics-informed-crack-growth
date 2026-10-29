"""Synthetic specimen table and event-grid targets.

``generate_specimen_table`` draws a small fatigue test campaign: initial
defect sizes, stresses, geometry and defect type, plus an "observed" cycle
count to the short-to-long crack transition. The observed life is the
synthetic physics prediction times a systematic, feature-dependent correction
and log-normal scatter, which mimics a physics model that is directionally
right but biased. The hybrid network's job is to learn that correction while
keeping the curve shapes physically plausible.

Run ``python -m picg.data --out examples/synthetic_specimens.csv`` to write a
table to disk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .physics import (
    DEFAULT_MATERIAL,
    SyntheticMaterial,
    short_crack_threshold,
    simulate_specimen,
    transition_threshold,
)

TARGET = "observed_cycles"
NUMERIC_FEATURES = ["a0_mm", "c0_mm", "sigma_max", "sigma_min", "w_mm", "t_mm"]
ENGINEERED_FEATURES = ["a0_over_c0", "sigma_range", "stress_ratio", "log_a0_mm", "log_c0_mm", "log_sigma_max"]
CATEGORICAL_FEATURES = ["defect_type"]
DEFAULT_KNOTS = (0.0, 0.2, 0.4, 0.65, 0.85, 1.0, 1.2)


def _draw_specimen(rng: np.random.Generator) -> dict[str, object]:
    defect_type = "embedded" if rng.random() < 0.35 else "surface"
    c0_mm = float(np.exp(rng.uniform(np.log(0.08), np.log(0.40))))
    a0_mm = float(c0_mm * rng.uniform(0.5, 1.1))
    sigma_max = float(rng.uniform(320.0, 620.0))
    stress_ratio = float(rng.choice([-1.0, 0.1, 0.3]))
    return {
        "a0_mm": round(a0_mm, 4),
        "c0_mm": round(c0_mm, 4),
        "sigma_max": round(sigma_max, 1),
        "sigma_min": round(stress_ratio * sigma_max, 1),
        "w_mm": float(rng.choice([8.0, 10.0, 12.0])),
        "t_mm": float(rng.choice([4.0, 5.0, 6.0])),
        "defect_type": defect_type,
    }


def _hidden_log_bias(row: dict[str, object]) -> float:
    """Systematic physics-model error that the network should learn."""

    stress_ratio = float(row["sigma_min"]) / float(row["sigma_max"])
    return (
        0.55
        + 0.35 * (row["defect_type"] == "embedded")
        - 0.45 * (stress_ratio - 0.1)
        + 0.60 * np.log(float(row["a0_mm"]) / float(row["c0_mm"]))
        - 0.40 * np.tanh((float(row["sigma_max"]) - 470.0) / 120.0)
    )


def generate_specimen_table(
    n_specimens: int = 56,
    seed: int = 2026,
    noise_log_std: float = 0.30,
    material: SyntheticMaterial = DEFAULT_MATERIAL,
) -> pd.DataFrame:
    """Draw ``n_specimens`` propagating specimens with an observed life."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    while len(rows) < n_specimens:
        row = _draw_specimen(rng)
        sim = simulate_specimen(**row, material=material)
        if sim is None:
            continue
        physics_cycles = physics_crossing_cycle(sim, material)
        log_obs = np.log(physics_cycles) + _hidden_log_bias(row) + rng.normal(0.0, noise_log_std)
        row["specimen"] = f"S{len(rows) + 1:03d}"
        row[TARGET] = float(np.round(np.exp(log_obs)))
        rows.append(row)
    return pd.DataFrame(rows)[["specimen", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET]]


def add_engineered_features(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["a0_over_c0"] = out["a0_mm"] / out["c0_mm"]
    out["sigma_range"] = out["sigma_max"] - out["sigma_min"]
    out["stress_ratio"] = out["sigma_min"] / out["sigma_max"]
    out["log_a0_mm"] = np.log(out["a0_mm"])
    out["log_c0_mm"] = np.log(out["c0_mm"])
    out["log_sigma_max"] = np.log(out["sigma_max"])
    return out


def feature_columns() -> tuple[list[str], list[str]]:
    return NUMERIC_FEATURES + ENGINEERED_FEATURES, list(CATEGORICAL_FEATURES)


def make_event_grid(event_s_max: float = 1.2, n_points: int = 97, knots: tuple[float, ...] = DEFAULT_KNOTS) -> np.ndarray:
    """Fixed event-coordinate grid; ``s = 1`` is the physics transition."""

    base = np.linspace(0.0, event_s_max, n_points)
    return np.unique(np.round(np.concatenate([base, np.asarray(knots, dtype=float)]), 8)).astype(np.float32)


def event_coordinate_to_cycle(event_s: np.ndarray, crossing_cycle: float, post_threshold_cycles: float) -> np.ndarray:
    """``s <= 1`` scales with the crossing cycle; ``s > 1`` adds a fixed tail."""

    event_s = np.asarray(event_s, dtype=float)
    return np.where(
        event_s <= 1.0,
        event_s * crossing_cycle,
        crossing_cycle + (event_s - 1.0) * post_threshold_cycles,
    )


def physics_crossing_cycle(sim: dict[str, np.ndarray | float], material: SyntheticMaterial = DEFAULT_MATERIAL) -> float:
    """Interpolated first cycle where either front reaches the transition."""

    cycle = np.asarray(sim["cycle"])
    curve = np.maximum(
        short_crack_threshold(sim["extension_a_m"], material),
        short_crack_threshold(sim["extension_c_m"], material),
    )
    level = transition_threshold(material)
    idx = int(np.flatnonzero(curve >= level)[0])
    if idx == 0:
        return float(cycle[0])
    frac = (level - curve[idx - 1]) / max(curve[idx] - curve[idx - 1], 1e-12)
    return float(cycle[idx - 1] + frac * (cycle[idx] - cycle[idx - 1]))


@dataclass
class Dataset:
    """Specimen features plus physics reference curves on the event grid."""

    table: pd.DataFrame
    event_grid: np.ndarray
    physics_crossing: np.ndarray  # (n,)
    delta_a: np.ndarray  # (n, grid) physics extension, depth front [m]
    delta_c: np.ndarray  # (n, grid) physics extension, surface front [m]
    cycles: np.ndarray  # (n,) observed cycles


def build_dataset(
    table: pd.DataFrame,
    event_grid: np.ndarray | None = None,
    post_threshold_cycles: float = 10_000.0,
    material: SyntheticMaterial = DEFAULT_MATERIAL,
) -> Dataset:
    """Run the physics model per specimen and resample onto the event grid.

    Only static specimen descriptors go into the physics model, so the same
    computation is available for an unseen specimen at prediction time.
    """

    event_grid = make_event_grid() if event_grid is None else event_grid
    rows, crossings, curves_a, curves_c = [], [], [], []
    for idx, row in table.reset_index(drop=True).iterrows():
        sim = simulate_specimen(
            row["a0_mm"], row["c0_mm"], row["sigma_max"], row["sigma_min"],
            row["w_mm"], row["t_mm"], row["defect_type"],
            material=material, post_threshold_cycles=post_threshold_cycles,
        )
        if sim is None:
            continue
        crossing = physics_crossing_cycle(sim, material)
        grid_cycles = event_coordinate_to_cycle(event_grid, crossing, post_threshold_cycles)
        rows.append(idx)
        crossings.append(crossing)
        curves_a.append(np.interp(grid_cycles, sim["cycle"], sim["extension_a_m"]))
        curves_c.append(np.interp(grid_cycles, sim["cycle"], sim["extension_c_m"]))
    kept = add_engineered_features(table.reset_index(drop=True).iloc[rows].reset_index(drop=True))
    return Dataset(
        table=kept,
        event_grid=event_grid.astype(np.float32),
        physics_crossing=np.asarray(crossings, dtype=np.float32),
        delta_a=np.asarray(curves_a, dtype=np.float32),
        delta_c=np.asarray(curves_c, dtype=np.float32),
        cycles=kept[TARGET].to_numpy(dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic specimen table.")
    parser.add_argument("--n-specimens", type=int, default=56)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("examples/synthetic_specimens.csv"))
    args = parser.parse_args()
    table = generate_specimen_table(args.n_specimens, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Wrote {len(table)} synthetic specimens to {args.out}")


if __name__ == "__main__":
    main()
