"""Training, prediction and the nested-CV runner for the hybrid decoder."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import tensorflow as tf

from . import losses
from .cv import fit_transform_features, inner_splits, outer_splits
from .data import Dataset, feature_columns
from .decoder import (
    DecoderConfig,
    bounded_log_warp_tf,
    bounded_terminal_ratio_tf,
    build_decoder_network,
    cycle_grid_tf,
    decode_extension_curves_tf,
    density_knots_from_curves,
    threshold_from_extension_tf,
)
from .metrics import cycle_metrics
from .physics import transition_extension, transition_threshold


@dataclass(frozen=True)
class TrainConfig:
    hidden_units: tuple[int, ...] = (32,)
    dropout_rate: float = 0.0
    l2_strength: float = 1e-4
    learning_rate: float = 8e-4
    batch_size: int = 16
    max_epochs: int = 160
    patience: int = 20
    w_cycle: float = 1.0
    w_shape: float = 1.0
    w_ratio: float = 1.0
    w_concavity: float = 1.0
    w_warp: float = 0.1
    crossing_beta: float = 18.0
    decoder: DecoderConfig = DecoderConfig()

    @property
    def name(self) -> str:
        hidden = "x".join(str(u) for u in self.hidden_units)
        return (
            f"h{hidden}_shape{self.w_shape:.3g}_ratio{self.w_ratio:.3g}"
            f"_concav{self.w_concavity:.3g}_warp{self.w_warp:.3g}"
        )

    def parameter_count(self, n_features: int) -> int:
        total, previous = 0, n_features
        for units in self.hidden_units:
            total += previous * units + units
            previous = units
        n_outputs = 2 * len(self.decoder.knots) + 2
        return total + previous * n_outputs + n_outputs


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


@dataclass
class Batch:
    x: np.ndarray
    physics_crossing: np.ndarray
    delta_a: np.ndarray
    delta_c: np.ndarray
    cycles: np.ndarray


def _batch(dataset: Dataset, x: np.ndarray, idx: np.ndarray) -> Batch:
    return Batch(x, dataset.physics_crossing[idx], dataset.delta_a[idx], dataset.delta_c[idx], dataset.cycles[idx])


def _forward(model, x, physics_crossing, grid, knot_grid, config: TrainConfig, training: bool):
    density_a, density_c, ratio_raw, warp_raw = model(x, training=training)
    ratio = bounded_terminal_ratio_tf(ratio_raw, config.decoder)
    log_warp = bounded_log_warp_tf(warp_raw, config.decoder)
    delta_a, delta_c = decode_extension_curves_tf(density_a, density_c, ratio, knot_grid, grid, transition_extension())
    threshold_a = threshold_from_extension_tf(delta_a)
    threshold_c = threshold_from_extension_tf(delta_c)
    cycles = cycle_grid_tf(log_warp, physics_crossing, grid, config.decoder)
    return delta_a, delta_c, threshold_a, threshold_c, ratio, log_warp, cycles


def train_model(train: Batch, grid: np.ndarray, config: TrainConfig, seed: int) -> tf.keras.Model:
    """Fit one decoder network; early stopping on the training objective."""

    set_seed(seed)
    knot_grid = np.asarray(config.decoder.knots, dtype=np.float32)
    init_warp = float(np.median(np.log(np.clip(train.cycles, 1, None) / np.clip(train.physics_crossing, 1, None))))
    i1 = int(np.argmin(np.abs(grid - 1.0)))
    model = build_decoder_network(
        train.x.shape[1],
        len(knot_grid),
        config.hidden_units,
        config.dropout_rate,
        config.l2_strength,
        config.decoder,
        initial_density_a=np.median(density_knots_from_curves(train.delta_a, grid, knot_grid), axis=0),
        initial_density_c=np.median(density_knots_from_curves(train.delta_c, grid, knot_grid), axis=0),
        initial_log_warp=float(np.clip(init_warp, -0.8 * config.decoder.max_log_cycle_warp, 0.8 * config.decoder.max_log_cycle_warp)),
        initial_terminal_ratio=float(np.median(train.delta_c[:, i1] / train.delta_a[:, i1])),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=1.0)
    grid_t = tf.constant(grid, tf.float32)
    knots_t = tf.constant(knot_grid)
    level = transition_threshold()
    tensors = [tf.constant(v, tf.float32) for v in (train.x, train.physics_crossing, train.delta_a, train.delta_c, train.cycles)]

    @tf.function
    def step(idx):
        x, crossing, ref_a, ref_c, cycles = (tf.gather(t, idx) for t in tensors)
        with tf.GradientTape() as tape:
            delta_a, delta_c, th_a, th_c, ratio, log_warp, cycle_grid = _forward(
                model, x, crossing, grid_t, knots_t, config, training=True
            )
            pred = losses.soft_first_crossing_cycle(th_a, th_c, cycle_grid, level, config.crossing_beta)
            total = (
                config.w_cycle * losses.log_cycle_loss(pred, cycles)
                + config.w_shape * losses.shape_loss(delta_a, delta_c, ref_a, ref_c, grid_t)
                + config.w_ratio * losses.terminal_ratio_loss(ratio, ref_a, ref_c, grid_t)
                + config.w_concavity * losses.threshold_concavity_loss(th_a, th_c, grid_t)
                + config.w_warp * losses.warp_size_loss(log_warp, config.decoder.max_log_cycle_warp)
            )
            if model.losses:
                total += tf.add_n(model.losses)
        grads = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total

    n = len(train.cycles)
    batch_size = min(config.batch_size, n)
    best, best_weights, wait = np.inf, None, 0
    for _ in range(config.max_epochs):
        order = np.random.permutation(n)
        epoch_loss = np.mean([float(step(tf.constant(order[i : i + batch_size]))) for i in range(0, n, batch_size)])
        if epoch_loss < best - 1e-6:
            best, best_weights, wait = epoch_loss, model.get_weights(), 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    if best_weights is not None:
        model.set_weights(best_weights)
    return model


def predict_cycles(model, batch: Batch, grid: np.ndarray, config: TrainConfig) -> np.ndarray:
    """Hard first crossing of the decoded threshold curves, interpolated."""

    knots_t = tf.constant(np.asarray(config.decoder.knots, dtype=np.float32))
    _, _, th_a, th_c, _, _, cycle_grid = _forward(
        model, tf.constant(batch.x), tf.constant(batch.physics_crossing), tf.constant(grid), knots_t, config, False
    )
    curve = np.maximum(th_a.numpy(), th_c.numpy())
    cycle_grid = cycle_grid.numpy()
    level = transition_threshold()
    out = np.full(len(curve), np.nan)
    for i, (values, cycles) in enumerate(zip(curve, cycle_grid)):
        hit = np.flatnonzero(values >= level - 1e-6)
        if hit.size == 0:
            continue
        j = int(hit[0])
        if j == 0:
            out[i] = cycles[0]
            continue
        frac = np.clip((level - values[j - 1]) / max(values[j] - values[j - 1], 1e-12), 0.0, 1.0)
        out[i] = cycles[j - 1] + frac * (cycles[j] - cycles[j - 1])
    return out


# --------------------------------------------------------------------------
# Hyper-parameter search (inner loop) and nested CV (outer loop)
# --------------------------------------------------------------------------


def search_space(base: TrainConfig, n_loss_profiles: int = 4, seed: int = 2026) -> list[TrainConfig]:
    """Loss-weight profiles (Latin hypercube, log scale) x two backbones."""

    ranges = {"w_ratio": (1e-2, 10.0), "w_shape": (1e-2, 10.0), "w_concavity": (1e-2, 10.0), "w_warp": (1e-3, 3.0)}
    rng = np.random.default_rng(seed)
    n_sampled = max(n_loss_profiles - 2, 1)
    lhs = np.column_stack([(rng.permutation(n_sampled) + rng.random(n_sampled)) / n_sampled for _ in ranges])
    profiles = [
        {"w_ratio": 1.0, "w_shape": 1.0, "w_concavity": 1.0, "w_warp": 0.1},
        {name: lo for name, (lo, _) in ranges.items()},
    ]
    for row in lhs:
        profiles.append(
            {name: float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) for (name, (lo, hi)), u in zip(ranges.items(), row)}
        )
    backbones = [dict(hidden_units=(32,), dropout_rate=0.0), dict(hidden_units=(32, 16), dropout_rate=0.05)]
    return [replace(base, **arch, **weights) for arch in backbones for weights in profiles[:n_loss_profiles]]


@dataclass(frozen=True)
class NestedCVConfig:
    n_outer: int = 5
    n_inner: int = 3
    n_loss_profiles: int = 4
    inner_epochs: int = 100
    inner_patience: int = 12
    seed: int = 2026
    base: TrainConfig = TrainConfig()


def _select(scores: pd.DataFrame) -> pd.Series:
    """One-standard-error rule: simplest candidate within 1 SE of the best."""

    best = scores.loc[scores["inner_mae"].idxmin()]
    band = best["inner_mae"] + best["inner_mae_std"] / np.sqrt(max(best["n_inner"], 1))
    eligible = scores[scores["inner_mae"] <= band]
    return eligible.sort_values(["parameter_count", "inner_mae"], kind="mergesort").iloc[0]


def run_nested_cv(dataset: Dataset, cfg: NestedCVConfig, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return blind outer-fold predictions and the inner-search score table.

    Preprocessing and hyper-parameter choice happen inside each outer training
    set, so the blind specimens never influence scaling, loss weights or the
    architecture used to predict them.
    """

    numeric, categorical = feature_columns()
    table, grid = dataset.table, dataset.event_grid
    candidates = search_space(cfg.base, cfg.n_loss_profiles, cfg.seed)
    predictions, all_scores = [], []
    for fold, (train_idx, test_idx) in enumerate(outer_splits(len(table), cfg.n_outer, cfg.seed), start=1):
        rows = []
        splits = inner_splits(train_idx, cfg.n_inner, seed=cfg.seed + fold)
        for cand in candidates:
            inner_cfg = replace(cand, max_epochs=cfg.inner_epochs, patience=cfg.inner_patience)
            maes = []
            for split_id, (in_train, in_val) in enumerate(splits, start=1):
                x_tr, x_val = fit_transform_features(table, in_train, in_val, numeric, categorical)
                model = train_model(_batch(dataset, x_tr, in_train), grid, inner_cfg, seed=cfg.seed + 1000 * fold + split_id)
                pred = predict_cycles(model, _batch(dataset, x_val, in_val), grid, inner_cfg)
                maes.append(cycle_metrics(dataset.cycles[in_val], pred)["mae_log"])
                tf.keras.backend.clear_session()
            rows.append(
                {
                    "outer_fold": fold,
                    "candidate": cand.name,
                    "inner_mae": float(np.mean(maes)),
                    "inner_mae_std": float(np.std(maes)),
                    "n_inner": len(maes),
                    "parameter_count": cand.parameter_count(len(numeric) + 2),
                }
            )
        scores = pd.DataFrame(rows)
        chosen = _select(scores)
        selected = next(c for c in candidates if c.name == chosen["candidate"])
        scores["selected"] = scores["candidate"].eq(selected.name)
        all_scores.append(scores)

        x_tr, x_te = fit_transform_features(table, train_idx, test_idx, numeric, categorical)
        model = train_model(_batch(dataset, x_tr, train_idx), grid, selected, seed=cfg.seed + fold)
        pred = predict_cycles(model, _batch(dataset, x_te, test_idx), grid, selected)
        tf.keras.backend.clear_session()
        fold_metrics = cycle_metrics(dataset.cycles[test_idx], pred)
        log(f"  fold {fold}/{cfg.n_outer}: selected {selected.name}  blind MAE(log) {fold_metrics['mae_log']:.3f}")
        predictions.append(
            pd.DataFrame(
                {
                    "outer_fold": fold,
                    "specimen": table.iloc[test_idx]["specimen"].to_numpy(),
                    "observed_cycles": dataset.cycles[test_idx],
                    "physics_cycles": dataset.physics_crossing[test_idx],
                    "hybrid_cycles": pred,
                    "selected_candidate": selected.name,
                }
            )
        )
    return pd.concat(predictions, ignore_index=True), pd.concat(all_scores, ignore_index=True)
