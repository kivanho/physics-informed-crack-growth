"""Training, prediction and the nested-CV runner for the hybrid decoder.

Training budget (one strategy, used everywhere)
------------------------------------------------
There is no early stopping on the training objective. Every network is trained
for a fixed number of epochs. Inside each outer training set, every candidate
configuration is trained on each inner split for ``epochs`` epochs while the
inner-validation MAE (log cycles) is recorded after every epoch. A candidate's
score is the minimum over epochs of its mean inner-validation MAE, and the
epoch at which that minimum occurs is the candidate's chosen epoch count.
The selected candidate is refit on the whole outer training set for exactly
that many epochs and evaluated once on the blind fold. All epoch selection
therefore uses outer-training data only. (The final fit sees about one third
more specimens per epoch than the inner fits; epochs, not gradient steps, are
held fixed.)

Numerical checks
----------------
Every optimisation step returns the individual loss terms and a per-variable
flag for non-finite gradients. A non-finite value raises
:class:`NumericalError` naming the epoch, step, loss term(s) and parameter(s).
Nothing is clipped or replaced to hide the failure.

Penalty weights
---------------
A penalty whose weight is exactly ``0`` is *disabled*: the term is not
computed and does not enter the graph. A small non-zero weight (for example
``1e-2``) is a different experiment, in which the term is still active.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from typing import Callable

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
from .metrics import InvalidPredictionError, cycle_metrics
from .physics import transition_extension, transition_threshold

PENALTY_NAMES = ("shape", "ratio", "concavity", "warp")
TERM_NAMES = ("cycle", *PENALTY_NAMES, "l2", "total")


class NumericalError(RuntimeError):
    """A non-finite loss term or gradient was produced during training."""


@dataclass(frozen=True)
class TrainConfig:
    hidden_units: tuple[int, ...] = (32,)
    dropout_rate: float = 0.0
    l2_strength: float = 1e-4
    learning_rate: float = 8e-4
    batch_size: int = 16
    max_epochs: int = 100
    w_cycle: float = 1.0
    w_shape: float = 1.0
    w_ratio: float = 1.0
    w_concavity: float = 1.0
    w_warp: float = 0.1
    decoder: DecoderConfig = DecoderConfig()

    @property
    def name(self) -> str:
        hidden = "x".join(str(u) for u in self.hidden_units)
        return (
            f"h{hidden}_shape{self.w_shape:.3g}_ratio{self.w_ratio:.3g}"
            f"_concav{self.w_concavity:.3g}_warp{self.w_warp:.3g}"
        )

    def disabled_penalties(self) -> tuple[str, ...]:
        return tuple(name for name in PENALTY_NAMES if getattr(self, f"w_{name}") == 0.0)

    def parameter_count(self, n_features: int) -> int:
        total, previous = 0, n_features
        for units in self.hidden_units:
            total += previous * units + units
            previous = units
        n_outputs = 2 * len(self.decoder.knots) + 2
        return total + previous * n_outputs + n_outputs

    def as_dict(self) -> dict:
        return {
            "hidden_units": list(self.hidden_units),
            "dropout_rate": self.dropout_rate,
            "l2_strength": self.l2_strength,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "w_cycle": self.w_cycle,
            "w_shape": self.w_shape,
            "w_ratio": self.w_ratio,
            "w_concavity": self.w_concavity,
            "w_warp": self.w_warp,
            "disabled_penalties": list(self.disabled_penalties()),
            "decoder": {
                "knots": list(self.decoder.knots),
                "max_log_cycle_warp": self.decoder.max_log_cycle_warp,
                "min_terminal_ratio": self.decoder.min_terminal_ratio,
                "max_terminal_ratio": self.decoder.max_terminal_ratio,
                "post_threshold_cycles": self.decoder.post_threshold_cycles,
            },
        }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow and request deterministic TF ops.

    Reruns on the same machine and TensorFlow version should match; results
    can still differ slightly across platforms, CPU instruction sets or
    TensorFlow releases.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


@dataclass
class Batch:
    x: np.ndarray
    physics_crossing: np.ndarray
    delta_a: np.ndarray
    delta_c: np.ndarray
    cycles: np.ndarray
    specimen: np.ndarray | None = None


def _batch(dataset: Dataset, x: np.ndarray, idx: np.ndarray) -> Batch:
    return Batch(
        x,
        dataset.physics_crossing[idx],
        dataset.delta_a[idx],
        dataset.delta_c[idx],
        dataset.cycles[idx],
        dataset.table.iloc[idx]["specimen"].to_numpy(),
    )


@dataclass
class TrainingHistory:
    """Per-epoch loss terms (training mean) and optional validation MAE."""

    terms: dict[str, list[float]] = field(default_factory=lambda: {name: [] for name in TERM_NAMES})
    validation_mae: list[float] = field(default_factory=list)
    epochs_run: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "terms": self.terms,
            "validation_mae": self.validation_mae,
            "epochs_run": self.epochs_run,
            "seconds": self.seconds,
        }


def _variable_label(variable) -> str:
    return getattr(variable, "path", None) or variable.name


def fit_network(
    model: tf.keras.Model,
    loss_terms: Callable[[tf.Tensor], dict[str, tf.Tensor]],
    n_train: int,
    config: TrainConfig,
    epochs: int,
    validation_fn: Callable[[], float] | None = None,
    context: str = "",
) -> TrainingHistory:
    """Generic fixed-budget loop with finite-loss and finite-gradient checks.

    ``loss_terms(idx)`` returns a dict of scalar tensors; the key ``"total"``
    is optimised, the others are recorded. A non-finite term or gradient
    raises :class:`NumericalError` with the epoch, step, term and parameter.
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=1.0)
    variables = model.trainable_variables
    labels = [_variable_label(v) for v in variables]
    optimizer.build(variables)  # create slots eagerly; apply_gradients runs inside a tf.cond below

    @tf.function
    def step(idx):
        with tf.GradientTape() as tape:
            terms = loss_terms(idx)
        grads = tape.gradient(terms["total"], variables)
        bad_grad = tf.stack([tf.logical_not(tf.reduce_all(tf.math.is_finite(g))) for g in grads])
        ok = tf.logical_and(tf.math.is_finite(terms["total"]), tf.logical_not(tf.reduce_any(bad_grad)))
        # Apply only finite updates; the Python side raises right after a failure.
        if ok:
            optimizer.apply_gradients(zip(grads, variables))
        ordered = tf.stack([terms.get(name, tf.constant(0.0)) for name in TERM_NAMES])
        return ordered, bad_grad

    history = TrainingHistory()
    batch_size = min(config.batch_size, n_train)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n_train)
        epoch_terms = []
        for step_id, i in enumerate(range(0, n_train, batch_size), start=1):
            ordered, bad_grad = step(tf.constant(order[i : i + batch_size]))
            ordered, bad_grad = ordered.numpy(), bad_grad.numpy()
            if not np.all(np.isfinite(ordered)):
                bad_terms = [name for name, value in zip(TERM_NAMES, ordered) if not np.isfinite(value)]
                raise NumericalError(
                    f"{context}non-finite loss term(s) {bad_terms} at epoch {epoch}, step {step_id} "
                    f"(terms: {dict(zip(TERM_NAMES, map(float, ordered)))})"
                )
            if bad_grad.any():
                bad_params = [label for label, flag in zip(labels, bad_grad) if flag]
                raise NumericalError(
                    f"{context}non-finite gradient for parameter(s) {bad_params} at epoch {epoch}, step {step_id}"
                )
            epoch_terms.append(ordered)
        means = np.mean(epoch_terms, axis=0)
        for name, value in zip(TERM_NAMES, means):
            history.terms[name].append(float(value))
        history.epochs_run = epoch
        if validation_fn is not None:
            try:
                history.validation_mae.append(float(validation_fn()))
            except InvalidPredictionError as exc:
                raise NumericalError(f"{context}invalid validation prediction at epoch {epoch}: {exc}") from exc
    history.seconds = time.perf_counter() - start
    return history


# --------------------------------------------------------------------------
# Decoder forward pass, loss and training
# --------------------------------------------------------------------------


def _forward(model, x, physics_crossing, grid, knot_grid, config: TrainConfig, training: bool):
    density_a, density_c, ratio_raw, warp_raw = model(x, training=training)
    ratio = bounded_terminal_ratio_tf(ratio_raw, config.decoder)
    log_warp = bounded_log_warp_tf(warp_raw, config.decoder)
    delta_a, delta_c = decode_extension_curves_tf(density_a, density_c, ratio, knot_grid, grid, transition_extension())
    threshold_a = threshold_from_extension_tf(delta_a)
    threshold_c = threshold_from_extension_tf(delta_c)
    cycles = cycle_grid_tf(log_warp, physics_crossing, grid, config.decoder)
    return delta_a, delta_c, threshold_a, threshold_c, ratio, log_warp, cycles


def decoder_loss_terms(model, batch_tensors, ref_threshold, grid_t, knots_t, config: TrainConfig, idx) -> dict:
    """All loss terms for one mini-batch. Disabled penalties (weight 0) are skipped."""

    x, crossing, ref_a, ref_c, cycles = (tf.gather(t, idx) for t in batch_tensors)
    ref_th_a, ref_th_c = (tf.gather(t, idx) for t in ref_threshold)
    delta_a, delta_c, th_a, th_c, ratio, log_warp, cycle_grid = _forward(
        model, x, crossing, grid_t, knots_t, config, training=True
    )
    pred = losses.first_crossing_cycle(th_a, th_c, cycle_grid, transition_threshold())
    terms = {"cycle": config.w_cycle * losses.log_cycle_loss(pred, cycles)}
    if config.w_shape != 0.0:
        terms["shape"] = config.w_shape * losses.shape_loss(delta_a, delta_c, ref_a, ref_c, grid_t)
    if config.w_ratio != 0.0:
        terms["ratio"] = config.w_ratio * losses.terminal_ratio_loss(ratio, ref_a, ref_c, grid_t)
    if config.w_concavity != 0.0:
        terms["concavity"] = config.w_concavity * losses.threshold_concavity_loss(th_a, th_c, ref_th_a, ref_th_c, grid_t)
    if config.w_warp != 0.0:
        terms["warp"] = config.w_warp * losses.warp_size_loss(log_warp, config.decoder.max_log_cycle_warp)
    if model.losses:
        terms["l2"] = tf.add_n(model.losses)
    terms["total"] = tf.add_n(list(terms.values()))
    return terms


def build_initialised_decoder(train: Batch, grid: np.ndarray, config: TrainConfig) -> tf.keras.Model:
    knot_grid = np.asarray(config.decoder.knots, dtype=np.float32)
    init_warp = float(np.median(np.log(np.clip(train.cycles, 1, None) / np.clip(train.physics_crossing, 1, None))))
    i1 = int(np.argmin(np.abs(grid - 1.0)))
    return build_decoder_network(
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


def train_model_with_history(
    train: Batch,
    grid: np.ndarray,
    config: TrainConfig,
    seed: int,
    epochs: int | None = None,
    validation: Batch | None = None,
    context: str = "",
) -> tuple[tf.keras.Model, TrainingHistory]:
    """Fit one decoder network for a fixed number of epochs.

    With ``validation`` given, the validation MAE (log cycles) is recorded after
    every epoch; an invalid validation prediction raises ``NumericalError``.
    """

    set_seed(seed)
    epochs = config.max_epochs if epochs is None else epochs
    model = build_initialised_decoder(train, grid, config)
    grid_t = tf.constant(grid, tf.float32)
    knots_t = tf.constant(np.asarray(config.decoder.knots, dtype=np.float32))
    tensors = [tf.constant(v, tf.float32) for v in (train.x, train.physics_crossing, train.delta_a, train.delta_c, train.cycles)]
    ref_threshold = [threshold_from_extension_tf(tensors[2]), threshold_from_extension_tf(tensors[3])]

    def loss_terms(idx):
        return decoder_loss_terms(model, tensors, ref_threshold, grid_t, knots_t, config, idx)

    validation_fn = None
    if validation is not None:

        def validation_fn():
            pred = predict_cycles(model, validation, grid, config)
            return cycle_metrics(validation.cycles, pred, validation.specimen)["mae_log"]

    history = fit_network(model, loss_terms, len(train.cycles), config, epochs, validation_fn, context)
    return model, history


def train_model(train: Batch, grid: np.ndarray, config: TrainConfig, seed: int, epochs: int | None = None) -> tf.keras.Model:
    """Fit one decoder network for ``epochs`` (default ``config.max_epochs``) epochs."""

    model, _ = train_model_with_history(train, grid, config, seed, epochs)
    return model


def predict_cycles(model, batch: Batch, grid: np.ndarray, config: TrainConfig) -> np.ndarray:
    """Cycles to crack nucleation: the same interpolated first crossing as in training."""

    knots_t = tf.constant(np.asarray(config.decoder.knots, dtype=np.float32))
    _, _, th_a, th_c, _, _, cycle_grid = _forward(
        model, tf.constant(batch.x), tf.constant(batch.physics_crossing), tf.constant(grid), knots_t, config, False
    )
    return losses.first_crossing_cycle(th_a, th_c, cycle_grid, transition_threshold()).numpy().astype(float)


def predict_curves(model, batch: Batch, grid: np.ndarray, config: TrainConfig) -> dict[str, np.ndarray]:
    """Decoded extension and threshold curves plus the ratio and warp, as NumPy."""

    knots_t = tf.constant(np.asarray(config.decoder.knots, dtype=np.float32))
    delta_a, delta_c, th_a, th_c, ratio, log_warp, cycles = _forward(
        model, tf.constant(batch.x), tf.constant(batch.physics_crossing), tf.constant(grid), knots_t, config, False
    )
    return {
        "delta_a": delta_a.numpy(),
        "delta_c": delta_c.numpy(),
        "threshold_a": th_a.numpy(),
        "threshold_c": th_c.numpy(),
        "terminal_ratio": ratio.numpy(),
        "log_warp": log_warp.numpy(),
        "cycle_grid": cycles.numpy(),
    }


# --------------------------------------------------------------------------
# Hyper-parameter search (inner loop) and nested CV (outer loop)
# --------------------------------------------------------------------------


def search_space(
    base: TrainConfig,
    n_loss_profiles: int = 4,
    seed: int = 2026,
    include_small_penalties: bool = True,
) -> list[TrainConfig]:
    """Loss-weight profiles x two backbones (32 and 32-16 units).

    The ``n_loss_profiles`` profiles are, in order: the default weights; an
    optional *small-penalty* profile (every penalty weight at the bottom of
    its range, a nearly unregularised but still penalised network; this is not
    a disabled-penalty ablation); and Latin-hypercube samples on a log scale
    for the remaining slots.
    """

    ranges = {"w_ratio": (1e-2, 10.0), "w_shape": (1e-2, 10.0), "w_concavity": (1e-2, 10.0), "w_warp": (1e-3, 3.0)}
    profiles = [{"w_ratio": 1.0, "w_shape": 1.0, "w_concavity": 1.0, "w_warp": 0.1}]
    if include_small_penalties:
        profiles.append({name: lo for name, (lo, _) in ranges.items()})
    n_sampled = max(n_loss_profiles - len(profiles), 0)
    if n_sampled:
        rng = np.random.default_rng(seed)
        lhs = np.column_stack([(rng.permutation(n_sampled) + rng.random(n_sampled)) / n_sampled for _ in ranges])
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
    include_small_penalties: bool = True
    epochs: int = 100
    min_epochs: int = 5
    seed: int = 2026
    base: TrainConfig = TrainConfig()
    candidates: tuple[TrainConfig, ...] | None = None  # explicit list overrides search_space

    def candidate_list(self) -> list[TrainConfig]:
        if self.candidates is not None:
            return list(self.candidates)
        return search_space(self.base, self.n_loss_profiles, self.seed, self.include_small_penalties)


def select_epoch(validation_curves: list[list[float]], min_epochs: int) -> tuple[int, float]:
    """Epoch (1-based) minimising the mean inner-validation MAE, and that mean."""

    curves = np.asarray(validation_curves, dtype=float)
    if curves.ndim != 2 or curves.shape[1] == 0:
        raise ValueError("validation curves must be a non-empty (n_splits, n_epochs) array")
    mean = curves.mean(axis=0)
    start = min(max(min_epochs, 1), mean.size)
    best = int(np.argmin(mean[start - 1 :])) + start
    return best, float(mean[best - 1])


@dataclass
class InnerSearchResult:
    scores: pd.DataFrame
    selected: TrainConfig
    chosen_epochs: int
    histories: list[dict]


def inner_search(
    dataset: Dataset,
    train_idx: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[TrainConfig],
    cfg: NestedCVConfig,
    seed_base: int,
    fold: int,
) -> InnerSearchResult:
    """Score every candidate on the inner splits; choose config and epoch count."""

    numeric, categorical = feature_columns()
    grid = dataset.event_grid
    rows, histories = [], []
    for cand_id, cand in enumerate(candidates):
        curves = []
        n_features = None
        for split_id, (in_train, in_val) in enumerate(splits, start=1):
            x_tr, x_val = fit_transform_features(dataset.table, in_train, in_val, numeric, categorical)
            n_features = x_tr.shape[1]
            context = f"[fold {fold} candidate {cand.name} inner split {split_id}] "
            _, history = train_model_with_history(
                _batch(dataset, x_tr, in_train), grid, cand, seed=seed_base + 1000 * fold + split_id,
                epochs=cfg.epochs, validation=_batch(dataset, x_val, in_val), context=context,
            )
            curves.append(history.validation_mae)
            histories.append(
                {"outer_fold": fold, "stage": "inner", "candidate": cand.name, "inner_split": split_id, **history.as_dict()}
            )
            tf.keras.backend.clear_session()
        chosen_epoch, score = select_epoch(curves, cfg.min_epochs)
        rows.append(
            {
                "outer_fold": fold,
                "candidate_id": cand_id,
                "candidate": cand.name,
                "inner_mae": score,
                "inner_mae_at_last_epoch": float(np.mean([c[-1] for c in curves])),
                "inner_mae_split_range": float(np.ptp([c[chosen_epoch - 1] for c in curves])),
                "chosen_epochs": chosen_epoch,
                "budget_epochs": cfg.epochs,
                "n_inner": len(curves),
                "parameter_count": cand.parameter_count(n_features),
                "disabled_penalties": ",".join(cand.disabled_penalties()),
            }
        )
    scores = pd.DataFrame(rows)
    # Lowest mean inner-validation MAE; ties broken by fewer parameters. The
    # inner splits overlap, so their spread is reported but not used as a
    # standard error.
    best = scores.sort_values(["inner_mae", "parameter_count"], kind="mergesort").iloc[0]
    scores["selected"] = scores["candidate_id"].eq(best["candidate_id"])
    return InnerSearchResult(scores, candidates[int(best["candidate_id"])], int(best["chosen_epochs"]), histories)


@dataclass
class DecoderFoldResult:
    """Everything one outer fold produces for the decoder (or an ablation of it)."""

    pred: np.ndarray
    curves: dict[str, np.ndarray]
    search: InnerSearchResult
    final_history: TrainingHistory
    final_seed: int
    parameter_count: int
    seconds: float


def decoder_fold(
    dataset: Dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    cfg: NestedCVConfig,
    fold: int,
    init_seed_offset: int = 0,
    search: InnerSearchResult | None = None,
) -> DecoderFoldResult:
    """Inner search on ``splits`` (all inside ``train_idx``), final fit, blind prediction.

    Pass a previous ``search`` to reuse its selection and only redo the final
    fit with another ``init_seed_offset`` (initialisation-sensitivity runs).
    """

    started = time.perf_counter()
    numeric, categorical = feature_columns()
    grid = dataset.event_grid
    if search is None:
        search = inner_search(dataset, train_idx, splits, cfg.candidate_list(), cfg, cfg.seed, fold)
    x_tr, x_te = fit_transform_features(dataset.table, train_idx, test_idx, numeric, categorical)
    final_seed = cfg.seed + fold + init_seed_offset
    model, final_history = train_model_with_history(
        _batch(dataset, x_tr, train_idx), grid, search.selected, seed=final_seed,
        epochs=search.chosen_epochs, context=f"[fold {fold} final fit {search.selected.name}] ",
    )
    test_batch = _batch(dataset, x_te, test_idx)
    pred = predict_cycles(model, test_batch, grid, search.selected)
    curves = predict_curves(model, test_batch, grid, search.selected)
    tf.keras.backend.clear_session()
    return DecoderFoldResult(
        pred, curves, search, final_history, final_seed,
        search.selected.parameter_count(x_tr.shape[1]), time.perf_counter() - started,
    )


@dataclass
class CVResult:
    predictions: pd.DataFrame
    scores: pd.DataFrame
    histories: list[dict]
    folds: list[dict]


def run_nested_cv(
    dataset: Dataset,
    cfg: NestedCVConfig,
    log=print,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    outer_seed: int | None = None,
    init_seed_offset: int = 0,
) -> CVResult:
    """Blind outer-fold predictions plus the inner-search tables and histories.

    Preprocessing, loss weights, architecture and epoch count are all chosen
    inside each outer training set, so the blind specimens never influence the
    model that predicts them. Pass ``folds`` to share outer splits with other
    variants; ``init_seed_offset`` changes only the network initialisation and
    batch order of the final fits (for initialisation-sensitivity runs).
    """

    table = dataset.table
    outer_seed = cfg.seed if outer_seed is None else outer_seed
    folds = outer_splits(len(table), cfg.n_outer, outer_seed) if folds is None else folds
    predictions, all_scores, histories, fold_records = [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        splits = inner_splits(train_idx, cfg.n_inner, seed=outer_seed + fold)
        result = decoder_fold(dataset, train_idx, test_idx, splits, cfg, fold, init_seed_offset)
        search = result.search
        all_scores.append(search.scores)
        histories.extend(search.histories)
        histories.append(
            {"outer_fold": fold, "stage": "final", "candidate": search.selected.name, "inner_split": None, **result.final_history.as_dict()}
        )
        specimens = table.iloc[test_idx]["specimen"].to_numpy()
        validity = cycle_metrics(dataset.cycles[test_idx], result.pred, specimens, on_invalid="mark")
        fold_records.append(
            {
                "outer_fold": fold,
                "train_specimens": table.iloc[train_idx]["specimen"].tolist(),
                "test_specimens": specimens.tolist(),
                "selected_candidate": search.selected.name,
                "selected_config": search.selected.as_dict(),
                "chosen_epochs": search.chosen_epochs,
                "final_fit_seed": result.final_seed,
                "parameter_count": result.parameter_count,
                "status": validity["status"],
                "invalid_specimens": validity["invalid_specimens"],
                "seconds": result.seconds,
            }
        )
        if log is not None:
            mae = validity["mae_log"] if validity["status"] == "VALID" else float("nan")
            log(
                f"  fold {fold}/{len(folds)}: {search.selected.name} @ {search.chosen_epochs} epochs  "
                f"blind MAE(log) {mae:.3f}  [{validity['status']}]  ({result.seconds:.0f} s)"
            )
        predictions.append(
            pd.DataFrame(
                {
                    "outer_fold": fold,
                    "specimen": specimens,
                    "observed_cycles": dataset.cycles[test_idx],
                    "physics_cycles": dataset.physics_crossing[test_idx],
                    "hybrid_cycles": result.pred,
                    "terminal_ratio": result.curves["terminal_ratio"],
                    "log_warp": result.curves["log_warp"],
                    "selected_candidate": search.selected.name,
                    "chosen_epochs": search.chosen_epochs,
                }
            )
        )
    return CVResult(pd.concat(predictions, ignore_index=True), pd.concat(all_scores, ignore_index=True), histories, fold_records)
