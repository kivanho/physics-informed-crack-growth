"""Model variants evaluated on identical outer folds.

Every variant implements ``fit_predict`` for one outer fold: it receives the
outer training indices, the blind test indices and the inner splits (all
drawn from the training indices), fits everything it needs on the training
side only and returns predictions for the blind specimens. Hyper-parameters
(ridge penalty, network width, epoch count) are chosen on the inner splits.

Variants, from simplest to most complex:

1. ``physics``: the synthetic physics model's crossing cycle, no fitting.
2. ``constant``: one multiplicative correction, ``exp(mean log(N_obs / N_phys))``
   over the training fold.
3. ``ridge``: ridge regression from the preprocessed specimen features to the
   log correction ``log(N_obs / N_phys)``; penalty chosen on the inner splits.
4. ``warp_net``: a small MLP that outputs only the bounded cycle warp
   (the decoder's timing head without the curve heads); width and epoch
   count chosen on the inner splits.
5. ``decoder``: the full physics-informed knot decoder with its inner search
   over loss-weight profiles and backbones.

Ablations of the decoder keep one fixed configuration (no profile search; the
epoch count is still chosen on the inner splits) and set individual penalty
weights to exactly ``0``, which removes the term from the graph. The
``small_penalties`` ablation keeps every term active at the bottom of its
search range, to show the difference between "small" and "disabled".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import Ridge

from .cv import fit_transform_features
from .data import Dataset, feature_columns
from .decoder import DecoderConfig
from .metrics import cycle_metrics
from .model import (
    NestedCVConfig,
    TrainConfig,
    _batch,
    decoder_fold,
    fit_network,
    select_epoch,
    set_seed,
)


@dataclass
class FoldOutput:
    """What one variant produces for one outer fold."""

    pred: np.ndarray
    selected: dict
    parameter_count: int
    seconds: float
    chosen_epochs: int | None = None
    inner_scores: pd.DataFrame | None = None
    histories: list[dict] = field(default_factory=list)
    curves: dict[str, np.ndarray] | None = None  # decoded curves for the blind specimens, if any
    search: object | None = None  # reusable inner-search result (decoder variants)


class Variant:
    name: str = "variant"
    description: str = ""
    decodes_curves: bool = False

    def fit_predict(self, dataset: Dataset, train_idx, test_idx, splits, fold: int, seed: int) -> FoldOutput:
        raise NotImplementedError


def _log_correction(dataset: Dataset, idx) -> np.ndarray:
    return np.log(dataset.cycles[idx].astype(float)) - np.log(dataset.physics_crossing[idx].astype(float))


class PhysicsVariant(Variant):
    name = "physics"
    description = "synthetic physics model, no fitting"

    def fit_predict(self, dataset, train_idx, test_idx, splits, fold, seed):
        start = time.perf_counter()
        pred = dataset.physics_crossing[test_idx].astype(float)
        return FoldOutput(pred, {}, 0, time.perf_counter() - start)


class ConstantCorrectionVariant(Variant):
    name = "constant"
    description = "physics x one multiplicative correction fitted on the training fold"

    def fit_predict(self, dataset, train_idx, test_idx, splits, fold, seed):
        start = time.perf_counter()
        shift = float(np.mean(_log_correction(dataset, train_idx)))
        pred = dataset.physics_crossing[test_idx].astype(float) * np.exp(shift)
        return FoldOutput(pred, {"log_correction": shift}, 1, time.perf_counter() - start)


class RidgeCorrectionVariant(Variant):
    name = "ridge"
    description = "ridge regression features -> log(N_obs / N_phys); alpha chosen on inner splits"

    def __init__(self, alphas: tuple[float, ...] = tuple(float(a) for a in np.logspace(-2, 3, 11))):
        self.alphas = alphas

    def fit_predict(self, dataset, train_idx, test_idx, splits, fold, seed):
        start = time.perf_counter()
        numeric, categorical = feature_columns()
        rows = []
        for alpha in self.alphas:
            maes = []
            for in_train, in_val in splits:
                x_tr, x_val = fit_transform_features(dataset.table, in_train, in_val, numeric, categorical)
                model = Ridge(alpha=alpha).fit(x_tr, _log_correction(dataset, in_train))
                pred = dataset.physics_crossing[in_val] * np.exp(model.predict(x_val))
                maes.append(cycle_metrics(dataset.cycles[in_val], pred)["mae_log"])
            rows.append({"outer_fold": fold, "candidate": f"alpha{alpha:.3g}", "alpha": alpha, "inner_mae": float(np.mean(maes)), "n_inner": len(maes)})
        scores = pd.DataFrame(rows)
        best = scores.loc[scores["inner_mae"].idxmin()]
        scores["selected"] = scores["alpha"].eq(best["alpha"])
        x_tr, x_te = fit_transform_features(dataset.table, train_idx, test_idx, numeric, categorical)
        model = Ridge(alpha=float(best["alpha"])).fit(x_tr, _log_correction(dataset, train_idx))
        pred = dataset.physics_crossing[test_idx].astype(float) * np.exp(model.predict(x_te))
        return FoldOutput(
            pred, {"alpha": float(best["alpha"])}, int(x_tr.shape[1] + 1), time.perf_counter() - start,
            inner_scores=scores,
        )


@dataclass(frozen=True)
class WarpNetConfig:
    hidden_units: tuple[int, ...] = (8,)
    l2_strength: float = 1e-4
    learning_rate: float = 8e-4
    batch_size: int = 16
    w_warp: float = 0.1
    max_log_cycle_warp: float = DecoderConfig().max_log_cycle_warp

    @property
    def name(self) -> str:
        return f"warp_h{'x'.join(str(u) for u in self.hidden_units)}_warp{self.w_warp:.3g}"

    def parameter_count(self, n_features: int) -> int:
        total, previous = 0, n_features
        for units in self.hidden_units:
            total += previous * units + units
            previous = units
        return total + previous + 1


def build_warp_network(n_features: int, config: WarpNetConfig, initial_log_warp: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(n_features,), name="static_features")
    regularizer = tf.keras.regularizers.l2(config.l2_strength) if config.l2_strength else None
    x = inputs
    for units in config.hidden_units:
        x = tf.keras.layers.Dense(units, activation="swish", kernel_regularizer=regularizer)(x)
    unit = float(np.clip(initial_log_warp / config.max_log_cycle_warp, -0.95, 0.95))
    out = tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.Constant(np.arctanh(unit)), name="cycle_warp")(x)
    return tf.keras.Model(inputs, out, name="warp_only")


def _warp_predict(model, x, physics_crossing, config: WarpNetConfig) -> tf.Tensor:
    log_warp = config.max_log_cycle_warp * tf.tanh(tf.reshape(model(x, training=False), [-1]))
    return physics_crossing * tf.exp(log_warp)


def train_warp_network(train, config: WarpNetConfig, seed: int, epochs: int, validation=None, context: str = ""):
    set_seed(seed)
    init_warp = float(np.median(np.log(np.clip(train.cycles, 1, None) / np.clip(train.physics_crossing, 1, None))))
    model = build_warp_network(train.x.shape[1], config, float(np.clip(init_warp, -0.8 * config.max_log_cycle_warp, 0.8 * config.max_log_cycle_warp)))
    x_t = tf.constant(train.x, tf.float32)
    phys_t = tf.constant(train.physics_crossing, tf.float32)
    cycles_t = tf.constant(train.cycles, tf.float32)

    def loss_terms(idx):
        x, phys, cycles = tf.gather(x_t, idx), tf.gather(phys_t, idx), tf.gather(cycles_t, idx)
        log_warp = config.max_log_cycle_warp * tf.tanh(tf.reshape(model(x, training=True), [-1]))
        pred = phys * tf.exp(log_warp)
        terms = {"cycle": tf.reduce_mean(tf.square(tf.math.log1p(pred) - tf.math.log1p(cycles)))}
        if config.w_warp != 0.0:
            terms["warp"] = config.w_warp * tf.reduce_mean(tf.square(log_warp / config.max_log_cycle_warp))
        if model.losses:
            terms["l2"] = tf.add_n(model.losses)
        terms["total"] = tf.add_n(list(terms.values()))
        return terms

    validation_fn = None
    if validation is not None:

        def validation_fn():
            pred = _warp_predict(model, tf.constant(validation.x), tf.constant(validation.physics_crossing), config).numpy()
            return cycle_metrics(validation.cycles, pred, validation.specimen)["mae_log"]

    loop_config = TrainConfig(learning_rate=config.learning_rate, batch_size=config.batch_size)
    history = fit_network(model, loss_terms, len(train.cycles), loop_config, epochs, validation_fn, context)
    return model, history


class WarpOnlyVariant(Variant):
    name = "warp_net"
    description = "small MLP predicting only the bounded cycle warp; width and epochs chosen on inner splits"

    def __init__(self, candidates=(WarpNetConfig((8,)), WarpNetConfig((32,))), epochs: int = 100, min_epochs: int = 5):
        self.candidates = list(candidates)
        self.epochs = epochs
        self.min_epochs = min_epochs

    def fit_predict(self, dataset, train_idx, test_idx, splits, fold, seed):
        start = time.perf_counter()
        numeric, categorical = feature_columns()
        rows, histories = [], []
        for cand_id, cand in enumerate(self.candidates):
            curves, n_features = [], None
            for split_id, (in_train, in_val) in enumerate(splits, start=1):
                x_tr, x_val = fit_transform_features(dataset.table, in_train, in_val, numeric, categorical)
                n_features = x_tr.shape[1]
                _, history = train_warp_network(
                    _batch(dataset, x_tr, in_train), cand, seed + 1000 * fold + split_id, self.epochs,
                    validation=_batch(dataset, x_val, in_val), context=f"[warp_net fold {fold} {cand.name} inner split {split_id}] ",
                )
                curves.append(history.validation_mae)
                histories.append({"outer_fold": fold, "stage": "inner", "candidate": cand.name, "inner_split": split_id, **history.as_dict()})
                tf.keras.backend.clear_session()
            chosen, score = select_epoch(curves, self.min_epochs)
            rows.append(
                {
                    "outer_fold": fold, "candidate_id": cand_id, "candidate": cand.name, "inner_mae": score,
                    "chosen_epochs": chosen, "budget_epochs": self.epochs, "n_inner": len(curves),
                    "parameter_count": cand.parameter_count(n_features),
                }
            )
        scores = pd.DataFrame(rows)
        best = scores.sort_values(["inner_mae", "parameter_count"], kind="mergesort").iloc[0]
        scores["selected"] = scores["candidate_id"].eq(best["candidate_id"])
        selected = self.candidates[int(best["candidate_id"])]
        x_tr, x_te = fit_transform_features(dataset.table, train_idx, test_idx, numeric, categorical)
        model, final_history = train_warp_network(
            _batch(dataset, x_tr, train_idx), selected, seed + fold, int(best["chosen_epochs"]),
            context=f"[warp_net fold {fold} final fit {selected.name}] ",
        )
        pred = _warp_predict(model, tf.constant(x_te), tf.constant(dataset.physics_crossing[test_idx]), selected).numpy().astype(float)
        tf.keras.backend.clear_session()
        histories.append({"outer_fold": fold, "stage": "final", "candidate": selected.name, "inner_split": None, **final_history.as_dict()})
        return FoldOutput(
            pred, {"hidden_units": list(selected.hidden_units), "w_warp": selected.w_warp, "l2_strength": selected.l2_strength},
            selected.parameter_count(x_tr.shape[1]), time.perf_counter() - start,
            chosen_epochs=int(best["chosen_epochs"]), inner_scores=scores, histories=histories,
        )


class DecoderVariant(Variant):
    decodes_curves = True

    def __init__(self, cv: NestedCVConfig, name: str = "decoder", description: str = "full physics-informed knot decoder, inner search over loss profiles and backbones"):
        self.cv = cv
        self.name = name
        self.description = description

    def fit_predict(self, dataset, train_idx, test_idx, splits, fold, seed, init_seed_offset: int = 0, search=None):
        cfg = replace(self.cv, seed=seed)
        result = decoder_fold(dataset, train_idx, test_idx, splits, cfg, fold, init_seed_offset, search)
        histories = list(result.search.histories) if search is None else []
        histories.append(
            {"outer_fold": fold, "stage": "final", "candidate": result.search.selected.name, "inner_split": None, **result.final_history.as_dict()}
        )
        return FoldOutput(
            result.pred, result.search.selected.as_dict(), result.parameter_count, result.seconds,
            chosen_epochs=result.search.chosen_epochs, inner_scores=result.search.scores, histories=histories,
            curves=result.curves, search=result.search,
        )


def ablation_variants(base: TrainConfig, cv: NestedCVConfig) -> list[DecoderVariant]:
    """Fixed-configuration decoders: default weights, true single-term and all-term
    removals (weight exactly 0, term not computed), and the small-but-active
    comparison profile."""

    fixed = replace(base, hidden_units=(32,), dropout_rate=0.0)
    specs = [
        ("decoder_fixed_default", {}, "decoder, default weights, no profile search (epochs chosen on inner splits)"),
        ("ablation_no_shape", {"w_shape": 0.0}, "shape penalty disabled (weight 0, term removed)"),
        ("ablation_no_ratio", {"w_ratio": 0.0}, "terminal-ratio penalty disabled"),
        ("ablation_no_concavity", {"w_concavity": 0.0}, "concavity penalty disabled"),
        ("ablation_no_warp", {"w_warp": 0.0}, "warp-size penalty disabled"),
        ("ablation_no_penalties", {"w_shape": 0.0, "w_ratio": 0.0, "w_concavity": 0.0, "w_warp": 0.0}, "all four physics penalties disabled"),
        ("ablation_small_penalties", {"w_shape": 1e-2, "w_ratio": 1e-2, "w_concavity": 1e-2, "w_warp": 1e-3}, "all penalties active at the bottom of the search range (not disabled)"),
    ]
    variants = []
    for name, overrides, description in specs:
        cfg = replace(cv, candidates=(replace(fixed, **overrides),))
        variants.append(DecoderVariant(cfg, name=name, description=description))
    return variants
