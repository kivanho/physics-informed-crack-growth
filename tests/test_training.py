"""Training-loop checks, budget strategy, baselines on shared folds, end-to-end smoke."""

import json

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from picg import benchmark
from picg.baselines import ConstantCorrectionVariant, PhysicsVariant, RidgeCorrectionVariant, ablation_variants
from picg.cv import fit_transform_features, inner_splits, outer_splits
from picg.data import build_dataset, feature_columns, generate_specimen_table
from picg.model import (
    NestedCVConfig,
    NumericalError,
    TrainConfig,
    _batch,
    decoder_loss_terms,
    fit_network,
    select_epoch,
    train_model_with_history,
)


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(generate_specimen_table(24, seed=7))


def _first_fold(dataset):
    train_idx, test_idx = outer_splits(len(dataset.cycles), 3, seed=1)[0]
    numeric, categorical = feature_columns()
    x_tr, x_te = fit_transform_features(dataset.table, train_idx, test_idx, numeric, categorical)
    return train_idx, test_idx, x_tr, x_te


def test_select_epoch_uses_mean_curve_and_respects_min_epochs():
    curves = [[1.0, 0.5, 0.25, 0.25, 0.5], [1.0, 0.1, 0.25, 0.25, 0.5]]  # mean: 1.0 0.3 0.25 0.25 0.5
    assert select_epoch(curves, min_epochs=1) == (3, 0.25)  # first epoch reaching the minimum
    assert select_epoch(curves, min_epochs=4) == (4, 0.25)
    assert select_epoch(curves, min_epochs=50)[0] == 5  # capped at the budget
    with pytest.raises(ValueError):
        select_epoch([[]], 1)


def test_disabled_penalty_is_not_computed(dataset):
    train_idx, _, x_tr, _ = _first_fold(dataset)
    train = _batch(dataset, x_tr, train_idx)
    tf.keras.utils.set_random_seed(0)
    from picg.decoder import threshold_from_extension_tf
    from picg.model import build_initialised_decoder

    grid_t = tf.constant(dataset.event_grid)
    tensors = [tf.constant(v, tf.float32) for v in (train.x, train.physics_crossing, train.delta_a, train.delta_c, train.cycles)]
    ref = [threshold_from_extension_tf(tensors[2]), threshold_from_extension_tf(tensors[3])]
    idx = tf.constant(np.arange(4))
    for disabled in ("shape", "ratio", "concavity", "warp"):
        config = TrainConfig(**{f"w_{disabled}": 0.0})
        assert config.disabled_penalties() == (disabled,)
        model = build_initialised_decoder(train, dataset.event_grid, config)
        terms = decoder_loss_terms(model, tensors, ref, grid_t, tf.constant(config.decoder.knots, tf.float32), config, idx)
        assert disabled not in terms, f"{disabled} term must be absent when its weight is exactly 0"
        assert set(terms) >= {"cycle", "total"}
    small = TrainConfig(w_shape=1e-2)
    assert small.disabled_penalties() == ()
    model = build_initialised_decoder(train, dataset.event_grid, small)
    terms = decoder_loss_terms(model, tensors, ref, grid_t, tf.constant(small.decoder.knots, tf.float32), small, idx)
    assert "shape" in terms and float(terms["shape"]) > 0.0


def test_training_history_has_all_terms_and_validation_curve(dataset):
    train_idx, test_idx, x_tr, x_te = _first_fold(dataset)
    _, history = train_model_with_history(
        _batch(dataset, x_tr, train_idx), dataset.event_grid, TrainConfig(), seed=0, epochs=4,
        validation=_batch(dataset, x_te, test_idx),
    )
    assert history.epochs_run == 4
    assert len(history.validation_mae) == 4 and all(np.isfinite(history.validation_mae))
    for name, values in history.terms.items():
        assert len(values) == 4, name
        assert all(np.isfinite(values)), name


def test_non_finite_loss_term_raises_with_epoch_term_and_step():
    model = tf.keras.Sequential([tf.keras.Input(shape=(2,)), tf.keras.layers.Dense(1, name="head")])
    calls = {"n": 0}

    def loss_terms(idx):
        x = tf.ones((4, 2))
        out = tf.reduce_mean(model(x))
        bad = tf.constant(float("nan"))
        return {"cycle": out, "shape": bad, "total": out + bad}

    with pytest.raises(NumericalError) as info:
        fit_network(model, loss_terms, n_train=4, config=TrainConfig(batch_size=4), epochs=3, context="[unit] ")
    message = str(info.value)
    assert "epoch 1" in message and "step 1" in message and "'shape'" in message and "[unit]" in message


def test_non_finite_gradient_raises_naming_the_parameter():
    model = tf.keras.Sequential([tf.keras.Input(shape=(1,)), tf.keras.layers.Dense(1, use_bias=False, name="head")])

    def loss_terms(idx):
        # finite loss value, infinite gradient: d/dw sqrt(w*x) at w*x -> 0 with x = 0 gives nan
        kernel = model.trainable_variables[0]
        value = tf.sqrt(tf.abs(kernel[0, 0]) * 0.0)
        return {"cycle": value, "total": value}

    with pytest.raises(NumericalError, match="non-finite gradient for parameter"):
        fit_network(model, loss_terms, n_train=4, config=TrainConfig(batch_size=4), epochs=1)


def test_fixed_budget_runs_exact_epoch_count(dataset):
    train_idx, _, x_tr, _ = _first_fold(dataset)
    _, history = train_model_with_history(_batch(dataset, x_tr, train_idx), dataset.event_grid, TrainConfig(), seed=0, epochs=7)
    assert history.epochs_run == 7


def test_baselines_fit_inside_training_fold_only(dataset):
    """Changing blind-fold targets must not change a baseline's blind predictions."""

    train_idx, test_idx = outer_splits(len(dataset.cycles), 3, seed=1)[0]
    splits = inner_splits(train_idx, 2, seed=5)
    for variant in (PhysicsVariant(), ConstantCorrectionVariant(), RidgeCorrectionVariant(alphas=(0.1, 10.0))):
        pred = variant.fit_predict(dataset, train_idx, test_idx, splits, 1, 0).pred
        shifted = build_dataset(generate_specimen_table(24, seed=7))
        shifted.cycles = shifted.cycles.copy()
        shifted.cycles[test_idx] *= 10.0  # blind targets only
        pred_shifted = variant.fit_predict(shifted, train_idx, test_idx, splits, 1, 0).pred
        np.testing.assert_allclose(pred, pred_shifted, rtol=1e-6, err_msg=variant.name)
        assert pred.shape == (len(test_idx),) and np.all(np.isfinite(pred)) and np.all(pred > 0)


def test_ablation_variants_disable_terms_exactly():
    variants = ablation_variants(TrainConfig(), NestedCVConfig(n_inner=1, epochs=2))
    by_name = {v.name: v.cv.candidates[0] for v in variants}
    assert by_name["ablation_no_shape"].disabled_penalties() == ("shape",)
    assert by_name["ablation_no_penalties"].disabled_penalties() == ("shape", "ratio", "concavity", "warp")
    assert by_name["ablation_small_penalties"].disabled_penalties() == ()
    assert by_name["decoder_fixed_default"].disabled_penalties() == ()
    assert all(len(v.cv.candidate_list()) == 1 for v in variants)


def test_benchmark_smoke_end_to_end(tmp_path):
    """Tiny benchmark: identical folds for every variant, complete coverage, valid status."""

    out = tmp_path / "smoke"
    code = benchmark.main(["--smoke", "--out", str(out), "--n-specimens", "16", "--init-seeds", "1"])
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "VALID"
    assert manifest["git"]["commit"] and manifest["environment"]["packages"]["tensorflow"]
    assert manifest["data"]["fingerprint_sha256"]
    predictions = pd.read_csv(out / "predictions.csv")
    assert len(predictions) == manifest["data"]["n_evaluated"]
    columns = [c for c in predictions.columns if c.endswith("_cycles") and c != "observed_cycles"]
    assert {"physics_cycles", "constant_cycles", "ridge_cycles", "warp_net_cycles", "decoder_cycles", "ablation_no_penalties_cycles"} <= set(columns)
    for column in columns:
        assert predictions[column].notna().all() and (predictions[column] > 0).all(), column
    metrics = json.loads((out / "metrics.json").read_text())
    for name, m in metrics.items():
        assert m["status"] == "VALID", name
        assert m["n_valid"] == m["n_expected"] == m["denominator"] == len(predictions), name
        assert m["invalid_specimens"] == []
    assert metrics["decoder"]["curve_fidelity"]["monotone_fraction"] == 1.0
    assert "not measured truth" in metrics["decoder"]["curve_fidelity"]["reference"]
    folds = json.loads((out / "folds.json").read_text())
    tested = [s for f in folds for s in f["test_specimens"]]
    assert sorted(tested) == sorted(predictions["specimen"])  # every specimen blind exactly once
    for f in folds:
        for split in f["inner_splits"]:
            assert set(split["train"]) | set(split["validation"]) <= set(f["train_specimens"])
            assert not (set(split["validation"]) & set(f["test_specimens"]))
    histories = json.loads((out / "histories.json").read_text())
    assert any(h["stage"] == "final" and h["variant"] == "decoder" for h in histories)
    assert (out / "init_seeds.json").exists() and (out / "inner_search.csv").exists()
    inner = pd.read_csv(out / "inner_search.csv")
    assert inner.groupby(["variant", "outer_fold"])["selected"].sum().eq(1).all()


def test_benchmark_exit_code_2_when_a_variant_is_invalid(tmp_path, monkeypatch):
    class BrokenVariant(PhysicsVariant):
        name = "broken"
        description = "returns a nan prediction"

        def fit_predict(self, dataset, train_idx, test_idx, splits, fold, seed):
            out = super().fit_predict(dataset, train_idx, test_idx, splits, fold, seed)
            out.pred = out.pred.copy()
            out.pred[0] = np.nan
            return out

    monkeypatch.setattr(benchmark, "build_variants", lambda args: [PhysicsVariant(), BrokenVariant()])
    out = tmp_path / "invalid"
    code = benchmark.main(["--smoke", "--out", str(out), "--n-specimens", "12", "--init-seeds"])
    assert code == 2
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["broken"]["status"] == "INVALID" and metrics["broken"]["n_invalid"] == 2
    assert len(metrics["broken"]["invalid_specimens"]) == 2
    assert metrics["physics"]["status"] == "VALID"
    assert json.loads((out / "manifest.json").read_text())["status"] == "INVALID"
