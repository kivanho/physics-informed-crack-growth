import numpy as np
import pandas as pd
import tensorflow as tf

from picg.cv import fit_transform_features, inner_splits, outer_splits
from picg.data import build_dataset, feature_columns, generate_specimen_table, make_event_grid
from picg.decoder import (
    DecoderConfig,
    decode_extension_curves_tf,
    monotone_cubic_interpolate_np,
    threshold_from_extension_tf,
)
from picg.losses import first_crossing_cycle, threshold_concavity_loss
from picg.metrics import cycle_metrics
from picg.physics import (
    invert_short_crack_threshold,
    short_crack_threshold,
    transition_extension,
    transition_threshold,
)


def _reference_dataset(n=24, seed=4):
    return build_dataset(generate_specimen_table(n, seed=seed))


def test_threshold_inversion_round_trip():
    extension = np.array([0.0, 1e-6, 50e-6, 500e-6])
    recovered, valid = invert_short_crack_threshold(short_crack_threshold(extension))
    assert valid.all()
    np.testing.assert_allclose(recovered, extension, atol=1e-10)


def test_decoder_is_monotone_for_any_raw_output():
    grid = make_event_grid()
    config = DecoderConfig()
    knots = tf.constant(config.knots, tf.float32)
    raw = tf.random.normal((64, len(config.knots)), stddev=5.0, seed=0)
    raw_c = tf.random.normal((64, len(config.knots)), stddev=5.0, seed=1)
    ratio = tf.fill((64,), 0.8)
    delta_a, delta_c = decode_extension_curves_tf(raw, raw_c, ratio, knots, tf.constant(grid), transition_extension())
    for curve in (delta_a, delta_c, threshold_from_extension_tf(delta_a)):
        assert np.all(np.diff(curve.numpy(), axis=1) >= -1e-9)
    i1 = int(np.argmin(np.abs(grid - 1.0)))
    np.testing.assert_allclose(delta_a.numpy()[:, i1], transition_extension(), rtol=1e-4)


def test_pchip_stays_within_positive_knots():
    knots = np.array([0.0, 0.2, 0.4, 0.65, 0.85, 1.0, 1.2])
    values = np.array([3.0, 0.01, 2.0, 0.05, 4.0, 0.02, 1.0])
    curve = monotone_cubic_interpolate_np(np.linspace(0, 1.2, 500), knots, values)
    assert curve.min() >= values.min() - 1e-12


def test_metrics_late_count():
    result = cycle_metrics(np.array([100.0, 100.0, 100.0]), np.array([200.0, 50.0, 100.0]))
    assert result["late"] == 1
    assert result["mae_log"] > 0


def test_preprocessing_is_fit_on_train_only():
    table = build_dataset(generate_specimen_table(20, seed=1)).table
    # Shift the evaluation rows so that statistics fitted on them (or on all
    # rows) would differ clearly from the training-subset statistics.
    table.loc[15:, "sigma_max"] += 300.0
    numeric, categorical = feature_columns()
    train_idx, eval_idx = np.arange(15), np.arange(15, 20)
    x_train, x_eval = fit_transform_features(table, train_idx, eval_idx, numeric, categorical)
    col = numeric.index("sigma_max")
    train_values = table.iloc[train_idx]["sigma_max"].to_numpy()
    mean, std = train_values.mean(), train_values.std()
    np.testing.assert_allclose(x_train[:, col], (train_values - mean) / std, rtol=1e-4, atol=1e-5)
    eval_values = table.iloc[eval_idx]["sigma_max"].to_numpy()
    np.testing.assert_allclose(x_eval[:, col], (eval_values - mean) / std, rtol=1e-4, atol=1e-5)
    assert x_eval[:, col].mean() > 1.0  # transformed with train statistics, not re-centred


def _numpy_first_crossing(curve, cycles, level):
    hit = np.flatnonzero(curve >= level - 1e-6)
    if hit.size == 0:
        return cycles[-1]
    j = int(hit[0])
    if j == 0:
        return cycles[0]
    frac = np.clip((level - curve[j - 1]) / (curve[j] - curve[j - 1]), 0.0, 1.0)
    return cycles[j - 1] + frac * (cycles[j] - cycles[j - 1])


def test_first_crossing_matches_reference_and_has_gradients():
    rng = np.random.default_rng(0)
    grid = make_event_grid()
    level = transition_threshold()
    extension = np.cumsum(rng.uniform(0.0, 4e-6, size=(16, grid.size)), axis=1)
    th_a = threshold_from_extension_tf(tf.constant(extension, tf.float32))
    th_c = threshold_from_extension_tf(tf.constant(0.7 * extension, tf.float32))
    cycles = tf.constant(np.outer(rng.uniform(1e4, 1e6, 16), grid), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(cycles)
        crossing = first_crossing_cycle(th_a, th_c, cycles, level)
    expected = [
        _numpy_first_crossing(np.maximum(a, c), n, level)
        for a, c, n in zip(th_a.numpy(), th_c.numpy(), cycles.numpy())
    ]
    np.testing.assert_allclose(crossing.numpy(), expected, rtol=1e-5)
    assert np.all(np.isfinite(tape.gradient(crossing, cycles).numpy()))


def test_concavity_penalty_is_zero_on_physics_reference():
    dataset = _reference_dataset()
    grid = tf.constant(dataset.event_grid)
    ref_a = threshold_from_extension_tf(tf.constant(dataset.delta_a))
    ref_c = threshold_from_extension_tf(tf.constant(dataset.delta_c))
    assert float(threshold_concavity_loss(ref_a, ref_c, ref_a, ref_c, grid)) < 1e-12
    convex = tf.tile(4.0 + 0.75 * tf.constant(dataset.event_grid)[None, :] ** 4, [ref_a.shape[0], 1])
    assert float(threshold_concavity_loss(convex, convex, ref_a, ref_c, grid)) > 1e-3


def test_splits_do_not_overlap():
    for train_idx, test_idx in outer_splits(56):
        assert not set(train_idx) & set(test_idx)
        for in_train, in_val in inner_splits(train_idx):
            assert set(in_train) | set(in_val) <= set(train_idx)
            assert not set(in_val) & set(test_idx)


def test_synthetic_table_is_reproducible():
    pd.testing.assert_frame_equal(generate_specimen_table(5, seed=3), generate_specimen_table(5, seed=3))
