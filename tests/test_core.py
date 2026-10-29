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
from picg.metrics import cycle_metrics
from picg.physics import invert_short_crack_threshold, short_crack_threshold, transition_extension


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
    table = generate_specimen_table(20, seed=1)
    dataset = build_dataset(table)
    numeric, categorical = feature_columns()
    train_idx, test_idx = np.arange(15), np.arange(15, 20)
    x_train, _ = fit_transform_features(dataset.table, train_idx, test_idx, numeric, categorical)
    np.testing.assert_allclose(x_train[:, : len(numeric)].mean(axis=0), 0.0, atol=1e-5)


def test_splits_do_not_overlap():
    for train_idx, test_idx in outer_splits(56):
        assert not set(train_idx) & set(test_idx)
        for in_train, in_val in inner_splits(train_idx):
            assert set(in_train) | set(in_val) <= set(train_idx)
            assert not set(in_val) & set(test_idx)


def test_synthetic_table_is_reproducible():
    pd.testing.assert_frame_equal(generate_specimen_table(5, seed=3), generate_specimen_table(5, seed=3))
