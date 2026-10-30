"""Finite values and gradients of the PCHIP interpolation and the decoder.

Regression tests for the audit finding that a constant seven-knot vector gave
``nan`` gradients: the reciprocal secant slopes were formed before the
zero/sign mask, and ``tf.where`` does not stop ``0 * inf`` in the backward pass.
"""

import numpy as np
import pytest
import tensorflow as tf

from picg.data import make_event_grid
from picg.decoder import (
    DecoderConfig,
    SLOPE_FLOOR,
    _pchip_slopes_np,
    _pchip_slopes_tf,
    bounded_log_warp_tf,
    bounded_terminal_ratio_tf,
    build_decoder_network,
    cycle_grid_tf,
    decode_extension_curves_tf,
    monotone_cubic_interpolate_np,
    monotone_cubic_interpolate_tf,
    threshold_from_extension_tf,
)
from picg.losses import first_crossing_cycle, shape_loss
from picg.physics import transition_extension, transition_threshold

CONFIG = DecoderConfig()
KNOTS = np.asarray(CONFIG.knots, dtype=np.float32)
GRID = make_event_grid()

KNOT_CASES = {
    "constant": [1.0] * 7,
    "constant_small": [1e-6] * 7,
    "pairwise_equal": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0],
    "flat_then_rise": [0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0],
    "sign_changes": [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0],
    "plateau_in_middle": [1.0, 2.0, 2.0, 2.0, 1.0, 0.5, 0.5],
    "very_small_slopes": [1.0, 1.0 + 1e-7, 1.0 + 2e-7, 1.0 + 3e-7, 1.0, 1.0, 1.0],
    "below_slope_floor": [1.0, 1.0 + 1e-13, 1.0 + 2e-13, 1.0 + 3e-13, 1.0 + 4e-13, 1.0, 1.0],
    "mixed_scales": [1e-4, 5.0, 1e-3, 1e-3, 40.0, 0.2, 0.2],
    "irregular_smooth": [0.3, 1.2, 0.9, 2.5, 2.0, 3.1, 4.0],
    "monotone_smooth": [0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9],
}


def _tf_interp_and_grad(values):
    y = tf.Variable(tf.constant([values], tf.float32))
    with tf.GradientTape() as tape:
        out = monotone_cubic_interpolate_tf(tf.constant(KNOTS), tf.constant(GRID), y)
        objective = tf.reduce_sum(out * tf.constant(GRID)[None, :])
    return out.numpy()[0], tape.gradient(objective, y).numpy()[0]


@pytest.mark.parametrize("name", sorted(KNOT_CASES))
def test_pchip_forward_and_backward_are_finite(name):
    values = KNOT_CASES[name]
    out, grad = _tf_interp_and_grad(values)
    assert np.all(np.isfinite(out)), f"{name}: non-finite interpolant"
    assert np.all(np.isfinite(grad)), f"{name}: non-finite gradient {grad}"


@pytest.mark.parametrize("name", sorted(KNOT_CASES))
def test_pchip_tf_matches_numpy(name):
    values = KNOT_CASES[name]
    out, _ = _tf_interp_and_grad(values)
    ref = monotone_cubic_interpolate_np(GRID, KNOTS, np.asarray(values, dtype=float))
    scale = max(float(np.max(np.abs(ref))), 1e-6)
    np.testing.assert_allclose(out, ref, rtol=2e-5, atol=2e-6 * scale)
    tf_slopes = _pchip_slopes_tf(tf.constant(KNOTS), tf.constant([values], tf.float32)).numpy()[0]
    np_slopes = _pchip_slopes_np(KNOTS.astype(float), np.asarray(values, dtype=float))
    np.testing.assert_allclose(tf_slopes, np_slopes, rtol=1e-4, atol=1e-6 * max(float(np.max(np.abs(np_slopes))), 1.0))


def test_constant_knots_give_constant_curve_and_zero_internal_slopes():
    out, grad = _tf_interp_and_grad(KNOT_CASES["constant"])
    np.testing.assert_allclose(out, 1.0, rtol=0, atol=1e-6)
    slopes = _pchip_slopes_tf(tf.constant(KNOTS), tf.constant([[1.0] * 7], tf.float32)).numpy()[0]
    np.testing.assert_array_equal(slopes, 0.0)
    assert np.all(np.isfinite(grad))
    # The interpolant of constant knots is a convex combination of the knots,
    # so the gradient of its weighted sum is the (positive) weight mass.
    assert np.all(grad >= -1e-6) and grad.sum() > 0


@pytest.mark.parametrize("name", ["flat_then_rise", "plateau_in_middle", "monotone_smooth", "pairwise_equal"])
def test_pchip_is_shape_preserving(name):
    values = np.asarray(KNOT_CASES[name], dtype=float)
    out, _ = _tf_interp_and_grad(values.tolist())
    for i in range(len(KNOTS) - 1):
        inside = (GRID >= KNOTS[i]) & (GRID <= KNOTS[i + 1])
        lo, hi = min(values[i], values[i + 1]), max(values[i], values[i + 1])
        assert np.all(out[inside] >= lo - 1e-5), f"{name}: undershoot in segment {i}"
        assert np.all(out[inside] <= hi + 1e-5), f"{name}: overshoot in segment {i}"
        if values[i] == values[i + 1]:
            np.testing.assert_allclose(out[inside], values[i], atol=1e-6)
        else:
            sign = np.sign(values[i + 1] - values[i])
            assert np.all(sign * np.diff(out[inside]) >= -1e-6), f"{name}: non-monotone segment {i}"


def test_slope_floor_is_below_anything_the_decoder_produces():
    # Knots are softplus outputs + EPSILON on a grid with spacing >= 0.15, so
    # a secant slope below SLOPE_FLOOR needs knot differences below 1.5e-13,
    # which is far below float32 resolution for knots of order 1e-8 or more.
    assert SLOPE_FLOOR <= 1e-12


def test_decoder_curves_positive_monotone_and_finite_for_degenerate_knots():
    raw_cases = tf.constant(
        [
            [0.0] * 7,
            [-30.0] * 7,
            [30.0] * 7,
            [-30.0, -30.0, -30.0, 30.0, 30.0, 30.0, 30.0],
            [5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0],
        ],
        tf.float32,
    )
    ratio = tf.fill((5,), 0.8)
    delta_a, delta_c = decode_extension_curves_tf(
        raw_cases, raw_cases, ratio, tf.constant(KNOTS), tf.constant(GRID), transition_extension()
    )
    for curve in (delta_a, delta_c):
        c = curve.numpy()
        assert np.all(np.isfinite(c))
        assert np.all(c[:, 1:] > 0.0)
        assert np.all(np.diff(c, axis=1) >= -1e-12)


def test_gradients_through_decoder_parameters_are_finite_for_degenerate_outputs():
    """Backward pass through the real network with biases that produce constant knots."""

    n_features = 4
    model = build_decoder_network(
        n_features, len(KNOTS), (8,), 0.0, 0.0, CONFIG,
        initial_density_a=np.ones(7, dtype=np.float32),
        initial_density_c=np.ones(7, dtype=np.float32),
    )
    # zero the hidden -> knot kernels so every specimen gets exactly constant knots
    for name in ("density_a_knots", "density_c_knots"):
        layer = model.get_layer(name)
        kernel, bias = layer.get_weights()
        layer.set_weights([np.zeros_like(kernel), bias])
    x = tf.constant(np.random.default_rng(0).normal(size=(6, n_features)), tf.float32)
    crossing = tf.constant([1e5, 2e5, 3e5, 4e5, 5e5, 6e5], tf.float32)
    observed = crossing * 1.5
    grid_t, knots_t = tf.constant(GRID), tf.constant(KNOTS)
    ref_a = tf.tile(tf.constant(GRID * transition_extension())[None, :], [6, 1])
    with tf.GradientTape() as tape:
        da_raw, dc_raw, ratio_raw, warp_raw = model(x, training=False)
        ratio = bounded_terminal_ratio_tf(ratio_raw, CONFIG)
        log_warp = bounded_log_warp_tf(warp_raw, CONFIG)
        delta_a, delta_c = decode_extension_curves_tf(da_raw, dc_raw, ratio, knots_t, grid_t, transition_extension())
        th_a, th_c = threshold_from_extension_tf(delta_a), threshold_from_extension_tf(delta_c)
        cycles = cycle_grid_tf(log_warp, crossing, grid_t, CONFIG)
        pred = first_crossing_cycle(th_a, th_c, cycles, transition_threshold())
        loss = tf.reduce_mean(tf.square(tf.math.log1p(pred) - tf.math.log1p(observed)))
        loss += shape_loss(delta_a, delta_c, ref_a, 0.8 * ref_a, grid_t)
    grads = tape.gradient(loss, model.trainable_variables)
    assert np.isfinite(float(loss))
    for variable, grad in zip(model.trainable_variables, grads):
        assert grad is not None, variable.name
        assert np.all(np.isfinite(grad.numpy())), f"non-finite gradient for {variable.name}"
    knot_variables = model.get_layer("density_a_knots").trainable_variables
    assert len(knot_variables) == 2  # kernel and bias of the knot head were part of the checked set


@pytest.mark.parametrize("name", ["monotone_smooth", "irregular_smooth", "sign_changes"])
def test_pchip_gradient_matches_finite_differences(name):
    """Central differences in float64 NumPy against the TF gradient, away from
    knots with equal values or sign changes exactly at a knot (non-differentiable
    points of the Fritsch-Carlson rule)."""

    values = np.asarray(KNOT_CASES[name], dtype=float)
    weights = GRID.astype(float)

    def objective(v):
        return float(np.sum(monotone_cubic_interpolate_np(GRID, KNOTS, v) * weights))

    y = tf.Variable(tf.constant([values], tf.float32))
    with tf.GradientTape() as tape:
        out = monotone_cubic_interpolate_tf(tf.constant(KNOTS), tf.constant(GRID), y)
        obj = tf.reduce_sum(out * tf.constant(GRID)[None, :])
    analytic = tape.gradient(obj, y).numpy()[0]
    eps = 1e-5 * max(1.0, float(np.max(np.abs(values))))
    numeric = np.zeros_like(values)
    for i in range(len(values)):
        plus, minus = values.copy(), values.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric[i] = (objective(plus) - objective(minus)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-3, atol=2e-3)
