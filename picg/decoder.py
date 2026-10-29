"""Knot decoder: 16 network outputs -> two monotone crack-growth curves.

Per specimen the network emits

* 7 raw density knots for the depth front ``a``,
* 7 raw density knots for the surface front ``c``,
* 1 raw terminal ratio ``c/a`` (extension at ``s = 1``),
* 1 raw cycle warp.

The decoder maps them to curves that are monotone *by construction*:

1. ``softplus`` makes every density knot strictly positive.
2. A shape-preserving cubic (PCHIP) interpolates the knots onto the event
   grid ``s`` in ``[0, 1.2]``; it stays between neighbouring knot values, so
   the density stays positive.
3. A cumulative trapezoid integral turns density into crack extension, which
   is therefore non-decreasing.
4. The depth curve is normalised so that ``delta_a(s=1)`` equals the
   transition extension; the surface curve is scaled by the bounded ratio.
5. The threshold law maps extension to a threshold curve (monotone again).
6. The event axis becomes cycles: ``N(s) = s * N_physics * exp(warp)`` for
   ``s <= 1``, with a bounded ``warp``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .data import DEFAULT_KNOTS
from .physics import DEFAULT_MATERIAL, SyntheticMaterial

EPSILON = 1e-8


@dataclass(frozen=True)
class DecoderConfig:
    knots: tuple[float, ...] = DEFAULT_KNOTS
    max_log_cycle_warp: float = 3.2
    min_terminal_ratio: float = 0.5
    max_terminal_ratio: float = 1.1
    post_threshold_cycles: float = 10_000.0


# --------------------------------------------------------------------------
# Shape-preserving cubic interpolation (PCHIP), NumPy and TensorFlow
# --------------------------------------------------------------------------


def _pchip_slopes_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(x)
    delta = np.diff(y) / np.maximum(h, EPSILON)
    slopes = np.zeros_like(y, dtype=float)
    slopes[0], slopes[-1] = delta[0], delta[-1]
    for i in range(1, len(y) - 1):
        d_left, d_right = delta[i - 1], delta[i]
        if d_left * d_right > 0.0:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / d_left + w2 / d_right)
    return slopes


def monotone_cubic_interpolate_np(eval_grid: np.ndarray, knot_grid: np.ndarray, knot_values: np.ndarray) -> np.ndarray:
    """Evaluate a C1 shape-preserving cubic through one knot vector."""

    x = np.asarray(knot_grid, dtype=float)
    y = np.asarray(knot_values, dtype=float)
    s = np.asarray(eval_grid, dtype=float)
    slopes = _pchip_slopes_np(x, y)
    right = np.clip(np.searchsorted(x, s, side="right"), 1, len(x) - 1)
    left = right - 1
    h = x[right] - x[left]
    t = (s - x[left]) / np.maximum(h, EPSILON)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * y[left] + h10 * h * slopes[left] + h01 * y[right] + h11 * h * slopes[right]


def _pchip_slopes_tf(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    h = x[1:] - x[:-1]
    delta = (y[:, 1:] - y[:, :-1]) / tf.maximum(h[None, :], EPSILON)
    d_left, d_right = delta[:, :-1], delta[:, 1:]
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    denominator = w1[None, :] / d_left + w2[None, :] / d_right
    safe = tf.where(tf.abs(denominator) > EPSILON, denominator, tf.ones_like(denominator))
    internal = (w1[None, :] + w2[None, :]) / safe
    internal = tf.where(d_left * d_right > 0.0, internal, tf.zeros_like(internal))
    return tf.concat([delta[:, :1], internal, delta[:, -1:]], axis=1)


def monotone_cubic_interpolate_tf(knot_grid: tf.Tensor, eval_grid: tf.Tensor, knot_values: tf.Tensor) -> tf.Tensor:
    """Differentiable batch PCHIP: ``knot_values`` is ``(batch, n_knots)``."""

    x = tf.cast(knot_grid, tf.float32)
    s = tf.cast(eval_grid, tf.float32)
    y = tf.cast(knot_values, tf.float32)
    slopes = _pchip_slopes_tf(x, y)
    right = tf.clip_by_value(tf.searchsorted(x, s, side="right"), 1, tf.shape(x)[0] - 1)
    left = right - 1
    x_left = tf.gather(x, left)
    h = tf.gather(x, right) - x_left
    t = (s - x_left) / tf.maximum(h, EPSILON)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return (
        h00[None, :] * tf.gather(y, left, axis=1)
        + (h10 * h)[None, :] * tf.gather(slopes, left, axis=1)
        + h01[None, :] * tf.gather(y, right, axis=1)
        + (h11 * h)[None, :] * tf.gather(slopes, right, axis=1)
    )


# --------------------------------------------------------------------------
# Decoder steps
# --------------------------------------------------------------------------


def cumulative_trapezoid_tf(values: tf.Tensor, grid: tf.Tensor) -> tf.Tensor:
    ds = grid[1:] - grid[:-1]
    increments = 0.5 * (values[:, 1:] + values[:, :-1]) * ds[None, :]
    return tf.concat([tf.zeros_like(increments[:, :1]), tf.cumsum(increments, axis=1)], axis=1)


def bounded_terminal_ratio_tf(raw: tf.Tensor, config: DecoderConfig) -> tf.Tensor:
    width = config.max_terminal_ratio - config.min_terminal_ratio
    return config.min_terminal_ratio + width * tf.nn.sigmoid(tf.reshape(raw, [-1]))


def bounded_log_warp_tf(raw: tf.Tensor, config: DecoderConfig) -> tf.Tensor:
    return config.max_log_cycle_warp * tf.tanh(tf.reshape(raw, [-1]))


def decode_extension_curves_tf(
    density_a_raw: tf.Tensor,
    density_c_raw: tf.Tensor,
    terminal_ratio: tf.Tensor,
    knot_grid: tf.Tensor,
    event_grid: tf.Tensor,
    delta_transition: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Raw density knots -> monotone extension curves for both fronts [m]."""

    curves = []
    for raw in (density_a_raw, density_c_raw):
        knots = tf.nn.softplus(raw) + EPSILON
        density = tf.maximum(monotone_cubic_interpolate_tf(knot_grid, event_grid, knots), EPSILON)
        mass = cumulative_trapezoid_tf(density, event_grid)
        at_transition = tf.gather(mass, tf.argmin(tf.abs(event_grid - 1.0)), axis=1)[:, None]
        curves.append(mass * (delta_transition / tf.maximum(at_transition, EPSILON)))
    delta_a, delta_c = curves
    return delta_a, delta_c * terminal_ratio[:, None]


def threshold_from_extension_tf(extension: tf.Tensor, material: SyntheticMaterial = DEFAULT_MATERIAL) -> tf.Tensor:
    a_i = material.intrinsic_length_m
    a_eh = material.el_haddad_length_m
    extension = tf.nn.relu(extension)
    return material.long_crack_threshold * tf.sqrt((extension + a_i) / (extension + a_i + a_eh))


def cycle_grid_tf(log_warp: tf.Tensor, physics_crossing: tf.Tensor, event_grid: tf.Tensor, config: DecoderConfig) -> tf.Tensor:
    """Event coordinate -> cycles, anchored on the physics crossing cycle."""

    scale = (physics_crossing * tf.exp(log_warp))[:, None]
    s = event_grid[None, :]
    return tf.where(s <= 1.0, s * scale, scale + (s - 1.0) * config.post_threshold_cycles)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


def _inverse_softplus(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 1e-6, None)
    return np.log(np.expm1(values)).astype(np.float32)


def build_decoder_network(
    n_features: int,
    n_knots: int,
    hidden_units: tuple[int, ...],
    dropout_rate: float,
    l2_strength: float,
    config: DecoderConfig,
    initial_density_a: np.ndarray | None = None,
    initial_density_c: np.ndarray | None = None,
    initial_log_warp: float = 0.0,
    initial_terminal_ratio: float = 0.8,
) -> tf.keras.Model:
    """Small MLP with four heads: 7 + 7 + 1 + 1 = 16 decoder parameters."""

    inputs = tf.keras.Input(shape=(n_features,), name="static_features")
    regularizer = tf.keras.regularizers.l2(l2_strength) if l2_strength else None
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="swish", kernel_regularizer=regularizer)(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    def bias(values):
        return "zeros" if values is None else tf.keras.initializers.Constant(values)

    ratio_unit = (initial_terminal_ratio - config.min_terminal_ratio) / (config.max_terminal_ratio - config.min_terminal_ratio)
    ratio_unit = float(np.clip(ratio_unit, 1e-3, 1 - 1e-3))
    warp_unit = float(np.clip(initial_log_warp / config.max_log_cycle_warp, -0.95, 0.95))
    density_a = tf.keras.layers.Dense(
        n_knots, bias_initializer=bias(None if initial_density_a is None else _inverse_softplus(initial_density_a)),
        name="density_a_knots",
    )(x)
    density_c = tf.keras.layers.Dense(
        n_knots, bias_initializer=bias(None if initial_density_c is None else _inverse_softplus(initial_density_c)),
        name="density_c_knots",
    )(x)
    ratio = tf.keras.layers.Dense(
        1, bias_initializer=tf.keras.initializers.Constant(np.log(ratio_unit / (1 - ratio_unit))), name="terminal_ratio"
    )(x)
    warp = tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.Constant(np.arctanh(warp_unit)), name="cycle_warp")(x)
    return tf.keras.Model(inputs, [density_a, density_c, ratio, warp], name="knot_decoder")


def density_knots_from_curves(delta_curves: np.ndarray, event_grid: np.ndarray, knot_grid: np.ndarray) -> np.ndarray:
    """Approximate positive density knots from reference extension curves.

    Used only to initialise the output biases near a physical shape.
    """

    rows = []
    for row in np.asarray(delta_curves, dtype=float):
        at_knots = np.interp(knot_grid, event_grid, row)
        segment = np.maximum(np.diff(at_knots) / np.maximum(np.diff(knot_grid), EPSILON), EPSILON)
        density = np.concatenate([segment[:1], 0.5 * (segment[:-1] + segment[1:]), segment[-1:]])
        rows.append(np.maximum(density / max(float(np.median(density)), EPSILON), 1e-3))
    return np.asarray(rows, dtype=np.float32)
