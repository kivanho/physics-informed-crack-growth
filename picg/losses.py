"""Training objective: log-cycle MSE plus four physics penalties.

    L = w_cycle  * MSE(log1p N_pred, log1p N_obs)
      + w_shape  * W1 distance between normalised extension curves and the
                   physics reference (curve shape)
      + w_ratio  * Huber(c/a terminal ratio - physics ratio)
      + w_concav * convex curvature of the threshold curve at 17 anchors,
                   beyond what the physics reference curve itself has
      + w_warp   * (log warp / max warp)^2  (keeps the cycle warp small)

``N_pred`` is the cycle count to crack nucleation, i.e. the first crossing of
the short-to-long-crack transition threshold. Training and prediction use the
same function for it (``first_crossing_cycle``): a linearly interpolated hard
crossing, which is differentiable almost everywhere with respect to the
threshold curve and the cycle axis.
"""

from __future__ import annotations

import tensorflow as tf

EPSILON = 1e-7
CROSSING_TOLERANCE = 1e-6


def _pretransition(values: tf.Tensor, grid: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    idx = tf.where(grid <= 1.0 + 1e-6)[:, 0]
    return tf.gather(values, idx, axis=1), tf.gather(grid, idx)


def huber(error: tf.Tensor, delta: float) -> tf.Tensor:
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    return tf.reduce_mean(0.5 * quadratic**2 + delta * (abs_error - quadratic))


def first_crossing_cycle(
    threshold_a: tf.Tensor,
    threshold_c: tf.Tensor,
    cycle_grid: tf.Tensor,
    level: float,
    tolerance: float = CROSSING_TOLERANCE,
) -> tf.Tensor:
    """Interpolated first cycle where ``max(threshold_a, threshold_c) >= level``.

    The crossing is located on the grid and linearly interpolated between the
    last point below and the first point at or above ``level``. Gradients flow
    through the two bracketing threshold values and cycle-grid points. A curve
    that never reaches ``level`` returns the last grid cycle (censored).
    """

    curve = tf.maximum(threshold_a, threshold_c)
    cycle_grid = tf.cast(cycle_grid, curve.dtype)
    hit = curve >= level - tolerance
    any_hit = tf.reduce_any(hit, axis=1)
    j = tf.argmax(tf.cast(hit, tf.int32), axis=1, output_type=tf.int32)
    j_prev = tf.maximum(j - 1, 0)
    v_hi = tf.gather(curve, j, axis=1, batch_dims=1)
    v_lo = tf.gather(curve, j_prev, axis=1, batch_dims=1)
    n_hi = tf.gather(cycle_grid, j, axis=1, batch_dims=1)
    n_lo = tf.gather(cycle_grid, j_prev, axis=1, batch_dims=1)
    frac = tf.clip_by_value((level - v_lo) / tf.maximum(v_hi - v_lo, 1e-12), 0.0, 1.0)
    crossing = tf.where(j > 0, n_lo + frac * (n_hi - n_lo), n_hi)
    return tf.where(any_hit, crossing, cycle_grid[:, -1])


def log_cycle_loss(pred_cycles: tf.Tensor, true_cycles: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(tf.math.log1p(pred_cycles) - tf.math.log1p(true_cycles)))


def shape_loss(pred_a, pred_c, ref_a, ref_c, grid, smooth_eps: float = 1e-4) -> tf.Tensor:
    """Wasserstein-1 distance between terminal-normalised extension curves.

    A monotone curve divided by its value at ``s = 1`` is the CDF of its
    growth allocation, so the area between two such curves is the W1 distance
    between where the two models place crack growth in time.
    """

    def one_front(pred, ref):
        pred_w, s = _pretransition(pred, grid)
        ref_w, _ = _pretransition(ref, grid)
        error = pred_w / tf.maximum(pred_w[:, -1:], EPSILON) - ref_w / tf.maximum(ref_w[:, -1:], EPSILON)
        smooth_abs = tf.sqrt(tf.square(error) + smooth_eps**2) - smooth_eps
        area = tf.reduce_sum(0.5 * (smooth_abs[:, 1:] + smooth_abs[:, :-1]) * (s[1:] - s[:-1])[None, :], axis=1)
        return tf.reduce_mean(area / (s[-1] - s[0]))

    return 0.5 * (one_front(pred_a, ref_a) + one_front(pred_c, ref_c))


def terminal_ratio_loss(pred_ratio, ref_a, ref_c, grid, delta: float = 0.05) -> tf.Tensor:
    ref_a_w, _ = _pretransition(ref_a, grid)
    ref_c_w, _ = _pretransition(ref_c, grid)
    ref_ratio = ref_c_w[:, -1] / tf.maximum(ref_a_w[:, -1], EPSILON)
    return huber(pred_ratio - ref_ratio, delta)


def normalised_curvature(values: tf.Tensor, grid: tf.Tensor, n_anchors: int = 17) -> tf.Tensor:
    """Second derivative on ``n_anchors`` uniform anchors of ``s <= 1``.

    Divided by each curve's range so the value is independent of grid and
    units. Positive means convex (bending upwards).
    """

    window, s = _pretransition(values, grid)
    n_points = tf.shape(s)[0]
    x = tf.linspace(s[0], s[-1], tf.minimum(n_anchors, n_points))
    right = tf.clip_by_value(tf.searchsorted(s, x, side="left"), 1, n_points - 1)
    left = right - 1
    x_left, x_right = tf.gather(s, left), tf.gather(s, right)
    frac = (x - x_left) / tf.maximum(x_right - x_left, EPSILON)
    y_left, y_right = tf.gather(window, left, axis=1), tf.gather(window, right, axis=1)
    y = y_left + frac[None, :] * (y_right - y_left)
    dx = tf.maximum(x[1:] - x[:-1], EPSILON)
    slopes = (y[:, 1:] - y[:, :-1]) / dx[None, :]
    curvature = (slopes[:, 1:] - slopes[:, :-1]) / tf.maximum(0.5 * (dx[1:] + dx[:-1]), EPSILON)[None, :]
    spread = tf.maximum(tf.reduce_max(y, axis=1, keepdims=True) - tf.reduce_min(y, axis=1, keepdims=True), EPSILON)
    return curvature / spread


def threshold_concavity_loss(
    threshold_a,
    threshold_c,
    ref_threshold_a,
    ref_threshold_c,
    grid,
    tolerance: float = 1e-3,
    n_anchors: int = 17,
) -> tf.Tensor:
    """Penalise convex bends of the threshold curve beyond the reference.

    Physical threshold curves are predominantly concave-down, so convex bends
    are penalised. Where the physics reference curve is itself convex (for
    example small cracks at low stress early in life), the decoded curve may
    be as convex as the reference without penalty; only excess convexity
    counts. The penalty is therefore exactly zero on the reference curves and
    never pulls the decoder away from the physics shape it is fitted to.
    """

    def one_front(values, reference):
        pred = normalised_curvature(values, grid, n_anchors)
        allowed = tf.nn.relu(normalised_curvature(reference, grid, n_anchors))
        return tf.reduce_mean(tf.square(tf.nn.relu(pred - allowed - tolerance)))

    return 0.5 * (one_front(threshold_a, ref_threshold_a) + one_front(threshold_c, ref_threshold_c))


def warp_size_loss(log_warp: tf.Tensor, max_log_warp: float) -> tf.Tensor:
    return tf.reduce_mean(tf.square(log_warp / max_log_warp))
