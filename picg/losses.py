"""Training objective: log-cycle MSE plus four physics penalties.

    L = w_cycle  * MSE(log1p N_pred, log1p N_obs)
      + w_shape  * W1 distance between normalised extension curves and the
                   physics reference (curve shape)
      + w_ratio  * Huber(c/a terminal ratio - physics ratio)
      + w_concav * positive curvature of the threshold curve at 17 anchors
      + w_warp   * (log warp / max warp)^2  (keeps the cycle warp small)

The cycle loss uses a differentiable soft first-crossing so gradients flow
through the whole decoder.
"""

from __future__ import annotations

import tensorflow as tf

EPSILON = 1e-7


def _pretransition(values: tf.Tensor, grid: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    idx = tf.where(grid <= 1.0 + 1e-6)[:, 0]
    return tf.gather(values, idx, axis=1), tf.gather(grid, idx)


def huber(error: tf.Tensor, delta: float) -> tf.Tensor:
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    return tf.reduce_mean(0.5 * quadratic**2 + delta * (abs_error - quadratic))


def soft_first_crossing_cycle(
    threshold_a: tf.Tensor,
    threshold_c: tf.Tensor,
    cycle_grid: tf.Tensor,
    level: float,
    beta: float = 18.0,
) -> tf.Tensor:
    """Expected first cycle where ``max(threshold_a, threshold_c) >= level``."""

    hit = tf.nn.sigmoid(beta * (tf.maximum(threshold_a, threshold_c) - level))
    survival = tf.math.cumprod(1.0 - hit + 1e-6, axis=1, exclusive=True)
    tail = tf.reduce_prod(1.0 - hit + 1e-6, axis=1)
    return tf.reduce_sum(hit * survival * cycle_grid, axis=1) + tail * cycle_grid[:, -1]


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


def threshold_concavity_loss(threshold_a, threshold_c, grid, tolerance: float = 1e-3, n_anchors: int = 17) -> tf.Tensor:
    """Penalise convex bends of the threshold curve on ``s <= 1``.

    Curvature is measured on ``n_anchors`` uniform anchors and divided by
    each curve's range so the penalty is independent of grid and units.
    """

    def one_front(values):
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
        return tf.reduce_mean(tf.square(tf.nn.relu(curvature / spread - tolerance)))

    return 0.5 * (one_front(threshold_a) + one_front(threshold_c))


def warp_size_loss(log_warp: tf.Tensor, max_log_warp: float) -> tf.Tensor:
    return tf.reduce_mean(tf.square(log_warp / max_log_warp))
