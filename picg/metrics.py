"""Life-prediction metrics in log cycles."""

from __future__ import annotations

import numpy as np


def cycle_metrics(true_cycles: np.ndarray, pred_cycles: np.ndarray) -> dict[str, float | int]:
    """MAE / RMSE of ``log1p`` cycles and the late-prediction count.

    A prediction is *late* (non-conservative) when it exceeds the observed
    life, i.e. the model would let a component run past its real transition.
    """

    true_cycles = np.asarray(true_cycles, dtype=float)
    pred_cycles = np.asarray(pred_cycles, dtype=float)
    valid = np.isfinite(true_cycles) & np.isfinite(pred_cycles) & (true_cycles > 0) & (pred_cycles > 0)
    error = np.log1p(pred_cycles[valid]) - np.log1p(true_cycles[valid])
    if error.size == 0:
        return {"n": int(len(true_cycles)), "n_valid": 0, "mae_log": np.nan, "rmse_log": np.nan, "late": 0}
    return {
        "n": int(len(true_cycles)),
        "n_valid": int(valid.sum()),
        "mae_log": float(np.mean(np.abs(error))),
        "rmse_log": float(np.sqrt(np.mean(np.square(error)))),
        "late": int(np.sum(error > 0.0)),
    }
