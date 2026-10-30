"""Metrics for predicted cycles to crack nucleation, in log cycles.

Every metric is computed over *all* expected specimens or not at all. A
prediction that is missing, non-finite or non-positive makes the evaluation
``INVALID``; it is never dropped so that the remaining specimens can be scored
as if coverage were complete (the audit case ``y_pred = [100, nan]`` against
``y_true = [100, 100]`` used to report an MAE of 0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

STATUS_VALID = "VALID"
STATUS_INVALID = "INVALID"
METRIC_KEYS = ("mae_log", "rmse_log", "bias_log", "late", "late_fraction")


class InvalidPredictionError(ValueError):
    """Raised when predictions cannot be evaluated as a complete set."""


@dataclass
class PredictionValidity:
    """Coverage report for one set of predictions against one set of targets."""

    n_expected: int
    n_valid: int
    invalid_specimens: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def n_invalid(self) -> int:
        return self.n_expected - self.n_valid

    @property
    def status(self) -> str:
        return STATUS_VALID if self.n_invalid == 0 and self.n_expected > 0 else STATUS_INVALID

    def describe(self) -> str:
        if self.status == STATUS_VALID:
            return f"{STATUS_VALID}: {self.n_valid}/{self.n_expected} predictions valid"
        detail = "; ".join(f"{sid}: {reason}" for sid, reason in self.reasons.items())
        return f"{STATUS_INVALID}: {self.n_valid}/{self.n_expected} predictions valid ({detail})"

    def as_dict(self) -> dict:
        data = asdict(self)
        data["n_invalid"] = self.n_invalid
        data["status"] = self.status
        return data


def _reason(value: float) -> str | None:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    if value <= 0.0:
        return f"non-positive ({value:g})"
    return None


def validate_predictions(true_cycles, pred_cycles, specimen_ids=None) -> PredictionValidity:
    """Check shape, coverage and finiteness. Target problems raise immediately.

    Invalid *targets* are a data error (nothing can be evaluated against them)
    and raise ``InvalidPredictionError``. Invalid *predictions* are recorded
    per specimen so the caller can fail or mark the run, never skip them.
    """

    true_cycles = np.asarray(true_cycles, dtype=float)
    pred_cycles = np.asarray(pred_cycles, dtype=float)
    if true_cycles.ndim != 1 or pred_cycles.ndim != 1:
        raise InvalidPredictionError(
            f"expected 1-d arrays, got shapes true={true_cycles.shape} pred={pred_cycles.shape}"
        )
    if true_cycles.shape != pred_cycles.shape:
        raise InvalidPredictionError(
            f"shape mismatch: {true_cycles.shape[0]} targets but {pred_cycles.shape[0]} predictions"
        )
    if true_cycles.size == 0:
        raise InvalidPredictionError("no specimens to evaluate (empty input)")
    if specimen_ids is None:
        specimen_ids = [str(i) for i in range(true_cycles.size)]
    specimen_ids = [str(s) for s in specimen_ids]
    if len(specimen_ids) != true_cycles.size:
        raise InvalidPredictionError(f"{len(specimen_ids)} specimen ids for {true_cycles.size} specimens")
    bad_targets = [f"{sid}: {_reason(v)}" for sid, v in zip(specimen_ids, true_cycles) if _reason(v)]
    if bad_targets:
        raise InvalidPredictionError("invalid observed cycles: " + "; ".join(bad_targets))
    reasons = {sid: r for sid, v in zip(specimen_ids, pred_cycles) if (r := _reason(v))}
    return PredictionValidity(
        n_expected=int(true_cycles.size),
        n_valid=int(true_cycles.size - len(reasons)),
        invalid_specimens=list(reasons),
        reasons=reasons,
    )


def cycle_metrics(true_cycles, pred_cycles, specimen_ids=None, on_invalid: str = "raise") -> dict:
    """MAE / RMSE / mean bias of ``log1p`` cycles and the late-prediction count.

    A prediction is *late* (non-conservative) when it exceeds the observed
    cycles to crack nucleation, i.e. the model would let a component run past
    its real transition. ``late_fraction`` and every mean use the same
    denominator, ``n_expected`` (also returned as ``denominator``).

    ``on_invalid="raise"`` raises :class:`InvalidPredictionError` for any
    non-finite or non-positive prediction. ``on_invalid="mark"`` instead
    returns ``status="INVALID"`` with every metric set to ``nan`` and the
    affected specimen ids listed; metrics are never computed over a subset.
    """

    if on_invalid not in ("raise", "mark"):
        raise ValueError("on_invalid must be 'raise' or 'mark'")
    validity = validate_predictions(true_cycles, pred_cycles, specimen_ids)
    result = {
        "status": validity.status,
        "n": validity.n_expected,
        "n_expected": validity.n_expected,
        "n_valid": validity.n_valid,
        "n_invalid": validity.n_invalid,
        "invalid_specimens": validity.invalid_specimens,
        "denominator": validity.n_expected,
    }
    if validity.status == STATUS_INVALID:
        if on_invalid == "raise":
            raise InvalidPredictionError(validity.describe())
        result.update({key: float("nan") for key in METRIC_KEYS})
        result["late"] = None
        return result
    true_cycles = np.asarray(true_cycles, dtype=float)
    pred_cycles = np.asarray(pred_cycles, dtype=float)
    error = np.log1p(pred_cycles) - np.log1p(true_cycles)
    result.update(
        {
            "mae_log": float(np.mean(np.abs(error))),
            "rmse_log": float(np.sqrt(np.mean(np.square(error)))),
            "bias_log": float(np.mean(error)),
            "late": int(np.sum(error > 0.0)),
            "late_fraction": float(np.mean(error > 0.0)),
        }
    )
    return result


def paired_bootstrap_mae_difference(
    true_cycles, pred_a, pred_b, n_boot: int = 2000, seed: int = 0, specimen_ids=None
) -> dict:
    """Percentile bootstrap over specimens of ``MAE(a) - MAE(b)`` in log cycles.

    This quantifies the sampling uncertainty of the metric *given these two
    fixed prediction sets*: it resamples blind specimens, not models, so it
    says nothing about split or initialisation variability. Both prediction
    sets must be complete and valid.
    """

    for name, pred in (("a", pred_a), ("b", pred_b)):
        validity = validate_predictions(true_cycles, pred, specimen_ids)
        if validity.status != STATUS_VALID:
            raise InvalidPredictionError(f"prediction set {name}: {validity.describe()}")
    true_cycles = np.asarray(true_cycles, dtype=float)
    err_a = np.abs(np.log1p(np.asarray(pred_a, dtype=float)) - np.log1p(true_cycles))
    err_b = np.abs(np.log1p(np.asarray(pred_b, dtype=float)) - np.log1p(true_cycles))
    rng = np.random.default_rng(seed)
    n = true_cycles.size
    draws = rng.integers(0, n, size=(n_boot, n))
    diffs = err_a[draws].mean(axis=1) - err_b[draws].mean(axis=1)
    return {
        "point": float(err_a.mean() - err_b.mean()),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "n_boot": int(n_boot),
        "n_specimens": int(n),
        "method": "percentile bootstrap over blind specimens, models held fixed",
    }
