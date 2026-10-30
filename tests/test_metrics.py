"""Invalid predictions must never improve, or silently enter, reported metrics."""

import numpy as np
import pytest

from picg.metrics import (
    STATUS_INVALID,
    STATUS_VALID,
    InvalidPredictionError,
    cycle_metrics,
    paired_bootstrap_mae_difference,
    validate_predictions,
)

TRUE = np.array([100.0, 100.0])


def test_audit_case_nan_prediction_raises_instead_of_scoring_zero():
    with pytest.raises(InvalidPredictionError, match="1/2 predictions valid"):
        cycle_metrics(TRUE, np.array([100.0, np.nan]))


def test_audit_case_marked_invalid_with_nan_metrics_and_ids():
    result = cycle_metrics(TRUE, np.array([100.0, np.nan]), specimen_ids=["S001", "S002"], on_invalid="mark")
    assert result["status"] == STATUS_INVALID
    assert result["n_expected"] == 2 and result["n_valid"] == 1 and result["n_invalid"] == 1
    assert result["invalid_specimens"] == ["S002"]
    assert np.isnan(result["mae_log"]) and np.isnan(result["rmse_log"]) and np.isnan(result["bias_log"])
    assert result["late"] is None
    assert result["denominator"] == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -5.0])
def test_each_invalid_value_kind_is_detected_with_reason(bad):
    validity = validate_predictions(TRUE, np.array([100.0, bad]), specimen_ids=["a", "b"])
    assert validity.status == STATUS_INVALID
    assert validity.invalid_specimens == ["b"]
    assert "b" in validity.reasons and validity.reasons["b"] != ""
    with pytest.raises(InvalidPredictionError):
        cycle_metrics(TRUE, np.array([100.0, bad]))


def test_shape_mismatch_raises_in_both_modes():
    for mode in ("raise", "mark"):
        with pytest.raises(InvalidPredictionError, match="shape mismatch"):
            cycle_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), on_invalid=mode)
    with pytest.raises(InvalidPredictionError, match="1-d"):
        cycle_metrics(np.ones((2, 2)), np.ones((2, 2)))


def test_empty_inputs_raise_in_both_modes():
    for mode in ("raise", "mark"):
        with pytest.raises(InvalidPredictionError, match="empty"):
            cycle_metrics(np.array([]), np.array([]), on_invalid=mode)


def test_invalid_targets_always_raise():
    with pytest.raises(InvalidPredictionError, match="observed"):
        cycle_metrics(np.array([100.0, np.nan]), np.array([100.0, 100.0]), on_invalid="mark")
    with pytest.raises(InvalidPredictionError, match="observed"):
        cycle_metrics(np.array([100.0, 0.0]), np.array([100.0, 100.0]), on_invalid="mark")


def test_wrong_number_of_specimen_ids_raises():
    with pytest.raises(InvalidPredictionError, match="specimen ids"):
        cycle_metrics(TRUE, TRUE, specimen_ids=["only_one"])


def test_complete_coverage_reports_counts_and_shared_denominator():
    true = np.array([100.0, 100.0, 100.0, 100.0])
    pred = np.array([200.0, 50.0, 100.0, 300.0])
    result = cycle_metrics(true, pred, specimen_ids=list("abcd"))
    assert result["status"] == STATUS_VALID
    assert result["n_expected"] == result["n_valid"] == result["denominator"] == 4
    assert result["n_invalid"] == 0 and result["invalid_specimens"] == []
    assert result["late"] == 2 and result["late_fraction"] == 0.5
    expected_error = np.log1p(pred) - np.log1p(true)
    assert result["mae_log"] == pytest.approx(np.mean(np.abs(expected_error)))
    assert result["bias_log"] == pytest.approx(np.mean(expected_error))


def test_invalid_prediction_cannot_lower_mae():
    true = np.array([100.0, 100.0, 100.0])
    good = cycle_metrics(true, np.array([100.0, 100.0, 1000.0]))
    assert good["mae_log"] > 0.5
    marked = cycle_metrics(true, np.array([100.0, 100.0, np.nan]), on_invalid="mark")
    assert marked["status"] == STATUS_INVALID
    assert not (marked["mae_log"] < good["mae_log"])  # nan compares False: never "better"


def test_bootstrap_requires_complete_valid_sets():
    true = np.array([100.0, 200.0, 300.0, 400.0])
    with pytest.raises(InvalidPredictionError):
        paired_bootstrap_mae_difference(true, np.array([100.0, 200.0, np.nan, 400.0]), true)
    out = paired_bootstrap_mae_difference(true, true * 2.0, true, n_boot=200)
    assert out["ci95_low"] <= out["point"] <= out["ci95_high"]
    assert out["n_specimens"] == 4
