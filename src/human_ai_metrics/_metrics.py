"""Binary human–AI interaction metrics, using a fixed 16-scenario taxonomy.

All functions operate at the *finding* level. A positive decision means the
finding is present and a negative decision means it is absent.

The four observed vectors do not establish causal automation bias. The words
"overreliance" and "underreliance" identify observable decision patterns.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

import numpy as np

__all__ = [
    "scenario_labels", "scenario_counts", "human_ai_metrics",
    "metric_details", "overreliance_rate", "underreliance_rate",
    "correct_ai_reliance_rate", "correct_self_reliance_rate",
    "ai_error_correction_rate", "ai_error_propagation_rate",
    "human_error_correction_rate", "human_error_persistence_rate",
    "human_induced_error_rate", "final_omission_rate",
    "final_commission_rate", "METRIC_SCENARIOS", "SCENARIO_DESCRIPTIONS",
]

# The order of each row is (y_true, y_ai, y_human_initial, y_human_final).
# +: finding present, -: finding absent.
SCENARIO_DESCRIPTIONS = {
    1: ("+", "+", "+", "+", "correct concordance", "true positive"),
    2: ("+", "+", "+", "-", "reversal of correct agreement", "false negative"),
    3: ("+", "+", "-", "+", "correct AI reliance", "true positive"),
    4: ("+", "+", "-", "-", "underreliance", "false negative"),
    5: ("+", "-", "+", "+", "correct self-reliance", "true positive"),
    6: ("+", "-", "+", "-", "overreliance", "false negative"),
    7: ("+", "-", "-", "+", "correction of concordant error", "true positive"),
    8: ("+", "-", "-", "-", "concordant error retained", "false negative"),
    9: ("-", "+", "+", "+", "concordant error retained", "false positive"),
    10: ("-", "+", "+", "-", "correction of concordant error", "true negative"),
    11: ("-", "+", "-", "+", "overreliance", "false positive"),
    12: ("-", "+", "-", "-", "correct self-reliance", "true negative"),
    13: ("-", "-", "+", "+", "underreliance", "false positive"),
    14: ("-", "-", "+", "-", "correct AI reliance", "true negative"),
    15: ("-", "-", "-", "+", "reversal of correct agreement", "false positive"),
    16: ("-", "-", "-", "-", "correct concordance", "true negative"),
}

# Numerator and denominator are scenario IDs, not observation indices.
# Denominators are explicit opportunity sets, rather than all cases.
METRIC_SCENARIOS = OrderedDict([
    ("overreliance_rate", ((6, 11), (5, 6, 11, 12))),
    ("underreliance_rate", ((4, 13), (3, 4, 13, 14))),
    ("correct_ai_reliance_rate", ((3, 14), (3, 4, 13, 14))),
    ("correct_self_reliance_rate", ((5, 12), (5, 6, 11, 12))),
    ("ai_error_correction_rate", ((5, 7, 10, 12), (5, 6, 7, 8, 9, 10, 11, 12))),
    ("ai_error_propagation_rate", ((6, 8, 9, 11), (5, 6, 7, 8, 9, 10, 11, 12))),
    ("human_error_correction_rate", ((3, 7, 10, 14), (3, 4, 7, 8, 9, 10, 13, 14))),
    ("human_error_persistence_rate", ((4, 8, 9, 13), (3, 4, 7, 8, 9, 10, 13, 14))),
    ("human_induced_error_rate", ((2, 6, 11, 15), (1, 2, 5, 6, 11, 12, 15, 16))),
    ("final_omission_rate", ((2, 4, 6, 8), (1, 2, 3, 4, 5, 6, 7, 8))),
    ("final_commission_rate", ((9, 11, 13, 15), (9, 10, 11, 12, 13, 14, 15, 16))),
])


def _as_vector(name, values):
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional vector") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError("Input vectors must contain at least one observation")
    if any(_is_missing(v) for v in arr):
        raise ValueError(f"{name} contains missing or non-finite values")
    return arr


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _validate(y_true, y_ai, y_human_initial, y_human_final, *, pos_label, sample_weight):
    vectors = [
        _as_vector("y_true", y_true),
        _as_vector("y_ai", y_ai),
        _as_vector("y_human_initial", y_human_initial),
        _as_vector("y_human_final", y_human_final),
    ]
    n = len(vectors[0])
    if any(len(v) != n for v in vectors):
        raise ValueError("All four input vectors must have the same length")
    # Enforce a common binary label space without assuming the negative value is 0.
    values = [v.item() if isinstance(v, np.generic) else v for a in vectors for v in a]
    unique = []
    for v in values:
        if not any(v == u for u in unique):
            unique.append(v)
            if len(unique) > 2:
                raise ValueError("Only binary labels are supported across all four vectors")
    if not any(pos_label == u for u in unique):
        # Only a single observed negative class is valid; no positive examples required.
        if len(unique) != 1:
            raise ValueError("pos_label is not in the observed labels")
    # Computing equality against pos_label is robust for numeric and string labels.
    flags = [np.asarray(a == pos_label, dtype=bool) for a in vectors]
    if sample_weight is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.ndim != 1 or len(weights) != n:
            raise ValueError("sample_weight must be one-dimensional and match input length")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight values must be finite and nonnegative")
    return flags, weights


def _labels_from_flags(flags):
    truth, ai, initial, final = flags
    # Canonical IDs: positive-truth cases 1..8, negative-truth cases 9..16.
    # In each truth stratum, (AI, initial, final) is lexicographic with + first.
    return np.where(truth, 1, 9) + 4 * (~ai).astype(int) + 2 * (~initial).astype(int) + (~final).astype(int)


def scenario_labels(y_true, y_ai, y_human_initial, y_human_final, *, pos_label=1):
    """Return scenario IDs (1–16) for each finding, preserving input order.

    Parameters follow scikit-learn conventions. ``pos_label`` specifies the
    label meaning finding present; the other observed class means absent.
    """
    flags, _ = _validate(y_true, y_ai, y_human_initial, y_human_final,
                         pos_label=pos_label, sample_weight=None)
    return _labels_from_flags(flags)


def scenario_counts(y_true, y_ai, y_human_initial, y_human_final, *,
                    pos_label=1, sample_weight=None):
    """Return a length-16 array; array index 0 represents scenario 1.

    When sample_weight is supplied, counts are weighted sums (floats).
    """
    flags, weights = _validate(y_true, y_ai, y_human_initial, y_human_final,
                               pos_label=pos_label, sample_weight=sample_weight)
    ids = _labels_from_flags(flags)
    counts = np.bincount(ids, weights=weights, minlength=17)[1:17]
    return counts if sample_weight is not None else counts.astype(int)


def _zero_division(value):
    valid = (value == "warn" or value == 0 or value == 1 or
             (isinstance(value, (float, np.floating)) and np.isnan(value)))
    if not valid:
        raise ValueError("zero_division must be 'warn', 0, 1, or np.nan")


def _rate(numerator, denominator, zero_division, metric):
    if denominator:
        return float(numerator / denominator)
    if zero_division == "warn":
        warnings.warn(f"{metric} is ill-defined (zero denominator); returning 0.0. "
                      "Set zero_division to 0, 1, or np.nan to control this.",
                      RuntimeWarning, stacklevel=3)
        return 0.0
    return float(zero_division)


def _compute(y_true, y_ai, y_human_initial, y_human_final, *, pos_label,
             sample_weight, zero_division):
    _zero_division(zero_division)
    counts = scenario_counts(y_true, y_ai, y_human_initial, y_human_final,
                             pos_label=pos_label, sample_weight=sample_weight)
    result = OrderedDict()
    for name, (numerator_ids, denominator_ids) in METRIC_SCENARIOS.items():
        numerator = float(sum(counts[i - 1] for i in numerator_ids))
        denominator = float(sum(counts[i - 1] for i in denominator_ids))
        result[name] = {"value": _rate(numerator, denominator, zero_division, name),
                        "numerator": numerator, "denominator": denominator,
                        "numerator_scenarios": numerator_ids,
                        "denominator_scenarios": denominator_ids}
    return result


def metric_details(y_true, y_ai, y_human_initial, y_human_final, *,
                   pos_label=1, sample_weight=None, zero_division="warn"):
    """Return a mapping for every metric with rate, counts, and scenario IDs."""
    return _compute(y_true, y_ai, y_human_initial, y_human_final,
                    pos_label=pos_label, sample_weight=sample_weight,
                    zero_division=zero_division)


def human_ai_metrics(y_true, y_ai, y_human_initial, y_human_final, *,
                     pos_label=1, sample_weight=None, zero_division="warn"):
    """Compute all 11 rates as an ordered dict, with results between 0 and 1.

    Undefined rates use scikit-learn-like ``zero_division`` handling.
    For numerator/denominator transparency, see :func:`metric_details`.
    """
    details = metric_details(y_true, y_ai, y_human_initial, y_human_final,
                             pos_label=pos_label, sample_weight=sample_weight,
                             zero_division=zero_division)
    return {name: detail["value"] for name, detail in details.items()}


def _make_rate_function(metric_name):
    def metric(y_true, y_ai, y_human_initial, y_human_final, *,
               pos_label=1, sample_weight=None, zero_division="warn"):
        """Calculate a binary human–AI metric using the fixed scenario taxonomy."""
        # Only warn for this metric if undefined, not for every other metric.
        _zero_division(zero_division)
        counts = scenario_counts(y_true, y_ai, y_human_initial, y_human_final,
                                 pos_label=pos_label, sample_weight=sample_weight)
        num_ids, den_ids = METRIC_SCENARIOS[metric_name]
        numerator = float(sum(counts[i - 1] for i in num_ids))
        denominator = float(sum(counts[i - 1] for i in den_ids))
        return _rate(numerator, denominator, zero_division, metric_name)
    metric.__name__ = metric_name
    metric.__qualname__ = metric_name
    metric.__doc__ = (f"Compute {metric_name}. Numerator scenarios: "
                      f"{METRIC_SCENARIOS[metric_name][0]}; denominator scenarios: "
                      f"{METRIC_SCENARIOS[metric_name][1]}.")
    return metric


for _metric_name in METRIC_SCENARIOS:
    globals()[_metric_name] = _make_rate_function(_metric_name)
del _metric_name
