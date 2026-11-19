import numpy as np
import pytest

from human_ai_metrics import (
    METRIC_SCENARIOS, SCENARIO_DESCRIPTIONS,
    scenario_labels, scenario_counts, human_ai_metrics, metric_details,
    overreliance_rate, underreliance_rate,
    correct_ai_reliance_rate, correct_self_reliance_rate,
    ai_error_correction_rate, ai_error_propagation_rate,
    human_error_correction_rate, human_error_persistence_rate,
    human_induced_error_rate, final_omission_rate, final_commission_rate,
)


def all_scenarios():
    matrix = np.array([[1 if sign == "+" else 0 for sign in SCENARIO_DESCRIPTIONS[i][:4]]
                       for i in range(1, 17)])
    return [matrix[:, j] for j in range(4)]


def test_exact_scenario_ids():
    assert scenario_labels(*all_scenarios()).tolist() == list(range(1, 17))
    assert scenario_counts(*all_scenarios()).tolist() == [1] * 16


def test_rates_exact_by_numerator_denominator():
    inputs = all_scenarios()
    details = metric_details(*inputs)
    for name, (numerator, denominator) in METRIC_SCENARIOS.items():
        assert details[name]["numerator"] == len(numerator)
        assert details[name]["denominator"] == len(denominator)
        assert details[name]["value"] == pytest.approx(len(numerator) / len(denominator))
        assert human_ai_metrics(*inputs)[name] == details[name]["value"]
    assert overreliance_rate(*inputs) == 0.5
    assert underreliance_rate(*inputs) == 0.5
    assert correct_ai_reliance_rate(*inputs) == 0.5
    assert correct_self_reliance_rate(*inputs) == 0.5
    assert ai_error_correction_rate(*inputs) == 0.5
    assert ai_error_propagation_rate(*inputs) == 0.5
    assert human_error_correction_rate(*inputs) == 0.5
    assert human_error_persistence_rate(*inputs) == 0.5
    assert human_induced_error_rate(*inputs) == 0.5
    assert final_omission_rate(*inputs) == 0.5
    assert final_commission_rate(*inputs) == 0.5


def test_weighted_counts_and_rates():
    vectors = all_scenarios()
    weights = np.arange(1, 17, dtype=float)
    np.testing.assert_allclose(scenario_counts(*vectors, sample_weight=weights), weights)
    expected = (weights[5] + weights[10]) / sum(weights[i - 1] for i in (5, 6, 11, 12))
    assert overreliance_rate(*vectors, sample_weight=weights) == pytest.approx(expected)


def test_custom_string_labels_and_inputs():
    vectors = [["present" if x else "absent" for x in v] for v in all_scenarios()]
    np.testing.assert_array_equal(scenario_labels(*vectors, pos_label="present"), np.arange(1, 17))


def test_zero_division_and_no_opportunities():
    inputs = ([1, 1], [1, 1], [1, 1], [1, 1])
    assert overreliance_rate(*inputs, zero_division=np.nan) is np.nan or np.isnan(overreliance_rate(*inputs, zero_division=np.nan))
    assert overreliance_rate(*inputs, zero_division=1) == 1.0
    with pytest.warns(RuntimeWarning):
        assert overreliance_rate(*inputs) == 0.0
    with pytest.raises(ValueError, match="zero_division"):
        overreliance_rate(*inputs, zero_division=4)


@pytest.mark.parametrize("input_vectors", [
    ([1], [1, 0], [1], [1]),
    ([], [], [], []),
    ([[1]], [[1]], [[1]], [[1]]),
    ([1, np.nan], [1, 0], [1, 1], [1, 0]),
    ([1, 0, 2], [1, 0, 2], [1, 0, 2], [1, 0, 2]),
])
def test_invalid_vectors(input_vectors):
    with pytest.raises(ValueError):
        scenario_labels(*input_vectors)


def test_invalid_weights():
    v = ([0, 1], [0, 1], [0, 1], [0, 1])
    for w in ([1], [-1, 1], [float("nan"), 1]):
        with pytest.raises(ValueError):
            scenario_counts(*v, sample_weight=w)


def test_scenarios_metric_complement():
    v = all_scenarios()
    r = human_ai_metrics(*v)
    assert r["overreliance_rate"] + r["correct_self_reliance_rate"] == 1
    assert r["underreliance_rate"] + r["correct_ai_reliance_rate"] == 1
    assert r["ai_error_correction_rate"] + r["ai_error_propagation_rate"] == 1
    assert r["human_error_correction_rate"] + r["human_error_persistence_rate"] == 1
