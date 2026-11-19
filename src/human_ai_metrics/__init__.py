"""Metrics for binary human–AI decision interactions.

Examples
--------
>>> from human_ai_metrics import overreliance_rate
>>> overreliance_rate([1, 0], [0, 1], [1, 0], [0, 1])
1.0
"""

from ._metrics import (
    METRIC_SCENARIOS, SCENARIO_DESCRIPTIONS,
    scenario_labels, scenario_counts, human_ai_metrics, metric_details,
    overreliance_rate, underreliance_rate,
    correct_ai_reliance_rate, correct_self_reliance_rate,
    ai_error_correction_rate, ai_error_propagation_rate,
    human_error_correction_rate, human_error_persistence_rate,
    human_induced_error_rate, final_omission_rate, final_commission_rate,
)

__version__ = "0.1.0"

__all__ = [
    "METRIC_SCENARIOS", "SCENARIO_DESCRIPTIONS", "scenario_labels", "scenario_counts",
    "human_ai_metrics", "metric_details", "overreliance_rate", "underreliance_rate",
    "correct_ai_reliance_rate", "correct_self_reliance_rate", "ai_error_correction_rate",
    "ai_error_propagation_rate", "human_error_correction_rate", "human_error_persistence_rate",
    "human_induced_error_rate", "final_omission_rate", "final_commission_rate",
]
