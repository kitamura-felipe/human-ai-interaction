# human-ai-metrics

A lightweight, **scikit-learn-style** Python package to study binary interactions among a reference standard, an AI prediction, an independent initial human decision, and a final human decision.

## Install

```bash
pip install .
# For testing:
pip install '.[test]'
pytest
```

Once published to PyPI, the distribution can be installed as `pip install human-ai-metrics`. The source release itself is installable directly now.

## Quick start

```python
import numpy as np
from human_ai_metrics import (
    human_ai_metrics,
    metric_details,
    overreliance_rate,
    scenario_labels,
    scenario_counts,
)

# Ground truth, AI prediction, human BEFORE AI, human AFTER AI
# Positive = 1 (finding present); negative = 0 (finding absent)
y_true = np.array([1, 0, 1, 0])
y_ai = np.array([0, 1, 1, 0])
y_human_initial = np.array([1, 0, 0, 1])
y_human_final = np.array([0, 1, 1, 1])

print(scenario_labels(y_true, y_ai, y_human_initial, y_human_final))
# [ 6 11  3 13]

print(overreliance_rate(y_true, y_ai, y_human_initial, y_human_final))
# 1.0 (2 overreliance events / 2 opportunities)

print(human_ai_metrics(y_true, y_ai, y_human_initial, y_human_final,
                       zero_division=np.nan))

# Detailed provenance: values, numerator/denominator, and their scenario IDs
print(metric_details(y_true, y_ai, y_human_initial, y_human_final,
                     zero_division=np.nan)["overreliance_rate"])

# Row 1 ... row 16 frequencies, in that order
print(scenario_counts(y_true, y_ai, y_human_initial, y_human_final))
```

Functions accept lists, NumPy vectors, and Pandas Series. All four input vectors must have equal length. The `pos_label` argument may be changed, for instance `pos_label="present"`. Pass `sample_weight` for weighted counts and rates. A rate is a proportion in [0, 1], **not** a percentage.

Like scikit-learn, `zero_division="warn"` warns and returns zero for a metric without eligible cases. Alternatively, specify `zero_division=0`, `1`, or `np.nan`.

## Complete 16-scenario matrix

| ID | Truth | AI | Initial human | Final human | Interaction | Outcome |
|---:|:---:|:---:|:---:|:---:|---|---|
| 1 | + | + | + | + | Correct concordance | TP |
| 2 | + | + | + | − | Reversal of correct agreement | FN, omission |
| 3 | + | + | − | + | Correct AI reliance | TP |
| 4 | + | + | − | − | Underreliance | FN, omission |
| 5 | + | − | + | + | Correct self-reliance | TP |
| 6 | + | − | + | − | Overreliance | FN, omission |
| 7 | + | − | − | + | Correction of concordant error | TP |
| 8 | + | − | − | − | Concordant error retained | FN, omission |
| 9 | − | + | + | + | Concordant error retained | FP, commission |
| 10 | − | + | + | − | Correction of concordant error | TN |
| 11 | − | + | − | + | Overreliance | FP, commission |
| 12 | − | + | − | − | Correct self-reliance | TN |
| 13 | − | − | + | + | Underreliance | FP, commission |
| 14 | − | − | + | − | Correct AI reliance | TN |
| 15 | − | − | − | + | Reversal of correct agreement | FP, commission |
| 16 | − | − | − | − | Correct concordance | TN |

**Interpretation:** "follows AI" means the final answer matches AI in a case where AI and initial human disagree; it does *not* establish that AI caused the change. Cases 7 and 10 involve an initially incorrect AI and human decision, both corrected at final review; the four binary vectors cannot identify the source of correction. Rows 2 and 15 likewise do not identify why an initially correct agreement was reversed.

## Metrics: meanings and exact scenario formulas

Let `n_i` be the count of scenario `i` (weighted if `sample_weight` is provided), and define `S(A) = sum(n_i for i in A)`. Every rate below is `S(numerator IDs) / S(denominator IDs)`.

| Metric | Numerator IDs | Denominator IDs | Interpretation |
|---|---|---|---|
| `overreliance_rate` | 6, 11 | 5, 6, 11, 12 | AI was incorrect, initial human correct, but final decision followed incorrect AI |
| `underreliance_rate` | 4, 13 | 3, 4, 13, 14 | AI was correct, initial human incorrect, and final decision kept the incorrect human answer |
| `correct_ai_reliance_rate` | 3, 14 | 3, 4, 13, 14 | AI was correct and initial human incorrect; final decision matched correct AI |
| `correct_self_reliance_rate` | 5, 12 | 5, 6, 11, 12 | AI was incorrect and initial human correct; final decision preserved the correct human answer |
| `ai_error_correction_rate` | 5, 7, 10, 12 | 5, 6, 7, 8, 9, 10, 11, 12 | Among AI errors, share corrected in final decision, regardless of source of correction |
| `ai_error_propagation_rate` | 6, 8, 9, 11 | 5, 6, 7, 8, 9, 10, 11, 12 | Among AI errors, share still incorrect in final decision; does not prove AI causation |
| `human_error_correction_rate` | 3, 7, 10, 14 | 3, 4, 7, 8, 9, 10, 13, 14 | Among initially incorrect human answers, share corrected by final assessment |
| `human_error_persistence_rate` | 4, 8, 9, 13 | 3, 4, 7, 8, 9, 10, 13, 14 | Among initially incorrect human answers, share remaining incorrect |
| `human_induced_error_rate` | 2, 6, 11, 15 | 1, 2, 5, 6, 11, 12, 15, 16 | Among initially correct human answers, share that became incorrect at final assessment |
| `final_omission_rate` | 2, 4, 6, 8 | 1–8 | Final false-negative rate: among truly positive findings, share finally reported negative |
| `final_commission_rate` | 9, 11, 13, 15 | 9–16 | Final false-positive rate: among truly negative findings, share finally reported positive |

### Complements and partitions

Subject to nonzero denominators:

- `overreliance_rate + correct_self_reliance_rate == 1`
- `underreliance_rate + correct_ai_reliance_rate == 1`
- `ai_error_correction_rate + ai_error_propagation_rate == 1`
- `human_error_correction_rate + human_error_persistence_rate == 1`

The 16-scenario matrix is exhaustive **only for a binary finding decision**. It does not capture severity, laterality, modifiers, uncertainty, timing, cognitive workload, or productivity.

## API

All eleven metric functions share a signature:

```python
metric(y_true, y_ai, y_human_initial, y_human_final,
       *, pos_label=1, sample_weight=None, zero_division="warn")
```

Additional functions:

- `scenario_labels(...)` — one row ID per observation.
- `scenario_counts(...)` — length-16 array of row frequencies, in numeric order.
- `human_ai_metrics(...)` — a dict of all eleven rate values.
- `metric_details(...)` — metric value, numerator, denominator and scenario ID lists.
- `METRIC_SCENARIOS` and `SCENARIO_DESCRIPTIONS` — fixed, inspectable taxonomy.

### Study design caveats

An independent initial human assessment should be recorded **before** AI exposure. Otherwise, the apparent decision shifts are not reliably measurable. A valid reference standard is essential. In shared-control workflows, AI exposure may influence final decisions without a visible change of answer, and the four vectors alone cannot establish counterfactual or causal effects. Treat the metric names as operational classifications of observed agreement and correctness, not as proof of cognitive mechanism.

## License

MIT. See [LICENSE](LICENSE).
