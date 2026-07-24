from xctb.eval.metrics import (
    accuracy,
    sensitivity_specificity,
    auroc,
    binary_report,
)
from xctb.eval.deferral import (
    risk_coverage_curve,
    accuracy_at_coverage,
    generalization_gap_recovery,
    coverage_to_recover,
)
from xctb.eval.degradation_uncertainty import (
    spearman_correlation,
    uncertainty_vs_severity,
)

__all__ = [
    "accuracy",
    "sensitivity_specificity",
    "auroc",
    "binary_report",
    "risk_coverage_curve",
    "accuracy_at_coverage",
    "generalization_gap_recovery",
    "coverage_to_recover",
    "spearman_correlation",
    "uncertainty_vs_severity",
]
