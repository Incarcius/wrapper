"""Training: the one loop, its losses, its metrics, and its reproducibility.

Training-only. Nothing in ``pcml.deploy`` may import from here — the deployed
artifact must not depend on training code, datasets or teacher checkpoints.

``losses`` imports torch; ``metrics`` and ``reproducibility`` do not require it,
so they stay importable for offline analysis of a results file.
"""

from .metrics import (
    balanced_accuracy,
    classification_report,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    per_class_scores,
    reliability_bins,
)
from .reproducibility import RngStreams, describe_rng_env, seed_everything, stream_seed
from .tournament import (
    ArmSummary,
    Comparison,
    RunResult,
    adjudicate,
    format_summary_table,
    holm_adjust,
    min_achievable_p,
    paired_differences,
    permutation_test_paired,
    summarise,
    trend_spearman,
)

__all__ = [
    "ArmSummary",
    "Comparison",
    "RngStreams",
    "RunResult",
    "adjudicate",
    "format_summary_table",
    "holm_adjust",
    "min_achievable_p",
    "paired_differences",
    "permutation_test_paired",
    "summarise",
    "trend_spearman",
    "balanced_accuracy",
    "classification_report",
    "confusion_matrix",
    "describe_rng_env",
    "expected_calibration_error",
    "macro_f1",
    "per_class_scores",
    "reliability_bins",
    "seed_everything",
    "stream_seed",
]
