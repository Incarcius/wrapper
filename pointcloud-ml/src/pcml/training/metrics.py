"""Classification metrics, computed by hand from the confusion matrix.

Hand-rolled rather than delegated to scikit-learn, for two reasons. First, a
published number should not shift because a dependency changed its averaging or
zero-division default between versions. Second, the absent-class convention
below is a decision this project has to make explicitly, and inheriting a
library default would bury it. ``tests/test_metrics.py`` cross-validates every
function against scikit-learn where the conventions agree, so the hand-rolled
code is checked rather than merely trusted.

**The absent-class convention.** Splits are by recording session
(``docs/task_definition.md`` §7), so a validation or test split can legitimately
contain **no instances of a class** — `vehicle` is expected to be rare enough
for this to happen. Macro-F1 then has an ambiguity: score the absent class 0, or
leave it out?

This module **averages over classes present in ``y_true``** and reports the
absent ones separately in ``absent_classes``. Scoring an absent class 0 would
penalise a model for a class the split does not contain, and would make two
splits with different missing classes incomparable — which is fatal when the
whole benchmark is a comparison. scikit-learn's default is the opposite
(``zero_division=0``), so the numbers here deliberately differ from a naive
``f1_score(average="macro")`` call whenever a class is absent; the tests pin
that difference rather than hide it.

Any reported macro-F1 must therefore be quoted together with ``n_classes_scored``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "confusion_matrix",
    "per_class_scores",
    "classification_report",
    "macro_f1",
    "balanced_accuracy",
    "expected_calibration_error",
    "reliability_bins",
]


def _as_int_array(a: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(arr == np.floor(arr)):
            raise ValueError(f"{name} must contain integer class ids")
        arr = arr.astype(np.int64)
    return arr


def confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """``(num_classes, num_classes)`` counts, rows = true, columns = predicted.

    Predicted id ``-1`` (the reject outcome) is **not** accepted here: rejection
    is a system-level behaviour evaluated separately against a threshold, not a
    class the model can emit. Passing it is an error rather than a silent extra
    row.
    """
    t = _as_int_array(y_true, "y_true")
    p = _as_int_array(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError(f"y_true {t.shape} and y_pred {p.shape} must match")
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    for name, arr in (("y_true", t), ("y_pred", p)):
        if arr.size and (arr.min() < 0 or arr.max() >= num_classes):
            raise ValueError(
                f"{name} has ids outside [0, {num_classes - 1}]: "
                f"min={arr.min()}, max={arr.max()}. The reject id -1 is evaluated "
                "separately, not as a class."
            )

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if t.size:
        np.add.at(cm, (t, p), 1)
    return cm


def per_class_scores(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and support per class, from a confusion matrix.

    Zero-division convention, applied per class and *not* silently:
      * no predictions for a class -> precision 0 (nothing was claimed, so
        nothing was correct);
      * no support for a class -> recall and F1 are ``nan``, because the split
        cannot measure them. ``nan`` propagates loudly instead of averaging in
        as a 0.
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {cm.shape}")

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, np.nan)
        denom = precision + recall
        f1 = np.where(
            support > 0,
            np.where(denom > 0, 2.0 * precision * recall / denom, 0.0),
            np.nan,
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(np.int64),
    }


def macro_f1(cm: np.ndarray) -> float:
    """Mean F1 over classes **present in the reference**. See the module docstring."""
    scores = per_class_scores(cm)
    present = scores["support"] > 0
    if not present.any():
        return float("nan")
    return float(scores["f1"][present].mean())


def balanced_accuracy(cm: np.ndarray) -> float:
    """Mean recall over classes present in the reference."""
    scores = per_class_scores(cm)
    present = scores["support"] > 0
    if not present.any():
        return float("nan")
    return float(scores["recall"][present].mean())


def classification_report(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float] | np.ndarray] = None,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Everything the evaluation protocol asks for, in one JSON-ready dict.

    Args:
        class_names: resolved from the label map. Required for a report that is
            readable months later; positional ids are not.
        confidences: max softmax probability per sample. When given, calibration
            is included — a high confidence is not evidence of correctness
            (``docs/task_definition.md`` §7), so the benchmark reports it.

    Returns:
        ``overall_accuracy``, ``balanced_accuracy``, ``macro_f1``,
        ``n_classes_scored``, ``absent_classes``, ``per_class`` and
        ``confusion_matrix``, plus ``calibration`` when confidences are given.

    ``macro_f1`` is the primary selection metric: class imbalance here is severe
    enough that overall accuracy would be dominated by whichever class is most
    common.
    """
    t = _as_int_array(y_true, "y_true")
    cm = confusion_matrix(t, y_pred, num_classes)
    scores = per_class_scores(cm)
    present = scores["support"] > 0

    if class_names is None:
        names = [f"class_{i}" for i in range(num_classes)]
    else:
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries but num_classes is "
                f"{num_classes}; a mismatched label map would mislabel every row"
            )
        names = list(class_names)

    total = float(cm.sum())
    report: dict[str, Any] = {
        "n_samples": int(total),
        "overall_accuracy": float(np.diag(cm).sum() / total) if total else float("nan"),
        "balanced_accuracy": balanced_accuracy(cm),
        "macro_f1": macro_f1(cm),
        "n_classes_scored": int(present.sum()),
        "absent_classes": [names[i] for i in range(num_classes) if not present[i]],
        "per_class": {
            names[i]: {
                "precision": float(scores["precision"][i]),
                "recall": float(scores["recall"][i]),
                "f1": float(scores["f1"][i]),
                "support": int(scores["support"][i]),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": names,
    }

    if confidences is not None:
        correct = _as_int_array(y_pred, "y_pred") == t
        report["calibration"] = {
            "expected_calibration_error": expected_calibration_error(
                confidences, correct, n_bins=n_calibration_bins
            ),
            "n_bins": int(n_calibration_bins),
            "bins": reliability_bins(confidences, correct, n_bins=n_calibration_bins),
            "mean_confidence": float(np.mean(confidences)) if len(confidences) else float("nan"),
        }

    return report


def reliability_bins(
    confidences: Sequence[float] | np.ndarray,
    correct: Sequence[bool] | np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Equal-width confidence bins, for a reliability diagram.

    Equal-width (not equal-count) bins, because the diagram's x-axis is
    confidence and the reader expects the bins to tile [0, 1]. Empty bins are
    reported with ``count: 0`` rather than dropped, so the gaps are visible.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=bool)
    if conf.shape != hit.shape:
        raise ValueError(f"confidences {conf.shape} and correct {hit.shape} must match")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if conf.size and (conf.min() < 0.0 or conf.max() > 1.0 + 1e-9):
        raise ValueError("confidences must lie in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin is closed on the right so confidence exactly 1.0 is counted.
        in_bin = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        count = int(in_bin.sum())
        out.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "count": count,
                "mean_confidence": float(conf[in_bin].mean()) if count else float("nan"),
                "accuracy": float(hit[in_bin].mean()) if count else float("nan"),
            }
        )
    return out


def expected_calibration_error(
    confidences: Sequence[float] | np.ndarray,
    correct: Sequence[bool] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Weighted mean gap between confidence and accuracy across bins.

    ECE depends on the bin count, so it is always reported alongside ``n_bins``;
    comparing ECE values computed with different binnings is meaningless.
    """
    bins = reliability_bins(confidences, correct, n_bins=n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return float("nan")
    return float(
        sum(
            (b["count"] / total) * abs(b["mean_confidence"] - b["accuracy"])
            for b in bins
            if b["count"] > 0
        )
    )
