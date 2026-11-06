"""Metric tests, cross-validated against scikit-learn where conventions agree.

The cross-validation is the point: these numbers go into a paper, so the
hand-rolled implementations are checked against an independent one rather than
merely reviewed. Where the conventions deliberately differ — the absent-class
rule — the tests pin the difference instead of hiding it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcml.training.metrics import (
    balanced_accuracy,
    classification_report,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    per_class_scores,
    reliability_bins,
)

SIX = ("person", "vegetation", "pole", "structure", "net_fence", "vehicle")


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestConfusionMatrix:
    def test_counts_and_orientation(self):
        """Rows are true, columns predicted — getting this backwards silently
        transposes precision and recall."""
        y_true = [0, 0, 1, 1, 2]
        y_pred = [0, 1, 1, 1, 0]
        cm = confusion_matrix(y_true, y_pred, 3)
        assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
        assert cm[0].sum() == 2  # support of class 0
        assert cm[:, 1].sum() == 3  # times class 1 was predicted

    def test_matches_sklearn(self):
        sk = pytest.importorskip("sklearn.metrics")
        g = rng(1)
        y_true = g.integers(0, 6, 400)
        y_pred = g.integers(0, 6, 400)
        ours = confusion_matrix(y_true, y_pred, 6)
        theirs = sk.confusion_matrix(y_true, y_pred, labels=list(range(6)))
        assert np.array_equal(ours, theirs)

    def test_empty_input(self):
        cm = confusion_matrix([], [], 6)
        assert cm.shape == (6, 6)
        assert cm.sum() == 0

    def test_rejects_the_reject_id(self):
        """-1 is a system-level outcome, not a class the model emits."""
        with pytest.raises(ValueError, match="reject id -1"):
            confusion_matrix([0, 1], [0, -1], 3)

    def test_rejects_out_of_range_and_mismatched_shapes(self):
        with pytest.raises(ValueError, match="outside"):
            confusion_matrix([0, 1], [0, 5], 3)
        with pytest.raises(ValueError, match="must match"):
            confusion_matrix([0, 1, 2], [0, 1], 3)


class TestPerClassScores:
    def test_perfect_prediction(self):
        cm = confusion_matrix([0, 1, 2], [0, 1, 2], 3)
        s = per_class_scores(cm)
        assert np.allclose(s["precision"], 1.0)
        assert np.allclose(s["recall"], 1.0)
        assert np.allclose(s["f1"], 1.0)

    def test_matches_sklearn_when_all_classes_present(self):
        sk = pytest.importorskip("sklearn.metrics")
        g = rng(2)
        y_true = g.integers(0, 4, 300)
        y_pred = g.integers(0, 4, 300)
        s = per_class_scores(confusion_matrix(y_true, y_pred, 4))
        p, r, f, _ = sk.precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(4)), zero_division=0
        )
        assert np.allclose(s["precision"], p)
        assert np.allclose(s["recall"], r)
        assert np.allclose(s["f1"], f)

    def test_absent_class_gives_nan_not_zero(self):
        """A class the split cannot measure must not quietly average in as 0."""
        # class 2 never appears in y_true
        cm = confusion_matrix([0, 0, 1], [0, 1, 1], 3)
        s = per_class_scores(cm)
        assert np.isnan(s["recall"][2])
        assert np.isnan(s["f1"][2])
        assert s["support"][2] == 0

    def test_class_never_predicted_has_zero_precision(self):
        cm = confusion_matrix([0, 1], [0, 0], 2)
        s = per_class_scores(cm)
        assert s["precision"][1] == 0.0

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            per_class_scores(np.zeros((2, 3)))


class TestAbsentClassConvention:
    """The deliberate divergence from scikit-learn, pinned."""

    def test_macro_f1_skips_absent_classes(self):
        # 3 classes, class 2 absent from the reference, classes 0 and 1 perfect.
        cm = confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1], 3)
        assert macro_f1(cm) == pytest.approx(1.0)

    def test_sklearn_would_disagree_and_that_is_the_point(self):
        """sklearn scores the absent class 0 and averages over all 3 -> 2/3.

        Ours averages over the 2 present classes -> 1.0. Scoring a model 0.67
        for perfect predictions, because the split happens not to contain a
        class, would make two splits incomparable — which is fatal when the
        whole benchmark is a comparison between arms.
        """
        sk = pytest.importorskip("sklearn.metrics")
        y_true, y_pred = [0, 0, 1, 1], [0, 0, 1, 1]
        theirs = sk.f1_score(
            y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
        )
        ours = macro_f1(confusion_matrix(y_true, y_pred, 3))
        assert theirs == pytest.approx(2.0 / 3.0)
        assert ours == pytest.approx(1.0)

    def test_report_names_the_absent_classes(self):
        report = classification_report([0, 0, 1], [0, 0, 1], 6, class_names=SIX)
        assert report["n_classes_scored"] == 2
        assert set(report["absent_classes"]) == set(SIX[2:])

    def test_macro_f1_matches_sklearn_when_nothing_is_absent(self):
        sk = pytest.importorskip("sklearn.metrics")
        g = rng(3)
        y_true = np.concatenate([np.arange(6), g.integers(0, 6, 300)])
        y_pred = g.integers(0, 6, len(y_true))
        ours = macro_f1(confusion_matrix(y_true, y_pred, 6))
        theirs = sk.f1_score(y_true, y_pred, labels=list(range(6)), average="macro")
        assert ours == pytest.approx(theirs)

    def test_all_absent_is_nan_not_a_number(self):
        assert np.isnan(macro_f1(confusion_matrix([], [], 3)))
        assert np.isnan(balanced_accuracy(confusion_matrix([], [], 3)))


class TestBalancedAccuracy:
    def test_matches_sklearn(self):
        sk = pytest.importorskip("sklearn.metrics")
        g = rng(4)
        y_true = np.concatenate([np.arange(5), g.integers(0, 5, 200)])
        y_pred = g.integers(0, 5, len(y_true))
        ours = balanced_accuracy(confusion_matrix(y_true, y_pred, 5))
        assert ours == pytest.approx(sk.balanced_accuracy_score(y_true, y_pred))

    def test_insensitive_to_imbalance_unlike_accuracy(self):
        """The reason it is reported: a majority-class predictor scores ~1/K."""
        y_true = [0] * 990 + [1] * 10
        y_pred = [0] * 1000
        cm = confusion_matrix(y_true, y_pred, 2)
        assert balanced_accuracy(cm) == pytest.approx(0.5)
        report = classification_report(y_true, y_pred, 2)
        assert report["overall_accuracy"] == pytest.approx(0.99)


class TestClassificationReport:
    def test_shape_and_keys(self):
        g = rng(5)
        y_true = np.concatenate([np.arange(6), g.integers(0, 6, 200)])
        y_pred = g.integers(0, 6, len(y_true))
        report = classification_report(y_true, y_pred, 6, class_names=SIX)
        for key in (
            "n_samples",
            "overall_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "n_classes_scored",
            "absent_classes",
            "per_class",
            "confusion_matrix",
        ):
            assert key in report
        assert set(report["per_class"]) == set(SIX)
        assert report["n_samples"] == len(y_true)

    def test_is_json_serialisable(self):
        """Reports go straight into reports/*.json."""
        import json

        g = rng(6)
        y_true = g.integers(0, 6, 60)
        y_pred = g.integers(0, 6, 60)
        conf = g.uniform(0.2, 1.0, 60)
        report = classification_report(
            y_true, y_pred, 6, class_names=SIX, confidences=conf
        )
        json.dumps(report)  # must not raise

    def test_rejects_mismatched_class_names(self):
        with pytest.raises(ValueError, match="class_names has"):
            classification_report([0, 1], [0, 1], 6, class_names=("a", "b"))

    def test_calibration_included_only_with_confidences(self):
        assert "calibration" not in classification_report([0, 1], [0, 1], 2)
        report = classification_report([0, 1], [0, 1], 2, confidences=[0.9, 0.8])
        assert "calibration" in report
        assert report["calibration"]["n_bins"] == 10


class TestCalibration:
    def test_perfectly_calibrated_gives_near_zero_ece(self):
        """Confidence c on a sample correct with probability c -> ECE ~ 0."""
        g = rng(7)
        conf = g.uniform(0.5, 1.0, 20000)
        correct = g.uniform(0, 1, 20000) < conf
        assert expected_calibration_error(conf, correct, n_bins=10) < 0.02

    def test_overconfident_model_has_large_ece(self):
        conf = np.full(1000, 0.99)
        correct = np.zeros(1000, dtype=bool)
        correct[:500] = True  # 50% accurate while claiming 99%
        assert expected_calibration_error(conf, correct) == pytest.approx(0.49, abs=0.01)

    def test_bins_tile_the_unit_interval_and_report_gaps(self):
        bins = reliability_bins([0.05, 0.95], [True, False], n_bins=10)
        assert len(bins) == 10
        assert bins[0]["lower"] == 0.0
        assert bins[-1]["upper"] == 1.0
        assert bins[0]["count"] == 1 and bins[-1]["count"] == 1
        # Empty bins are reported, not dropped.
        assert bins[5]["count"] == 0
        assert np.isnan(bins[5]["accuracy"])

    def test_confidence_exactly_one_is_counted(self):
        """An off-by-one in the last bin edge silently drops saturated samples."""
        bins = reliability_bins([1.0, 1.0], [True, False], n_bins=4)
        assert bins[-1]["count"] == 2
        assert bins[-1]["accuracy"] == pytest.approx(0.5)

    def test_ece_depends_on_bin_count(self):
        """Documents why n_bins is always reported alongside ECE."""
        g = rng(8)
        conf = g.uniform(0, 1, 2000)
        correct = g.uniform(0, 1, 2000) < conf
        coarse = expected_calibration_error(conf, correct, n_bins=2)
        fine = expected_calibration_error(conf, correct, n_bins=50)
        assert coarse != pytest.approx(fine, abs=1e-6)

    def test_rejects_out_of_range_confidences(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            expected_calibration_error([1.5], [True])

    def test_empty_is_nan(self):
        assert np.isnan(expected_calibration_error([], []))
