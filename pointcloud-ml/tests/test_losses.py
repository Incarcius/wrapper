"""Distillation loss tests.

The properties asserted here are the ones that, if wrong, produce a plausible
number rather than a crash — which is the dangerous failure mode for a
benchmark. In particular: the T^2 scaling, the teacher being detached, and
"batchmean" rather than "mean".
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from pcml.training.losses import (  # noqa: E402
    DistillationObjective,
    FeatureProjector,
    cross_entropy_loss,
    feature_loss,
    kd_loss,
    kd_per_sample,
    teacher_confidence,
)

K = 6


def logits(b: int = 8, k: int = K, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, k, generator=g) * scale


class TestKdDivergence:
    def test_identical_logits_give_zero(self):
        x = logits()
        assert float(kd_loss(x, x.clone(), temperature=4.0)) == pytest.approx(0.0, abs=1e-6)

    def test_is_invariant_to_a_logit_shift(self):
        """Softmax is shift-invariant, so the loss must be too. A failure here
        means the divergence was computed on logits rather than log-probs."""
        s, t = logits(seed=1), logits(seed=2)
        base = float(kd_loss(s, t, 3.0))
        shifted = float(kd_loss(s + 7.0, t - 4.0, 3.0))
        assert shifted == pytest.approx(base, rel=1e-5)

    def test_matches_batchmean_reference(self):
        s, t = logits(seed=3), logits(seed=4)
        T = 2.5
        ref = F.kl_div(
            F.log_softmax(s / T, dim=1),
            F.log_softmax(t / T, dim=1),
            log_target=True,
            reduction="batchmean",
        ) * (T**2)
        assert float(kd_loss(s, t, T)) == pytest.approx(float(ref), rel=1e-6)

    @pytest.mark.filterwarnings("ignore:reduction:UserWarning")
    def test_is_not_the_mean_reduction(self):
        """torch's "mean" also divides by K, scaling the loss by 1/6 here and
        therefore silently scaling beta. This pins that we did not use it.

        torch warns about `reduction="mean"` here; that warning is the whole
        point of the comparison, so it is suppressed rather than fixed.
        """
        s, t = logits(seed=5), logits(seed=6)
        T = 2.0
        wrong = F.kl_div(
            F.log_softmax(s / T, dim=1),
            F.log_softmax(t / T, dim=1),
            log_target=True,
            reduction="mean",
        ) * (T**2)
        ours = kd_loss(s, t, T)
        assert float(ours) == pytest.approx(float(wrong) * K, rel=1e-5)

    def test_t_squared_keeps_gradient_magnitude_stable_across_temperature(self):
        """The reason the T^2 factor exists: without it the KD gradient shrinks
        as ~1/T^2 and `beta` stops meaning the same thing between ablation runs.
        """
        t = logits(seed=7)
        grads = {}
        for T in (1.0, 2.0, 4.0, 8.0):
            s = logits(seed=8).requires_grad_(True)
            kd_loss(s, t, T).backward()
            grads[T] = float(s.grad.norm())
        # Without T^2 this would fall by ~64x from T=1 to T=8.
        assert grads[8.0] / grads[1.0] > 0.25
        assert grads[8.0] / grads[1.0] < 4.0

    def test_teacher_is_detached(self):
        """No arm may backpropagate into its teacher — including mutual
        learning, where the peer must be updated by its own loss."""
        s = logits(seed=9).requires_grad_(True)
        t = logits(seed=10).requires_grad_(True)
        kd_loss(s, t, 4.0).backward()
        assert s.grad is not None
        assert t.grad is None

    def test_per_sample_shape_and_consistency_with_reduction(self):
        s, t = logits(seed=11), logits(seed=12)
        per = kd_per_sample(s, t, 3.0)
        assert per.shape == (8,)
        assert float(per.mean()) == pytest.approx(float(kd_loss(s, t, 3.0)), rel=1e-6)

    def test_is_non_negative(self):
        for seed in range(5):
            assert float(kd_loss(logits(seed=seed), logits(seed=seed + 100), 4.0)) >= 0.0

    def test_stable_at_low_temperature_and_large_logits(self):
        """Where a naive exp/log implementation produces NaN."""
        s = logits(seed=13, scale=50.0)
        t = logits(seed=14, scale=50.0)
        value = float(kd_loss(s, t, temperature=0.1))
        assert value == value  # not NaN
        assert value < float("inf")

    def test_rejects_bad_temperature_and_shape(self):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            kd_loss(logits(), logits(), 0.0)
        with pytest.raises(ValueError, match="must match"):
            kd_loss(logits(b=4), logits(b=5), 4.0)


class TestConfidenceWeighting:
    def test_weights_are_normalised_so_beta_keeps_its_meaning(self):
        """Turning weighting on must not also change the loss magnitude, or the
        ablation confounds "weighting helped" with "beta got smaller"."""
        s, t = logits(seed=15), logits(seed=16)
        uniform = torch.full((8,), 0.3)
        assert float(kd_loss(s, t, 4.0, weights=uniform)) == pytest.approx(
            float(kd_loss(s, t, 4.0)), rel=1e-6
        )

    def test_zero_weights_give_zero_loss(self):
        s, t = logits(seed=17), logits(seed=18)
        assert float(kd_loss(s, t, 4.0, weights=torch.zeros(8))) == pytest.approx(0.0)

    def test_max_prob_confidence_range_and_ordering(self):
        confident = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        unsure = torch.zeros(1, K)
        c_hi = float(teacher_confidence(confident, "max_prob"))
        c_lo = float(teacher_confidence(unsure, "max_prob"))
        assert c_hi > 0.99
        assert c_lo == pytest.approx(1.0 / K)
        assert c_hi > c_lo

    def test_neg_entropy_spans_zero_to_one(self):
        confident = torch.tensor([[50.0] + [0.0] * (K - 1)])
        unsure = torch.zeros(1, K)
        assert float(teacher_confidence(confident, "neg_entropy")) == pytest.approx(1.0, abs=1e-5)
        assert float(teacher_confidence(unsure, "neg_entropy")) == pytest.approx(0.0, abs=1e-6)

    def test_confidence_carries_no_gradient(self):
        t = logits(seed=19).requires_grad_(True)
        teacher_confidence(t, "max_prob").sum()
        assert t.grad is None

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown confidence mode"):
            teacher_confidence(logits(), "vibes")


class TestFeatureLoss:
    def test_identical_features_give_zero(self):
        f = torch.randn(8, 512)
        assert float(feature_loss(f, f.clone(), "cosine")) == pytest.approx(0.0, abs=1e-6)

    def test_scale_invariant(self):
        """The teacher's embedding scale is arbitrary, so magnitude must not
        drive the loss — otherwise the student matches a scale carrying no
        information."""
        s, t = torch.randn(8, 512), torch.randn(8, 512)
        base = float(feature_loss(s, t, "cosine"))
        scaled = float(feature_loss(s, t * 100.0, "cosine"))
        assert scaled == pytest.approx(base, rel=1e-5)

    def test_opposite_directions_are_maximally_penalised(self):
        f = torch.randn(4, 64)
        assert float(feature_loss(f, -f, "cosine")) == pytest.approx(2.0, rel=1e-5)

    def test_teacher_features_detached(self):
        s = torch.randn(4, 64, requires_grad=True)
        t = torch.randn(4, 64, requires_grad=True)
        feature_loss(s, t, "mse").backward()
        assert s.grad is not None
        assert t.grad is None

    def test_width_mismatch_names_the_projector(self):
        with pytest.raises(ValueError, match="FeatureProjector"):
            feature_loss(torch.randn(4, 128), torch.randn(4, 512))

    def test_rejects_unknown_mode(self):
        f = torch.randn(4, 8)
        with pytest.raises(ValueError, match="unknown feature mode"):
            feature_loss(f, f, "earthmover")


class TestFeatureProjector:
    def test_maps_student_width_up_to_teacher_width(self):
        proj = FeatureProjector(128, 512)
        assert proj(torch.randn(8, 128)).shape == (8, 512)

    def test_output_is_usable_by_feature_loss(self):
        proj = FeatureProjector(128, 512)
        loss = feature_loss(proj(torch.randn(8, 128)), torch.randn(8, 512))
        assert float(loss.detach()) >= 0.0

    def test_rejects_degenerate_dims(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            FeatureProjector(0, 512)


class TestDistillationObjective:
    def targets(self, b: int = 8) -> torch.Tensor:
        g = torch.Generator().manual_seed(42)
        return torch.randint(0, K, (b,), generator=g)

    def test_arm0_is_plain_cross_entropy(self):
        obj = DistillationObjective(alpha=1.0, beta=0.0, gamma=0.0)
        s, y = logits(), self.targets()
        total, parts = obj(s, y)
        assert not obj.uses_teacher
        assert float(total) == pytest.approx(float(cross_entropy_loss(s, y)), rel=1e-6)
        assert set(parts) == {"ce", "total"}

    def test_kd_arm_requires_a_target_source(self):
        obj = DistillationObjective(alpha=0.5, beta=0.5)
        assert obj.uses_teacher
        with pytest.raises(ValueError, match="requires teacher_logits"):
            obj(logits(), self.targets())

    def test_components_are_reported_separately(self):
        """An ablation needs to see which term moved, not only the sum."""
        obj = DistillationObjective(alpha=0.5, beta=0.5, gamma=0.1)
        total, parts = obj(
            logits(seed=20),
            self.targets(),
            teacher_logits=logits(seed=21),
            student_features=torch.randn(8, 64),
            teacher_features=torch.randn(8, 64),
        )
        assert set(parts) == {"ce", "kd", "feature", "total"}
        assert parts["total"] == pytest.approx(
            0.5 * parts["ce"] + 0.5 * parts["kd"] + 0.1 * parts["feature"], rel=1e-5
        )

    def test_weights_scale_the_total(self):
        s, y, t = logits(seed=22), self.targets(), logits(seed=23)
        a, _ = DistillationObjective(alpha=1.0, beta=1.0)(s, y, teacher_logits=t)
        b, _ = DistillationObjective(alpha=2.0, beta=2.0)(s, y, teacher_logits=t)
        assert float(b) == pytest.approx(2.0 * float(a), rel=1e-5)

    def test_feature_arm_requires_both_feature_tensors(self):
        obj = DistillationObjective(alpha=1.0, gamma=0.5)
        with pytest.raises(ValueError, match="student_features and"):
            obj(logits(), self.targets(), student_features=torch.randn(8, 64))

    def test_confidence_weighting_is_logged(self):
        obj = DistillationObjective(alpha=0.5, beta=0.5, confidence_weighting="max_prob")
        _, parts = obj(logits(seed=24), self.targets(), teacher_logits=logits(seed=25))
        assert "mean_teacher_confidence" in parts
        assert 0.0 <= parts["mean_teacher_confidence"] <= 1.0

    def test_class_weights_change_the_loss(self):
        s, y = logits(seed=26), self.targets()
        flat = DistillationObjective(alpha=1.0)
        weighted = DistillationObjective(
            alpha=1.0, class_weights=torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )
        assert float(weighted(s, y)[0]) != pytest.approx(float(flat(s, y)[0]))

    def test_gradient_flows_to_the_student_only(self):
        s = logits(seed=27).requires_grad_(True)
        t = logits(seed=28).requires_grad_(True)
        obj = DistillationObjective(alpha=0.5, beta=0.5)
        obj(s, self.targets(), teacher_logits=t)[0].backward()
        assert s.grad is not None
        assert t.grad is None

    def test_rejects_all_zero_weights(self):
        with pytest.raises(ValueError, match="nothing to optimise"):
            DistillationObjective(alpha=0.0, beta=0.0, gamma=0.0)

    def test_rejects_negative_weights(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            DistillationObjective(alpha=-1.0)
