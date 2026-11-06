"""Distillation objectives.

One composition serves every arm in ``docs/distillation_plan.md``:

    L = alpha * CE(student, y)
      + beta  * T^2 * KL(teacher/T || student/T)
      + gamma * FEAT(student_embedding, teacher_embedding)

Arms differ only in where the teacher logits come from (frozen checkpoint, EMA
of the student, or a live peer) and in the weights — not in this code. That is
what lets the benchmark claim the arms share a code path rather than merely
asserting that nothing else changed.

Three details here are the ones distillation implementations get wrong:

* **The ``T^2`` factor.** Dividing logits by ``T`` scales the softmax gradients
  by roughly ``1/T^2``. Without the correction, ``beta`` silently changes
  meaning every time ``T`` is tuned, and the two become impossible to
  disentangle in an ablation.
* **Log-space divergence.** ``log_target=True`` consumes the teacher's
  log-probabilities directly instead of exponentiating and re-logging them,
  which is where KD losses lose precision and occasionally produce NaNs.
* **Reduction.** ``"batchmean"`` is the mathematically correct reduction for KL;
  torch's ``"mean"`` also divides by the class count, quietly scaling the loss
  by 1/K (1/6 here) and therefore scaling the effective ``beta`` too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "cross_entropy_loss",
    "kd_per_sample",
    "kd_loss",
    "teacher_confidence",
    "feature_loss",
    "FeatureProjector",
    "DistillationObjective",
]

ConfidenceMode = Literal["max_prob", "neg_entropy"]
FeatureMode = Literal["cosine", "mse"]


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Hard-label supervision, optionally class-weighted.

    Class weights come from the training-split distribution and are recorded in
    the label map, so an experiment is reproducible from config alone. Imbalance
    here is severe (``vehicle`` may have a handful of distinct instances), which
    is also why macro-F1 rather than accuracy is the selection metric.
    """
    return F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def kd_per_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Temperature-scaled KL divergence per sample, ``(B,)``, already ``T^2``-scaled.

    Returned per sample so that confidence weighting (§6 of the plan) can be
    applied before reduction. The teacher is detached here, so no arm can
    accidentally backpropagate into its teacher — including the mutual-learning
    arm, where the "teacher" is a live peer that must be updated by its *own*
    loss, not by its partner's.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student logits {tuple(student_logits.shape)} and teacher logits "
            f"{tuple(teacher_logits.shape)} must match"
        )

    student_log_p = F.log_softmax(student_logits / temperature, dim=1)
    teacher_log_p = F.log_softmax(teacher_logits.detach() / temperature, dim=1)

    # reduction="none" gives (B, K); summing over classes gives the per-sample KL.
    per_class = F.kl_div(student_log_p, teacher_log_p, log_target=True, reduction="none")
    return per_class.sum(dim=1) * (temperature**2)


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduced KD loss. Equivalent to ``reduction="batchmean"`` when unweighted.

    Args:
        weights: optional per-sample weights, ``(B,)``. Normalised to mean 1 so
            that turning confidence weighting on does not also change the
            effective magnitude of ``beta`` — otherwise the ablation would
            confound "weighting helps" with "beta got smaller".
    """
    per_sample = kd_per_sample(student_logits, teacher_logits, temperature)
    if weights is None:
        return per_sample.mean()

    if weights.shape != per_sample.shape:
        raise ValueError(
            f"weights {tuple(weights.shape)} must be per-sample {tuple(per_sample.shape)}"
        )
    w = weights.detach()
    mean_w = w.mean()
    if float(mean_w) <= 0:
        # Every sample was discarded. Returning 0 keeps the step well-defined.
        return per_sample.sum() * 0.0
    return (per_sample * (w / mean_w)).mean()


def teacher_confidence(
    teacher_logits: torch.Tensor,
    mode: ConfidenceMode = "max_prob",
) -> torch.Tensor:
    """Per-sample confidence in ``[0, 1]``, for confidence-weighted distillation.

    The point (plan §6): on rare classes the teacher's soft targets are
    themselves poorly estimated, and unweighted KD propagates that uncertainty
    as though it were knowledge. Down-weighting the samples the teacher is
    unsure about stops the student imitating its confusion.

    Args:
        mode: ``"max_prob"`` is the maximum softmax probability. ``"neg_entropy"``
            is ``1 - H/log(K)``, which uses the whole distribution rather than
            only its peak and is the better choice when K is small, since
            ``max_prob`` saturates.
    """
    with torch.no_grad():
        probs = F.softmax(teacher_logits.detach(), dim=1)
        if mode == "max_prob":
            return probs.max(dim=1).values
        if mode == "neg_entropy":
            k = probs.shape[1]
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            return 1.0 - entropy / torch.log(torch.tensor(float(k), device=probs.device))
    raise ValueError(f"unknown confidence mode {mode!r}; expected max_prob or neg_entropy")


def feature_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    mode: FeatureMode = "cosine",
) -> torch.Tensor:
    """Align global embeddings, on **L2-normalised** vectors.

    Normalising first is not optional. The teacher's embedding scale is
    arbitrary (it is a max-pool over unbounded activations), so unnormalised MSE
    would mostly penalise magnitude rather than direction — the student would
    spend capacity matching a scale that carries no information.

    Note what this does *and does not* reach in this architecture: the embedding
    sits before the classifier head, and the head holds ~66% of the deploy
    model's parameters. So feature distillation directly supervises only about a
    third of the student's weights, which is why its expected effect here is
    smaller than the literature on backbone-dominated models suggests.
    """
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"student features {tuple(student_features.shape)} and teacher features "
            f"{tuple(teacher_features.shape)} must match — use a FeatureProjector "
            "when the embedding widths differ"
        )

    s = F.normalize(student_features, dim=1)
    t = F.normalize(teacher_features.detach(), dim=1)

    if mode == "cosine":
        return (1.0 - (s * t).sum(dim=1)).mean()
    if mode == "mse":
        return F.mse_loss(s, t)
    raise ValueError(f"unknown feature mode {mode!r}; expected cosine or mse")


class FeatureProjector(nn.Module):
    """Maps the student embedding up to the teacher's width. Discarded after training.

    Projects **student -> teacher**, never the reverse: projecting the teacher
    down would let the projection discard exactly the information being
    transferred, and the student would be rewarded for matching a lossy summary.

    Costs nothing at inference because it is not part of the deployed model —
    the student is exported without it.
    """

    def __init__(self, student_dim: int, teacher_dim: int, bias: bool = False) -> None:
        super().__init__()
        if student_dim < 1 or teacher_dim < 1:
            raise ValueError(
                f"embedding dims must be >= 1, got student={student_dim}, "
                f"teacher={teacher_dim}"
            )
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.proj = nn.Linear(student_dim, teacher_dim, bias=bias)

    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        return self.proj(student_features)


@dataclass
class DistillationObjective:
    """The composed loss. One object configures every arm.

    Args:
        alpha: weight on hard-label cross-entropy.
        beta: weight on the KD term. ``0`` disables it, which is arm 0.
        gamma: weight on the feature term. ``0`` disables it.
        temperature: KD temperature.
        confidence_weighting: ``None``, ``"max_prob"`` or ``"neg_entropy"``.
        feature_mode: ``"cosine"`` or ``"mse"``.
        class_weights: per-class CE weights, or ``None`` for unweighted.
        label_smoothing: applied to the CE term only.

    Calling it returns ``(total, components)``; ``components`` is logged every
    step so an ablation can see which term actually moved, rather than only the
    sum.
    """

    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    temperature: float = 4.0
    confidence_weighting: Optional[ConfidenceMode] = None
    feature_mode: FeatureMode = "cosine"
    class_weights: Optional[torch.Tensor] = None
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.alpha == 0 and self.beta == 0 and self.gamma == 0:
            raise ValueError("all loss weights are zero: there is nothing to optimise")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

    @property
    def uses_teacher(self) -> bool:
        """Whether this arm needs teacher outputs at all (arm 0 does not)."""
        return self.beta > 0 or self.gamma > 0

    def __call__(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        components: dict[str, float] = {}
        total = student_logits.sum() * 0.0  # zero that carries grad and device

        if self.alpha > 0:
            ce = cross_entropy_loss(
                student_logits,
                targets,
                class_weights=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            total = total + self.alpha * ce
            components["ce"] = float(ce.detach())

        if self.beta > 0:
            if teacher_logits is None:
                raise ValueError(
                    f"beta={self.beta} requires teacher_logits, got None. An arm with "
                    "a KD term must be given a target source."
                )
            weights = (
                teacher_confidence(teacher_logits, self.confidence_weighting)
                if self.confidence_weighting is not None
                else None
            )
            kd = kd_loss(
                student_logits, teacher_logits, self.temperature, weights=weights
            )
            total = total + self.beta * kd
            components["kd"] = float(kd.detach())
            if weights is not None:
                components["mean_teacher_confidence"] = float(weights.mean())

        if self.gamma > 0:
            if student_features is None or teacher_features is None:
                raise ValueError(
                    f"gamma={self.gamma} requires both student_features and "
                    "teacher_features"
                )
            feat = feature_loss(student_features, teacher_features, self.feature_mode)
            total = total + self.gamma * feat
            components["feature"] = float(feat.detach())

        components["total"] = float(total.detach())
        return total, components
