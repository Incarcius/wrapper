"""The distillation tournament: aggregating runs and adjudicating between arms.

This module decides what the benchmark is allowed to conclude. It is separate
from the training loop on purpose — the decision rule is pre-registered
(``docs/publication_scope.md``), so it must be implementable and testable
*before* any result exists, and it must not be adjustable after seeing them.

Three design commitments, each of which answers an objection a reviewer would
otherwise raise:

**1. Comparisons are paired by seed.** Arm A at seed 3 is compared against the
baseline at seed 3. Pairing removes the variance that seeds contribute in
common, which is most of it, and is strictly more powerful than comparing two
independent means. Unmatched seeds are an error rather than a silent drop,
because quietly comparing 5 of A against 4 of the baseline is how a spurious
difference appears.

**2. Significance is tested by exact paired permutation**, not a t-test. With a
handful of seeds the normality assumption behind a t-test is unjustifiable, and
a permutation test needs no distributional assumption: under the null, the sign
of each paired difference is symmetric, so enumerating all 2^n sign assignments
gives an exact p-value.

This has a consequence worth knowing **before** choosing a seed count. The
smallest two-sided p an exact paired permutation test can return is ``2 / 2^n``:

=====  ===================
seeds  minimum two-sided p
=====  ===================
5      0.0625
6      0.0312
8      0.0078
10     0.0020
=====  ===================

So **five seeds cannot produce p < 0.05 no matter how large the effect**. Six is
the floor if p-values are to be reported at all, and eight is the sensible
choice. :func:`min_achievable_p` computes this, and
:func:`permutation_test_paired` returns it alongside every p-value so a
non-significant result can be read as "underpowered by construction" rather than
"no effect".

**3. The default verdict is INCONCLUSIVE.** An arm is adopted only by clearing
the pre-registered bar. "Did not clear it" is reported as such, never upgraded
to a win by reaching for a different metric or dropping a seed.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Sequence

import numpy as np

__all__ = [
    "RunResult",
    "ArmSummary",
    "Comparison",
    "Verdict",
    "summarise",
    "paired_differences",
    "min_achievable_p",
    "permutation_test_paired",
    "holm_adjust",
    "trend_spearman",
    "adjudicate",
    "format_summary_table",
]

Verdict = Literal["ADOPT", "REJECT", "INCONCLUSIVE", "UNDERPOWERED"]
Alternative = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True)
class RunResult:
    """One training run. The unit the tournament aggregates.

    ``scale`` is the training-set size in instances per class, which is the
    independent variable of the scaling study; ``None`` means the full dataset.
    ``provenance`` carries the hashes, seeds and step counts that make the run
    reconstructible — a result without it cannot go in a paper.
    """

    arm: str
    seed: int
    dataset: str
    metrics: dict[str, float]
    scale: Optional[int] = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str) -> float:
        if name not in self.metrics:
            raise KeyError(
                f"run {self.arm}/seed{self.seed} has no metric {name!r}; "
                f"available: {sorted(self.metrics)}"
            )
        return float(self.metrics[name])


@dataclass(frozen=True)
class ArmSummary:
    """One arm at one (dataset, scale), across seeds."""

    arm: str
    dataset: str
    scale: Optional[int]
    metric: str
    values: tuple[float, ...]
    seeds: tuple[int, ...]

    @property
    def n_seeds(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")

    @property
    def std(self) -> float:
        """Sample standard deviation (ddof=1): we are estimating, not describing."""
        if self.n_seeds < 2:
            return float("nan")
        return float(np.std(self.values, ddof=1))

    @property
    def sem(self) -> float:
        if self.n_seeds < 2:
            return float("nan")
        return self.std / math.sqrt(self.n_seeds)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.mean:.4f} ± {self.std:.4f} (n={self.n_seeds})"


@dataclass(frozen=True)
class Comparison:
    """An arm against the baseline, paired by seed."""

    arm: str
    baseline: str
    dataset: str
    scale: Optional[int]
    metric: str
    differences: tuple[float, ...]
    seeds: tuple[int, ...]
    p_value: float
    min_p: float
    p_adjusted: Optional[float] = None

    @property
    def n_pairs(self) -> int:
        return len(self.differences)

    @property
    def mean_difference(self) -> float:
        return float(np.mean(self.differences)) if self.differences else float("nan")

    @property
    def std_difference(self) -> float:
        if self.n_pairs < 2:
            return float("nan")
        return float(np.std(self.differences, ddof=1))

    @property
    def effect_size(self) -> float:
        """Paired Cohen's d. Reported because with few seeds it says more than p.

        An exactly-consistent effect gives infinity rather than a huge finite
        number. ``np.std`` on identical values returns ~1e-18 rather than 0, so
        the guard is relative to the mean; an ``== 0`` test would silently
        return 6.6e15 for a perfectly consistent difference.
        """
        sd = self.std_difference
        mean = self.mean_difference
        if not np.isfinite(sd):
            return float("nan")
        if sd <= max(abs(mean), 1e-12) * 1e-9:
            return 0.0 if mean == 0 else float("inf")
        return mean / sd

    @property
    def is_underpowered(self) -> bool:
        """True when no effect size could have reached significance."""
        return self.min_p > 0.05


def summarise(
    results: Iterable[RunResult], metric: str = "macro_f1"
) -> list[ArmSummary]:
    """Group runs by (arm, dataset, scale) and summarise across seeds.

    Raises:
        ValueError: on a duplicated (arm, dataset, scale, seed). A duplicate
            usually means a re-run was appended rather than replacing the
            original, and silently averaging the two would understate variance.
    """
    buckets: dict[tuple[str, str, Optional[int]], dict[int, float]] = {}
    for r in results:
        key = (r.arm, r.dataset, r.scale)
        bucket = buckets.setdefault(key, {})
        if r.seed in bucket:
            raise ValueError(
                f"duplicate run for arm={r.arm!r} dataset={r.dataset!r} "
                f"scale={r.scale} seed={r.seed}. Replace the earlier result "
                "rather than appending a second one."
            )
        bucket[r.seed] = r.metric(metric)

    out: list[ArmSummary] = []
    for (arm, dataset, scale), by_seed in sorted(
        buckets.items(), key=lambda kv: (kv[0][1], str(kv[0][2]), kv[0][0])
    ):
        seeds = tuple(sorted(by_seed))
        out.append(
            ArmSummary(
                arm=arm,
                dataset=dataset,
                scale=scale,
                metric=metric,
                values=tuple(by_seed[s] for s in seeds),
                seeds=seeds,
            )
        )
    return out


def paired_differences(
    arm: Sequence[RunResult],
    baseline: Sequence[RunResult],
    metric: str = "macro_f1",
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """``(arm - baseline)`` per shared seed, and the seeds used.

    Raises:
        ValueError: if the seed sets differ. Comparing 5 runs of one arm against
            4 of another is how a spurious difference gets published; the
            mismatch must be fixed, not absorbed.
    """
    arm_by_seed = {r.seed: r.metric(metric) for r in arm}
    base_by_seed = {r.seed: r.metric(metric) for r in baseline}
    if not arm_by_seed or not base_by_seed:
        raise ValueError("both arms need at least one run to be compared")

    if set(arm_by_seed) != set(base_by_seed):
        missing_arm = sorted(set(base_by_seed) - set(arm_by_seed))
        missing_base = sorted(set(arm_by_seed) - set(base_by_seed))
        raise ValueError(
            "paired comparison needs identical seed sets; "
            f"arm is missing {missing_arm}, baseline is missing {missing_base}. "
            "Run the gap rather than comparing unequal sets."
        )

    seeds = tuple(sorted(arm_by_seed))
    diffs = tuple(arm_by_seed[s] - base_by_seed[s] for s in seeds)
    return diffs, seeds


def min_achievable_p(n_pairs: int, alternative: Alternative = "two-sided") -> float:
    """Smallest p an exact paired permutation test can return for ``n_pairs``.

    ``2 / 2^n`` two-sided, ``1 / 2^n`` one-sided. The reason to compute this
    *before* choosing a seed count: at n=5 the two-sided floor is 0.0625, so no
    effect however large can reach p < 0.05.
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    one_sided = 1.0 / (2.0**n_pairs)
    return min(1.0, 2.0 * one_sided) if alternative == "two-sided" else one_sided


def permutation_test_paired(
    differences: Sequence[float],
    alternative: Alternative = "two-sided",
    max_exact: int = 20,
    n_resamples: int = 100_000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Exact paired sign-flip permutation test on the mean difference.

    Under the null the sign of each paired difference is symmetric, so every one
    of the ``2^n`` sign assignments is equally likely. For ``n <= max_exact`` all
    are enumerated, which makes the p-value exact rather than approximate; above
    that, signs are sampled.

    Returns:
        ``(p_value, min_achievable_p)``. The second value is returned with every
        test so a non-significant result can be distinguished from an
        underpowered one.
    """
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("differences must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(d)):
        raise ValueError("differences contain NaN or inf")

    n = d.size
    observed = float(d.mean())
    floor = min_achievable_p(n, alternative)

    if np.allclose(d, 0.0):
        return 1.0, floor

    if n <= max_exact:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        means = signs @ d / n
    else:  # pragma: no cover - large-n path
        gen = rng if rng is not None else np.random.default_rng(0)
        signs = gen.choice([1.0, -1.0], size=(n_resamples, n))
        means = signs @ d / n

    if alternative == "two-sided":
        count = int(np.count_nonzero(np.abs(means) >= abs(observed) - 1e-12))
    elif alternative == "greater":
        count = int(np.count_nonzero(means >= observed - 1e-12))
    elif alternative == "less":
        count = int(np.count_nonzero(means <= observed + 1e-12))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")

    return float(count / means.shape[0]), floor


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, preserving input order.

    The tournament makes many comparisons — arms × scales × datasets — so an
    unadjusted 0.05 would produce false positives by construction. Holm is
    chosen over plain Bonferroni because it is uniformly more powerful while
    controlling the same family-wise error rate, and over Benjamini-Hochberg
    because a false *adoption* here means shipping a technique that does not
    work, so family-wise control is the right trade.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return []
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)  # enforce monotonicity
        adjusted[idx] = min(1.0, running)
    return [float(x) for x in adjusted]


def trend_spearman(
    scales: Sequence[float], gains: Sequence[float]
) -> tuple[float, float]:
    """Spearman rank correlation between training-set size and KD gain.

    The scaling study's primary statistic. Rank-based rather than a fitted slope
    because the relationship is not assumed linear and the point counts are
    small; monotonicity is the claim, not a coefficient.

    Returns:
        ``(rho, p_two_sided)``. The p-value comes from an exact permutation over
        orderings for small n, so it is valid at the sample sizes involved.
    """
    x = np.asarray(scales, dtype=np.float64)
    y = np.asarray(gains, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"scales {x.shape} and gains {y.shape} must match")
    if x.size < 3:
        raise ValueError(f"need at least 3 scales for a trend, got {x.size}")

    def rank(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(a.size, dtype=np.float64)
        # Average ties so repeated values do not bias the correlation.
        for value in np.unique(a):
            mask = a == value
            if mask.sum() > 1:
                r[mask] = r[mask].mean()
        return r

    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0, 1.0
    rho = float(np.corrcoef(rx, ry)[0, 1])

    # Exact permutation over orderings of y.
    n = x.size
    if n <= 8:
        count = total = 0
        for perm in itertools.permutations(range(n)):
            r = float(np.corrcoef(rx, ry[list(perm)])[0, 1])
            total += 1
            if abs(r) >= abs(rho) - 1e-12:
                count += 1
        return rho, float(count / total)

    gen = np.random.default_rng(0)  # pragma: no cover - large-n path
    draws = 20_000
    count = 0
    for _ in range(draws):
        r = float(np.corrcoef(rx, gen.permutation(ry))[0, 1])
        if abs(r) >= abs(rho) - 1e-12:
            count += 1
    return rho, float(count / draws)


def adjudicate(
    comparison: Comparison,
    min_effect: float = 0.0,
    alpha: float = 0.05,
) -> Verdict:
    """Apply the pre-registered rule to one comparison.

    The rule, fixed before any result exists:

    * ``UNDERPOWERED`` — the seed count made significance unreachable
      (``min_p > alpha``). Reported as a property of the design, not of the arm.
    * ``ADOPT`` — the adjusted p-value clears ``alpha`` **and** the mean
      improvement exceeds ``min_effect``. Both, because a statistically
      detectable but practically irrelevant gain is not a reason to ship extra
      machinery.
    * ``REJECT`` — significant in the wrong direction.
    * ``INCONCLUSIVE`` — everything else, and the default.

    ``min_effect`` should be set from what a deployment would actually notice,
    not from what the data turned out to show.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if comparison.min_p > alpha:
        return "UNDERPOWERED"

    p = comparison.p_adjusted if comparison.p_adjusted is not None else comparison.p_value
    if p > alpha:
        return "INCONCLUSIVE"
    if comparison.mean_difference > min_effect:
        return "ADOPT"
    if comparison.mean_difference < -min_effect:
        return "REJECT"
    return "INCONCLUSIVE"


def format_summary_table(
    summaries: Sequence[ArmSummary],
    comparisons: Sequence[Comparison] = (),
    min_effect: float = 0.0,
) -> str:
    """A markdown table, so a paper table is generated rather than transcribed.

    Transcribing numbers by hand into a manuscript is where they stop matching
    the runs that produced them.
    """
    by_key = {(c.arm, c.dataset, c.scale): c for c in comparisons}
    lines = [
        "| arm | dataset | scale | n | mean | std | vs baseline | p (adj) | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        c = by_key.get((s.arm, s.dataset, s.scale))
        scale = "full" if s.scale is None else str(s.scale)
        if c is None:
            delta = p_text = verdict = "—"
        else:
            delta = f"{c.mean_difference:+.4f}"
            shown = c.p_adjusted if c.p_adjusted is not None else c.p_value
            p_text = f"{shown:.4f}"
            verdict = adjudicate(c, min_effect=min_effect)
        lines.append(
            f"| {s.arm} | {s.dataset} | {scale} | {s.n_seeds} | "
            f"{s.mean:.4f} | {s.std:.4f} | {delta} | {p_text} | {verdict} |"
        )
    return "\n".join(lines)
