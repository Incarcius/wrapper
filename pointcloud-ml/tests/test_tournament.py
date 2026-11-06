"""Tournament adjudication tests.

These are the tests that decide whether a published claim is defensible, so
they are written against the *properties* the design commits to rather than
against the implementation:

* comparisons are paired by seed, and a seed mismatch is an error;
* the permutation test is exact, and reports the floor its seed count imposes;
* multiple comparisons are adjusted;
* the default verdict is INCONCLUSIVE, and no path upgrades a weak result.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcml.training.tournament import (
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


def runs(arm: str, values: dict[int, float], dataset: str = "proxy", scale=None):
    return [
        RunResult(arm=arm, seed=s, dataset=dataset, scale=scale, metrics={"macro_f1": v})
        for s, v in values.items()
    ]


def comparison(diffs, p=0.01, min_p=0.03, p_adj=None) -> Comparison:
    return Comparison(
        arm="arm1",
        baseline="arm0",
        dataset="proxy",
        scale=None,
        metric="macro_f1",
        differences=tuple(diffs),
        seeds=tuple(range(len(diffs))),
        p_value=p,
        min_p=min_p,
        p_adjusted=p_adj,
    )


class TestSeedCountFloor:
    """The arithmetic that constrains the experiment design."""

    @pytest.mark.parametrize(
        "n,expected",
        [(5, 0.0625), (6, 0.03125), (8, 0.0078125), (10, 0.001953125)],
    )
    def test_two_sided_floor_is_two_over_2_to_the_n(self, n, expected):
        assert min_achievable_p(n) == pytest.approx(expected)

    def test_five_seeds_cannot_reach_significance(self):
        """The finding that changes the plan: at n=5 the floor exceeds 0.05, so
        no effect however large can be called significant two-sided."""
        assert min_achievable_p(5) > 0.05
        assert min_achievable_p(6) < 0.05

    def test_an_enormous_effect_at_five_seeds_still_cannot_reach_0_05(self):
        """Empirical confirmation, not just arithmetic."""
        huge = [10.0, 11.0, 12.0, 13.0, 14.0]  # unmistakable, consistent effect
        p, floor = permutation_test_paired(huge)
        assert p == pytest.approx(0.0625)
        assert p > 0.05
        assert floor == pytest.approx(0.0625)

    def test_the_same_effect_at_six_seeds_does_reach_it(self):
        huge = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        p, floor = permutation_test_paired(huge)
        assert p == pytest.approx(0.03125)
        assert p < 0.05
        assert floor < 0.05

    def test_one_sided_floor_is_half_the_two_sided(self):
        assert min_achievable_p(5, "greater") == pytest.approx(0.03125)

    def test_rejects_zero_pairs(self):
        with pytest.raises(ValueError, match="n_pairs must be >= 1"):
            min_achievable_p(0)


class TestPermutationTest:
    def test_exactness_on_a_hand_checkable_case(self):
        """n=3, all differences positive and equal: only the all-positive and
        all-negative assignments reach |mean| >= observed, so p = 2/8."""
        p, _ = permutation_test_paired([1.0, 1.0, 1.0])
        assert p == pytest.approx(2.0 / 8.0)

    def test_all_zero_differences_give_p_one(self):
        p, _ = permutation_test_paired([0.0, 0.0, 0.0, 0.0])
        assert p == 1.0

    def test_mixed_signs_are_not_significant(self):
        p, _ = permutation_test_paired([1.0, -1.0, 0.9, -1.1, 0.2, -0.3])
        assert p > 0.5

    def test_one_sided_is_more_powerful_than_two_sided(self):
        diffs = [0.5, 0.6, 0.4, 0.7, 0.55, 0.62]
        two, _ = permutation_test_paired(diffs, "two-sided")
        one, _ = permutation_test_paired(diffs, "greater")
        assert one < two
        assert one == pytest.approx(two / 2.0)

    def test_direction_matters_for_one_sided(self):
        diffs = [-0.5, -0.6, -0.4, -0.7, -0.55, -0.62]
        greater, _ = permutation_test_paired(diffs, "greater")
        less, _ = permutation_test_paired(diffs, "less")
        assert less < greater

    def test_is_invariant_to_scaling_the_differences(self):
        """A rank/sign-based test must not care about units."""
        a, _ = permutation_test_paired([0.01, 0.02, 0.015, 0.03, 0.025, 0.02])
        b, _ = permutation_test_paired([1.0, 2.0, 1.5, 3.0, 2.5, 2.0])
        assert a == pytest.approx(b)

    def test_rejects_empty_and_nonfinite(self):
        with pytest.raises(ValueError, match="non-empty"):
            permutation_test_paired([])
        with pytest.raises(ValueError, match="NaN or inf"):
            permutation_test_paired([1.0, float("nan")])

    def test_rejects_unknown_alternative(self):
        with pytest.raises(ValueError, match="unknown alternative"):
            permutation_test_paired([1.0, 2.0], "sideways")


class TestPairing:
    def test_differences_are_matched_by_seed_not_by_position(self):
        """Order independence: a dict ordering must not change the result."""
        arm = runs("arm1", {3: 0.7, 1: 0.6, 2: 0.65})
        base = runs("arm0", {1: 0.5, 2: 0.55, 3: 0.6})
        diffs, seeds = paired_differences(arm, base)
        assert seeds == (1, 2, 3)
        assert diffs == pytest.approx((0.1, 0.1, 0.1))

    def test_seed_mismatch_is_an_error(self):
        """Comparing 3 runs against 2 is how a spurious difference appears."""
        arm = runs("arm1", {1: 0.7, 2: 0.7, 3: 0.7})
        base = runs("arm0", {1: 0.5, 2: 0.5})
        with pytest.raises(ValueError, match="identical seed sets"):
            paired_differences(arm, base)

    def test_error_names_the_missing_seeds(self):
        arm = runs("arm1", {1: 0.7})
        base = runs("arm0", {1: 0.5, 9: 0.5})
        with pytest.raises(ValueError, match=r"arm is missing \[9\]"):
            paired_differences(arm, base)

    def test_empty_side_is_an_error(self):
        with pytest.raises(ValueError, match="at least one run"):
            paired_differences([], runs("arm0", {1: 0.5}))

    def test_pairing_is_more_powerful_than_unpaired(self):
        """Why pairing: seeds contribute shared variance, and removing it is
        what makes a small consistent effect detectable at all."""
        # Large spread across seeds, but a consistent +0.02 within each seed.
        base_vals = {s: v for s, v in zip(range(6), [0.40, 0.55, 0.62, 0.48, 0.70, 0.35])}
        arm_vals = {s: v + 0.02 for s, v in base_vals.items()}
        diffs, _ = paired_differences(runs("arm1", arm_vals), runs("arm0", base_vals))
        p_paired, _ = permutation_test_paired(diffs)
        assert p_paired == pytest.approx(0.03125)  # significant

        # Unpaired, the between-seed spread (~0.12) swamps a 0.02 shift.
        spread = float(np.std(list(base_vals.values()), ddof=1))
        assert spread > 0.1


class TestSummarise:
    def test_groups_by_arm_dataset_scale(self):
        results = (
            runs("arm0", {1: 0.5, 2: 0.6}, scale=20)
            + runs("arm0", {1: 0.7, 2: 0.8}, scale=200)
            + runs("arm1", {1: 0.55, 2: 0.65}, scale=20)
        )
        summaries = summarise(results)
        assert len(summaries) == 3
        keys = {(s.arm, s.scale) for s in summaries}
        assert keys == {("arm0", 20), ("arm0", 200), ("arm1", 20)}

    def test_mean_std_and_n(self):
        s = summarise(runs("arm0", {1: 0.4, 2: 0.6}))[0]
        assert s.n_seeds == 2
        assert s.mean == pytest.approx(0.5)
        assert s.std == pytest.approx(float(np.std([0.4, 0.6], ddof=1)))

    def test_std_uses_sample_ddof_not_population(self):
        """We are estimating a population std from a few seeds, not describing
        the sample; ddof=0 would understate the spread the decision rests on."""
        s = summarise(runs("arm0", {1: 0.0, 2: 1.0}))[0]
        assert s.std == pytest.approx(0.7071, abs=1e-4)  # ddof=1
        assert s.std != pytest.approx(0.5)  # ddof=0

    def test_single_seed_has_nan_std_not_zero(self):
        s = summarise(runs("arm0", {1: 0.5}))[0]
        assert s.n_seeds == 1
        assert np.isnan(s.std)

    def test_duplicate_seed_is_an_error(self):
        dupes = [
            RunResult(arm="a", seed=1, dataset="p", metrics={"macro_f1": 0.5}),
            RunResult(arm="a", seed=1, dataset="p", metrics={"macro_f1": 0.7}),
        ]
        with pytest.raises(ValueError, match="duplicate run"):
            summarise(dupes)

    def test_missing_metric_names_what_is_available(self):
        r = [RunResult(arm="a", seed=1, dataset="p", metrics={"accuracy": 0.9})]
        with pytest.raises(KeyError, match="macro_f1"):
            summarise(r)


class TestHolmAdjust:
    def test_single_p_is_unchanged(self):
        assert holm_adjust([0.03]) == pytest.approx([0.03])

    def test_step_down_weights(self):
        """Smallest p gets x m, next x (m-1), and so on — then monotonised.

        0.03 * 1 = 0.03, but Holm's step-down cannot report a larger p as more
        significant than a smaller one, so it is raised to the running maximum
        of 0.04. That monotonicity is the property, not an artefact.
        """
        adj = holm_adjust([0.01, 0.02, 0.03])
        assert adj[0] == pytest.approx(0.03)  # 0.01 * 3
        assert adj[1] == pytest.approx(0.04)  # 0.02 * 2
        assert adj[2] == pytest.approx(0.04)  # 0.03 * 1, raised to the running max

    def test_is_monotone_non_decreasing_in_rank(self):
        raw = [0.001, 0.009, 0.02, 0.04, 0.5]
        adj = np.array(holm_adjust(raw))
        order = np.argsort(raw)
        assert np.all(np.diff(adj[order]) >= -1e-12)

    def test_preserves_input_order(self):
        adj = holm_adjust([0.5, 0.001])
        assert adj[0] > adj[1]

    def test_clamped_to_one(self):
        assert max(holm_adjust([0.4, 0.5, 0.6])) <= 1.0

    def test_is_less_conservative_than_bonferroni(self):
        raw = [0.01, 0.02, 0.03, 0.04]
        holm = holm_adjust(raw)
        bonferroni = [min(1.0, p * len(raw)) for p in raw]
        assert all(h <= b + 1e-12 for h, b in zip(holm, bonferroni))
        assert any(h < b for h, b in zip(holm, bonferroni))

    def test_empty_and_invalid(self):
        assert holm_adjust([]) == []
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            holm_adjust([1.5])


class TestTrend:
    def test_detects_a_decaying_gain(self):
        """The scaling study's headline shape: KD gain falls as data grows."""
        scales = [20, 50, 100, 500, 2000]
        gains = [0.08, 0.05, 0.03, 0.01, -0.002]
        rho, p = trend_spearman(scales, gains)
        assert rho == pytest.approx(-1.0)
        assert p < 0.05

    def test_detects_an_increasing_gain(self):
        rho, _ = trend_spearman([20, 50, 100, 500], [0.0, 0.02, 0.04, 0.07])
        assert rho == pytest.approx(1.0)

    def test_flat_gain_is_not_a_trend(self):
        """The boring outcome must be reported as boring, not as a weak trend."""
        rho, p = trend_spearman([20, 50, 100, 500, 2000], [0.03] * 5)
        assert rho == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_noisy_non_monotone_gain_is_not_significant(self):
        rho, p = trend_spearman([20, 50, 100, 500, 2000], [0.03, 0.01, 0.04, 0.02, 0.03])
        assert p > 0.05

    def test_insensitive_to_the_scale_axis_units(self):
        """Rank-based, so log vs linear spacing cannot change the conclusion."""
        gains = [0.08, 0.05, 0.03, 0.01]
        a, _ = trend_spearman([20, 50, 100, 500], gains)
        b, _ = trend_spearman([1, 2, 3, 4], gains)
        assert a == pytest.approx(b)

    def test_needs_at_least_three_points(self):
        with pytest.raises(ValueError, match="at least 3 scales"):
            trend_spearman([20, 50], [0.1, 0.2])


class TestAdjudicate:
    def test_underpowered_is_reported_as_a_design_property(self):
        c = comparison([0.1] * 5, p=0.0625, min_p=0.0625)
        assert adjudicate(c) == "UNDERPOWERED"

    def test_adopt_requires_significance_and_effect(self):
        c = comparison([0.05] * 6, p=0.03, min_p=0.031, p_adj=0.03)
        assert adjudicate(c, min_effect=0.01) == "ADOPT"

    def test_significant_but_trivial_effect_is_not_adopted(self):
        """A detectable but irrelevant gain is no reason to ship machinery."""
        c = comparison([0.001] * 8, p=0.008, min_p=0.0078, p_adj=0.008)
        assert adjudicate(c, min_effect=0.01) == "INCONCLUSIVE"

    def test_significant_in_the_wrong_direction_is_rejected(self):
        c = comparison([-0.05] * 8, p=0.008, min_p=0.0078, p_adj=0.008)
        assert adjudicate(c, min_effect=0.01) == "REJECT"

    def test_adjusted_p_is_used_when_present(self):
        """Raw p would adopt; adjusted must not."""
        c = comparison([0.05] * 8, p=0.008, min_p=0.0078, p_adj=0.4)
        assert adjudicate(c, min_effect=0.01) == "INCONCLUSIVE"

    def test_default_is_inconclusive(self):
        c = comparison([0.01, -0.01, 0.02, -0.02, 0.0, 0.01], p=0.9, min_p=0.031, p_adj=1.0)
        assert adjudicate(c) == "INCONCLUSIVE"

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            adjudicate(comparison([0.1] * 6), alpha=1.5)


class TestEffectSize:
    def test_paired_cohens_d(self):
        c = comparison([0.02, 0.03, 0.04, 0.03, 0.02, 0.04])
        expected = np.mean(c.differences) / np.std(c.differences, ddof=1)
        assert c.effect_size == pytest.approx(expected)

    def test_zero_variance_with_an_effect_is_infinite(self):
        assert comparison([0.05] * 6).effect_size == float("inf")

    def test_is_underpowered_flag_tracks_the_floor(self):
        assert comparison([0.1] * 5, min_p=0.0625).is_underpowered is True
        assert comparison([0.1] * 6, min_p=0.03125).is_underpowered is False


class TestTable:
    def test_generates_markdown_with_a_verdict_column(self):
        summaries = summarise(runs("arm1", {s: 0.6 for s in range(6)}))
        c = comparison([0.05] * 6, p=0.03, min_p=0.031, p_adj=0.03)
        table = format_summary_table(summaries, [c], min_effect=0.01)
        assert "| arm | dataset | scale |" in table
        assert "ADOPT" in table
        assert "+0.0500" in table

    def test_arms_without_a_comparison_show_a_dash(self):
        table = format_summary_table(summarise(runs("arm0", {1: 0.5, 2: 0.6})))
        assert "—" in table
