# Publication Pipeline — step by step, with gates

**Recorded:** 6 Nov 2026. Companion to
[`publication_scope.md`](publication_scope.md) (what to claim) and
[`distillation_plan.md`](distillation_plan.md) (how the arms work). This document
is the **order of operations**, and what each stage is allowed to conclude.

---

## 0. A word on "irrefutable"

No empirical result is irrefutable, and a paper that claims to be invites the
reviewer to find the hole. What *is* achievable is **hard to dismiss**, and that
is a property built in beforehand, not argued for afterwards. Eight things do
almost all of the work:

| # | Commitment | The objection it pre-empts |
|---|---|---|
| 1 | **Pre-registration**, committed and dated before the first training run | "you picked the metric/rule that made your method win" |
| 2 | **Paired** comparisons by seed | "your difference is between-seed variance" |
| 3 | **Exact permutation tests**, not t-tests | "n=8 does not justify a normality assumption" |
| 4 | **Holm adjustment** across all comparisons | "you made 30 comparisons and reported the lucky one" |
| 5 | **≥8 seeds** (§1 below) | "underpowered" |
| 6 | **Baseline tuned as hard as the arms**, with the budget documented | "your baseline was crippled" — the most common and most fatal |
| 7 | **Every arm reported**, including the ones that lost | "what happened to the arms in your plan that aren't in your table?" |
| 8 | **Test split touched once**, at the end | "you selected on test" |

All eight are cheap *if decided now* and impossible to retrofit. That asymmetry
is the entire reason this document exists.

## 1. Correction: the seed count must be 8, not 5

Established today while building `tournament.py`, and it supersedes the "≥5
seeds" in `publication_scope.md`.

Significance is tested by **exact paired permutation**: under the null, the sign
of each paired difference is symmetric, so all `2^n` sign assignments are
enumerated. That makes the p-value exact — and it imposes a floor:

| seeds | minimum two-sided p |
|---:|---:|
| 5 | **0.0625** |
| 6 | 0.0312 |
| 8 | 0.0078 |
| 10 | 0.0020 |

**At five seeds no effect, however large, can reach p < 0.05.** Verified
empirically, not just derived: a perfectly consistent +10-point difference
across 5 seeds returns p = 0.0625
(`test_an_enormous_effect_at_five_seeds_still_cannot_reach_0_05`).

Six is the bare floor; **eight** is the choice, because Holm adjustment across
~30 comparisons needs headroom below 0.05 and six leaves none. Every p-value is
reported alongside its floor, so a null result reads as "underpowered by
construction" or "no effect", never ambiguously.

The cost is small: these models train in 1–2 GPU-minutes, so 8 seeds instead of
5 is a few extra GPU-hours across the whole study.

## 2. The stages

Each stage names what it needs, what it produces, and **the gate** — the
condition for proceeding. A failed gate is a finding, not a blocker.

### Stage 0 — Pre-register · *no data or hardware needed*

Commit a dated `docs/preregistration.md` fixing: the hypotheses, the arm list,
the primary metric (**macro-F1**), the seed count (**8**), `min_effect`, the
adjudication rule, and the stopping rule. The rule is already implemented in
`tournament.adjudicate`, which is what makes pre-registration meaningful rather
than aspirational — the decision procedure is code, written before the results.

**Gate:** nothing trains until this is committed.

### Stage 1 — Validate the implementation · *ModelNet40*

Train the deploy PointNet on ModelNet40 and compare against the published
~89% overall accuracy. Our input contract is already ModelNet's convention
(1024 points, unit sphere), so this needs no pipeline changes.

**Gate:** within ~2 points of published. Outside that, there is a bug, and every
later number would have been uninterpretable. **This is the stage that gives the
study a reference point it currently does not have** — without it we cannot tell
a hard task from a broken trainer.

*Produces:* a sanity table. Labelled CAD-domain; never quoted as task accuracy.

### Stage 2 — Power analysis · *proxy, subsampled*

arm 0 vs arm 1, 8 seeds, at the **target** scale (15–20 instances per class).
Measure the seed-to-seed spread.

**Gate:** if the spread swamps plausible arm differences, the paper's claims are
restricted to the ladder (Stage 4) and the field data becomes qualitative only.
Deciding that here costs one afternoon; discovering it after Stage 4 wastes the
study.

### Stage 3 — The capacity gate · *proxy, full*

Train the 37k student and the 2M teacher from scratch under identical protocols.

**Gate:** if the student matches the teacher, **there is no capacity gap for
distillation to close**. The study is then reframed around that negative result
(a legitimate contribution, `publication_scope.md` §2d) and the remaining arms
are not built. If a gap exists, it is the headroom every later number is
measured against.

*Produces:* the paper's Table 1.

### Stage 4 — The ladder · *proxy, 5 scales × arms × 8 seeds* — **the main experiment**

Arms across training-set sizes ≈ 20 / 50 / 100 / 500 / 2000 instances per class.
`trend_spearman` on gain-versus-scale is the primary statistic; `summarise` and
`paired_differences` feed `adjudicate` per cell.

A trend across five scales is far more robust than any single pairwise
comparison, which is the point: it tests a **slope**, retaining power where
individual cells are noisy.

**Gate:** none — every outcome is reportable. A decaying gain is Finding A; a
flat one is a calibration result; an increasing one is a surprise worth chasing.

*Produces:* Findings A and B. ≈ 8–12 GPU-hours.

### Stage 5 — On-device measurement · *Jetson Orin Nano* — **runs in parallel**

Needs **no training and no data**: the three ONNX models are already exported
and parity-verified to 2.98e-08. Measure latency (median/p90/p95/p99) across
widths, precisions (fp32/fp16/int8) and region counts `M`, with warm-up, power
mode, thermal state and sustained throughput recorded.

**Gate:** does latency scale with model width? **If it does not** — plausible,
because a 250k-parameter model may be kernel-launch bound rather than FLOP bound
— then pruning has no purpose here and that is itself a finding worth
publishing.

*Produces:* Finding C. **This is the critical path for a robotics venue**, and
it is blocked on nothing but physical access to the device.

### Stage 6 — Transfer check · *field data, validation only*

Take the arm selected on the proxy and confirm it on real Mid-70 crops. If it
fails to beat arm 0 here, the pre-registered rule applies: **ship arm 0**, and
report the non-transfer — which is Finding B and the more interesting outcome.

**Gate:** the field **test** split is opened exactly once, after this, for the
final number only.

### Stage 7 — Write-up

Tables generated by scripts, never transcribed — `format_summary_table` exists
so the manuscript's numbers cannot drift from the runs that produced them. Add
`REPRODUCING.md` mapping each table to the command that made it, built up as the
tables are generated rather than reconstructed at submission.

## 3. Dependency structure

```
Stage 0  pre-register ─────────────────┐
                                       ▼
Stage 1  ModelNet40 validation ──► Stage 2  power ──► Stage 3  capacity gate
                                                            │
                                                            ▼
                                                      Stage 4  the ladder ──┐
                                                                            ▼
Stage 5  Jetson  (independent — start as soon as the device exists) ──► Stage 6 ──► Stage 7
```

Stage 5 shares no dependency with 1–4. If the Jetson becomes available at any
point, it should be measured immediately rather than queued behind training.

## 4. Venue: keep both open, decide from the results

Deciding at write-up time is sound, and it is nearly free, because the two
venues' *experiments* overlap far more than their framing does. Keeping both
open costs exactly two additions:

1. **A second architecture in the ladder** (DGCNN or PointNet++). An ML venue
   will ask whether the finding is a property of knowledge distillation or of
   one small PointNet, and only a second architecture family answers that.
   Already required for the Stage 4 architecture comparison, so this is not new
   work.
2. **The Jetson measurements.** Already required for deployment.

Both were on the roadmap regardless, so the dual track adds essentially nothing.

**Calibrated expectation**, since the aim is a strong venue:

| Target | Realistic? |
|---|---|
| **RA-L** (journal, presented at ICRA/IROS) | **Best fit.** Values real-hardware validation; **rolling submission**, so no deadline pressure — which suits a project gated on device access and fieldwork |
| **ICRA / IROS** main track | Achievable for a well-executed on-device study with a real sensor |
| WACV, or CVPR/ICCV/ECCV efficiency workshops | Achievable |
| **NeurIPS / ICML / ICLR** main track | A stretch for an applied KD study. Would need Finding A to be a clean, architecture-general scaling law — possible, but not the base case |

The decision rule I would use: **if Finding A generalises across both
architectures and the curve is clean, aim ML; otherwise RA-L**, where the Mid-70
and the Jetson are assets rather than incidental detail.

## 5. What is built, and what remains

**Built and tested** (53 tests in `tests/test_tournament.py`, 83 across the
training modules):

- `tournament.py` — pairing, exact permutation tests, Holm, Spearman trend,
  the adjudication rule, markdown table generation
- `metrics.py` — macro-F1 with the absent-class convention, calibration
- `losses.py` — the single objective covering every arm
- `reproducibility.py` — independent RNG streams

**Remaining:** the training loop itself, the EMA wrapper, checkpointing and run
manifests, the arm configs, and the matrix runner. None of it is blocked; the
loop is next.
