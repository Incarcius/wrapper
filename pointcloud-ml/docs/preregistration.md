# Pre-registration — distillation paradigm study

**Committed:** 6 Nov 2026, before any training run. No model has been trained
and no accuracy number exists in this repository at the time of writing.

**Amendments** are appended to §10 with a date and a reason. Silent edits are
not acceptable; a dated amendment is credible, an undated one is not.

**The decision rule is already code** — `pcml.training.tournament.adjudicate`,
with 53 tests — which is what makes this document a commitment rather than an
intention. The procedure that will declare a winner was written before any
result existed and can be inspected.

---

## 1. The thesis

Not a list of comparisons, but one mechanism and its consequences.

> **In the low-data regime, the teacher is data-starved too.**

Offline knowledge distillation is nearly always evaluated on benchmark-scale
datasets, where the teacher is strong because it had thousands of examples to
learn from. Edge deployment does not look like that: you have one small bespoke
dataset, and *both* models must be trained from it. A teacher trained on 20
instances per class has little knowledge to transfer.

If that is right, two things follow. First, offline KD's advantage should shrink
or disappear at small scale — not because distillation stops working, but
because there is no longer a knowledgeable teacher. Second, paradigms that need
**no separately-trained teacher** — EMA self-distillation, Born-Again, mutual
learning — should do comparatively *better* at small scale, because they do not
depend on a resource that scarcity has removed.

The published ordering of KD paradigms may therefore be an artifact of the data
scale at which the field measures them. That is the claim this study tests.

## 2. Design

### 2.1 Two teacher conditions — the key design decision

Whether the teacher sees the same subsample as the student changes the question
being asked, so both are run and pre-registered separately:

| Condition | Teacher trained on | Question it answers |
|---|---|---|
| **S** (same) | the **same subsample** as the student | "Does KD help when this is *all* the data you have?" — the realistic deployment condition |
| **F** (full) | the **full** dataset | "Does a strong teacher help a data-starved student?" — an upper bound, closer to a transfer setting |

Comparing S against F isolates how much of KD's value comes purely from the
teacher having seen more data. Only the offline arms (3, 4) have both
conditions; the self and mutual arms have no separately-trained teacher and
therefore exist only in condition S, which is itself part of the point.

### 2.2 Arms

Fixed. All are reported, including those that lose.

| # | Arm | Paradigm | Conditions |
|---|---|---|---|
| 0 | from scratch | none — **the baseline** | S |
| 1 | EMA mean-teacher + two-view consistency | self | S |
| 2 | Born-Again | self | S |
| 3 | offline KD, wide-PointNet teacher | offline | S, F |
| 4 | offline KD, DGCNN or PointNet++ teacher | offline, cross-architecture | S, F |
| 5 | mutual learning (249k ↔ 37k) | online | S |
| 6 | best of 1–5 plus feature distillation | — | as inherited |

### 2.3 Scales, seeds, datasets

- **Scales:** 20, 50, 100, 500, 2000 instances per class (and full, where the
  dataset allows).
- **Seeds:** **8**, fixed in advance. Exact paired permutation testing has a
  two-sided floor of `2/2^n`, which is 0.0625 at n=5 — so five seeds could not
  reach p < 0.05 for any effect size, and six leaves no headroom for Holm
  adjustment across ~30 comparisons.
- **Datasets:** a public proxy with real statistical power (SemanticKITTI
  instance crops preferred for class alignment; ScanObjectNN as the lighter
  fallback), and the Mid-70 field data as the transfer target.
- **Architectures:** the compact PointNet student throughout, plus **one
  additional architecture family** in the ladder, so that any finding can be
  attributed to distillation rather than to one small PointNet.

## 3. Primary metric

**Macro-F1**, with the absent-class convention implemented in
`pcml.training.metrics`: averaged over classes present in the reference, with
absent classes reported separately. Chosen over accuracy because class
imbalance here is severe enough that accuracy would be dominated by whichever
class is most common.

Secondary, reported but not decisive: balanced accuracy, per-class F1, expected
calibration error, and on-device latency.

## 4. `min_effect` = 0.02 absolute macro-F1

The gain below which distillation is not worth adopting. Reasoning, in the order
it actually bears:

1. **Unit of meaning.** With K=6, one class's F1 moving by 0.12 shifts macro-F1
   by 0.02. So 0.02 is roughly "one class got meaningfully better" — a natural
   granularity for this task rather than an arbitrary round number.
2. **Resolvability.** Expected seed standard deviation is 0.01–0.03 on the proxy
   at full scale. At n=8 the standard error is σ/2.83, so a 0.02 mean difference
   sits at roughly 2–3 standard errors: detectable. At the target scale
   (σ ≈ 0.03–0.06) it is **not** resolvable, which is precisely why selection
   happens on the proxy and the field data confirms rather than decides.
3. **Cost of adoption.** Adopting KD costs one-off teacher training, a second
   checkpoint to version, and three extra hyperparameters (T, β, γ). The
   *deployed artifact is unchanged* — all arms export the same architecture — so
   the cost is real but bounded, which argues against a high bar. The
   alternative is free, which argues against a trivial one.
4. **Calibration against the literature.** Typical point-cloud KD gains are
   ~1–3 macro-F1 points. A 0.02 threshold means we would call a
   literature-typical gain at the low end "not worth adopting". That is
   deliberate: those gains are measured at benchmark scale, which §1 argues is
   the unrepresentative regime.

**Scope:** `min_effect` governs the **adoption** decision only. The scaling and
ranking hypotheses test a slope and a rank correlation respectively and do not
use it, so this threshold cannot blunt the study's main claims.

**Field data uses non-inferiority, not superiority.** The field set cannot
support a superiority test at this effect size (point 2). An arm selected on the
proxy is confirmed on field *validation* if it is no worse than arm 0 by more
than 0.02. Failing that, the pre-registered fallback applies: **ship arm 0**.

## 5. Hypotheses

Each states a direction, a test, and what would falsify it. H1 and H5 follow
from the §1 mechanism; if both fail, the thesis is wrong.

### H1 — KD gain is non-monotone in data scale under condition S, peaking at intermediate scale

Two forces oppose each other: at tiny scale the teacher is itself too weak to
teach, and at large scale the student can learn directly. The gain should
therefore be largest in between.

- **Test:** the scale maximising arm 3's gain over arm 0 is an interior point of
  the ladder, and the gain there exceeds the gain at both endpoints by more than
  `min_effect`.
- **Falsified by:** monotone decay (Spearman ρ significantly negative), monotone
  increase, or a flat profile (|ρ| small and no interior peak).
- **Note:** under condition **F** the teacher's quality is fixed, so H1 predicts
  **monotone decay** there instead (`trend_spearman` ρ < 0, p < 0.05). The
  contrast between S and F is the sharpest test of the mechanism.

### H2 — Paradigm ranking is unstable at small scale

- **Test:** Kendall τ between the arm ranking at the smallest scale and at the
  largest. Predicted **τ < 0.5**.
- **Falsified by:** τ ≥ 0.5 with the top-ranked arm preserved.

### H3 — Self and mutual arms beat offline KD at the smallest scales, in condition S

The direct consequence of §1: at small scale the offline teacher is starved,
while arms 1, 2 and 5 do not depend on one.

- **Test:** at the two smallest scales, the best of arms {1, 2, 5} beats arm 3
  by more than `min_effect`, Holm-adjusted p < 0.05.
- **Falsified by:** arm 3 matching or beating all of {1, 2, 5} at small scale.

### H4 — The wide-PointNet teacher distils better than the cross-architecture teacher, despite lower standalone accuracy

DGCNN's advantage lives partly in a dynamically rebuilt kNN graph that a student
with no neighbourhood operator cannot represent.

- **Test:** paired comparison of student macro-F1 under arm 3 versus arm 4 at
  full scale. Predicted arm 3 ≥ arm 4 even where teacher(4) > teacher(3).
- **Falsified by:** arm 4's student significantly better.

### H5 — On-device latency does not scale with model width at small `M`

Measured on x86 CPU, the 249k → 37k reduction gave ~4.6× (PyTorch) and ~5×
(ONNX Runtime) at M=1. On the Orin Nano's GPU a 249k-parameter model is expected
to be kernel-launch and memory-transfer bound rather than FLOP bound.

- **Test:** measured median latency ratio (deploy ÷ student) at M=1 on the
  Jetson. Predicted **< 2×**, rising toward the FLOP ratio as `M` grows.
- **Falsified by:** a ratio at or above the ~5× seen on CPU.
- **Consequence if supported:** structured pruning has no latency purpose on
  this device, which is a finding in its own right.

## 6. Decision rules

Implemented in `tournament.adjudicate`; nothing here is decided by judgement
after the fact.

- Comparisons are **paired by seed**. A seed-set mismatch is an error, not a
  silent drop.
- Significance by **exact paired permutation** on the mean difference. Every
  p-value is reported with the floor its seed count imposes, so "not
  significant" and "underpowered by construction" stay distinguishable.
- **Holm–Bonferroni** across all comparisons in a family. Family-wise control
  rather than FDR, because a false adoption here means shipping a technique that
  does not work.
- `ADOPT` requires adjusted p < 0.05 **and** mean gain > `min_effect`. The
  default verdict is `INCONCLUSIVE`.

## 7. Stopping rule

Exactly 8 seeds per (arm, condition, scale, dataset) cell. **No inspecting
results and adding seeds**; optional stopping invalidates the p-values. If a
cell needs more seeds for a reason unrelated to its outcome (a crashed run), the
replacement uses the next unused seed index and the substitution is recorded.

## 8. Exclusion rule

A run may be discarded only for a mechanical failure: a crash, an out-of-memory
kill, a NaN loss, or a corrupted checkpoint. Every exclusion is logged with its
reason and the seed is re-run. **A run is never excluded for having produced an
unexpected number.**

## 9. Split discipline

- Technique selection: proxy dataset, validation split.
- Confirmation: field dataset, validation split.
- The field **test** split is opened **once**, after all selection is complete,
  for the final reported number only.
- Hyperparameters (T, β, γ, learning rate) are tuned on the proxy validation
  split, and the chosen values are then frozen and reused on the field data
  without re-tuning.
- **The baseline is tuned with the same budget as the arms**, and that budget is
  reported. An undertuned baseline is the most common and most fatal objection
  to a distillation study.

## 10. What will be reported regardless of outcome

- Every arm, every condition, every scale, every seed count.
- All four possible verdicts, including `UNDERPOWERED`.
- The capacity-gate result, even if it shows no gap and makes the rest moot.
- H5's latency result, even if it shows pruning to be pointless here.
- A statement of which hypotheses were supported, falsified, or left undecided.

Confirmatory analysis is exactly what this document specifies. Anything else we
run later is labelled **exploratory** and reported separately.

### Amendments

*(none yet)*
