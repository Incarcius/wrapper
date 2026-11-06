# Publication Scope

**Recorded:** 6 Nov 2026, after the owner raised writing this up for a
conference if the results are publishable.

This document is deliberately sceptical. The purpose is to decide **now** what
would and would not constitute a publishable contribution, because that decision
changes the experiment design — and retrofitting rigor onto a finished study is
not possible.

---

## 1. What is *not* publishable on its own

**"We compared offline, online and self-distillation on point clouds."** A
paradigm comparison is a well-trodden format; the paradigms themselves are
established, and a ranking on a standard benchmark with a standard architecture
is incremental. Reviewers would reasonably ask what was learned that a reader
could not have predicted.

So the comparison is the *method*, not the contribution. The contribution has to
be the thing the comparison reveals.

## 2. What is genuinely novel here

Four candidates, in descending order of how defensible I think they are:

**(a) Does the paradigm *ranking* transfer between datasets?**
Nearly every distillation study establishes a ranking on one dataset and
implicitly assumes it generalises. This project has a forced two-dataset
structure — technique selected on a large proxy with real statistical power
(SemanticKITTI instance crops), then confirmed on a small real target
(Mid-70 field data). Measuring whether the ranking *survives* that transfer is a
question almost nobody asks, and a negative answer would be a genuinely useful
caution to the field.

**(b) Accuracy at fixed, on-device latency — not FLOPs.**
Every arm in this study produces the **identical architecture**; only the weights
differ. So the comparison is cleanly "accuracy at a fixed latency budget",
with the latency *measured on the deployment hardware* rather than estimated from
parameter counts. Most KD papers report accuracy alone and never run on an edge
device. We already have ONNX export verified to 2.98e-08 parity, which makes the
on-device measurement credible rather than approximate.

**(c) The real-time constraint scales with scene complexity, not frame rate.**
Per-frame cost is `M` regions × one forward pass, where `M` is the number of
objects in view. A model comfortable at `M=1` can miss the deadline at `M=10`.
This is specific to cluster-then-classify perception pipelines and is, as far as
I can tell, not a constraint the distillation literature engages with at all.

**(d) A properly powered negative result.**
"Distillation does not help in the small-real-data edge regime; a small model
trained from scratch matches it" is publishable **if** the study has the
statistical power to support the claim and pre-registered the decision rule.
Negative results of this kind are rarer and more useful than another 0.4-point
improvement. This is a real possible outcome — see `distillation_plan.md` §9 —
and the design should treat it as a finding rather than a failure.

### Proposed framing

> *Do knowledge-distillation paradigm rankings transfer? An on-device study of
> LiDAR instance classification under a real-time budget.*

(a) is the question, (b) and (c) are what make the setting non-generic, and (d)
is an acceptable answer.

## 2b. Raising the bar from "publishable" to "worth reading"

**Added 6 Nov 2026**, and it supersedes the single-comparison design implied
above.

### The unlock: make data scale the independent variable

The design above is hostage to the field dataset, and a pairwise arm comparison
at 15–20 instances per class is statistically hopeless (§4). The fix is to stop
treating the small sample as a limitation and make it an **axis**: run the arms
across a ladder of training-set sizes on public data — on the order of 20, 50,
100, 500 and 2000 instances per class — and study KD gain *as a function of
scale*.

This changes the statistics qualitatively. One pairwise difference at N=20 is
noise. A **monotone trend over six scales × five seeds** tests a slope rather
than a difference, and retains power even when every individual point is noisy.
It is the single change that makes a credible study affordable with the data and
compute actually available.

### The three findings, as a risk portfolio

| Finding | Claim | Risk | Reward | Blocked on |
|---|---|---|---|---|
| **A** scaling curve | KD's advantage decays with training-set size and vanishes below ~N instances per class — a negative result with a number in it | Medium (the curve may be flat) | High | public data only |
| **B** ranking instability | The *ranking* of paradigms is unstable in the low-data regime, so selecting a KD method from published benchmarks is unsafe at edge data scale | Gamble (rankings may be stable) | Highest | public data + field data |
| **C** on-device Pareto | At matched latency on the Orin Nano, which lever actually pays: width, KD, pruning or quantisation | Low — it is a measurement | Moderate | **the Jetson** |

**C guarantees a paper exists; A and B decide whether it is a good one.** So
proceeding is not a bet on a single outcome, and even "rankings *are* stable" is
a useful calibration result rather than a dead end.

### What the ladder buys that nothing else does

It **disentangles domain shift from data scale**. Same dataset at several sizes
isolates scale; proxy-versus-field isolates domain. Being able to state that
rankings are robust to domain shift but break below some N — or the converse —
is what turns an observation into a result. No single-dataset study can separate
those two, and most do not try.

### Compute, revised

6 scales × 5 arms × 5 seeds is about 150 proxy runs. At 250k parameters and a
few thousand samples a student run is 1–2 minutes on a GPU, ~5 with the 2 M
teacher live, so the ladder is roughly **8–12 GPU-hours** — one day on one
rented GPU, not a cluster. The models being small is what makes the study cheap,
and it is the same property that makes the deployment target interesting.

### Venue fork, which changes what to prioritise

- **Robotics (ICRA / IROS / RA-L):** lead with the deployed system, the real
  Mid-70 and the on-device measurement. **Finding C is the spine**, A and B
  support it. These venues value real-hardware validation, and the Mid-70's
  non-repetitive scan is a differentiator rather than an incidental detail.
- **ML (NeurIPS / ICML / CVPR efficiency workshops):** lead with the scaling law
  and the transfer question. **Finding A is the spine**, the Jetson is
  validation. Higher bar, needs a clean curve, indifferent to the sensor.

Current recommendation: **robotics**, because it plays to hardware that already
exists here and treats the sensor as an asset.

### What keeps it from being good, even if accepted

- A single-dataset, single-scale comparison with a declared winner.
- A headline of "our pipeline works".
- Reporting only the arm that won.
- Any accuracy claim resting on the geometric simulation.

## 3. What this changes about the experiment design

Rigor requirements that supersede the looser numbers in `distillation_plan.md`:

| Item | Engineering plan | Publication requirement |
|---|---|---|
| Seeds per arm | ≥3 | **≥8** — see the correction below. Report mean ± std, never best-of |
| Statistics | "beats baseline by more than seed spread" | Paired comparisons against arm 0, with the multiple-comparison problem stated honestly (7 arms × 2 datasets is ~14 comparisons) |
| Baseline tuning | "same protocol" | The from-scratch baseline must be **tuned as hard as the distilled arms**. An undertuned baseline is the most common reason a KD paper is rejected, and the easiest for a reviewer to suspect |
| Ablations | optional | Required: temperature, `beta`, feature-vs-logit, confidence weighting. Each as a separate table |
| Power analysis | nice to have | **Required** — the subsampling experiment in §4 below. A study on 15–20 instances per class with no power analysis is not credible |
| On-device numbers | "pending" | Mandatory: warm-up protocol, median/p90/p95/p99, power mode, thermal state, sustained throughput over a documented run |
| Provenance | good practice | Already satisfied: config/dataset/label-map hashes, seeds, environment manifest, code revision |

## 4. The experiment that must come first

Before the main matrix: take the proxy dataset, **subsample it to the target
regime** (15–20 instances per class), and measure the seed-to-seed spread of
arm 0 versus arm 1 at five seeds.

This answers whether the study can resolve anything at all at the target data
scale. If the spread is ±5 macro-F1, no ranking on the real data will be
meaningful, and the paper's claim has to be restricted to the proxy with the
target used only as a qualitative check. Discovering that *after* running the
full matrix would waste the entire study.

It is also cheap, needs no field data, and is unblocked today apart from
obtaining the proxy dataset.

## 5. What would make this unpublishable

Stated plainly so none of it happens by drift:

- **Synthetic data only.** The geometric field simulation validates formats and
  contracts. It is not a sensor model, and no accuracy claim may rest on it.
- **CPU-only latency.** The central framing in §2(b) is on-device. Without a
  Jetson measurement the paper has no contribution, only a hypothesis. **This
  makes the Jetson the critical path, not an optimisation.**
- **One seed.** Non-negotiable.
- **Selecting on the test split.** If the technique is chosen by looking at
  field test scores, that split is burned and every subsequent number is
  optimistic. Selection happens on the proxy and on validation only; the field
  test split is touched exactly once.
- **Quietly dropping arms that did badly.** Every arm that is run gets reported,
  including the ones that lost.
- **Claiming CAD-domain results as real-world.** Any ModelNet40 number is
  labelled as implementation validation, never as task performance.

## 6. Compute implication

5 seeds × 7 arms × 2 datasets ≈ 70 runs, plus ablations and the power analysis —
realistically 100+ training runs. Student-scale arms are minutes each on CPU, but
anything with the 2 M-parameter teacher in the loop is hours.

**CPU-only is not sufficient for a publishable study.** This is the point at
which a GPU stops being a convenience. The mitigation already identified still
helps: train each teacher **once**, freeze it, and reuse the frozen teacher
across all offline arms and seeds — which is offline distillation's structural
advantage under this constraint.

## 7. Artifact release

The repository is already close to releasable: configs are files rather than
flags, every result carries dataset and label-map hashes, the export path is
verified, and the test suite covers the data pipeline. What is missing for a
release is the training code (in progress), the results aggregation, and a
`REPRODUCING.md` that maps each table in the paper to the exact command that
produced it.

Worth keeping that mapping as the tables are generated, rather than
reconstructing it at submission time.

## 8. Honest assessment of the odds

The contribution in §2(a) is real but depends on an empirical outcome we do not
control: if the ranking transfers perfectly and distillation gives a clean gain
on both datasets, the result is a competent applied study and probably a
workshop paper. If the ranking *fails* to transfer, or if §2(d) holds and
nothing beats the baseline, it is a more interesting paper and a harder one to
write well.

Either way the engineering is the same, which is the useful part: **nothing in
this document asks for work that the deployment does not already need.** The
extra seeds and the power analysis are the only genuine additions, and both are
cheap compared with the field data collection.
