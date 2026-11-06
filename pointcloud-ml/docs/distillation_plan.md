# Distillation Plan (Phase Five / Stage 6)

**Recorded:** 15 Oct 2026. **Status: planned, not implemented.** Gated on data —
there is no trained teacher and no dataset, so nothing here has been run.

Two questions, answered separately:

- **Which paradigm** — offline, online, or self-distillation? §2. All three are
  run and compared; §7 is the protocol that makes the comparison fair.
- **Which mechanism** — what the loss actually is. §3–§4.

---

## 1. Why this task is unusually favourable for distillation

The standard warning is: *do not compare teacher and student point-wise unless
the corresponding points are aligned.* For us that is satisfied for free:

- Teacher and student consume the **same region**, resampled by the **same
  seeded sampler** to the **same 1024 points** (`pcml.data.sampling` is
  deterministic by design — audit finding A1).
- Both emit one `(M, 6)` logit tensor for the same `M` regions in the same order.

So region *i*'s teacher and student logits describe exactly the same points. No
spatial alignment, no interpolation, no masking of invalid points. This holds for
every paradigm below.

## 2. Paradigm: offline, online, or self-distillation

### 2.1 What each one means here

**Offline.** Teacher is trained to convergence first, then frozen; the student
learns from fixed teacher outputs. Two stages.

**Online.** Teacher and student are trained *simultaneously* and both are
updated. Two sub-variants are relevant:
- *Mutual learning (DML)* — two peers train together, each learning from the
  other's softened predictions plus its own hard-label loss. No pre-trained
  teacher exists at any point.
- *Self-ensemble / multi-head (ONE)* — one shared trunk with several heads; the
  ensemble of head logits acts as the teacher.

**Self-distillation.** Teacher and student are the same network, or the same
architecture:
- *Born-Again (BAN)* — train the compact model, then train an architecturally
  identical model using the first as teacher. Sequential, same size throughout.
- *EMA / mean teacher* — the teacher is an exponential moving average of the
  student's own weights, updated every step. Combined with our augmentation, it
  becomes a consistency objective between two augmented views of one crop.
- *Deep-to-shallow (Be-Your-Own-Teacher)* — deeper layers supervise auxiliary
  classifiers on shallower ones.

### 2.2 Fit against this project, with the measured numbers

Parameter distribution, measured today (`K=6`):

| Config | Total | Backbone MLP | T-Nets | Classifier head |
|---|---:|---:|---:|---:|
| `pointnet_deploy` | 249,542 | 83,776 (33.6%) | 0 | **165,766 (66.4%)** |
| `pointnet_student` | 37,414 | 11,872 (31.7%) | 0 | **25,542 (68.3%)** |
| `pointnet_teacher` | 1,983,759 | 310,464 (15.7%) | **1,014,089 (51.1%)** | 659,206 (33.2%) |

Largest single layer in the deploy model is `head.0` (512→256) at 52.5% of all
parameters. In the teacher, `feature_tnet.fc.6` (256→4096) is 26.4%.

These two facts decide several paradigm questions:

| Paradigm | Verdict | Reasoning |
|---|---|---|
| **Offline** | ✅ **run it** | The only paradigm where the expensive teacher is trained **once** and reused across every arm and every seed. Under our constraint — no local GPU — that amortisation is decisive: the teacher can be trained once on a rented GPU, frozen, and then every offline run is cheap. Also the only way to use a DGCNN or PointNet++ teacher at all |
| **Online — mutual learning** | ✅ **run it** | Natural fit: we *want* two models anyway (the 249k deploy model and the 37k student), and mutual learning improves both in one run. Peers of similar size avoid the capacity-gap problem that hurts a 2M→37k offline transfer. Expected to be strong precisely because our binding constraint is small data, and peer disagreement acts as a data-dependent regulariser |
| **Online — multi-head (ONE)** | ❌ **skip** | Would be cheap if the trunk dominated. It does not: the head is 66% of parameters, so each extra head costs ~165k — more than half a whole extra model. The shared-trunk economy that makes ONE attractive does not exist in this architecture |
| **Self — EMA / mean teacher** | ✅ **run it** | Nearly free: no second backward pass, just an EMA weight copy and one extra forward. Needs **no teacher at all**, so it is not blocked on the Stage 4 comparison. Pairs naturally with the 180° yaw augmentation already configured — consistency between two yaw-augmented views of one crop is a strong signal on small data |
| **Self — Born-Again** | ✅ **run it** | Cheap, sequential, same architecture. Directly tests the §9 question: *is a big teacher needed at all, or does any teacher-shaped regulariser do the work?* |
| **Self — deep-to-shallow** | ❌ **skip** | The backbone is five conv layers. There is no meaningful "deep" to supervise a "shallow" from; intermediate exits at layer 2 of 5 would classify from 64-d point features before any pooling. Architecturally inapplicable, and saying so is cheaper than discovering it |

### 2.3 One subtlety about the teacher's T-Nets

Half the teacher's parameters (51.1%) are T-Nets, which regress an alignment
*transform* rather than build a representation — and the student has no T-Net at
all. So:

- Feature distillation from the teacher's **embedding** (§4) is unaffected: the
  embedding is computed *after* the transforms are applied, so their effect is
  baked into the target the student matches.
- But the T-Nets' actual knowledge — *how to canonicalise an arbitrarily-oriented
  cluster* — cannot be transferred to a student that has no mechanism to apply a
  transform. The student must instead learn orientation robustness from
  augmentation.

That interacts with the open decision in `task_definition.md` §9.5 (yaw
augmentation vs PCA canonicalisation): if canonicalisation wins there, the
teacher's T-Nets become largely redundant and a teacher without them is both
cheaper and a better-matched source. Worth testing a no-T-Net teacher variant.

## 3. The loss mechanics (shared by all paradigms)

```
L = alpha * CE(student_logits, y)
  + beta  * T^2 * KL( softmax(teacher_logits / T) || softmax(student_logits / T) )
  + gamma * L_feature                                   # section 4, optional
```

Details that are easy to get wrong and do matter:

- **The `T^2` factor is not cosmetic.** Dividing logits by `T` shrinks softmax
  gradients by roughly `1/T^2`; without the correction, `beta` changes meaning
  every time `T` is tuned and the two become entangled.
- **Compute the divergence in log space:**
  `F.kl_div(F.log_softmax(student/T, 1), F.log_softmax(teacher/T, 1),
  log_target=True, reduction="batchmean")`. `log_target=True` avoids an
  exp-then-log round trip on the teacher side, which is where KD losses lose
  precision and occasionally produce NaNs at low `T`. Use `"batchmean"`, not
  `"mean"` — the latter also divides by `num_classes`, silently scaling the loss
  by 1/6 here.
- **Teacher in `eval()` under `torch.inference_mode()`.** Augmentation is
  stochastic per sample, so teacher logits cannot be precomputed and cached; the
  teacher runs live. Budget for it: the teacher costs ~4x the deploy model per
  forward pass (measured 11.13 vs 2.51 ms per region).
- Start at `alpha + beta = 1`, sweeping `beta` over {0.3, 0.5, 0.7, 0.9} and `T`
  over {2, 3, 4, 6}.

For **mutual learning**, the same KL term appears symmetrically: each peer's loss
takes the other's detached logits as target. For **EMA**, the target comes from
the averaged weights and no gradient flows into the teacher at all.

## 4. Feature distillation: match the embedding, **not** the per-point features

`PointNetClassifier.encode()` was deliberately exposed for this: it returns the
post-max-pool global embedding, `(B, embed_dim)`. That is the tensor to match.

**Why the embedding.** Everything after the max-pool is a plain MLP head, so the
pooled vector *is* the model's whole representation of the region — a global
shape signature where each dimension is set by one critical point on the surface.
Matching it transfers the learned shape descriptors fairly directly.

**Why not per-point features.** Point correspondence *is* available (§1), so
matching the pre-pool `(B, C, N)` activations is tempting. But max-pooling keeps
only the per-channel maximum and **discards the other N−1 values**. Forcing the
student to reproduce pre-pool activations spends its capacity replicating
representations the teacher itself throws away, competing with the objective.

**Dimension mismatch.** Teacher 1024-d, deploy student 512-d, aggressive student
128-d. Use a single `nn.Linear(student_embed, teacher_embed)` on the **student**
side, trained jointly and **discarded after training** — zero inference cost.
Project student→teacher, not the reverse: projecting the teacher down would
discard the very information being transferred. Loss on L2-normalised embeddings
(MSE or cosine), because the teacher's embedding scale is arbitrary and
unnormalised MSE would mostly penalise magnitude.

## 5. Teacher selection is itself an experiment

Once data exists and DGCNN / PointNet++ SSG are implemented for Stage 4:

| Teacher | Hypothesis |
|---|---|
| PointNet-wide | Best representational compatibility — same inductive bias, same embedding semantics |
| PointNet-wide, no T-Nets | Cheaper, and better matched to a T-Net-less student (§2.3) |
| DGCNN | Probably the most accurate teacher, but its knowledge lives partly in a kNN graph the student cannot represent |
| PointNet++ SSG | Hierarchical features; middle ground |

**Falsifiable prediction:** the wide PointNet will distil best *despite* probably
being the least accurate teacher, because cross-architecture distillation
transfers less of what makes the stronger teacher strong. If the measurement
contradicts this, the prediction was wrong and the better teacher wins — the
point of recording it is that the experiment is informative either way.

Selection is always on the **student's** macro-F1, never the teacher's.

## 6. Protecting the rare classes

`vehicle` is expected to be the rarest class (possibly 3–4 distinct instances
available). Vanilla KD has a specific failure mode here: the KD term is averaged
over the batch, so it is dominated by the abundant classes, and on rare classes
it propagates the teacher's own *uncertainty* as if it were knowledge.

1. **Confidence-weighted distillation** — scale each sample's KD term by the
   teacher's max softmax probability (or `1 - normalised entropy`), so the
   student is not asked to imitate the teacher's confusion.
2. **Class-weighted cross-entropy** on the hard-label term, weights from the
   training-split distribution (`label_map.yaml: class_weights`).
3. **Report per-class F1 deltas.** KD can raise the aggregate while quietly
   destroying a rare class, so a rare-class regression blocks adoption even when
   macro-F1 improves.

## 7. The comparison protocol

### 7.1 The arms

| # | Arm | Paradigm | Needs a teacher? |
|---|---|---|---|
| 0 | Student, from scratch | none — **the baseline that matters** | no |
| 1 | Student, EMA mean-teacher + augmentation consistency | self | no |
| 2 | Student, Born-Again (identical-architecture teacher) | self | no (its own predecessor) |
| 3 | Student, offline KD from PointNet-wide | offline | yes |
| 4 | Student, offline KD from best Stage-4 model | offline, cross-architecture | yes |
| 5 | Student ↔ deploy model, mutual learning | online | no (peer) |
| 6 | Best of 1–5, plus feature distillation | — | depends |

Arms 0–2 and 5 need **no pre-trained teacher**, so they can run as soon as data
exists. Arms 3–4 wait on the Stage 4 comparison. That dependency ordering is
deliberate: it front-loads the arms that are both cheapest and most likely to
win on a small dataset.

### 7.2 What is held constant

Identical across all arms: dataset version, split manifests, label map,
augmentation config, optimiser, LR schedule, batch size, and **number of
student gradient steps**.

Equal *student steps*, not equal wall-clock. Online and self-distillation arms
give the student extra signal per step; that extra signal is the thing under
test, so the step budget must not also vary. Wall-clock and total FLOPs are
reported separately as a cost column, including the teacher's pre-training cost
for offline arms — which is the hidden two-stage expense that online methods do
not have, and must not be concealed.

**≥3 seeds per arm.** A 1-point macro-F1 difference on a few thousand crops is
indistinguishable from seed noise; a single-run improvement is not evidence. The
decision statistic is the seed mean with its spread.

### 7.3 The operational columns are constant — and that simplifies this phase

Every arm 0–4 and 6 produces the **identical deployed architecture**. Only the
weights differ. So parameter count, checkpoint size, latency, peak memory and
export behaviour are *the same by construction* across arms, and this phase's
comparison is purely about accuracy and calibration per unit of training cost.

The exception is arm 5, which yields two usable models (37k and 249k) from one
run — worth noting as a side benefit, and it means arm 5 reports two rows.

### 7.4 Acceptance

A paradigm is adopted only if its student beats **arm 0** by more than the
seed-to-seed spread, without regressing any class. Beating the *teacher* is not
the test; beating an equally-trained student is. If nothing clears that bar,
distillation is dropped as complexity that bought nothing — which has to be a
real possible outcome, not a formality.

## 8. Compute feasibility

Rough estimate from the measured per-region forward costs (0.38 ms/region for
the student, 2.51 for deploy, 11.13 for the teacher, all x86 CPU), assuming
~5,000 crops per epoch, 200 epochs, and backward ≈ 2× forward:

| Arm scale | Per run (est.) | × 3 seeds |
|---|---|---|
| Student-only (arms 0, 1, 2) | ~20 min | ~1 h each |
| Mutual learning (arm 5, both models) | ~2.5 h | ~7 h |
| Offline with live 2M teacher (arms 3, 4) | ~8 h | ~24 h |

**Estimates, not measurements** — they ignore data loading, augmentation and
clustering cost, and they assume a dataset size that does not exist yet. Two
conclusions survive the uncertainty, though:

1. The student-scale arms are affordable on **CPU alone**, because the models
   are genuinely tiny. The whole no-teacher half of the matrix is an afternoon.
2. Anything with the 2M teacher in the training loop is where CPU-only hurts.
   That is the practical argument for **training the teacher once on a rented
   GPU and freezing it**, then reusing that frozen teacher across arms 3, 4, 6
   and all their seeds — which is precisely offline distillation's structural
   advantage here, and a reason not to dismiss it in favour of online methods on
   elegance grounds.

## 9. The honest risk: this phase may be unnecessary

Distillation pays off when the student is **capacity-limited relative to the
task**. Signs that it may not be, here:

- `K = 6`, and the six classes were chosen to be *geometrically* distinct —
  thin-vertical vs large-planar vs volumetric-irregular. Far easier than
  ModelNet40's 40 fine-grained categories.
- A realistic first dataset is a few thousand crops from 15–20 distinct
  instances per class. On that, a 249k-parameter model is likely limited by
  **data**, not capacity.
- The measured capacity span is 37,414 to 1,983,759 parameters — 53×. If the 37k
  model matches the 2M model when both are trained from scratch, there is no gap
  for any paradigm to close.

**So the first experiment in Stage 6 is not distillation at all.** It is training
the student and the teacher from scratch on identical data and measuring whether
a capacity gap exists. If it does not, the paradigm comparison is skipped
entirely and the effort goes to data, which will be the real constraint.

Prior, recorded for falsification: on a small dataset with geometrically
distinct classes, the **self-distillation and mutual-learning arms will beat
offline KD**, because the binding constraint is regularisation rather than
teacher capacity — and arm 0 may well beat all of them.

## 10. What is already in place

- `PointNetClassifier.encode()` — the feature-distillation hook, exposed and tested.
- `configs/models/pointnet_{teacher,deploy,student}.yaml` — the capacity range,
  so "teacher", "peer" and "student" are config, not code.
- Deterministic seeded sampling — makes teacher and student see bit-identical
  inputs, which §1 depends on.
- `model_factory.build_model` — one config file constructs any arm's models.

Not yet written: `src/pcml/distillation/` (losses, projection head, EMA wrapper,
mutual-learning loop). Deliberately deferred until §9's capacity question has an
answer, since that answer determines whether any of it is needed.

### Noted in passing, for Phase Six (pruning)

The measurement in §2.2 is also the strongest available pruning signal: **two
thirds of the deploy model sits in the classifier head**, and `head.0` (512→256)
alone is 52.5% of all parameters. Narrowing the global embedding from 512 to 256
would shrink both `head.0` *and* the final conv simultaneously — by far the
highest-yield structured change, and a much better first cut than pruning the
backbone convolutions. For the teacher, the equivalent observation is that its
T-Nets are 51.1% of its parameters.
