# Task Definition

**Status:** v1, 15 Oct 2026. Decided with the project owner.
**Supersedes:** nothing. **Blocks:** dataset design, model head width, loss choice, evaluation code.

Every statement below is tagged:

- **[DECIDED]** — settled; code may depend on it.
- **[PROVISIONAL]** — a working assumption chosen so work can continue. Labelled at every use site. Revisit before final training.
- **[PENDING]** — unknown. Named explicitly, with what is needed to resolve it.

---

## 1. The prediction target

**[DECIDED]** Task formulation **C: local-region classification**, specifically *per-instance
classification of spatial clusters*.

One `(class_id, confidence)` pair is predicted for each **object cluster** extracted from a
cleaned Livox Mid-70 frame. Clusters come from DBSCAN over the de-noised scene, not from the
network.

This is explicitly **not**:

| Rejected formulation | Why not |
|---|---|
| A — whole-cloud classification | The ROI is a ±10 m *room*. One label for a room, drawn from an object vocabulary, is meaningless. This is finding **C2** in the wrapper's code review. |
| B — point-wise semantic segmentation | Cleaner for scene understanding and handles touching objects, but needs scene-level per-point labels: the most expensive annotation there is. Not justified at proof-of-work stage. Remains the long-term destination if the goal becomes "understand a city block". |
| D — 3D object detection with boxes | Clustering already supplies instance separation and a centroid. Learned box regression adds annotation cost (oriented 3D boxes) and architecture cost for no benefit the downstream consumer has asked for. |
| E — hybrid segmentation + clustering | Strictly more expensive than C and only pays off once per-point labels exist. |

### Pipeline shape this implies

```
Mid-70 frame (~10k pts, 10 Hz)
  → ROI crop                       existing: pipeline.roi_filter
  → outlier removal                existing: pipeline.remove_outliers
  → [NEW] ground-plane removal     optional, ablation-gated
  → [NEW] DBSCAN instance clustering    → M clusters
  → per cluster: normalise + FPS to 1024 pts   existing stages, re-scoped per cluster
  → [NEW] batch M clusters into one (M, 3, 1024) tensor
  → [NEW] classifier               one forward pass per frame, not M passes
  → M × (class_id, confidence) + geometry
  → [NEW] publish to ROS 2
```

The re-scoping of normalise + FPS from *scene* to *cluster* is the fix for **C2**. It also makes
the existing **C1** fix (voxel/FPS before normalisation) mandatory rather than cosmetic, because
cluster-level normalisation is what the model is trained on.

## 2. Granularity and cardinality

**[DECIDED]**

- One prediction per cluster. No point-level output.
- Cluster count `M` is data-dependent and unbounded in principle.
- **[PROVISIONAL]** `M` is capped at `max_clusters_per_frame: 16`, selected by descending point
  count. Rationale: bounds worst-case latency and memory, which an unbounded `M` does not. The cap
  must be re-derived from measured cluster-count distributions on real recordings — see
  [§8 unknowns](#8-pending-items).
- **[PROVISIONAL]** Clusters smaller than `min_cluster_points: 50` are dropped before
  classification, not classified as a reject class. 1024-point FPS from a 20-point cluster is 98%
  duplicates and carries almost no shape information.

## 3. Class labels

**[DECIDED — confirmed by the owner, 15 Oct 2026]** `K = 6`, for a college cricket/football field:

| id | name | id | name |
|---:|---|---:|---|
| 0 | `person` | 3 | `structure` |
| 1 | `vegetation` | 4 | `net_fence` |
| 2 | `pole` | 5 | `vehicle` |

plus `-1` / `unknown` as the reject **outcome**, not a trained class. Definitions, annotation
policies and the reasoning that selected these six — every one is physically resolvable by a
Mid-70 at a safe altitude — are in [`../data/label_map.yaml`](../data/label_map.yaml) and
[`data_collection_plan.md`](data_collection_plan.md) §1–2. Excluded as unresolvable: cricket ball,
stumps, bags, line markings.

The structural rules, decided before the vocabulary existed and unchanged by it:

- **[DECIDED]** `K` is read from a single versioned `label_map.yaml`. It is never hard-coded in a
  model definition, a training script or the deployment adapter. Changing `K` must be a config
  change plus a retrain, never a code edit.
- **[DECIDED]** The label map is the artifact of record: `class_id → class_name` plus, for each
  class, its semantic definition and annotation policy. A checkpoint without its label map is
  considered unusable.
- **[DECIDED]** The set includes a **reject outcome**. Open-set behaviour is not optional here: a
  drone over a real scene will meet objects outside any training vocabulary, and a closed
  softmax will label them confidently and wrongly. Implemented as a confidence threshold plus an
  explicit `unknown` outcome in the output contract — see §5.
- **[DECIDED]** `unknown` is **not** a trained class in v1 (`reject.trained_as_class: false`). A
  background class built from an arbitrary grab-bag of negatives tends to absorb the rare real
  classes, and `vehicle` is expected to be the rarest. Negatives are still collected and labelled
  `unknown`; they calibrate the threshold rather than train a class. Open: if residual ground
  fragments dominate the false positives, promoting `ground_residue` to a trained 7th class is the
  first remedy to try — revisit once a confusion matrix exists.

## 4. Input contract

**[DECIDED]** per cluster, matching what the wrapper already produces:

| Property | Value | Source |
|---|---|---|
| Points per cluster | 1024 | `downsampling.target_points`, ModelNet convention |
| Coordinates | xyz only | wrapper drops columns ≥3 at `pipeline.py:214` |
| Frame | sensor frame, metres | Livox driver output, no transform applied |
| Normalisation | per-cluster centroid subtraction + unit-sphere scale | `pipeline.normalize`, re-scoped |
| Tensor layout | `(B, 3, N)` float32, contiguous, batched over clusters | config-selectable, finding **I5** |

**[DECIDED]** Intensity/reflectivity is **not** used. The wrapper already discards it, and the
training data would have to carry calibrated intensity for it to transfer. Revisiting this means
changing the ingestion callback as well, so it is a deliberate deferral, not an oversight.

### The scale problem

**[PROVISIONAL — flagged for ablation]** Unit-sphere normalisation destroys absolute scale. For
CAD-object benchmarks that is correct; for LiDAR objects it discards one of the most
discriminative features available — a shrub and a building reduce to the same unit sphere. The
planned mitigation is a small auxiliary feature vector concatenated at the classifier head, not
into the point stream:

`[extent_x, extent_y, extent_z, log(n_points), range_to_centroid, height_above_ground]`

All six are byproducts of clustering, cost nothing to compute, and are available at inference
time. This is an **ablation to run, not a decision taken** — it must earn its place against the
xyz-only baseline. It is recorded here because it changes the model's input signature, so the
head must be written to accept an optional feature vector from day one.

## 5. Output contract

**[DECIDED]** Per frame, a list of records. This is the contract the wrapper and any downstream
consumer code against:

```python
@dataclass(frozen=True)
class ClusterPrediction:
    cluster_id: int          # index within this frame only; not a track id
    class_id: int            # index into label_map.yaml; -1 == unknown/rejected
    class_name: str          # resolved from the label map, never from a literal
    confidence: float        # calibrated softmax max, [0, 1]
    n_points_raw: int        # points in the cluster BEFORE FPS — the honest density signal
    centroid_m: tuple[float, float, float]   # sensor frame, metres
    extent_m: tuple[float, float, float]     # axis-aligned, sensor frame, metres

@dataclass(frozen=True)
class FramePrediction:
    stamp: float                            # sensor header stamp, seconds
    predictions: list[ClusterPrediction]
    n_clusters_found: int                   # before the max_clusters cap
    n_clusters_classified: int              # after cap and min-size filter
    latencies_ms: dict[str, float]          # per-stage, extends StageTimings
```

`n_clusters_found` vs `n_clusters_classified` is deliberate: silent dropping of objects by the cap
or the size filter is exactly the failure that must be visible in logs rather than inferred.

**[PROVISIONAL]** ROS 2 publication (finding **I8**) uses
`vision_msgs/Detection3DArray`, which carries class, score and a 3D bounding box per instance.
`Classification2D` cannot express per-instance geometry, so it is the wrong message for this task.

## 6. Temporal behaviour

**[DECIDED]**

- Frames arrive at ~10 Hz; the system produces one `FramePrediction` per processed frame.
- Each frame is classified independently. No recurrent state in the model.
- Drop-oldest queueing is retained from the wrapper: a stale cloud is worse than a missing one on
  a moving platform.

**[PROVISIONAL]** Temporal aggregation is **off** in v1 and evaluated separately. Raw per-frame
model performance and post-processed system performance are reported as two distinct numbers, so
smoothing can never be mistaken for model accuracy. Aggregation requires cluster association
across frames (a tracker), which is new machinery; it is deferred until per-frame prediction
stability has actually been measured.

**[PROVISIONAL]** Frame accumulation (finding **I9**) is off in v1. The Mid-70's non-repetitive
rosette fills in over time, so accumulating 2–3 frames materially densifies clusters — the single
cheapest accuracy lever available. But while stationary it is free and while flying it requires
motion compensation from an external IMU (the Mid-70 has none). Accumulation is therefore an
ablation to run **stationary**, with the explicit caveat that its gains do not transfer to flight
until deskewing exists.

## 7. Evaluation

**[DECIDED]** Two levels, reported separately and never merged into one headline number.

**Level 1 — model, on ground-truth cluster crops:**
balanced accuracy; macro-F1; per-class precision / recall / F1; confusion matrix; per-class
performance at the rare-class end called out explicitly. Primary selection metric:
**macro-F1**, because class imbalance in LiDAR scenes is severe and overall accuracy would be
dominated by whatever is most common.

**Level 2 — pipeline, end to end on full frames:**
an object counts as correct only if clustering isolated it *and* the classifier labelled it
correctly. Reported as instance-level precision/recall per class. Clustering failures (merged
objects, split objects, missed objects) are attributed to the clustering stage and counted, not
hidden.

**Calibration:** reliability diagram and expected calibration error on the held-out set, plus
precision/recall as a function of the reject threshold. A high softmax score is not treated as
evidence of correctness.

**Splits:** by recording session, never by random frame. Consecutive Mid-70 frames of the same
scene are near-duplicates; random splitting would produce a validation score that measures
memorisation. Train / val / test manifests are fixed files under version control.

## 8. Pending items

| # | Unknown | Needed to resolve | Blocks |
|---|---|---|---|
| ~~P1~~ | ~~Class list and semantic definitions~~ | **RESOLVED 15 Oct 2026** — `data/label_map.yaml`, `K = 6` | — |
| P2 | Labelled data: location, format, volume, provenance | Owner to supply. Protocol in `data_collection_plan.md` §4 | Everything downstream of the data loader |
| P7 | Flight altitude and sensor mount angle | Owner to supply | Whether `person` is in scope at all: resolvable to ~12 m side-on but only ~8 m from nadir |
| P3 | Downstream consumer of the predictions | Who reads the ROS 2 topic, and what they do with it | FP/FN cost asymmetry, reject threshold, latency deadline |
| P4 | Real-time deadline | Is 10 Hz the requirement, or is the drone's control loop stricter? | Latency budget, `max_clusters_per_frame` |
| P5 | Flight regime at inference time | Stationary hover vs forward flight, and airspeed | Whether motion compensation is in scope for v1 |
| P6 | Cluster-count and cluster-size distributions | Real recordings + a profiling script | `max_clusters_per_frame`, `min_cluster_points`, batch shape |

### The dominant risk, recorded explicitly

**Training on complete CAD objects (ModelNet40-style) will not transfer to DBSCAN clusters from a
Mid-70 frame.** A ModelNet chair is a watertight, uniformly sampled, complete object. A Mid-70
cluster of a chair is a single-view, partial, range-dependent, sparse shell — sometimes merged
with the floor or a neighbouring object. These are different distributions, and the gap is not
closed by augmentation alone.

This means the label set in P1/P2 must consist of **real Mid-70 cluster crops**, or of synthetic
data generated with a sensor model realistic enough to be validated against real crops. If the
available labels turn out to be CAD-derived, that is not a blocker for building the pipeline, but
any accuracy number from it must be reported as CAD-domain performance and never as evidence of
real-world performance.

## 9. Consequences for architecture and compute

**[DECIDED]** recorded so the reasoning survives the decision:

1. **Cost scales with scene complexity, not frame rate.** `M` clusters per frame means the
   per-frame inference cost is `M ×` a single-object forward pass. A model that is comfortable at
   `M=1` can miss the deadline at `M=10`. This is the constraint that selects the architecture.
2. **Batching is mandatory, not an optimisation.** All `M` clusters are classified in one forward
   pass of shape `(M, 3, 1024)`. `M` separate calls would pay kernel-launch and
   Python-dispatch overhead `M` times over, which on an Orin Nano dominates the arithmetic.
   This requires a fixed point count per cluster — which the wrapper already guarantees.
3. **Architectures with per-layer neighbourhood search are penalised `M`-fold.** DGCNN's kNN
   graph rebuild costs ~5 N×N distance matrices per forward pass; at `M=8` that is the whole
   frame budget. PointNet performs zero neighbourhood searches.
4. **Fixed 1024 points per cluster wastes compute on small clusters and discards detail on large
   ones.** A multi-resolution input is possible but breaks the single-batch property in (2). Not
   pursued in v1; recorded as a known inefficiency.
5. **Rotation.** None of the candidate architectures is rotation-invariant. Clusters arrive at
   arbitrary yaw relative to the drone. Either canonicalise per cluster (PCA on the horizontal
   plane) or train with heavy yaw augmentation. **[PROVISIONAL]** yaw augmentation over the full
   360° plus small roll/pitch perturbation, since full canonicalisation has its own failure mode
   on near-symmetric objects where the PCA axis flips frame to frame — which would inject
   instability precisely where §6 wants stability. To be settled by ablation.
