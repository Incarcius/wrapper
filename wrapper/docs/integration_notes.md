# Classifier integration — what changed, and what was left alone

**Date:** 15 Oct 2026. **Scope:** wiring the per-cluster classification pipeline from
`pointcloud-ml/` into this wrapper. Contract: `pointcloud-ml/docs/wrapper_audit.md` §5 and
`pointcloud-ml/docs/task_definition.md` §4–§5.

The task is **per-object classification**: DBSCAN splits the scene, each cluster gets one class
label. Not whole-scene classification, not point-wise segmentation.

---

## Findings addressed

| Finding | What it was | What was done |
|---|---|---|
| **C1** | Voxel pre-pass and FPS ran *after* normalisation, so `downsampling.voxel_size_m` acted in unit-sphere units rather than metres | Resampling moved before normalisation and delegated to `pcml.data.sampling.resample_to_fixed_size`, which is the same function the training path uses. `voxel_size_m` is now a length in metres. |
| **C2** | The whole ±10 m ROI was normalised into one unit sphere and handed to a classifier trained on single objects | New clustering stage between outlier removal and sampling (`pcml.data.clustering.cluster_scene`); normalise + FPS re-scoped per cluster (`prepare_cluster_batch`). `process()` returns `FrameClusters` — M clusters plus their metric geometry — instead of one `(1024, 3)` array. `clustering.enabled: false` keeps the old whole-scene behaviour available for debugging and logs a warning when used. |
| **A1** | Three `np.random.choice` calls against NumPy's global RNG | One `np.random.Generator` owned by the pipeline, seeded from the new top-level `seed:` key. After C2 most clusters are smaller than 1024 points, so the padding path is the common case. |
| **A2** | `normalize()` upcast to float64; FPS and the voxel pass then ran at double the memory traffic | Everything stays float32. |
| **A4** | `StageTimings` had five fixed float fields and no cluster count | Extensible: named fields for `clustering` / `cluster_prep` / `inference` / `postprocess`, `__setitem__` routing unknown stages into `extra` rather than dropping them, `as_ms()` for `FramePrediction.latencies_ms`, and `n_clusters` (M) on the periodic log line. |
| **A5** | No test touched the voxel pre-pass — the branch containing C1 | `TestVoxelPrePassRunsInMetres` in `tests/test_pipeline.py`. The decisive test is an exact identity: `batch(cloud × k, v) == batch(cloud, v / k)`, which holds only if the voxel grid sees metres. Verified to invert under the old ordering: with voxel-after-normalisation that equality is false and `batch(cloud × k, v) == batch(cloud, v)` is true instead. |
| **A6** | No NaN/Inf guard after ingestion; `skip_nans` protects the ROS 2 path only | Non-finite rows dropped immediately after the shape check, via `pcml.data.clustering.finite_points_only`. |
| **I10** | Ground-plane removal absent, so `height_above_ground_m` was a minimum-z guess | New ground-removal stage between outlier removal and clustering, delegating to `pcml.data.ground.remove_ground` with the pipeline's seeded generator, and `plane.ground_z_at_origin()` feeding `prepare_cluster_batch(ground_z_m=...)`. See the section below: on this scene it is a prerequisite, not an ablation. |
| **I4** | Empty clouds became 1024 zeros and were classified anyway | `cli.process_frame` skips inference when no cluster survives or when every cluster's `SamplingStats.mode == "empty"`; `ClusterClassifier.classify_prepared` filters empty-mode entries independently, so a different front-end cannot reintroduce it. |
| **I5** | `(1, 3, 1024)` built inline in the main loop as a non-contiguous view with a hardcoded layout | Layout, dtype, device and contiguity are the adapter's (`pcml.deploy.adapter.to_model_input`) and the classifier's preallocated buffer's. `output.format: torch` is now rejected at startup with a message naming the replacement, rather than quietly doing something else. |
| **I6** | Config indexed with bare `[...]` on the per-frame path | `pipeline.validate_config` runs once in `__init__`: fills defaults, range-checks values, and rejects unknown sections and unknown keys. `ingestion` and `model` pass through untouched for their owners. A config error is now a startup failure. |
| **I8** | Subscribed to `/livox/lidar`, published nothing | `src/livox_wrapper/ros_publisher.py` publishes `vision_msgs/Detection3DArray` — class, score and a 3D box per instance. `Classification2D` cannot carry per-instance geometry. |

## Deliberately left alone

| Finding | Why |
|---|---|
| **C3** — staleness checked against the wrong clock | Ingestion-layer, orthogonal to the model work, and `livox_interface.py` is out of scope for this change. Still present. |
| **I1** — `get_latest_frame` returns the oldest frame | Same: real, ingestion-layer, untouched. |
| **A3** — `roi_filter` `mode: "range"` is spherical, not cylindrical | Changing it would change the meaning of an existing config key. Documented at the key and in `roi_filter()` instead; use `mode: "box"` for separate height limits. |
| `roi_filter` box values | **Widened, but not tuned.** The old ±10 m / z −2..3 box was sized for indoor proof-of-work and is wrong for a field: `z_max: 3.0` decapitated the floodlight masts and mature trees that the label map gives an `expected_extent_m` up to 15 m, and a ±10 m box sees a patch of grass on a 100–150 m field. Now ±75 m / z −30..30 with `max_range_m: 90.0` (the Mid-70's datasheet range at 10% reflectivity). Deliberately a permissive superset rather than a guess at the real crop, because an over-tight ROI deletes objects silently and unrecoverably while an over-wide one only costs latency and shows in the stage timings. Setting these properly needs P5 — flight regime, sensor mounting and altitude — which is unresolved; the ROI is in the sensor frame and nothing here transforms to body or world, so the sign and range of z depend entirely on how the sensor ends up pointing. **Owner to tighten.** |
| Frame accumulation (`I9`) | Off in v1 by decision — gains do not transfer to flight without deskewing, and the Mid-70 has no IMU (`task_definition.md` §6). |
| `docs/architecture.md` data-flow section | Predates the per-cluster re-scoping: it still describes a `(1024, 3)` pipeline output. Not rewritten here. |

## Config surfaces that moved underneath this change

Two things in `pointcloud-ml/configs/preprocess.yaml` changed while this integration was being
written, and both are handled explicitly rather than by silent defaulting:

* **`clustering:` became `region_proposal:`** and gained a pluggable `method`
  (`dbscan_instances` for discrete objects, `ground_tiles` for continuous surfaces).
  `ClusterClassifier` reads `region_proposal` when present and falls back to the older
  `clustering` keys, and it resolves the proposer through `pcml.data.regions.build_proposer`, so
  selecting `ground_tiles` upstream actually changes what it does. This wrapper's own
  `clustering:` section still mirrors only the `dbscan_instances` parameters; the key-by-key
  mapping is written out at that section in `config/pipeline_config.yaml`.
* **`ground_removal.enabled` became `true`**, and was reclassified upstream from an ablation to
  mandatory for the sports-field scene. It is now **wired on both paths**: this wrapper's
  `process()` runs it between outlier removal and clustering, and
  `ClusterClassifier.classify_frame` runs the same stage with the same parameters before region
  proposal. Both had to change together — if one removed the ground and the other did not, the
  same frame would cluster differently depending on which entry point a caller used.
  `classify_prepared` deliberately does **not** do it, because by the time clusters exist the
  ground is already gone and a second RANSAC fit would fit planes to the objects themselves.
  `wrapper/tests/test_integration.py` asserts the two routes reach identical cluster geometry.

## Why ground removal is a prerequisite here, not a preprocessing nicety

A mown field is a continuous sheet of returns. At the 0.5 m clustering radius every object
standing on it is connected to the grass, and therefore to every other object, so DBSCAN collapses
the frame into one component. Measured both ways in
`tests/test_pipeline.py::TestGroundRemoval`: on a synthetic field with four columns, leaving the
ground in yields **1 cluster holding >90% of the frame with a >10 m extent**, and removing it
yields **4 object-sized clusters**. There is no per-cluster classification to do in the first
state — there is one "object", and it is the field.

Two honest caveats:

* **A rejected fit deletes nothing.** If no plane clears `min_inlier_fraction` and `max_tilt_deg`,
  the frame passes through unchanged and the stage logs a warning. Removing a wrongly-fitted
  "ground" would delete real objects, so declining is the safer failure — but the frame will then
  cluster badly, which is why the outcome is logged rather than silent.
* **Density decides whether any of this works.** At the sparsity a single Mid-70 frame actually
  has over a whole field (~10k points over ~80 × 80 m is ~1 point/m², a mean spacing of ~0.95 m),
  the ground sheet is *wider-spaced than the 0.5 m DBSCAN radius*, so `outlier_removal.method:
  dbscan` classifies most of the ground as noise and deletes it before RANSAC ever sees it — the
  measurement script hit exactly this and reported "no ground plane found" on 15 of 20 frames
  until the scene was tightened to a low hover (~12 points/m²). That interaction between the
  outlier stage and the ground stage is a real ordering hazard and is **not** resolved here: the
  shipped `outlier_removal.method` is `open3d_statistical`, which does not threshold on a radius
  and so should not do this, but Open3D cannot be imported on this machine and that path is
  therefore untested. Worth a profiling pass on real recordings before trusting either stage at
  field-wide range.

## Known placeholders — not results

The class list (`task_definition.md` P1) is **resolved**: `pointcloud-ml/data/label_map.yaml` is
owner-confirmed as of 15 Oct 2026, K = 6 — `person`, `vegetation`, `pole`, `structure`,
`net_fence`, `vehicle` — with `class_id -1` / `unknown` reserved as the reject *output* and never
a trained class. `ClusterClassifier` builds the head at K = 6 from that file, and no model config
or script contains a class count.

The labelled data (P2) is **still open**, so there is no training run and no checkpoint.
`model.classifier_config` points at `pointcloud-ml/configs/deploy_placeholder.yaml`, which leaves
`checkpoint: null` and therefore builds a **randomly-initialised** model. Every `FramePrediction`
from it carries `is_placeholder=True` and the classifier logs a warning at startup.

The vocabulary being real makes the output *more* dangerous to quote, not less: a prediction of
`person` now looks like a result, when all it means is that random weights put the largest logit
at index 0. In the measurement below every cluster in every frame came back `vegetation` with
confidence 0.181 — the same class and the same confidence throughout, which is the signature of
an untrained network and is exactly what should be expected.

Note also that the reject threshold is `0.0`, taken from the label map, which **disables
rejection**: nothing is ever labelled `unknown`. That is deliberate — a real threshold can only
come from a precision/recall-vs-threshold curve on held-out data, and there is no trained model to
draw one from. Tests assert both halves of this, so the `0.0` reads as a decision rather than a
dead branch.

This configuration measures latency, memory, batch shapes, determinism and publisher plumbing. It
measures nothing about accuracy, and its class outputs must never be quoted as such.

## Not executed anywhere

* **The ROS 2 publisher.** `rclpy` and `vision_msgs` come from a ROS 2 install, not pip, and
  neither is available on the development machine. The conversion is a pure function over plain
  dicts (`detection_array_fields`) and is unit-tested; the `DetectionPublisher` class that copies
  those dicts into real messages has never been run. It also carries a `hasattr` branch for the
  `vision_msgs` 3.x flat hypothesis layout, which is likewise unverified — Humble ships 4.x.
* **Anything on the Jetson.** No device is reachable from this machine, so every on-device figure
  — latency, memory, TensorRT behaviour — remains pending.
* **Open3D paths.** Open3D is installed in the dev environment but fails to import (missing
  `libusb-1.0.so.0`), so the measurements below are from the scikit-learn DBSCAN and NumPy FPS
  fallbacks. The Open3D C++ paths are selected by the shipped config and are untested here.

## Measured, on this dev box only

x86 CPU (Intel Core 5 120U), WSL2 Ubuntu 24.04, Python 3.12, torch 2.14 CPU, Open3D unavailable.
20 synthetic frames of 10,000 points: a ground sheet plus 6 objects 2–12 m tall standing on it,
over a 24 × 24 m patch (a low hover — see the density caveat above). The shipped
`pipeline_config.yaml` with two keys overridden (`outlier_removal.method: dbscan`,
`model.publish.enabled: false`), after `warmup(10)`. Median / max per frame, milliseconds:

| Stage | Median | Max |
|---|---|---|
| finite filter (A6) | 0.4 | 1.3 |
| ROI crop | 0.3 | 1.1 |
| outlier removal (sklearn DBSCAN) | 38.8 | 90.8 |
| **ground removal (NumPy RANSAC, 200 iters)** | **7.0** | **18.6** |
| clustering (sklearn DBSCAN) | 3.2 | 12.5 |
| per-cluster resample + normalise | 1.2 | 3.8 |
| batching into the preallocated tensor | 0.1 | 0.3 |
| inference, one pass over all M clusters | 9.9 | 28.6 |
| post-process | 0.1 | 0.3 |
| **total** | **68.9** | **157.7** |

M was 0–6 per frame. Two observations worth keeping:

* **The region count and the classified count differ, visibly.** The first frame found 9 regions
  and classified 3; the rest were ground-skirt fragments below `min_region_points: 50`. That gap
  is reported in `FramePrediction` rather than inferred, which is the point of carrying both
  numbers.
* **Ground removal pays for itself.** Measured as an A/B on the *same* 20 frames, preprocessing
  only (no inference), so the comparison is the stage and not a change of fixture:

  | | stage on | stage off |
  |---|---|---|
  | planes fitted | 20/20 | — |
  | ground removal | 7.8 ms | 0.0 |
  | clustering | 2.1 ms | 22.4 ms |
  | per-cluster resample + normalise | 0.4 ms | 54.4 ms |
  | **preprocessing total** | **45.8 ms** | **113.9 ms** |
  | points reaching clustering | 554 | 7,220 |
  | M per frame | 0–3 | 8–16 |

  It costs 7.8 ms and removes 68 ms, because every stage after it then runs on the ~7% of points
  that are not ground. The earlier 344.5 ms median in this project's notes was measured on a
  ground-less fixture and is not comparable to either column.

  The `M` row is the part that matters more than the latency. With the stage off, `M` is 8–16 and
  hits the `max_clusters_per_frame: 16` cap — those are ground fragments being classified as
  objects, not the six real objects in the scene. Note this is the *shattering* failure mode
  rather than the merging one: which of the two a frame gets depends on ground density relative
  to the 0.5 m radius, and `tests/test_pipeline.py::TestGroundRemoval` demonstrates the merging
  case at a higher density. Either way the output is wrong, and neither is visible from the
  cluster count alone — the stage-off column looks healthier than the stage-on one.

  With the stage on, `M` is 0–3 against six objects present, because after ground removal this
  synthetic scene leaves only ~90 points per object and several fall below
  `min_region_points: 50`. That is the fixture being sparse, not a pipeline fault, but it is a
  reminder that `min_region_points` and the object point counts have to be re-derived together
  from real recordings (P6).

These numbers are **not** a Jetson budget and not evidence about the 100 ms frame deadline. Both
DBSCAN stages are the scikit-learn fallback, which is not the path the shipped config selects, and
the scene is synthetic. They are recorded as a baseline for the same code on the
same machine.
