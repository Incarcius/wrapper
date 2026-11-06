# Phase One — Wrapper Audit and Interface Specification

**Audited:** 15 Oct 2026 · `wrapper/` @ `4cbd903` ("Initial commit for wrapper") ·
remote `github.com/Incarcius/wrapper` · 3 Python modules, 1 config, 1 shell script, 5 docs, 1 test file.

This audit is independent of the wrapper's own `docs/code_review.md`. Where it confirms that
review, it says so; where it adds findings, they are marked **NEW**.

---

## 1. What actually exists

| Component | State | Evidence |
|---|---|---|
| ROS 2 ingestion | Implemented, **never run against hardware** | `livox_interface.py:92-129`; roadmap item 1 is entirely unchecked |
| Direct SDK ingestion | Stub that raises | `livox_interface.py:156` |
| ROI crop | Implemented, tested | `pipeline.py:70-87`; `test_pipeline.py:83-101` |
| Outlier removal | Implemented (3 methods), only DBSCAN tested | `pipeline.py:92-146` |
| Normalisation | Implemented, tested | `pipeline.py:151-167` |
| Voxel + FPS downsample | Implemented, **FPS tested, voxel path untested** | `pipeline.py:172-197`, `230-236` |
| Per-stage timing | Implemented | `StageTimings`, `pipeline.py:40-46` |
| Instance clustering | **Absent** | DBSCAN labels are computed then discarded at `pipeline.py:146` |
| Ground removal | **Absent** | roadmap item 7 |
| Frame accumulation | **Absent** | roadmap item 7 |
| **Any model** | **Absent** | `cli.py:65` is a `TODO` comment |
| Model adapter | 4 lines inline in the main loop | `cli.py:59-63` |
| ROS 2 output | **Absent** — subscribes, never publishes | finding I8 |
| Dataset / labels / training code | **Absent from the entire project** | `pointcloud-ml/` is new; `wrapper/` has none |

### The DGCNN question, settled

The brief states that "initial data preparation has been performed with DGCNN-oriented
preprocessing." That is accurate but narrower than it sounds. **There is no DGCNN implementation,
no DGCNN checkpoint and no DGCNN-specific code anywhere in the repository.** What exists is three
*conventions* that DGCNN happens to share with PointNet and PointNet++:

1. `target_points: 1024` — the ModelNet40 convention for all three architectures.
2. Unit-sphere normalisation — likewise shared.
3. The `(1, 3, 1024)` `(B, C, N)` tensor layout — shared by the common PyTorch reference
   implementations of all three, because they all use `Conv1d` over the point axis.

**Consequence: switching to PointNet costs nothing.** No preparation work is discarded, and the
preprocessing pipeline needs no architecture-driven changes. The wrapper's own review reached the
same conclusion ("roughly 85% of this review is model-independent"), and this audit confirms it by
inspection: the string "dgcnn" does not appear in any source file.

---

## 2. Data contract, as verified by reading the code

| Property | Value | Where |
|---|---|---|
| Ingestion output | `(N, 3)` float32, C-contiguous | `livox_interface.py:117-118` |
| Fields carried | x, y, z only. Intensity/tag/line/timestamp are **discarded at ingestion** | `read_points_numpy(field_names=["x","y","z"])` |
| NaN handling | `skip_nans=True` at ingestion; no further NaN/Inf guard downstream | `livox_interface.py:117` |
| Coordinate frame | Livox **sensor frame**, metres. No transform to body or world is applied anywhere | — |
| Pipeline input accepted | `(N, ≥3)`; columns ≥3 silently dropped | `pipeline.py:212-214` |
| Pipeline output | `(1024, 3)` float32, or torch tensor if `output.format: torch` | `pipeline.py:242-245` |
| Point ordering | Unspecified. FPS order from Open3D, then random padding | `pipeline.py:179-197` |
| Frame semantics | One 100 ms driver frame. No accumulation, no rolling window | `architecture.md:49-51` |
| Rate | ~10 Hz, ~10,000 points/frame (~100k pts/s sensor-limited) | `architecture.md:45-51` |
| Model input | `(1, 3, 1024)`, **non-contiguous**, CPU, unbatched | `cli.py:60` |

### Train/inference parity risks in this contract

Three properties of the inference path must be mirrored exactly by the training data pipeline, or
the model will be evaluated on inputs it will never see in flight:

1. **Sensor-frame coordinates, no gravity alignment.** A drone at non-zero roll/pitch produces
   clusters tilted in the sensor frame. Training crops must carry the same tilt distribution.
2. **xyz only.** No intensity, no normals, no height-above-ground — unless a preprocessing stage
   is added to *both* paths.
3. **Per-cluster unit-sphere normalisation** (after the C2 re-scoping). Absolute scale is
   discarded; see `task_definition.md` §4.

---

## 3. Findings

### Confirmed from the wrapper's own review

C1 (voxel runs after normalisation — units wrong), C2 (whole scene normalised as one sample),
C3 (staleness checked against the wrong clock), I1 (`get_latest_frame` returns the oldest frame),
I4 (empty clouds become all-zeros and get classified anyway), I5 (non-contiguous, hardcoded
layout), I6 (unvalidated config indexed on the hot path), I7 (repeated float64 conversions),
I8 (nothing published back to ROS 2). All verified by inspection; all still present in the code.

### NEW findings

**A1 — Preprocessing is stochastic and unseeded, so it is not reproducible.**
`pipeline.py:181`, `:191`, `:196` all call `np.random.choice` against NumPy's **global** RNG.
Three separate code paths — undersized-cluster padding, FPS size correction, and the random
sampling fallback — are therefore non-deterministic and unseedable from config.

Why it matters more than it looks: after the C2 fix, clusters are frequently *smaller* than 1024
points, so the random-padding path at `:181` becomes the common case, not an edge case. The same
physical cluster then yields a different model input on every run. That breaks (a) reproducible
evaluation, (b) any attempt to diff PyTorch against an exported ONNX/TensorRT engine
(export-parity testing needs bit-identical inputs), and (c) debugging of a single recorded frame.

*Fix:* a `np.random.Generator` owned by the pipeline, seeded from config, with the seed recorded
in every experiment manifest.

**A2 — `normalize()` silently upcasts to float64 and returns float64.**
`pipeline.py:156` does `.astype(np.float64)`; the result flows through FPS and is only cast back
at `pipeline.py:242`. So the most expensive stage (FPS, and the voxel pre-pass) operates on
float64 arrays — double the memory traffic, on a memory-bandwidth-bound device. Compounds I7.

**A3 — `roi_filter` `mode: "range"` is spherical, not cylindrical.**
`pipeline.py:84` takes the 3-vector norm, so the filter is a sphere centred on the sensor. For a
downward-or-forward-looking drone sensor, a horizontal radius with separate height limits
(cylinder) is almost always the intended shape. Not a bug, but a silent semantic that will
surprise whoever widens `max_range_m` for outdoor flight.

**A4 — `StageTimings` cannot express the pipeline this task needs.**
It has five fixed float fields. The per-cluster pipeline adds ground removal, clustering, per
cluster normalise+FPS (×M), batching, inference and post-processing. Per §11.3 of the brief these
must be measured separately. The dataclass needs to become extensible, and the
`log_every_n_frames` log line needs to carry `M`.

**A5 — No test exercises the voxel pre-pass, which is where the one critical bug lives.**
`test_pipeline.py` sets `voxel_size_m: None` in `BASE_CONFIG` (line 11), so the branch at
`pipeline.py:232-235` — the C1 bug — is never entered by the test suite. 10 tests, all passing,
and the most damaging defect in the codebase sits in untested code. This is the single most
instructive fact in the audit: the suite's green status carries less information than it appears to.

**A6 — No NaN/Inf guard after ingestion.**
`skip_nans=True` protects the ROS 2 path only. A frame loaded from a file, a recording, or the SDK
path can carry NaNs, which propagate silently: `normalize()` would produce an all-NaN cloud
(centroid is NaN), FPS would return NaNs, and the model would consume them. Needed because the
training/evaluation path loads from files, not from ROS 2.

---

## 4. Verified pipeline diagram (current state)

```
┌──────────────┐   UDP    ┌───────────────────┐  /livox/lidar    ┌────────────────────────────┐
│ Livox Mid-70 │─────────▶│ livox_ros2_driver │─────────────────▶│ LivoxInterface (thread)    │
│ ~100k pts/s  │          │ (C++, external)   │  PointCloud2     │ → (N,3) f32                │
└──────────────┘          └───────────────────┘  ~10 Hz          │ queue(2), drop-oldest      │
                                                                 └─────────────┬──────────────┘
                                                                               │ get_latest_frame()
                                            ┌──────────────────────────────────▼──────────────┐
                                            │ PointCloudPipeline.process()   [main thread]    │
                                            │  1 roi_filter      (N,3)  →  (N₁,3)             │
                                            │  2 remove_outliers (N₁,3) →  (N₂,3)             │
                                            │  3 normalize       (N₂,3) →  (N₂,3) f64  ← A2   │
                                            │  4 voxel + FPS     (N₂,3) →  (1024,3)    ← C1   │
                                            └──────────────────────────────────┬──────────────┘
                                                                               │ (1024,3) f32
                                                        ┌──────────────────────▼───────────────┐
                                                        │ cli.py:60  .T + expand_dims          │
                                                        │ → (1,3,1024) non-contiguous  ← I5    │
                                                        │ → TODO: model                 ← gap  │
                                                        └──────────────────────────────────────┘
                                                                     ⊗ nothing published  ← I8
```

Target state, with the per-cluster task definition applied — new stages marked `NEW`:

```
 ingest → roi → outliers → [NEW ground removal?] → [NEW DBSCAN cluster] → M clusters
                                                                            │
        ┌───────────────────────────────────────────────────────────────────┘
        ▼  per cluster (vectorised over M)
   normalise + FPS → (M,1024,3) → [NEW adapter] → (M,3,1024) contiguous, device
        │
        ▼
   [NEW PointNet] → (M,K) logits → [NEW postprocess: softmax, reject, label map]
        │
        ▼
   FramePrediction → [NEW ROS 2 publisher: vision_msgs/Detection3DArray]
```

---

## 5. Interface specification

The contract between the wrapper and the ML work. **One function and one class** — this is the
entire coupling, deliberately kept small so architectures can be swapped without touching the
wrapper.

### Ownership boundary

```
pointcloud-ml/              ← this project. Training, experiments, compression, evaluation.
  src/pcml/
    deploy/                 ← the ONLY subpackage that ships to the Jetson.
      adapter.py            ← implements the interface below. No torch.nn beyond the model.
      classifier.py         ← loads checkpoint + label map, runs inference, returns predictions.
wrapper/                    ← separate git repo, kept intact.
  src/livox_wrapper/
    cli.py                  ← gains ~5 lines: construct the classifier, call it, publish.
```

The deployed artifact depends on: a checkpoint (or TensorRT engine), a `label_map.yaml`, and a
`preprocess.yaml`. It depends on **no** dataset, no teacher checkpoint and no training code.

### The interface

```python
class ClusterClassifier:
    """Owns the model, the label map and the preprocessing config.

    Constructed once at startup; `classify_frame` is called per frame.
    Must not allocate per call on the steady-state path.
    """

    def __init__(self, config_path: str) -> None: ...

    def warmup(self, n_frames: int = 10) -> None:
        """Run n_frames of synthetic input to pay CUDA context and kernel
        autotuning costs before timing starts. Mandatory on Jetson: the first
        CUDA call can take seconds and would otherwise appear as a latency spike."""

    def classify_frame(
        self,
        points: np.ndarray,          # (N, 3) float32, sensor frame, metres, post-outlier-removal
        stamp: float,
    ) -> FramePrediction: ...        # see task_definition.md §5
```

Responsibilities, in order: validate input → cluster → per-cluster normalise + FPS → batch →
adapt layout/device → infer → softmax + reject threshold → resolve names via label map →
assemble `FramePrediction` with per-stage latencies.

**Invariants the wrapper may rely on:**

1. Never raises on malformed input. An empty, all-NaN or single-point cloud yields a
   `FramePrediction` with an empty prediction list and a populated diagnostic field. A perception
   module must not be able to crash the flight software.
2. Bounded memory and bounded time: at most `max_clusters_per_frame` clusters are classified, and
   the batch tensor is preallocated at `(max_clusters, 3, 1024)`.
3. Coordinate frames, units and class IDs are never silently changed. Centroids come back in the
   same sensor frame, in metres, that went in.
4. Output is deterministic given a fixed seed and a fixed input (requires finding A1 to be fixed).

### What this requires from the wrapper

| Change | Finding | Risk |
|---|---|---|
| Reorder voxel/FPS before normalisation | C1 | low, isolated to `process()` |
| Re-scope normalise + FPS to per-cluster | C2 | medium — changes `process()`'s return type |
| Config-driven adapter, contiguous, on-device | I5 | low |
| Seedable RNG | A1 (NEW) | low |
| Extensible `StageTimings` + cluster count in the log line | A4 (NEW) | low |
| NaN/Inf guard | A6 (NEW) | low |
| Publish `Detection3DArray` | I8 | low, additive |

**Nothing in `livox_interface.py` needs to change for the model work.** C3 and I1 are real bugs
but they are ingestion-layer and orthogonal; they are the wrapper owner's to fix.

---

## 6. Files: preserve / modify / create

**Preserve as-is (no technical reason to touch):**
`livox_interface.py`, `scripts/start_demo.sh`, `docs/setup_guide.md`, `docs/demo_runbook.md`,
`pyproject.toml`, `requirements.txt`, `.gitattributes`.

**Modify (minimally, each change traced to a finding):**
`pipeline.py` (C1, C2, A1, A2, A4, A6), `cli.py` (I5, I8, model wiring),
`config/pipeline_config.yaml` (new `clustering:` and `model:` sections),
`tests/test_pipeline.py` (A5 — a voxel-path test that would have caught C1).

**Create (in `pointcloud-ml/`, not in the wrapper):**
dataset loader, label map, preprocessing/augmentation, model definitions, training loop,
evaluation, distillation, pruning, export, benchmarks, and the `deploy/` subpackage above.

---

## 7. Reproducible baseline

Verified-working command for the existing test suite (run it before and after any wrapper change):

```bash
cd wrapper && python -m pytest -q
```

**Not yet executed in this environment.** Blocked, then unblocked, as follows:

| Blocker | Status |
|---|---|
| Windows host: Smart App Control (`VerifiedAndReputablePolicyState = 1`) blocks NumPy's compiled DLL, so `import numpy` fails under the Windows Python 3.11 install. It also blocks the agent's Git-Bash tool. | Cannot be disabled without reinstalling Windows — SAC is one-way. Worked around, not fixed. |
| No NVIDIA GPU on this laptop (`nvidia-smi` absent; Intel Core 5 120U, 23.6 GB RAM) | Permanent. CPU-only locally; real training needs cloud or the Jetson. |
| Workaround | WSL2 **Ubuntu-24.04**, Python 3.12.3, venv at `~/venvs/pcml`, pip bootstrapped from PyPA (no sudo available in that distro), torch CPU wheels. |

Python version note: the venv is 3.12 while JetPack 6 ships **3.10**. Not parity. Dependencies are
therefore pinned in a manifest and the export path (ONNX) is version-independent, but any
Python-version-sensitive behaviour must be re-verified on the device.
