# pointcloud-ml

Per-cluster point-cloud classification for a Livox Mid-70 on a Jetson Orin Nano,
built to plug into the existing [`wrapper/`](../wrapper) preprocessing pipeline.

**Status: Stage 1–2 complete. No model has been trained — there is no data yet.**

| Stage | State |
|---|---|
| 1. Discovery / wrapper audit | **done** — [`docs/wrapper_audit.md`](docs/wrapper_audit.md) |
| 2. Task + architecture decision | **done** — [`docs/task_definition.md`](docs/task_definition.md), [`reports/stage1_baseline.md`](reports/stage1_baseline.md) |
| 3. Data pipeline | **toolchain done, waiting on a recording** — ingestion, labelling, splits and the torch `Dataset` are built and tested end to end on synthetic recordings. [`data/README.md`](data/README.md), [`docs/data_collection_plan.md`](docs/data_collection_plan.md) |
| 4. Baseline experiments | **blocked on a recording only** — nothing else is missing |
| 5–8. Selection, distillation, pruning, export | not started |
| 9. Jetson deployment | blocked on hardware access |

## The decisions, in one place

| Question | Answer | Why |
|---|---|---|
| What does the model predict? | One class + confidence per **DBSCAN cluster** | A ±10 m scene cannot carry one object label (wrapper finding C2) |
| Which architecture ships? | **Compact PointNet**, `[64,64,64,128,512]`, 249k params | Zero neighbourhood searches; cost is `M` × one shared-MLP pass; every op is TensorRT-native |
| Which architecture trains? | A wider PointNet now; DGCNN / PointNet++ SSG once data exists | Teacher runs off-device and is distilled into the student (Phase Five) |
| How many forward passes per frame? | **One**, batched over `M` clusters | Measured 38% per-cluster saving vs `M=1`; `M` separate calls pay dispatch overhead `M` times |
| Is the DGCNN prep wasted? | No | It is three conventions (1024 points, unit sphere, `(B,3,N)`) that PointNet shares |

Reasoning in full: [`docs/task_definition.md`](docs/task_definition.md) §9.

## Repository layout

```
pointcloud-ml/
  configs/
    preprocess.yaml              clustering, sampling, normalisation, augmentation
    models/
      pointnet_deploy.yaml       the deployment candidate
      pointnet_teacher.yaml      distillation teacher (never deployed)
      pointnet_student.yaml      aggressive student (efficiency arm)
  data/                          no recordings yet — see data/README.md
    label_map.yaml               the class vocabulary, owner-confirmed. K comes from here
    label_map.template.yaml      the empty structural template
    raw/                         untouched recordings (not version-controlled)
    processed/                   generated cluster crops (not version-controlled)
    manifests/                   train/val/test, split BY SESSION (version-controlled)
  docs/
    task_definition.md           what is predicted, at what granularity, and why
    wrapper_audit.md             Phase One audit + the wrapper<->model interface spec
    data_collection_plan.md      class list, flight constraints, what to record
    domain_roadmap.md            what transfers to agriculture / road quality
    distillation_plan.md         Phase Five method, and when to skip it
    publication_scope.md         what would make the study publishable, and what would not
    publication_pipeline.md      the ordered stages, their gates, and the venue call
    preregistration.md           hypotheses, metric, min_effect and decision rule, dated
  reports/
    stage1_baseline.md           measured parameters / size / CPU latency
    model_summary.json           raw output of scripts/model_summary.py
    sensor_feasibility.json      what the Mid-70 can resolve, by object and range
  scripts/
    extract_crops.py             recordings -> cluster crops + index
    label_clusters.py            label once per object, propagate by position
    make_splits.py               train/val/test manifests, BY SESSION
    model_summary.py             measures params, checkpoint size, CPU latency
    sensor_feasibility.py        points-on-target vs range for the Mid-70
  src/pcml/
    data/
      io.py                      frame readers: .npy/.npz/.pcd/.ply + ROS 2 .db3
      config.py                  typed preprocess.yaml, plus its sha256
      sampling.py                deterministic voxel + FPS, in metres
      ground.py                  RANSAC ground split (ground kept as signal)
      regions.py                 pluggable region proposal: instances | tiles
      clustering.py              scene -> instances -> batched model input
      crops.py                   recording -> labelled-cluster crops, reproducibly
      labels.py                  label store, §5.2 propagation, ASCII cluster views
      label_mapping.py           loads and validates label_map.yaml; K lives here
      dataset.py                 torch Dataset (the only torch-dependent module)
      splits.py                  manifests split by session + distribution report
    models/
      pointnet.py                the architecture
      model_factory.py           config -> model
    training/
      reproducibility.py         independent named RNG streams (the fairness guarantee)
      metrics.py                 macro-F1 / balanced acc / calibration, vs sklearn in tests
      losses.py                  CE + temperature KD + feature distillation, one objective
      tournament.py              paired permutation tests, Holm, trend, the decision rule
    compression/
      export.py                  ONNX export with a dynamic batch axis + parity check
    deploy/
      adapter.py                 the ONLY coupling to the wrapper
      classifier.py              ClusterClassifier: the wrapper-facing inference entry
  tests/                         665 tests, all passing (run one process at a time)
    synthetic_recordings.py      synthetic sessions, incl. a real CDR .db3 writer
```

The field classification is a **proof of concept**; the intended deployment
domains are agriculture and road quality. What transfers, what does not, and an
honest per-domain sensor verdict: [`docs/domain_roadmap.md`](docs/domain_roadmap.md).

`src/pcml/deploy/` is the only subpackage that ships to the Jetson. It must never
import training code, datasets or teacher checkpoints.

## Environment

The Windows host cannot run this: Smart App Control blocks NumPy's compiled DLL,
and there is no NVIDIA GPU. Work happens in WSL2.

```bash
# one-time, inside WSL Ubuntu-24.04
python3 -m venv --without-pip ~/venvs/pcml
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/venvs/pcml/bin/python /tmp/get-pip.py
~/venvs/pcml/bin/python -m pip install numpy pyyaml pytest scikit-learn
~/venvs/pcml/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
~/venvs/pcml/bin/python -m pip install open3d onnx onnxruntime
```

Exact resolved versions: [`requirements-dev-lock.txt`](requirements-dev-lock.txt).

Open3D installs but does not import — `libusb-1.0.so.0` is missing. The code falls
back to scikit-learn DBSCAN and the NumPy FPS reference path, so correctness is
unaffected and both paths are tested. To get the faster C++ paths:

```bash
sudo apt install -y libusb-1.0-0 libgl1 libgomp1
```

## Commands

```bash
cd pointcloud-ml

# tests (529, no data or GPU needed). Run ONE process at a time: several tests
# assert on the real data/label_map.yaml and configs/preprocess.yaml by path, so
# two concurrent runs (or an edit mid-run) produce spurious failures.
~/venvs/pcml/bin/python -m pytest -q

# measured parameters / checkpoint size / CPU latency
~/venvs/pcml/bin/python scripts/model_summary.py

# what the Mid-70 can resolve, by object and range
~/venvs/pcml/bin/python scripts/sensor_feasibility.py

# wrapper baseline, to confirm nothing regressed
cd ../wrapper && ~/venvs/pcml/bin/python -m pytest -q
```

## From a recording to a training batch

This is the whole data path, in four commands. Every step is tested end to end
on synthetic recordings ([`tests/test_ingestion_end_to_end.py`](tests/test_ingestion_end_to_end.py)),
so the only thing missing is a real recording.

**0. Before the flight** — write the per-session metadata file and fill it in at
capture time. Altitude, mount angle and whether the drone was hovering cannot be
reconstructed from the points afterwards ([`§4.2`](docs/data_collection_plan.md)),
and the session id is what the splits are made by.

```bash
~/venvs/pcml/bin/python scripts/label_clusters.py \
    --write-meta-template data/raw/session_01_hover_12m.meta.json
```

**1. Record raw**, unprocessed, on the drone. Every preprocessing parameter in
this project is something we will want to re-run against the same bytes, so
saving only the 1024-point outputs would discard every future ablation.

```bash
ros2 bag record /livox/lidar -o session_01_hover_12m
```

**2. Extract cluster crops.** Ground removal, then region proposal, per frame.
Writes one `<session>_crops.npz` plus one `<session>_crops.json` per session.

```bash
~/venvs/pcml/bin/python scripts/extract_crops.py \
    data/raw --glob --out data/processed --summary
```

`--summary` prints the cluster-count and cluster-size distributions. Run it on
the first real recording and use it to re-derive `max_clusters_per_frame: 16` and
`min_cluster_points: 50` from PROVISIONAL to measured
([`task_definition.md`](docs/task_definition.md) P6).

**3. Label.** Each static object is labelled **once** and propagated to every
frame by centroid matching, so a 60-second hover over 8 objects costs 8
decisions rather than 4,800 ([`§5.2`](docs/data_collection_plan.md)). Text-first:
it prints the cluster geometry in metres plus two ASCII projections and takes a
single keypress, so it works over ssh with no display. Saved after every
decision, so it is resumable and Ctrl-C costs at most the cluster on screen.

```bash
~/venvs/pcml/bin/python scripts/label_clusters.py \
    data/processed/session_01_hover_12m_crops.json

# progress so far, changing nothing
~/venvs/pcml/bin/python scripts/label_clusters.py \
    data/processed/session_01_hover_12m_crops.json --report
```

Keys are the class ids from `data/label_map.yaml`, plus `u` for `unknown`, `m`
for a moving object (this frame only — propagation assumes static), `s` to skip
and `q` to save and quit. `--radius` changes the match radius (default 0.5 m, the
DBSCAN `eps_m`); `--repropagate` re-matches after changing it. Every label
records its provenance, `annotated` or `propagated`
([`§5.3`](docs/data_collection_plan.md)).

**4. Split by session** and write the manifests.

```bash
~/venvs/pcml/bin/python scripts/make_splits.py \
    --crops data/processed --out data/manifests \
    --report reports/splits.md
```

Splits are by **recording session, never by frame** — consecutive Mid-70 frames
are near-duplicates, so a random frame split would measure memorisation
([`task_definition.md`](docs/task_definition.md) §7). There is no option for a
frame-level split. The test manifest takes `annotated` labels only, so the final
number is not partly a measure of the propagation heuristic. Use
`--pin SESSION:test` to hold a recording out deliberately — a second location
for domain shift, or the range-calibration recording, which is diagnostic.

**5. Load it.** The manifests are what training consumes:

```python
from pcml.data.dataset import ClusterCropDataset

train = ClusterCropDataset.from_manifest("data/manifests/train.json")
points, aux, label = train[0]      # (3, 1024), (6,), int
```

`augment` defaults to on for `train` and off for `val`/`test`. Augmentation is
applied to the **raw metric** cluster before normalisation, because the
parameters are metric — `jitter_sigma_m: 0.01` is a centimetre of Mid-70 ranging
noise, not 1% of the object's size — and it is seeded from
`(seed, epoch, index, view)`, so an item is identical on every run regardless of
batch size, shuffle order or worker count.

For a consistency or mean-teacher objective, `n_views` returns several
independently augmented views of each crop, sharing one label:

```python
train = ClusterCropDataset.from_manifest("data/manifests/train.json", n_views=2)
points, aux, label = train[0]      # (2, 3, 1024), (2, 6), int
```

The default collate then yields `(B, V, 3, N)`, which flattens to `(B*V, 3, N)`
for a single forward pass. This belongs to the dataset rather than the trainer:
yaw commutes with centring and unit-sphere scaling, but the jitter and point
dropout are metric and applied before normalisation, so a second view made by
re-rotating the dataset's output would carry the same dropout pattern and the
same jitter — a weaker perturbation than it looks. `n_views >= 2` requires
augmentation to be on, because views that differ only by the resampling draw
satisfy a consistency loss trivially.

### Reading recordings directly

`open_recording` dispatches on what it is given: `.npy`, `.npz`, `.pcd`, `.ply`,
a ROS 2 `.db3` or a rosbag2 directory, or a directory of frame files.

```python
from pcml.data.io import open_recording

for points, stamp in open_recording("data/raw/session_01_hover_12m"):
    ...  # points: (n, 3) float32 metres, sensor frame; stamp: header seconds
```

Bags are read **without rclpy**, which is not installed and will not be on this
laptop: the rosbag2 sqlite tables are read directly and the CDR payload is
deserialised in [`io.py`](src/pcml/data/io.py). It reads each field's offset and
datatype **from the message header** rather than assuming `x,y,z` are the first
three float32s — a Livox `PointCloud2` carries intensity, `tag` and `line`, and
the stride differs by driver version, so slicing the first 12 bytes of each point
would return plausible-looking garbage. A missing `x`/`y`/`z` is a loud error,
never a reinterpretation.

## What is verified vs written

**Executed and passing:** all 529 tests; the model summary, sensor feasibility
and ONNX export scripts; the wrapper's own tests. The whole data path —
recording → crops → labels → session splits → `Dataset` → a batched forward pass
through the deployed PointNet, plus one training step that reduces the loss — runs
end to end on synthetic recordings, including a ROS 2 `.db3` bag written by the
test itself so the CDR parser is genuinely exercised.

**Written but not yet exercised against real data:** every module has tests, but
**no module has seen a real Mid-70 frame**, because none exists in this project.
The synthetic field is a geometric simulation, not a sensor model: it has no
incidence-angle falloff, no reflectivity variation and none of the rosette's
within-frame non-uniformity. It is enough to validate shapes, contracts and file
formats, and it is not enough to validate anything about accuracy.

**Not written yet:** training loop, distillation, pruning, quantisation,
ONNX/TensorRT export, Jetson benchmarks, the streaming inference component, and
the ROS 2 publisher. Each is blocked on either data or hardware, and is
deliberately not written speculatively.

**Never claimed:** any accuracy figure, any Jetson measurement, any statement that
this runs on a Jetson. It has not been run on one. The labels in the end-to-end
test are assigned to synthetic objects by construction, so the loss falling in
`test_one_training_step_reduces_the_loss_on_this_batch` is a check that gradients
flow — not a result.
