# Drone LiDAR perception — Livox Mid-70 on Jetson Orin Nano

Real-time per-object classification of LiDAR point clouds on a drone. A Livox
Mid-70 streams into a Jetson Orin Nano over ROS 2; the scene is split into
object instances and each one is classified onboard.

> **Status: no model has been trained, and there is no accuracy number anywhere
> in this repository.** The pipeline, the model, the export path and the
> benchmarking framework are built and tested; what is missing is labelled
> recordings and access to the device. Everything unmeasured is marked
> `pending` rather than estimated.

---

## The two subprojects

| | |
|---|---|
| **[`wrapper/`](wrapper/)** | The ROS 2 ingestion and preprocessing pipeline that runs on the drone. Subscribes to `/livox/lidar`, crops, de-noises, removes the ground, clusters the scene into objects, normalises and resamples each one, classifies, and publishes the results back to ROS 2. |
| **[`pointcloud-ml/`](pointcloud-ml/)** | The ML project: dataset tooling, the model, training and distillation benchmarking, compression, ONNX export, and the deployment adapter the wrapper calls. |

The boundary between them is deliberately one function and one class
(`pointcloud-ml/src/pcml/deploy/`), so the model can be swapped without
touching the pipeline. `pcml.deploy` is the only subpackage that ships to the
Jetson, and it must never import training code, datasets or teacher
checkpoints.

```
Livox Mid-70 ──▶ livox_ros2_driver ──▶ LivoxInterface ──▶ PointCloudPipeline
   (Ethernet)        (C++, external)      (ROS 2 sub)      crop → de-noise →
                                                           ground removal →
                                                           cluster → per-cluster
                                                           normalise + FPS
                                                                    │
                                                          (M, 3, 1024) │
                                                                    ▼
                                                        ClusterClassifier
                                                        one batched forward pass
                                                                    │
                                                                    ▼
                                            vision_msgs/Detection3DArray → ROS 2
```

## The decisions, in one place

| Question | Answer | Why |
|---|---|---|
| What does the model predict? | one class + confidence per **object cluster** | a ±10 m scene cannot carry one object label |
| Which architecture ships? | **compact PointNet**, 249,542 params | zero neighbourhood searches; per-frame cost is `M` clusters × one shared-MLP pass; every op is TensorRT-native |
| How many forward passes per frame? | **one**, batched over all clusters | measured 38% per-cluster saving versus one call per cluster |
| Classes | 6: `person`, `vegetation`, `pole`, `structure`, `net_fence`, `vehicle`, plus `unknown` as a reject outcome | each is resolvable by the Mid-70 at a safe altitude — the ball and stumps are not |
| Is ground removal optional? | **No** | with the ground left in, a mown field merges every object into one cluster |

Full reasoning: [`pointcloud-ml/docs/task_definition.md`](pointcloud-ml/docs/task_definition.md).

## Where to start reading

| If you want to… | Read |
|---|---|
| Understand what is predicted and why | [`task_definition.md`](pointcloud-ml/docs/task_definition.md) |
| Know what the sensor can actually resolve | [`data_collection_plan.md`](pointcloud-ml/docs/data_collection_plan.md) |
| Collect the data | same document, §4 — read §1 before flying |
| See the measured numbers | [`reports/stage1_baseline.md`](pointcloud-ml/reports/stage1_baseline.md) |
| Understand the wrapper's internals | [`wrapper/docs/architecture.md`](wrapper/docs/architecture.md) |
| Set up a Jetson from scratch | [`wrapper/docs/setup_guide.md`](wrapper/docs/setup_guide.md) |
| Run the distillation study | [`distillation_plan.md`](pointcloud-ml/docs/distillation_plan.md), then [`publication_pipeline.md`](pointcloud-ml/docs/publication_pipeline.md) |
| Know what was predicted before the runs | [`preregistration.md`](pointcloud-ml/docs/preregistration.md) |
| Know what transfers to agriculture / road quality | [`domain_roadmap.md`](pointcloud-ml/docs/domain_roadmap.md) |

## Tests

**Run one pytest process at a time.** Several tests assert on the content and
sha256 of the real `data/label_map.yaml` and `configs/preprocess.yaml` — which
is the point of them, since that is what catches a label map that would
silently invalidate a checkpoint. Two concurrent runs therefore produce
spurious failures.

```bash
cd pointcloud-ml && python -m pytest -q -p no:cacheprovider    # 665 tests
cd wrapper        && python -m pytest -q                        #  76 tests
```

Neither suite needs a LiDAR, a GPU, ROS 2 or any data: the ingestion path is
validated against synthetic recordings, including a genuine CDR-serialised
ROS 2 `.db3` bag written by the tests themselves.

## Development environment

Linux or WSL2. Python 3.10 (matching JetPack 6) through 3.12 are both covered
by CI.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy pyyaml pytest scikit-learn
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install onnx onnxruntime           # export parity checks
pip install open3d                     # optional: faster C++ FPS and DBSCAN
```

`open3d` is optional throughout — every path falls back to NumPy or
scikit-learn, and both paths are tested. On a Jetson, **install PyTorch from
NVIDIA's JetPack wheel, never PyPI**: a PyPI ARM64 build is CPU-only and the
model would silently never touch the GPU.

Exact resolved versions used for the recorded measurements:
[`pointcloud-ml/requirements-dev-lock.txt`](pointcloud-ml/requirements-dev-lock.txt).

## What is not in this repository, on purpose

- **Trained weights and compiled engines.** Regenerable, and the `.onnx` files
  that exist locally hold randomly initialised weights — exactly the kind of
  artifact that later gets mistaken for a trained one.
- **Raw and processed recordings.** Large, and the input to a reproducible
  pipeline rather than an output of it. The split manifests and the label map
  *are* versioned.
- **PDF exports of the documentation.** The markdown is the source of record.
