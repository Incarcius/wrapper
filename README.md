# Livox Mid-70 Point-Cloud Wrapper

Real-time preprocessing pipeline that turns raw **Livox Mid-70** LiDAR frames into clean,
fixed-size point clouds for a point-cloud classification network. It runs on an
**NVIDIA Jetson Orin Nano** with **ROS 2 Humble**.

> **Status: proof of work.** Ingestion and preprocessing are implemented; the classifier is
> not yet chosen or wired in. A code review found several issues to fix before the demo.
> See [Known issues](#known-issues).

---

## How it works

```
Livox Mid-70 ──Ethernet──▶ livox_ros2_driver ──/livox/lidar──▶ LivoxInterface
                           (C++ ROS 2 node)     PointCloud2     background thread,
                                                                 2-slot drop-oldest queue
                                                                        │  (N, 3) float32
                                                                        ▼
        PointCloudPipeline:  1. ROI crop → 2. outlier removal → 3. normalise → 4. voxel + FPS
                                                                        │  (1024, 3)
                                                                        ▼
                                                              classifier (TBD)  (1, 3, 1024)
```

The Mid-70 produces about 10,000 points per frame, 10 times a second. The wrapper listens to
those frames over ROS 2 and removes points outside a region of interest. It deletes isolated
noise points, centres and scales the cloud to fit a unit sphere, then samples exactly 1,024
evenly spread points: the fixed-size input a point-cloud classifier expects. Every stage is
timed and can be switched on or off from the config file.

- **Technical walkthrough:** [docs/architecture.md](docs/architecture.md)
- **Plain-English explanation:** [docs/code_review.md, Part 1](docs/code_review.md#part-1--how-the-wrapper-works-in-plain-english)

---

## Requirements

| | |
|---|---|
| **Compute** | NVIDIA Jetson Orin Nano, JetPack 6.x (Ubuntu 22.04) |
| **Sensor** | Livox Mid-70, Ethernet to the Jetson (host on `192.168.1.x`) |
| **Middleware** | ROS 2 Humble + [`livox_ros2_driver`](https://github.com/Livox-SDK/livox_ros2_driver) |
| **Python** | 3.8+ (3.10 on JetPack 6): `numpy`, `open3d`, `pyyaml`; `scikit-learn` optional |
| **Model runtime** | PyTorch from NVIDIA's JetPack wheel, not PyPI (needed once a classifier is added) |

Setting up a Jetson from scratch: **[docs/setup_guide.md](docs/setup_guide.md)**.

---

## Repository layout

```
wrapper/
├── README.md
├── pyproject.toml              package metadata, console script, pytest config
├── requirements.txt            pip dependencies for the Jetson
├── config/
│   └── pipeline_config.yaml    every tunable setting, with comments
├── scripts/
│   └── start_demo.sh           one-command start: ROS 2 → driver → pipeline
├── src/
│   └── livox_wrapper/
│       ├── livox_interface.py  ROS 2 subscriber → (N, 3) numpy frames
│       ├── pipeline.py         ROI → outliers → normalise → downsample
│       ├── cli.py              main loop; formats output for the classifier
│       └── __main__.py         enables `python -m livox_wrapper`
├── tests/
│   └── test_pipeline.py        unit tests (no LiDAR or ROS 2 needed)
└── docs/
    ├── setup_guide.md          one-time Jetson + ROS 2 + driver setup
    ├── demo_runbook.md         day-of checklist and troubleshooting
    ├── architecture.md         technical data-flow walkthrough
    ├── roadmap.md              remaining work, in priority order
    └── code_review.md          review findings and recommended fixes
```

---

## Quick start

### On the Jetson (after the one-time setup)

```bash
cd ~/wrapper
chmod +x scripts/start_demo.sh     # once; USB copies lose the executable bit
./scripts/start_demo.sh
```

The script sources ROS 2, starts `livox_ros2_driver` in the background and runs the pipeline.
Every 30 frames it logs per-stage latency:

```
2026-09-22 14:03:11,482 INFO livox_wrapper.pipeline: frame=30 roi=0.4ms outlier=11.8ms norm=0.3ms down=24.9ms total=37.4ms
```

Timing lines are the only regular output until a classifier is wired in. Press `Ctrl+C` to
stop. Day-of checklist and troubleshooting: [docs/demo_runbook.md](docs/demo_runbook.md).

### Running the pieces by hand

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# terminal 1: the driver (shows its errors, unlike the demo script)
ros2 launch livox_ros2_driver livox_lidar_launch.py

# terminal 2: the pipeline, from the repo root
PYTHONPATH=src:$PYTHONPATH python3 -m livox_wrapper --config config/pipeline_config.yaml
```

Or install the package once with `pip install -e .` and run `livox-wrapper --config config/pipeline_config.yaml`.

### Using the pipeline as a library

```python
import numpy as np
import yaml
from livox_wrapper import PointCloudPipeline

with open("config/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

pipeline = PointCloudPipeline(config)
points = np.load("frame.npy")              # (N, 3) or (N, 4+) array: x, y, z[, ...]
cloud, timings = pipeline.process(points)  # cloud: (1024, 3) float32
print(cloud.shape, f"{timings.total * 1000:.1f} ms")
```

---

## Configuration

Everything lives in [`config/pipeline_config.yaml`](config/pipeline_config.yaml). The settings you are most likely to change:

| Key | Default | What it does |
|---|---|---|
| `ingestion.ros2_topic` | `/livox/lidar` | PointCloud2 topic published by the driver |
| `roi_filter.mode` | `box` | `box` (x/y/z limits), `range` (radius) or `none` |
| `roi_filter.x_min` … `z_max` | ±10 m, z −2…3 m | Crop box, sized for indoor testing |
| `outlier_removal.method` | `open3d_statistical` | also `open3d_radius` (cheaper), `dbscan` (slow) or `none` |
| `downsampling.target_points` | `1024` | Points handed to the classifier |
| `downsampling.voxel_size_m` | `0.15` | Voxel pre-pass before FPS. **See known issue C1** |
| `output.format` | `numpy` | `numpy` or `torch` |
| `logging.log_every_n_frames` | `30` | How often per-stage timings are logged |

---

## Development

### Running the tests

The tests run on any machine. They need no LiDAR, ROS 2 or Open3D.

```bash
pip install numpy pyyaml pytest    # add scikit-learn to run the DBSCAN test too
python -m pytest
```

`pyproject.toml` puts `src/` on the test path, so no install step is needed.

### Line endings

`.gitattributes` forces LF line endings. Without it, Git on Windows (`core.autocrlf=true`)
checks out `start_demo.sh` with CRLF endings, and after copying to the Jetson it fails with
`/bin/bash^M: bad interpreter`. Keep editing on Windows as normal.

---

## Documentation

| Document | Read it when |
|---|---|
| [setup_guide.md](docs/setup_guide.md) | Setting up a Jetson from a blank SD card |
| [demo_runbook.md](docs/demo_runbook.md) | Running the demo, or something isn't working |
| [architecture.md](docs/architecture.md) | You need to understand or change the data flow |
| [roadmap.md](docs/roadmap.md) | Deciding what to work on next |
| [code_review.md](docs/code_review.md) | Full findings, with fixes and code snippets |

---

## Known issues

From the [code review](docs/code_review.md). These matter most before the demo:

| ID | Issue |
|---|---|
| [C1](docs/code_review.md#c1--the-voxel-downsample-runs-after-normalisation-so-its-units-are-wrong-88-of-the-points-fed-to-the-model-are-duplicates) | Voxel downsampling runs *after* normalisation, so `voxel_size_m` acts in unit-sphere units. In a simulated frame only ~121 real points survive, padded to 1,024 with duplicates. |
| [C2](docs/code_review.md#c2--the-wrapper-normalises-the-whole-scene-but-a-classification-network-classifies-single-objects) | The whole scene is normalised as one sample, but classifiers are trained on single objects. Needs a clustering step. |
| [C3](docs/code_review.md#c3--frame-staleness-is-checked-against-the-wrong-clock-it-can-silently-drop-100-of-frames) | The stale-frame check compares the sensor's clock with the Jetson's, which can silently drop every frame. |
| [C4](docs/code_review.md#c4--start_demosh-hides-the-drivers-output) | `start_demo.sh` discards the driver's output, so a failed driver launch looks like a quiet, healthy pipeline. |
| [I1](docs/code_review.md#i1--get_latest_frame-returns-the-oldest-frame-not-the-latest) | `get_latest_frame()` returns the oldest queued frame, not the newest. |
| [I2](docs/code_review.md#i2--an-exception-in-the-ingestion-thread-kills-ingestion-silently-the-main-loop-spins-forever) | Errors in the ingestion thread don't stop the main loop, which then waits silently for ever. |

## Choosing the classifier

The model is not chosen yet. The code review ([Part 3](docs/code_review.md#part-3--choosing-the-classifier))
recommends **starting with PointNet**: it is cheap, tolerant of sparse input and exports cleanly to
TensorRT. **PointNet++** is the planned upgrade. The only code that depends on the model is the
input-formatting block in [`cli.py`](src/livox_wrapper/cli.py), so switching models does not touch
the pipeline.
