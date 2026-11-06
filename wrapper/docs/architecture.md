# Architecture

How a point cloud travels from the Livox Mid-70 to the classifier, and which code handles each
step. For a non-technical version, see [code_review.md, Part 1](code_review.md#part-1--how-the-wrapper-works-in-plain-english).

```
┌────────────┐  Ethernet  ┌───────────────────┐  /livox/lidar   ┌──────────────────────────────┐
│ Livox      │───────────▶│ livox_ros2_driver │────────────────▶│ LivoxInterface               │
│ Mid-70     │  UDP       │ (C++ ROS 2 node)  │  PointCloud2    │ ingestion thread             │
└────────────┘            └───────────────────┘  ~10 Hz         │ PointCloud2 → (N,3) float32  │
                                                                │ bounded queue, drop-oldest   │
                                                                └──────────────┬───────────────┘
                                                                               │ get_latest_frame()
                                                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ PointCloudPipeline.process()                          main thread (cli.py loop)              │
│                                                                                              │
│  1 roi_filter ──▶ 2 remove_outliers ──▶ 3 normalize ──▶ 4 voxel pre-pass + FPS               │
│   box / range      statistical /         centre +         fixed point count                  │
│                    radius / DBSCAN       unit sphere      (target_points = 1024)             │
└──────────────────────────────────────────────────────────────────────────────┬───────────────┘
                                                                               │ (1024, 3) float32
                                                                               ▼
                                                             model-input formatting (cli.py)
                                                             (1, 3, 1024) → classifier (TBD)
```

---

## Modules

| Module | Responsibility |
|---|---|
| [`livox_interface.py`](../src/livox_wrapper/livox_interface.py) | `LivoxInterface`: subscribes to the driver's topic on a background thread and hands frames to the main loop |
| [`pipeline.py`](../src/livox_wrapper/pipeline.py) | `PointCloudPipeline`: the four processing stages, plus per-stage timing (`StageTimings`) |
| [`cli.py`](../src/livox_wrapper/cli.py) | `main()`: loads config, starts ingestion, runs the processing loop, formats output for the model |
| [`config/pipeline_config.yaml`](../config/pipeline_config.yaml) | Every threshold and on/off switch |

---

## 0. The sensor and driver

The Mid-70 does **not** spin. It steers its laser through rotating prisms in a
**non-repetitive rosette pattern** across a 70.4° circular field of view, at about 100,000
points per second. Consecutive frames trace *different* paths, so coverage improves the
longer you integrate. A single 100 ms frame is fairly sparse.

[`livox_ros2_driver`](https://github.com/Livox-SDK/livox_ros2_driver) (C++, not part of this
repository) talks to the sensor over UDP. It packages every 100 ms of points (the default
`publish_freq` is 10 Hz) into a `sensor_msgs/PointCloud2` message on `/livox/lidar`, so each
message holds about 10,000 points.

## 1. Ingestion: `LivoxInterface`

- `start()` spawns a daemon thread running `_run_ros2()`. That thread creates the
  `livox_pipeline_ingest` node, subscribes to `ingestion.ros2_topic`, and spins in
  50 ms steps until `stop()` is called.
- The callback `_cb()` extracts the `x`, `y`, `z` fields with
  `sensor_msgs_py.point_cloud2.read_points_numpy`. There is no per-point Python loop. The result
  is an `(N, 3)` float32 array.
- `_push_frame()` does two things:
  1. **Drops stale frames.** It compares the message's header timestamp with the Jetson's wall
     clock and drops frames older than `drop_stale_after_sec` (0.15 s).
     *Known issue [C3](code_review.md#c3--frame-staleness-is-checked-against-the-wrong-clock-it-can-silently-drop-100-of-frames):
     these are two different clocks.*
  2. **Queues without blocking.** The queue holds `frame_queue_size` (2) frames. When it is full,
     the oldest frame is discarded. On a moving platform a late frame is worse than a missing one.
- The main loop calls `get_latest_frame(timeout=1.0)`.
  *Known issue [I1](code_review.md#i1--get_latest_frame-returns-the-oldest-frame-not-the-latest):
  this returns the oldest queued frame, not the newest.*

`mode: "sdk"` (direct Livox SDK bindings) exists only as a stub. It raises `NotImplementedError`.

## 2. Processing: `PointCloudPipeline.process()`

The input is cast to float32, and any columns beyond x, y, z are dropped. Stages run cheapest
first, so that each stage shrinks the work for the next.

| # | Stage | Method | Config section | What it does |
|---|---|---|---|---|
| 1 | ROI crop | `roi_filter()` | `roi_filter` | Keeps points inside an axis-aligned box (default ±10 m × ±10 m, z −2…3 m) or within a radius (`range` mode, 40 m) |
| 2 | Outlier removal | `remove_outliers()` | `outlier_removal` | Removes isolated noise. `open3d_statistical` (default) drops points whose mean neighbour distance is unusually large; `open3d_radius` drops points with too few neighbours; `dbscan` keeps only points in dense clusters |
| 3 | Normalisation | `normalize()` | `normalization` | Subtracts the centroid, then divides by the largest distance from it, so the cloud fits a unit sphere |
| 4 | Downsampling | voxel pre-pass + `farthest_point_sampling()` | `downsampling` | If more than 3 × `target_points` remain, voxel-downsamples first to bound the cost. Farthest Point Sampling (FPS) then picks exactly `target_points` evenly spread points. If too few points remain, it pads by resampling with replacement. If none remain, it returns zeros |

> **Known issue [C1](code_review.md#c1--the-voxel-downsample-runs-after-normalisation-so-its-units-are-wrong-88-of-the-points-fed-to-the-model-are-duplicates):**
> the voxel pre-pass in stage 4 runs *after* stage 3 normalises the cloud, so
> `voxel_size_m: 0.15` means 15% of the scene radius, not 15 cm. In a simulated classroom-sized
> frame only ~121 unique points survive, and they are padded to 1,024.

`process()` returns `(result, timings)`:

- `result` is a `(target_points, 3)` float32 array, or a torch tensor if `output.format: "torch"`.
- `timings` is a `StageTimings` holding seconds for each stage and the total. Every
  `log_every_n_frames` frames, the pipeline logs:

```
INFO livox_wrapper.pipeline: frame=30 roi=0.4ms outlier=11.8ms norm=0.3ms down=24.9ms total=37.4ms
```

## 3. Model-input formatting (`cli.py`)

The `(1024, 3)` result is transposed and given a batch dimension: `(1, 3, 1024)`, i.e.
`(Batch, Channels, NumPoints)`. That is the layout used by the PyTorch reference implementations
of PointNet, PointNet++ and DGCNN. Some libraries (Open3D-ML, torch-points3d) expect
`(1, 1024, 3)` instead. This block is the only model-specific code in the repository.

## 4. Classification (not yet implemented)

The classifier is not chosen. The code review recommends PointNet to start, with PointNet++ as
the upgrade path ([code_review.md, Part 3](code_review.md#part-3--choosing-the-classifier)).

> **Known issue [C2](code_review.md#c2--the-wrapper-normalises-the-whole-scene-but-a-classification-network-classifies-single-objects):**
> the pipeline normalises the whole cropped scene (floor, walls, objects) as one sample. Standard
> classifiers are trained on single isolated objects (ModelNet40-style), so a clustering step is
> needed to split the scene into objects before classification.

---

## Threading and timing

- **Two threads.** The ingestion thread only converts and queues frames. All processing runs on
  the main thread. Both threads share Python's GIL, so a slow callback can delay processing and
  the reverse. The per-stage timing log is the first place to look if latency is unexpectedly
  high.
- **Latency budget.** At 10 Hz a new frame arrives every 100 ms. If `total` in the timing log
  regularly exceeds ~100 ms, the queue drops frames. That is by design, but it lowers the
  effective frame rate.
- **Shutdown.** `Ctrl+C` interrupts the main loop. `finally: ingest.stop()` sets the stop event
  and joins the ingestion thread, waiting at most 2 s.
