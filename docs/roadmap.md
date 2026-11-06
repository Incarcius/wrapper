# Roadmap

Remaining work, in priority order. IDs such as **C1** and **I2** refer to findings in
[code_review.md](code_review.md), which includes a suggested fix for each.

| # | Work item | Needs hardware? | Status |
|---|---|---|---|
| 1 | [Bring up ROS 2 ingestion on the Jetson](#1-bring-up-ros-2-ingestion-on-the-jetson) | Yes | To do |
| 2 | [Fix the pre-demo issues](#2-fix-the-pre-demo-issues) | No | To do |
| 3 | [Choose and integrate the classifier](#3-choose-and-integrate-the-classifier) | Partly | To do |
| 4 | [Resolve Jetson (ARM64) dependencies](#4-resolve-jetson-arm64-dependencies) | Yes | To do |
| 5 | [Extend the test suite](#5-extend-the-test-suite) | No | To do |
| 6 | [Segment the scene into objects](#6-segment-the-scene-into-objects) | No | To do |
| 7 | [Prepare for flight](#7-prepare-for-flight) | Yes | Later |
| 8 | [Building-footprint cropping (optional)](#8-building-footprint-cropping-optional) | No | Later |

---

## 1. Bring up ROS 2 ingestion on the Jetson

The pipeline uses the `ros2` ingestion mode. The `sdk` mode is only a stub.

- [ ] ROS 2 Humble installed on the Jetson ([setup guide, Phase 3](setup_guide.md#phase-3-install-ros-2-humble-and-the-livox-driver)).
- [ ] `livox_ros2_driver` built in `~/ros2_ws`. It builds its own copy of the Livox SDK, so no
      separate SDK install is needed.
- [ ] `ros2 topic hz /livox/lidar` shows ~10 Hz ([setup guide, Phase 4](setup_guide.md#phase-4-verify-the-lidar-stream)).
- [ ] `rclpy` and `sensor_msgs_py` import in `python3` after sourcing ROS 2.

## 2. Fix the pre-demo issues

These can be done now without hardware, and none depends on the model choice.

- [ ] **C1** Reorder downsampling so the voxel pre-pass and FPS run before normalisation.
- [ ] **C3** Measure frame age with a local monotonic clock, not the sensor timestamp.
- [ ] **C4** Stop discarding the driver's output in `start_demo.sh`, and check it is still running.
- [ ] **I2** and **I4** Make silent failures loud: surface ingestion-thread errors, and skip
      inference on empty or near-empty clouds.
- [ ] **I8 (partial)** Publish the processed cloud on a ROS 2 topic so it can be inspected in RViz.

## 3. Choose and integrate the classifier

The model is **not decided**. Candidates include PointNet, PointNet++ and DGCNN. The code
review's [Part 3](code_review.md#part-3--choosing-the-classifier) recommends starting with
**PointNet**, which is cheap and exports cleanly to TensorRT, and trying **PointNet++ (SSG)** next.

- [ ] Choose the architecture, and decide how orientation will be handled (training
      augmentation or canonicalisation). None of the candidates is rotation-invariant.
- [ ] Obtain trained weights (`.pt` / `.pth`) and the model class definition.
- [ ] Make the model-input layout configurable: `(B, 3, N)` vs `(B, N, 3)` (**I5**).
- [ ] Load the model in [`cli.py`](../src/livox_wrapper/cli.py) and pass it `model_input`. Use
      `model.eval()` and `torch.inference_mode()`, and run ~10 warm-up frames before timing.
- [ ] Later: export to ONNX and TensorRT for Jetson inference speed.

## 4. Resolve Jetson (ARM64) dependencies

- [ ] `pip3 install -r requirements.txt` succeeds, including `open3d`. Prebuilt ARM64 wheels exist
      only for particular Python versions.
- [ ] PyTorch installed from **NVIDIA's JetPack wheel**, not PyPI.
      `python3 -c "import torch; print(torch.cuda.is_available())"` prints `True` (**C6**).
- [ ] `scikit-learn` only if you use `outlier_removal.method: dbscan`.

## 5. Extend the test suite

The current tests pass (`python -m pytest`). Gaps from review **N4**:

- [ ] A test that enables `voxel_size_m` and asserts the output has mostly unique points. This
      would have caught C1.
- [ ] A test for `process()` when filtering removes every point.
- [ ] Tests for the config-validation errors in `PointCloudPipeline.__init__`.
- [ ] A generous latency-budget test.

## 6. Segment the scene into objects

Classifiers are trained on single objects, but the pipeline currently normalises the whole
cropped scene as one sample (**C2**).

- [ ] Add a clustering stage (Open3D `cluster_dbscan`) after outlier removal, then normalise,
      sample and classify **per cluster**.
- [ ] Budget for *N* inferences per frame, one per object in view.
- [ ] Longer term, weigh this against a semantic-segmentation model that labels every point.

## 7. Prepare for flight

Not needed for a stationary indoor demo. Needed before flying.

- [ ] **Motion compensation.** Each frame spans 100 ms, so a moving drone smears the cloud. The
      Mid-70 has no built-in IMU, so this needs IMU data from the flight controller. Consider a
      LiDAR-inertial odometry front-end such as FAST-LIO with that external IMU.
- [ ] **Frame accumulation (I9).** The Mid-70's rosette scan fills in over time, so accumulating
      2–3 frames gives denser clouds. This needs motion compensation when moving.
- [ ] **Ground-plane removal (I10).** RANSAC plane segmentation, which is robust to drone attitude.
- [ ] **Publish results to ROS 2 (I8).** Classifications and pipeline health, for the rest of the
      drone's software.
- [ ] Widen the crop box for outdoor use ([setup guide note](setup_guide.md#classroom-testing-vs-outdoor-flight)).

## 8. Building-footprint cropping (optional)

The DALi-CGAN reference paper crops point clouds using building footprints.

- [ ] If needed for outdoor architecture scans, get footprint coordinates for the target
      buildings.
- [ ] For simple rectangles, feed the bounds into the existing `roi_filter` box. For real
      polygons, use a library such as `shapely` or `PDAL`.
