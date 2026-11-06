# Livox Mid-70 Wrapper — Review & Plain-English Summary

**Reviewed:** 20 Sep 2026 · commit `4cbd903` · ~1,000 lines across 3 Python modules, 1 config, 1 shell script, 4 docs
**Target hardware:** Jetson Orin Nano (JetPack 6 / Ubuntu 22.04), Livox Mid-70, ROS 2 Humble
**Downstream model:** *undecided* — PointNet, PointNet++, DGCNN or another point-cloud classifier.
The code currently hardcodes DGCNN's `(1, 3, 1024)` convention; see [Part 3](#part-3--choosing-the-classifier) for what the choice does and doesn't change.

> **Updated 22 Sep 2026: the repository was reorganised after this review.** Paths and links now
> point to the new layout (`src/livox_wrapper/`, `scripts/`, renamed docs). Line numbers in
> `pipeline.py` and `livox_interface.py` are unchanged. Findings fully or partly resolved by the
> reorganisation are marked **✅**. Three factual statements in the original review were wrong and
> are corrected in place: the `livox_ros_driver2` advice (C4), the `sensor_msgs_py`-in-Foxy claim
> (C5), and the `/livox/imu` remarks (the Mid-70 has no built-in IMU). One new finding, **C7**
> (CRLF line endings), was found and fixed during the reorganisation.

---

# Part 1 — How the wrapper works, in plain English

## The one-paragraph version

The Livox Mid-70 is a laser scanner that fires out beams and records where they bounce back, producing a
cloud of about ten thousand 3D dots, ten times a second. A neural network can't eat that directly — it's
messy, it's a different size every time, and it's measured in metres instead of the tidy numbers the
network was trained on. This wrapper is the **prep kitchen** that sits between the two: it takes the raw
basket of dots, throws away the ones that are too far away, picks out the floating specks of dust and
noise, re-centres and shrinks everything to a standard size, and finally hands the network a perfectly
uniform tray of exactly 1,024 dots. It has to do all of that in under about 100 milliseconds, every
time, on a small computer bolted to a drone.

## The journey of one frame, step by step

**0 · The scanner speaks.** The Mid-70 is plugged into the Jetson over Ethernet. A separate piece of
software (the Livox ROS 2 driver, written in C++, not part of this wrapper) does the actual talking to
the hardware and broadcasts each batch of dots on a channel called `/livox/lidar`. Think of it as a radio
station that publishes a new "dot picture" ten times a second.

**1 · Tuning in.** [`livox_interface.py`](../src/livox_wrapper/livox_interface.py) is the radio receiver. It runs quietly
in the background on its own thread, listening to that channel and converting each broadcast from the
ROS 2 message format into a plain table of numbers — one row per dot, three columns (X, Y, Z). It drops
frames onto a very short conveyor belt (two slots). If the rest of the system is busy and the belt fills
up, it *throws the old frame away* rather than queueing it. On a moving drone, a point cloud from half a
second ago is worse than no point cloud at all, so this is deliberate.

**2 · Throwing away what's out of reach.** [`pipeline.py`](../src/livox_wrapper/pipeline.py) picks the frame up. First
and cheapest job: delete every dot outside an imaginary box around the drone (currently a ±10 m box, set
small for indoor testing). This is done first on purpose — the fewer dots survive this step, the less work
every later step has to do.

**3 · Cleaning off the noise.** LiDARs see things that aren't there: dust, rain, a single stray return off
a reflective surface. The wrapper looks at every dot and asks "do you have neighbours?" A dot floating on
its own in empty space gets deleted; a dot that's part of a dense surface (a wall, a chair) is kept.

**4 · Standardising the position and size.** The network was trained on objects that sit neatly at the
origin and fit inside a sphere of radius 1. A real scan of a chair sitting 10 m away at eye height looks
nothing like that. So the wrapper computes the average position of all the dots, slides everything so that
average lands on (0, 0, 0), then measures the furthest dot and shrinks everything until that dot sits
exactly on the radius-1 sphere.

**5 · Cutting the count to exactly 1,024.** The network demands exactly 1,024 dots — no more, no fewer.
Rather than picking at random (which could delete an entire chair leg and leave two clumps), the wrapper
uses **Farthest Point Sampling**: pick a dot, then repeatedly pick whichever remaining dot is furthest
from everything already chosen. The result is 1,024 dots spread evenly over the shape, preserving its
silhouette.

**6 · Handing it over.** [`cli.py`](../src/livox_wrapper/cli.py) takes the finished 1,024×3 table,
flips it on its side and wraps it in a "batch of one" to match the shape PyTorch expects —
`(1, 3, 1024)` — and that's the point where the classifier gets plugged in. *(Currently a TODO; the
model call itself isn't wired up yet, and the model itself is still undecided.)*

Throughout, a stopwatch runs on every stage, and every 30th frame the wrapper prints how many milliseconds
each step took. That's your tuning dial for the Jetson.

## What each file is for

| File | Plain-English job |
|---|---|
| `src/livox_wrapper/livox_interface.py` | The receiver. Listens to the LiDAR, hands over frames, drops old ones. |
| `src/livox_wrapper/pipeline.py` | The prep kitchen. Crop → clean → centre & scale → cut to 1,024 dots. |
| `src/livox_wrapper/cli.py` | The manager. Starts everything, loops forever, formats for PyTorch. |
| `config/pipeline_config.yaml` | The dials. Every threshold, on/off switch and size, in one place. |
| `scripts/start_demo.sh` | The big red button. Sources ROS 2, starts the driver, starts the pipeline. |
| `tests/test_pipeline.py` | Sanity checks you can run on your laptop without a LiDAR. |
| `docs/` | Setup runbooks for the Jetson and the demo. |

## What's genuinely good about it

- **The stage order is right.** Cheapest filter first, most expensive last. That's the correct instinct
  for a latency-bound embedded system.
- **Everything is a config knob.** You can profile and tune on-device without editing Python.
- **Per-stage timing is built in from day one.** Most people bolt this on after the first performance
  panic.
- **Drop-oldest instead of queue-up.** The right call for a flying platform, and it's documented as a
  deliberate choice rather than an accident.
- **Graceful degradation is at least thought about** — missing Open3D, missing scikit-learn and empty
  clouds all have explicit handling paths.
- **The runbooks are unusually good.** Step-by-step, beginner-safe, and they flag the Rufus/BalenaEtcher
  and JetPack 7 traps that actually bite people.

---

# Part 2 — Findings and improvements

Ranked by impact. Line references are to the current layout (identical to the reviewed commit for
`pipeline.py` and `livox_interface.py`).

## 🔴 Critical

### C1 — The voxel downsample runs *after* normalisation, so its units are wrong. ~88% of the points fed to the model are duplicates.

[`pipeline.py:228-236`](../src/livox_wrapper/pipeline.py#L228)

`voxel_size_m: 0.15` is documented and named as **metres**. But by the time it's applied, `normalize()`
has already shrunk the entire scene into a unit sphere. In that space 0.15 is not 15 cm — it's 15% of the
whole scene's radius.

I simulated a realistic ~10,000-point Mid-70 frame of a classroom-sized scene:

```
raw frame points                       : 10000
unit-sphere scale factor               : 13.03 m
"0.15 m" after normalisation is really : 1.95 real metres
points surviving voxel (current order) : 121
  -> FPS pads 121 -> 1024 by resampling WITH REPLACEMENT
  -> ~121 unique points (12%), ~903 duplicates (88%)

points surviving voxel (correct order) : 6175  -> real FPS to 1024, 0 duplicates
```

This is silent — nothing errors, nothing warns, the shape is still `(1024, 3)`. But the model receives a
121-point cloud smeared into 1,024 slots.

**This is critical regardless of which classifier you pick**, though the damage lands differently. I
measured both effects on the actual 121-unique/1024-row cloud:

| Model family | What duplicates do | Measured |
|---|---|---|
| **PointNet** | Global max-pool is **duplicate-invariant** — the global feature vector is bit-identical to what the 121 unique points alone would produce. No degenerate math… but the model genuinely only ever sees 121 points of geometry. | global feature identical to the 121-point cloud: **True** |
| **PointNet++** | Ball query / grouping fills local neighbourhoods with copies of the same point, so local features are computed over near-constant sets. Its internal FPS also re-picks duplicates. | — |
| **DGCNN** | EdgeConv's kNN finds duplicates at **zero distance**, so edge features `x_j - x_i` are exactly zero for those neighbours. | **8.3 of every 20 neighbours (41%) are zero-distance duplicates** |

So: PointNet degrades *gracefully* (it's just a sparse cloud), while PointNet++ and DGCNN degrade
*additionally* through corrupted neighbourhoods. In every case you've thrown away 98% of your real
geometry before the model ever sees it, and nothing in the logs says so.

> Correcting my own earlier framing: I described DGCNN's kNN graph as "collapsing." It doesn't fully
> collapse — no point's entire neighbourhood is duplicates (0 of 1024). ~41% of edges going dead is the
> accurate figure. Still serious, but less catastrophic than "collapse" implies.

**Fix — reorder the last two stages.** Voxel-downsample and FPS in metric space, then normalise the
surviving 1,024 points:

```python
# in process(), replace the normalize -> downsample block with:
t = time.perf_counter()
target_points = self.cfg["downsampling"]["target_points"]
voxel_size = self.cfg["downsampling"].get("voxel_size_m")
if voxel_size and _HAS_OPEN3D and len(pts) > target_points * 3:
    pcd = self._to_o3d(pts)
    pts = np.asarray(pcd.voxel_down_sample(voxel_size=voxel_size).points)
pts = self.farthest_point_sampling(pts, target_points)
timings.downsampling = time.perf_counter() - t

t = time.perf_counter()
pts = self.normalize(pts)          # now on 1024 points, in metric space
timings.normalization = time.perf_counter() - t
```

Two bonus wins: normalisation now costs ~nothing (1,024 points instead of ~10,000), and the centroid/scale
are computed on **exactly the points the model sees**, which is what ModelNet-style training does for
all three candidate architectures.

---

### C2 — The wrapper normalises the *whole scene*, but a classification network classifies *single objects*.

[`pipeline.py:146`](../src/livox_wrapper/pipeline.py#L146) · architectural

The original `docs/pipeline_flow.txt` (now [architecture.md](architecture.md), corrected) said the output is "98% confident it is a Chair." But nothing in the pipeline ever
isolates a chair. The ROI box is a ±10 m room; everything inside it — floor, walls, desks, the chair — gets
centred and squashed into one unit sphere together and handed to the model as a single sample.

**This is model-independent.** PointNet, PointNet++ and DGCNN classification heads are *all* trained on
isolated objects (ModelNet40-style), so whichever you pick will be asked to name "a room" using a
vocabulary of chairs and tables. Switching architectures does not fix this; it's a pipeline-shape problem.

The missing step is **instance clustering**: split the cleaned scene into separate blobs, then run
normalise → FPS → classify **per blob**.

The sharpest part of this: you're already computing exactly what you need and throwing it away.
`remove_outliers_dbscan` runs DBSCAN and then does `return work_points[labels != -1]` — it keeps only
whether a point was noise, and discards the cluster IDs that *are* your object instances.

```python
def segment_objects(self, points, eps=0.5, min_pts=10, min_cluster_size=50):
    """Split a cleaned scene into per-object clouds. Reuses the labels DBSCAN
    already computes — currently discarded in remove_outliers_dbscan()."""
    labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(points[:, :3])
    return [points[labels == c] for c in np.unique(labels)
            if c != -1 and (labels == c).sum() >= min_cluster_size]
```

Open3D's `cluster_dbscan` is the C++ equivalent and is much faster than scikit-learn's on the Jetson —
worth using instead for the real-time path.

**Since the model is still open, there's a second way out of this.** You have a genuine architectural
fork, and it's worth deciding deliberately rather than by default:

| Approach | Pipeline shape | Trade-off |
|---|---|---|
| **A. Cluster → classify** (current direction) | DBSCAN splits the scene → normalise + FPS **per cluster** → classification net per cluster | Keeps the simple ModelNet-style model. But cost scales with object count (*N* clusters = *N* inferences per frame — a real latency risk on the Orin), and DBSCAN merges touching objects. |
| **B. Semantic segmentation instead** | Feed the whole scene, get a label **per point**, no clustering needed | Architecturally cleaner for scene understanding and naturally handles touching objects. Needs a seg model (PointNet++ seg, RandLA-Net, KPConv, Cylinder3D) and scene-level training data, which is far more expensive to label than object crops. |

Approach A is the right call for a proof-of-work — it's less work and the groundwork is already written.
But if the end goal is "understand a city block from a drone," B is where this ends up, and knowing that
now may change how you spend labelling effort.

If per-object classification is genuinely out of scope for the proof-of-work, that's a fine decision —
but then the docs should say "classifies the scene", not "98% confident it is a Chair", so nobody is
surprised at the demo.

---

### C3 — Frame-staleness is checked against the wrong clock; it can silently drop 100% of frames.

[`livox_interface.py:65-68`](../src/livox_wrapper/livox_interface.py#L65)

```python
drop_after = self.cfg.get("drop_stale_after_sec", 0.15)
if time.time() - timestamp > drop_after:
    return
```

`timestamp` comes from `msg.header.stamp` — the **ROS/sensor clock**. `time.time()` is the **Jetson's wall
clock**. Livox drivers can stamp with the LiDAR's own internal clock (or a PTP/GPS clock). If those two
clocks differ by more than 0.15 s — which needs no NTP problem at all, just an unsynced device clock —
then *every single frame is dropped*, forever, with only a `logger.debug` to show for it. Your demo shows
nothing and the logs look clean at INFO level.

Even when the clocks agree, the margin is too thin: the Livox driver publishes at 10 Hz, and the header
stamp is typically the **start** of the 100 ms accumulation window, so a freshly-arrived frame is already
~100 ms "old" by this measure. 0.15 s leaves 50 ms of slack, and normal DDS jitter will eat that.

**Fix — measure age from local receive time, and keep the sensor stamp only as metadata:**

```python
def _push_frame(self, points, sensor_stamp):
    recv_time = time.monotonic()          # monotonic: immune to NTP steps
    ...
    # in the consumer, age = time.monotonic() - recv_time
```

and raise `drop_stale_after_sec` to ~0.25 s (2.5 frame periods). If you want true sensor-clock staleness
later, sync it properly (PTP) and check both.

---

### C4 — `start_demo.sh` hides the driver's output.

[`scripts/start_demo.sh:57`](../scripts/start_demo.sh#L57) · **✅ partly resolved:** the package name was fixed in the reorganisation

As reviewed, the script launched **`livox_ros_driver`** (the ROS 1 package name) while the setup runbook
built **`livox_ros2_driver`**, so the launch would have failed with "package not found". **The
reorganisation fixed the name.** It is now `DRIVER_PKG="livox_ros2_driver"` in the script's config block.

The underlying problem remains:

```bash
ros2 launch "$DRIVER_PKG" "$DRIVER_LAUNCH" > /dev/null 2>&1 &
```

`> /dev/null 2>&1` throws away everything the driver says. Suppose it fails for any reason: no LiDAR
found, network misconfigured, workspace not sourced. The script still prints
`[INFO] Driver started in background (PID: …)`, sleeps 3 s and starts Python. Python then sits in
`get_latest_frame(timeout=1.0)` returning `None` for ever. **A silent, confusing failure at exactly
the worst moment.** Fix: log to a file, and check that the driver is alive and publishing before
starting Python:

```bash
DRIVER_LOG="/tmp/livox_driver.log"

ros2 launch "$DRIVER_PKG" "$DRIVER_LAUNCH" > "$DRIVER_LOG" 2>&1 &
DRIVER_PID=$!
sleep 3
if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    echo "[ERROR] Driver died on startup. Last lines of $DRIVER_LOG:"
    tail -20 "$DRIVER_LOG"
    exit 1
fi
# confirm data is actually flowing before starting Python
timeout 5 ros2 topic echo /livox/lidar --once > /dev/null \
    || { echo "[ERROR] No data on /livox/lidar after 5s."; tail -20 "$DRIVER_LOG"; exit 1; }
```

> **Correction to the original review:** it suggested Mid-70 units might need `livox_ros_driver2`.
> That was wrong. `livox_ros_driver2` supports only Livox SDK 2 devices (HAP, Mid-360 and similar),
> not the Mid-70. **`livox_ros2_driver` is the right driver**; it supports the Mid-40, Mid-70,
> Tele-15, Horizon and Avia.

---

### C5 — The default `ROS2_DISTRO="foxy"` cannot run this code.

[`scripts/start_demo.sh:13`](../scripts/start_demo.sh#L13) · **✅ Resolved in the reorganisation:** the default is now `ROS2_DISTRO="${ROS_DISTRO:-humble}"`

`livox_interface.py` calls `sensor_msgs_py.point_cloud2.read_points_numpy`. Foxy's `sensor_msgs_py`
has no `read_points_numpy`; Humble's does (checked against both branches of `ros2/common_interfaces`).
Foxy also doesn't support Ubuntu 22.04, which is what JetPack 6 runs. Meanwhile the setup runbook
correctly installed **Humble** and told you to hand-edit line 11, a manual step that's easy to forget
under demo pressure.

> **Correction to the original review:** it said `sensor_msgs_py` doesn't exist in Foxy. It does;
> only `read_points_numpy` is missing. The conclusion is unchanged.

**Fix (applied):** default to `humble`, but use an already-sourced ROS environment if there is one:

```bash
ROS2_DISTRO="${ROS_DISTRO:-humble}"   # honour an already-sourced environment
```

Also add a fallback in the callback so older `sensor_msgs_py` builds still work:

```python
try:
    from sensor_msgs_py.point_cloud2 import read_points_numpy
except ImportError:
    from sensor_msgs_py.point_cloud2 import read_points
    def read_points_numpy(msg, field_names=None, skip_nans=True):
        return np.array(list(read_points(msg, field_names=field_names,
                                         skip_nans=skip_nans)), dtype=np.float32)
```

---

### C6 — `pip3 install -r requirements.txt` will give you a broken PyTorch on the Jetson.

[`requirements.txt`](../requirements.txt) · **✅ partly resolved:** torch was removed from `requirements.txt` in the reorganisation; the GPU check below is still open

As reviewed, `requirements.txt` listed `torch>=2.0`, and both runbooks said to `pip install` it. PyPI's torch is not the Jetson build. On ARM64 pip will either fail outright or install a
CPU-only wheel — so your model runs on the Orin's CPU while the GPU sits idle, and you spend the demo
wondering why inference takes 800 ms. `open3d` has the same class of problem (patchy ARM64 wheel coverage;
falling back to a source build takes hours).

**Fix (applied):** treat torch the way you already, correctly, treat `rclpy` — comment it out with a pointer, so the
one-liner install can't silently do the wrong thing:

```
numpy>=1.23
open3d>=0.18       # ARM64: check for a prebuilt wheel first, source builds take hours
scikit-learn>=1.2  # only if outlier_removal.method == "dbscan"
pyyaml>=6.0
#
# torch: do NOT pip-install on Jetson. Use the NVIDIA JetPack wheel:
#   https://forums.developer.nvidia.com/t/pytorch-for-jetson
# A PyPI torch on ARM64 is CPU-only — your model will not touch the GPU.
```

**Still open:** add a one-line GPU check to `scripts/start_demo.sh`, so you find out in 2 seconds rather than mid-demo:

```bash
$PYTHON_EXEC -c "import torch; assert torch.cuda.is_available(), 'torch has no CUDA — wrong wheel'" \
  || { echo "[ERROR] PyTorch cannot see the GPU. Install the NVIDIA JetPack wheel."; exit 1; }
```

### C7 — Every file is checked out with CRLF line endings, including the shell script.

**✅ Resolved in the reorganisation.** *This is a new finding; it was not in the original review.*

The repository is cloned on Windows with `core.autocrlf=true`, so Git writes every file to disk with
Windows (CRLF) line endings. That includes `start_demo.sh`, confirmed byte by byte. The runbooks say
to copy the folder to the Jetson by USB, and on Linux a CRLF shebang fails immediately:

```
/bin/bash^M: bad interpreter: No such file or directory
```

Even `bash start_demo.sh` would break, because every variable picks up a trailing `\r`, and paths
like `/opt/ros/humble\r/setup.bash` don't exist.

**Fixed:** a `.gitattributes` now forces LF endings for all text files, whatever the local Git
settings, and the working tree has been converted to LF. The demo runbook's troubleshooting table
explains how to repair a copy that's already affected.

---

## 🟠 Important

### I1 — `get_latest_frame()` returns the *oldest* frame, not the latest.

[`livox_interface.py:83-87`](../src/livox_wrapper/livox_interface.py#L83)

`queue.Queue.get()` is FIFO. With `maxsize=2` you can be handed a frame that's a full period old. This
directly contradicts the module docstring's promise ("a stale point cloud is often worse than a missing
one") — the producer carefully drops old frames, and then the consumer asks for the oldest survivor.
Verified:

```
q.put("frame_1"); q.put("frame_2");  q.get()  ->  "frame_1"   # oldest
```

**Fix — drain to the newest:**

```python
def get_latest_frame(self, timeout: float = 1.0):
    try:
        frame = self._queue.get(timeout=timeout)
    except queue.Empty:
        return None
    while True:                       # discard anything newer that arrived meanwhile
        try:
            frame = self._queue.get_nowait()
        except queue.Empty:
            return frame
```

Either that, or rename the method to `get_next_frame()` so the behaviour matches the name.

---

### I2 — An exception in the ingestion thread kills ingestion silently; the main loop spins forever.

[`livox_interface.py:92-102`](../src/livox_wrapper/livox_interface.py#L92)

Both `_run_ros2` and `_run_sdk` raise `RuntimeError` on missing dependencies. That exception unwinds a
**daemon thread** — the main loop in `cli.py` never sees it and just keeps calling
`get_latest_frame()`, getting `None`, and `continue`-ing. Forever. Note `_run_sdk` currently ends in an
unconditional `NotImplementedError`, so `mode: "sdk"` fails exactly this way today.

**Fix — record the failure and let the consumer see it:**

```python
def __init__(self, ...):
    ...
    self.error: Optional[BaseException] = None

def _run_guarded(self, fn):
    try:
        fn()
    except BaseException as e:            # noqa: BLE001 - must not vanish into a daemon thread
        logger.exception("Ingestion thread died")
        self.error = e
        self._stop_event.set()

def is_alive(self) -> bool:
    return self._thread is not None and self._thread.is_alive() and self.error is None
```

and in `cli.py`'s loop:

```python
if not ingest.is_alive():
    raise RuntimeError("Ingestion stopped") from ingest.error
if frame is None:
    consecutive_empty += 1
    if consecutive_empty == 5:
        logger.warning("No frames for 5s — is the driver publishing on %s?", topic)
    continue
```

---

### I3 — Default QoS (reliable, depth 10) works against the drop-stale design.

[`livox_interface.py:109-111`](../src/livox_wrapper/livox_interface.py#L109)

```python
self.create_subscription(PointCloud2, parent.cfg["ros2_topic"], self._cb, 10)
```

Passing `10` gives you the default **reliable** profile with a depth-10 queue. Two consequences on a drone:
DDS will retransmit lost point-cloud packets (pointless for a sensor stream where the next frame is 100 ms
away and better), and it will buffer up to **10 frames = 1 full second** of point clouds *before* your
carefully-bounded 2-slot queue ever sees them. You've built a 2-frame buffer sitting behind a hidden
10-frame one.

**Fix — sensor-data QoS, depth 1:**

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1)
self.create_subscription(PointCloud2, parent.cfg["ros2_topic"], self._cb, qos)
```

**Related Jetson deployment note:** PointCloud2 messages exceed the 64 KB UDP datagram limit and get
fragmented. Default kernel socket buffers will drop fragments under load, which shows up as
intermittently-empty or truncated clouds. Add to the runbook:

```bash
sudo sysctl -w net.core.rmem_max=2147483647   # persist in /etc/sysctl.d/ for boot
```

---

### I4 — Empty and near-empty clouds become all-zeros and get classified anyway.

[`pipeline.py:176-182`](../src/livox_wrapper/pipeline.py#L176)

If ROI + outlier removal wipes out the frame, FPS returns `np.zeros((1024, 3))` and `process()` returns it
with no indication anything went wrong. The model will confidently classify a cloud of 1,024 coincident
points at the origin. Same story, less obviously, when only 30 points survive and get padded to 1,024 with
replacement.

**Fix — return a quality signal and let the caller gate inference.** Add to `StageTimings` (or a small
`FrameStats`):

```python
@dataclass
class StageTimings:
    ...
    n_raw: int = 0
    n_after_filters: int = 0
    valid: bool = True        # False -> caller should skip inference
```

set `timings.valid = n_after_filters >= cfg.get("min_points_for_inference", 256)`, add that key to the
config, and in `cli.py`:

```python
if not timings.valid:
    logger.warning("Frame %d: only %d points survived — skipping inference.",
                   frame_no, timings.n_after_filters)
    continue
```

This also gives you the diagnostic you'll want at the demo: *"the ROI box is too tight"* is instantly
visible as `n_raw=9800 n_after_filters=12` instead of as mysteriously bad predictions.

---

### I5 — The model adapter is hardcoded to one framework's convention, and produces a non-contiguous tensor.

[`cli.py:59-63`](../src/livox_wrapper/cli.py#L59)

Two problems in three lines, and the second one matters more now that the model is undecided.

**(a) It's not contiguous.**

```python
model_input = np.expand_dims(processed.T, axis=0)   # C_CONTIGUOUS: False  (verified)
```

A transpose is a view with permuted strides. Many CUDA kernels — including the kNN, ball-query and gather
ops used by DGCNN and PointNet++ — either require contiguous input or force a hidden copy every frame.
Nothing here moves the tensor to the GPU either.

**(b) The `(B, 3, N)` layout is an assumption, not a standard.** It happens to be right for the common
PyTorch PointNet / PointNet++ / DGCNN reference implementations (they all use `Conv1d` over the point
axis), but **Open3D-ML, torch-points3d and most `torch_geometric` models expect `(B, N, 3)`** instead.
Right now switching model family means editing Python. It should be a config line.

**Fix — a small adapter driven by config, so the wrapper stays model-agnostic:**

```yaml
model:
  input_layout: "bcn"      # "bcn" = (B,3,N) for PointNet/PointNet++/DGCNN refs
                           # "bnc" = (B,N,3) for Open3D-ML / torch-points3d / PyG
  device: "cuda"
```

```python
def to_model_input(pts, layout="bcn", device="cuda"):
    """pts: (N,3) numpy or torch -> batched, contiguous, on-device tensor."""
    t = torch.as_tensor(np.ascontiguousarray(pts)) if isinstance(pts, np.ndarray) else pts
    t = t.unsqueeze(0)                                  # (1, N, 3)
    if layout == "bcn":
        t = t.transpose(1, 2)                           # (1, 3, N)
    return t.contiguous().to(device, non_blocking=True)
```

That one function is the entire coupling between this wrapper and the model. Keeping it that small is
what lets you A/B PointNet against PointNet++ without touching the pipeline.

While you're there, whichever model you land on: `model.eval()`, wrap inference in
`torch.inference_mode()`, and **run 10 warm-up frames before you start timing** — the first CUDA call on a
Jetson can take seconds of context/kernel setup and will otherwise look like a catastrophic latency spike
in your first timing log.

---

### I6 — Config is read with bare `[...]` deep inside the per-frame path.

[`pipeline.py:248`](../src/livox_wrapper/pipeline.py#L248) and throughout

`self.cfg["logging"]`, `self.cfg["output"]["format"]`, `cfg["x_min"]` and friends are indexed directly. A
config missing a section doesn't fail at startup — it fails on the first frame that reaches that line,
which for `"logging"` is frame 1 but for a rarely-taken branch could be much later. On a drone, "crashed
at frame 3,000" is a much worse failure mode than "refused to start."

**Fix — validate and fill defaults once, in `__init__`:**

```python
DEFAULTS = {
    "roi_filter":       {"enabled": True, "mode": "none"},
    "outlier_removal":  {"method": "none"},
    "normalization":    {"center": True, "scale_to_unit_sphere": True},
    "downsampling":     {"target_points": 1024, "method": "open3d_fps", "voxel_size_m": None},
    "output":           {"format": "numpy", "dtype": "float32"},
    "logging":          {"log_stage_timings": False, "log_every_n_frames": 30},
}

def _merge_defaults(cfg):
    out = {k: {**v, **cfg.get(k, {})} for k, v in DEFAULTS.items()}
    if out["roi_filter"].get("mode") == "box":
        missing = [k for k in ("x_min","x_max","y_min","y_max","z_min","z_max")
                   if k not in out["roi_filter"]]
        if missing:
            raise ValueError(f"roi_filter.mode='box' requires: {missing}")
    return out
```

Unknown keys should warn too — a typo like `voxel_size` instead of `voxel_size_m` currently just silently
disables the stage.

---

### I7 — Three separate NumPy↔Open3D float64 conversions per frame.

[`pipeline.py:264-268`](../src/livox_wrapper/pipeline.py#L264)

`_to_o3d` is called independently by outlier removal, the voxel downsample and FPS. Each call allocates a
fresh `PointCloud` and copies the whole array up to float64 — on ~10,000 points that's three ~240 KB
copies plus three allocations per frame, ~30 per second, all of it pure overhead. It also converts *back*
to NumPy between each stage.

**Fix:** convert once, keep the `PointCloud` alive across Open3D stages, and only come back to NumPy at the
end. Worth measuring rather than assuming — your per-stage timers already tell you whether it's material
on the Orin.

---

### I8 — The pipeline is a dead end: nothing is published back to ROS 2.

For a system whose whole backend is ROS 2, the wrapper subscribes but never publishes. Classification
results exist only inside one Python process. Nothing else on the drone — navigation, logging, telemetry,
a ground station, `ros2 bag` — can see them.

**Fix:** publish results and health from the same node. Even a minimal version unlocks the whole ROS 2
toolchain (`ros2 topic echo`, `rqt`, bag recording for post-flight analysis):

| Topic | Type | Contents |
|---|---|---|
| `/perception/classification` | `vision_msgs/Classification2D` (or a small custom msg) | class id, confidence, source frame stamp |
| `/perception/pipeline_health` | `diagnostic_msgs/DiagnosticArray` | frames in/dropped, per-stage ms, points surviving |
| `/perception/processed_cloud` | `sensor_msgs/PointCloud2` | the 1,024-point cloud, for RViz debugging |

That last one is worth doing **before the demo** — being able to see in RViz exactly what the model is
being fed is the single fastest way to debug C1 and C2 with your own eyes.

---

### I9 — No frame accumulation, which gives up the Mid-70's main advantage.

The Mid-70 uses a **non-repetitive (rosette) scan pattern** — that's its headline feature. A single 100 ms
frame is sparse and full of gaps, but because consecutive frames trace *different* paths, coverage keeps
improving the longer you integrate. Two or three accumulated frames give you a dramatically denser and
more complete surface than one.

Right now each frame is processed in complete isolation, so you're classifying the sparsest possible view.

**Fix — an optional accumulation buffer (only valid when the platform is near-stationary, or after motion
compensation):**

```yaml
ingestion:
  accumulate_frames: 3       # 1 = current behaviour; 3 ≈ 300 ms integration
```

For a static/hovering proof-of-work this is likely the **single cheapest accuracy win available** — no new
dependencies, maybe 15 lines. It trades latency for density, so make it a config knob rather than a
default.

---

### I10 — No ground-plane removal, so the floor dominates normalisation.

In my simulated frame the floor was half the points. The centroid and the unit-sphere scale are computed
over everything, so a large flat ground plane drags the object off-centre and inflates the scale — the
object you actually care about ends up occupying a small fraction of the unit sphere — the regime every
one of these architectures handles worst (and for PointNet++, the one where its absolute ball-query
radii stop matching the data at all).

The `z_min: -2.0` ROI is a crude proxy, and it breaks the moment the drone pitches, rolls or changes
altitude. Open3D's RANSAC plane fit is one call and robust to attitude changes:

```python
plane, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=100)
pcd = pcd.select_by_index(inliers, invert=True)
```

Add it as a toggleable stage between outlier removal and clustering.

---

## 🟡 Nice to have

**N1 · Do FPS on the GPU.** Open3D's `farthest_point_down_sample` is CPU-only and roughly O(N·K) — with
K=1024 that's the most expensive stage in the pipeline, and on the Orin it runs on the ARM cores while a
perfectly good GPU idles. Since torch is already a dependency, a ~20-line CUDA FPS (or
`torch_cluster.fps`) is typically 5–10× faster, and it keeps the data on the GPU where the model wants it
anyway — no extra host↔device copy before inference.
*Extra reason if you choose PointNet++:* it runs FPS **internally** in every set-abstraction layer
(1024→512→128). A GPU FPS kernel in the wrapper is then the same code path the model already needs, so
you can share one implementation instead of paying for a CPU one and a GPU one.

**N2 · Seed the RNG.** `np.random.choice` uses the global RNG, so runs aren't reproducible. Store
`self._rng = np.random.default_rng(cfg.get("seed"))` on the pipeline and use it throughout — it makes A/B
comparisons of config changes actually meaningful.

**N3 · Small code cleanups.**
- `output.dtype` in the config is never read — output is always float32. Either honour it or delete it.
- `import torch` sits inside `process()` ([`pipeline.py:244`](../src/livox_wrapper/pipeline.py#L244)) and re-runs the
  import machinery every frame. Hoist it to a module-level lazy import.
- ✅ ~~`run_pipeline.py` imports `time` but never uses it, and has an unreachable `pass` after
  `while True`.~~ Removed in the reorganisation.
- `KeyboardInterrupt` isn't caught, so a normal Ctrl+C prints a traceback over the demo output.
- `_push_frame`'s recovery path calls `put_nowait` a second time unguarded
  ([`livox_interface.py:81`](../src/livox_wrapper/livox_interface.py#L81)) — safe with one producer today, a latent bug
  if a second ingestion source is ever added.
- `read_points_numpy(...).reshape(-1, 3)` ([`livox_interface.py:117`](../src/livox_wrapper/livox_interface.py#L117))
  assumes all three of x/y/z are present; `xyz_fields` is built by filtering, so if a field is missing you
  get silently **mis-interleaved coordinates** rather than an error. Add
  `assert len(xyz_fields) == 3, f"cloud missing xyz fields: {field_names}"`.

**N4 · Test gaps.** The existing tests are a good foundation, but:
- **Nothing exercises `voxel_size_m`** — it's `None` in `BASE_CONFIG`, which is precisely why C1 (the most
  damaging bug here) went unnoticed. A test asserting the output has ≥900 *unique* rows for a realistic
  input would have caught it.
- No test for `process()` where filters remove everything (the all-zeros path).
- No test for the config-validation errors in `__init__`.
- No latency regression test — `test_stage_under_budget` with a generous ceiling would catch a
  pathological config before it reaches the drone.

**N5 · `start_demo.sh` hardening.**
- No `trap` — if the script dies between launching the driver and reaching the `kill`, the driver is
  orphaned and holds the LiDAR socket, so the *next* run fails too. Use `trap cleanup INT TERM EXIT`.
- `kill $DRIVER_PID` kills the `ros2 launch` wrapper, not its child nodes. Launch with `setsid` and kill
  the process group: `kill -TERM -$DRIVER_PGID`.
- The dependency check runs **before** ROS 2 is sourced, so it can't check `rclpy` — the one import most
  likely to be missing. Move it after sourcing and add `rclpy, sensor_msgs_py`.
- `2>/dev/null` on the check hides *which* package is missing. Show the error.
- Add Jetson performance mode before starting — it's free throughput:
  the highest `nvpmodel` power mode, plus `sudo jetson_clocks`.

**N6 · Documentation corrections.** ✅ *All resolved in the reorganisation. The docs were rewritten as
`architecture.md`, `setup_guide.md`, `demo_runbook.md` and `roadmap.md`. The table records what was fixed.*

| Where | Says | Should say |
|---|---|---|
| `pipeline_flow.txt` §1 | "The Livox Mid-70 **spins**" | It doesn't — it uses a non-repetitive Risley-prism rosette scan. This matters: it's the reason I9 (accumulation) works. |
| `pipeline_flow.txt` §2 | "further than 30 metres" | Config is a ±10 m box (`max_range_m: 40` for range mode). |
| `demo_runbook.md` §6 | "you will see the DGCNN matrices `(1,3,1024)` being printed" | Nothing is printed — only stage timings, every 30th frame. Either add the print or fix the expectation, so nobody thinks the demo is broken when it's working. |
| `pow_setup_runbook.md` Phase 4 | "will output a matrix shaped `(1024, 3)`" | Same. |
| `demo_runbook.md` title | "Jetson **Nano**" | Target is the Orin Nano; the two have very different performance envelopes and the stale references will mislead whoever tunes this next. |
| `pow_setup_runbook.md` Phase 3 | clones `livox_ros2_driver` | Must match `start_demo.sh`. See C4. |

**N7 · `.gitignore`** ✅ *Resolved in the reorganisation.* It didn't cover `*.pt`, `*.pth`, `*.pcd`, `*.bag`, `*.db3` or a `data/` directory — easy
to accidentally commit a 200 MB weights file or a recorded bag.

**N8 · Packaging.** ✅ *Mostly resolved.* The code is now the `livox_wrapper` package under `src/`,
with a `pyproject.toml`, package-relative imports, a `livox-wrapper` console script and no `sys.path`
hack in the tests. Still open: an `ament_python` package plus a ROS 2 launch file. As reviewed,
`run_pipeline.py` did `from pipeline import PointCloudPipeline`, which works only
because Python puts the script's own directory on `sys.path`; `python -m src.run_pipeline` fails, and the
tests need a `sys.path.insert` hack. Adding an `__init__.py` and a minimal `pyproject.toml` (or a
`setup.py` + `ament_python` package if you want `ros2 run`) makes imports work everywhere and would let
this be launched from a ROS 2 launch file alongside the driver — which would also make C4 disappear
entirely, since one launch file would start both.

---

# Part 3 — Choosing the classifier

The model is still open (PointNet / PointNet++ / DGCNN / something else). The good news first:

> **Roughly 85% of this review is model-independent.** Everything in ingestion, ROS 2, the shell script,
> the Jetson setup and the docs — C3, C4, C5, C6, I1, I2, I3, I4, I6, I7, I8, I9, I10 and N2–N8 — is
> unaffected by which network you pick. Only **C1** (how badly duplicates hurt), **C2** (which is
> unchanged, but see the segmentation fork) and **I5** (input layout) interact with the choice at all.
> Don't wait on the model decision to start fixing things.

Three things about the wrapper are safe to lock in now, whatever you choose:

- **`target_points: 1024`** is the ModelNet40 convention for PointNet, PointNet++ *and* DGCNN. Safe default.
- **Unit-sphere normalisation** is the shared convention across all three, so stage 4 stays as-is.
- **Everything upstream of the model adapter** is shared. Only `to_model_input()` (I5) should change.

## What actually differs

| | **PointNet** | **PointNet++** | **DGCNN** |
|---|---|---|---|
| Local geometry | None — per-point MLP + global max-pool | Hierarchical ball-query neighbourhoods | kNN graph, **rebuilt every layer** |
| Neighbour searches / forward pass | **0** | ~3 (over 1024→512→128 pts) | ~5 (each an N×N distance matrix, ~1M distances at N=1024) |
| Relative inference cost | **Cheapest by a wide margin** | Moderate (SSG); MSG ~2–3× that | Heaviest |
| Hurt by C1's duplicates? | Gracefully (max-pool is duplicate-invariant) | Yes — corrupted neighbourhoods | Yes — 41% of edges go dead |
| Accuracy on fine structure | Weakest (no local context) | Strong | Strong |
| **ONNX → TensorRT on Jetson** | **Clean** — Conv1d/BN/ReLU/MaxPool are all standard ops | **Hardest** — ball query, FPS and grouping are custom CUDA extensions needing pure-torch rewrites or TRT plugins | Exportable (matmul + topk + gather), but the per-layer N×N matrix is expensive and memory-hungry |

The last row is the one I'd weight most heavily for your hardware, and it's easy to overlook at model-
selection time. TensorRT typically buys 2–5× on Jetson, and **PointNet gets that essentially for free**
while PointNet++ requires real engineering to export at all. On an Orin Nano sharing a ~100 ms budget
with the wrapper, that gap can matter more than the raw accuracy difference between the architectures.

## Recommendation

**Start with PointNet, and keep the adapter pluggable.** Reasoning:

1. It's the cheapest by far, which leaves headroom for C2's per-cluster inference — remember approach A
   means *N* inferences per frame, not one. With DGCNN, four objects in view could blow your budget.
2. It's the most forgiving of the duplicate-point problem while you're still fixing C1.
3. It exports to TensorRT cleanly, so your optimisation path is open from day one.
4. It's the simplest thing that can produce a working end-to-end demo — and an end-to-end demo is what a
   proof-of-work needs.

Then, once C1 and C2 are fixed and you have a real accuracy number to beat, try **PointNet++ (SSG)** as
the upgrade. Local geometry genuinely helps on LiDAR data, and SSG is the better cost/accuracy point than
MSG on this hardware. I'd treat **DGCNN as the least attractive of the three here** — it carries DGCNN's
cost without a clear accuracy edge over PointNet++ on sparse LiDAR, since rebuilding a kNN graph every
layer is expensive precisely when points are sparse and neighbourhoods are unstable.

Two caveats that apply to all three:

- **None of them are rotation-invariant.** The T-Net/STN modules help but don't solve it. A chair scanned
  from a drone at 40° yaw looks different from the training set. Either canonicalise orientation before
  inference (PCA on the cluster, or the drone's yaw from the flight controller, since the Mid-70 has no built-in IMU) or
  train with heavy rotation augmentation. Worth deciding *before* you collect training data.
- **PointNet++ makes normalisation load-bearing.** Its ball-query radii are absolute numbers (0.1 / 0.2 /
  0.4 in the ModelNet configs) that assume unit-sphere-normalised input. If your scale is off, those radii
  are meaningless and the model silently sees empty or over-full neighbourhoods. That makes **C1 a
  hard blocker** for PointNet++ specifically, not just a quality issue.

## What to change in the wrapper right now

Nothing model-specific, beyond making the coupling explicit:

1. Implement the `to_model_input()` adapter from **I5** with `model.input_layout` in config.
2. ✅ **Done in the reorganisation:** `dgcnn_input` is now `model_input` in `cli.py`, and the comments
   and docs no longer present DGCNN as settled.
3. Add `model.num_points` to config instead of relying on `downsampling.target_points` meaning the same
   thing forever — if you ever try a 2048-point variant, these decouple.

---

# Suggested order of work

**None of the pre-demo items depend on the model decision** — items 1–6 below are all in the ingestion,
config and deployment layers. Start them now; pick the network in parallel.

**Before the demo — these are the ones that will bite you on the day:**

1. **C1** — reorder voxel/FPS/normalise. One code block; unblocks any real accuracy.
2. **C4:** stop hiding the driver's output. (The package name and the Humble default were fixed in the
   reorganisation.)
3. **C6** — verify `torch.cuda.is_available()` on the Jetson *now*, not during the demo.
4. **C3** — switch staleness to a local monotonic clock.
5. **I2 + I4** — make silent failures loud. Cheap, and turns every demo-day mystery into a log line.
6. **I8 (partial)** — publish the processed cloud so you can see it in RViz.

**Shortly after:**

7. **C2** — add clustering, so the classifier gets objects rather than rooms. This is the biggest accuracy lever
   and the DBSCAN groundwork is already written.
8. **I1, I3** — latency and QoS.
9. **I9, I10** — frame accumulation and ground removal; large accuracy gains for little code.
10. **I5, I6, I7** — contiguity, config validation, conversion overhead.

**When there's time:** N1 (GPU FPS) if the timing logs say FPS is the bottleneck, then N4 test coverage.
(N6, N7 and most of N8 were done in the reorganisation.)

---

# A note on scale

Both runbooks correctly flag that the ±10 m ROI must grow for city-scale flight — that warning is well
placed. Two things to add to it:

- **The Mid-70's specified range is ~90 m at 10% reflectivity**, so widening the box to ±100 m won't
  produce useful returns much past ~90 m; the true constraint is the sensor, not the config.
- **Point count doesn't grow the way you'd fear.** The Mid-70 emits a fixed ~100,000 points/second
  regardless of range — widening the ROI spreads the *same* budget over more volume, making the cloud
  *sparser*, not larger. So the Orin won't "crash from processing millions of points" from ROI width
  alone. The real risk at city scale is the opposite: objects become too sparse to classify, and
  accumulation (I9) plus IMU motion compensation become necessary rather than optional.

✅ Corrected in the reorganised setup guide and config comments.

---

# One thing not yet addressed at all: motion compensation

Each frame accumulates over ~100 ms. On a drone moving at even 5 m/s, the sensor travels 0.5 m during a
single frame, so points captured early and late in the window are recorded in *different* physical
positions but stamped into one common frame. The cloud is smeared, and the smear scales with velocity —
so a model tuned indoors while hovering will degrade in a way that looks random but is actually
proportional to airspeed.

This doesn't need solving for a stationary indoor proof-of-work, but it will need solving before flight.
The standard approach is per-point deskewing using IMU data, or a LIO front-end such as FAST-LIO or
Point-LIO that outputs an already-deskewed, odometry-registered cloud. **The Mid-70 has no built-in
IMU.** Its spec sheet lists none, unlike the Avia and Horizon. So the IMU data must come from the flight
controller or an add-on IMU, time-synchronised with the LiDAR. ✅ Now tracked in
[roadmap.md](roadmap.md#7-prepare-for-flight).

> **Correction to the original review:** it said `livox_ros_driver2` would publish Mid-70 IMU data on
> `/livox/imu`. Neither part is true: that driver doesn't support the Mid-70, and the Mid-70 has no IMU.

