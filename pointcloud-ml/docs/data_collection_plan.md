# Data Collection Plan — College Cricket / Football Field

**Date:** 15 Oct 2026. **Scene:** college sports field (cricket + football).
**Reproduce the numbers:** `python scripts/sensor_feasibility.py` →
[`../reports/sensor_feasibility.json`](../reports/sensor_feasibility.json)

Read §1 before flying. It constrains the class list, the altitude and the mount
angle, and it is cheaper to read than to rediscover from unusable recordings.

---

## 1. What the Mid-70 can actually resolve

The Mid-70 spreads a **fixed** budget of ~10,000 points per frame over a 70.4°
cone. Density therefore falls as 1/R², and a wider view is a *sparser* view, not
a bigger one. Taking 50 points as the minimum for a classifiable cluster:

| Target | Max range, 1 frame | Max range, 3 frames (0.3 s) |
|---|---:|---:|
| Building wall segment | 59 m | 102 m |
| Tree canopy (small) | 42 m | 72 m |
| Practice net | 37 m | 65 m |
| Car, side-on | 26 m | 46 m |
| Floodlight / fence pole | 20 m | 35 m |
| Football goalpost | 16 m | 28 m |
| Bench | 15 m | 25 m |
| **Person, oblique / side-on** | **12 m** | **21 m** |
| **Person, nadir (from above)** | **8 m** | **14 m** |
| Sports bag | 7 m | 11 m |
| Cricket stumps | 5 m | 9 m |
| Cricket ball | 0.8 m | 1.4 m |

Ground density, nadir-looking, one frame: 64 pts/m² at 10 m altitude, 16 at 20 m,
7 at 30 m.

### Three consequences you need to accept before collecting

**1. Point the sensor obliquely, not straight down.** A person seen from above
presents ~0.35 m² and is classifiable only below 8 m — not a sensible altitude to
fly over people. Seen from the side the same person presents ~0.85 m², which
buys 12 m single-frame and 21 m with accumulation. **A 30–45° forward-down mount
roughly doubles the useful range on everything upright**, which is almost
everything you care about. Record the mount angle; it is part of the data.

**2. Cricket is mostly out of reach.** Stumps clear the threshold only inside
~5 m, and a cricket **ball is invisible** — ~3 points at 10 m, 0 beyond. If
anything about this project depends on tracking the ball or the stumps, the
Mid-70 is the wrong sensor and that should be settled now, not after collection.
The *field* and its *infrastructure* are well within reach; the *game* is not.

**3. Hover beats flying, for data collection.** Three-frame accumulation is a
~1.7× range gain for free, but only while stationary: moving at 5 m/s smears a
100 ms frame by 0.5 m, and the Mid-70 has no IMU to deskew with. So collect the
training set from a hover and treat moving flight as the robustness/domain-shift
set, measured separately.

## 2. Proposed class list

Six classes plus a reject outcome, chosen so every class is physically
resolvable at a safe altitude. This is a proposal — edit it, then it becomes
`data/label_map.yaml`.

| id | name | What it is | Resolvable to |
|---:|---|---|---|
| 0 | `person` | Players, umpires, bystanders. Standing or walking | ~12 m (1 fr) / 21 m (3 fr) |
| 1 | `vegetation` | Perimeter trees, shrubs, hedges | ~42 m |
| 2 | `pole` | Floodlight masts, goalpost uprights, fence posts, stumps-of-a-sightscreen — thin vertical structures | ~20 m |
| 3 | `structure` | Pavilion, perimeter wall, dugout, scoreboard — large planar surfaces | ~59 m |
| 4 | `net_fence` | Cricket practice nets, boundary fencing, cage mesh. Permeable, so returns are sparse and double-surfaced | ~37 m |
| 5 | `vehicle` | Cars, bikes, the pitch roller parked at the edge | ~26 m |
| -1 | `unknown` | **Output, not a trained class.** Produced by the confidence threshold | — |

Deliberately excluded: cricket ball, stumps, bags, litter, line markings. Not
resolvable, or not resolvable at a safe altitude. Including a class the sensor
cannot see teaches the model to guess.

Two notes on the list:
- `pole` vs `structure` vs `vegetation` is a *shape* distinction — thin-vertical,
  large-planar, volumetric-irregular — which is exactly what a geometric
  classifier is good at. That is why these are likely to work.
- `net_fence` is the interesting one. Mesh gives partial returns from both the
  near and far surface, so clusters look hollow and double-walled. Worth keeping
  separate rather than folding into `structure`.

Expect severe imbalance: trees and poles are present in every frame, people
occasionally, vehicles rarely. That is handled with class-weighted loss and
macro-F1 as the selection metric, both already decided — but it does mean the
**rare classes need deliberate staging**, see §4.

## 3. Ground removal is now mandatory, and it is implemented

A mown field is the most important object in the scene and it is not a class.
Measured on simulated fields ([`../tests/test_ground.py`](../tests/test_ground.py)):

There are **three** regimes, set by ground spacing against the 0.5 m DBSCAN
radius — and the middle one is the trap:

| Ground spacing | Altitude (nadir) | What happens | Usable? |
|---|---|---|---|
| ≲0.25 m | ≲15 m | Grass bridges every object into **one component holding >90% of the frame**, field-sized | No |
| **~0.30–0.35 m** | **~20–30 m** | Grass fragments, and fragments are **large enough to clear the 50-point floor** — 3–10 of them per frame pass as objects. Largest measured ~62 points | **No — and it looks like it worked** |
| ≳0.45 m | ≳30 m | Grass fragments heavily (30+ components) but every fragment is below the floor, so the filter absorbs them. Only real objects survive | Degrades gracefully |

> **Corrected 4 Nov 2026.** An earlier version of this section claimed the
> low-density regime produced object-sized false positives. It does not — the
> filter catches them. The false-object problem is real but belongs to the
> *transitional* band, which is squarely where a 20–30 m flight sits. The
> original test passed for the wrong reason (a too-sparse simulated tree
> splitting in two, not grass fragments); both the test and the claim are fixed.

So `ground_removal` moves from "ablation" to "prerequisite". It is now
implemented as RANSAC plane fitting in
[`pcml/data/ground.py`](../src/pcml/data/ground.py), with a horizontality gate so
it cannot latch onto the pavilion wall, and a refusal to delete anything when
the fit is rejected. On a playing field this is close to the ideal case for
RANSAC: genuinely planar, overwhelmingly the largest surface, near-level. Fitted
tilt on the simulated field is 0.01° with 86–95% inliers.

Good news attached: after ground removal, **objects on a field separate far more
cleanly than objects in a room**. The cluster-then-classify approach suits this
scene better than it would suit the original indoor test.

### Wrapper config changes this forces

| Key | Current | Needs to be | Why |
|---|---|---|---|
| `roi_filter.z_max` | `3.0` m | `≥ 20` m | Floodlight masts and mature trees are 8–15 m tall. A 3 m ceiling decapitates every tall class |
| `roi_filter.x/y` | ±10 m | ±40 m or `range` mode | A field is 100–150 m across; ±10 m sees a patch of grass |
| `ground_removal.enabled` | `false` | `true` | §3 |
| `outlier_removal.method` | `open3d_statistical` | keep, but re-tune | Thresholds set for an indoor ±10 m box; grass is a different noise regime |

## 4. What to record

### 4.1 The single most valuable recording: a range calibration

Before anything else, 2 minutes that make every later decision evidence-based:

> Hover at a fixed altitude with the sensor at its intended mount angle. Have
> one person stand still at a measured 5 m, then 10, 15, 20, 25, 30 m from the
> sensor's ground track. Record ~10 seconds at each distance and write down the
> distances.

This yields a measured points-vs-range curve for the class that matters most, and
either validates §1's geometry or corrects it. Everything in §1 is an optimistic
upper bound — it ignores incidence angle, reflectivity and the rosette's
non-uniformity within a single frame. **Measure it once rather than trusting it.**

### 4.2 Per-session metadata, recorded at capture time

Cannot be reconstructed later. One small text file per recording:

- session id, date, time
- altitude (m) and sensor mount angle (degrees below horizontal)
- stationary hover / moving, and approximate airspeed
- which field (cricket / football) and which part of it
- what is in view, and roughly where
- weather: dry / wet grass, dust. Wet surfaces return less
- anything staged, and at what measured distance

### 4.3 Coverage to aim for

| Content | How |
|---|---|
| `person` | Staged: 1 person, then 3–4 people, at 5/10/15/20/25 m. Standing, walking, sitting, arms raised. **Also deliberately overlapping**, 0.3 m apart — DBSCAN will merge them, and you need those examples labelled as the merge case they are |
| `vegetation` | Orbit the perimeter trees; capture each distinct tree from 3–4 bearings |
| `pole` | Floodlight masts, both goalposts, fence posts. From several bearings — a thin cylinder looks identical from most angles, which is both why it is easy and why you need few bearings |
| `structure` | Pavilion, perimeter wall, scoreboard, dugout, from varying ranges |
| `net_fence` | Practice nets from the front, at an angle, and edge-on. Edge-on is the hard case |
| `vehicle` | Whatever is parked at the edge, from 2–3 bearings |
| Negatives | Empty field, sightscreens, roller, bins, bags — recorded and labelled `unknown` so the reject threshold has something to calibrate against |

### 4.4 How much — and the counting mistake to avoid

**Frames are not samples.** A stationary tree at 10 Hz yields 10 near-identical
clusters per second; 10,000 frames of that tree is **one tree**. What matters is
*distinct physical objects × distinct viewpoints*.

Target, per class:

- **≥ 15–20 distinct physical instances** (different trees, different poles,
  different people), each from **3–4 bearings** and **2–3 ranges**.
- That yields ~100–250 genuinely independent cluster examples per class, which
  frames then multiply into several thousand training crops.
- `person` is the easiest to hit, because the same person at a new distance and
  pose is a legitimately new sample. `vehicle` will be the hardest — a college
  field may only have 3–4 vehicles available.

Below ~10 distinct instances for a class, the model memorises those specific
objects and the validation score will not survive a new location. If a class
cannot reach that, merge it into `unknown` rather than shipping a class that
only recognises one pole.

### 4.5 Format: record raw, decide later

Save **unprocessed ROS 2 bags** of `/livox/lidar`:

```bash
ros2 bag record /livox/lidar -o session_01_hover_12m
```

Do not preprocess before saving, and do not save only the 1024-point outputs.
Every preprocessing parameter in this project — voxel size, DBSCAN radius,
ground threshold, accumulation count — is something we will want to re-run
against the same raw data. Discarding raw points discards every future ablation.
Disk is cheap: ~10k points/frame × 10 Hz ≈ 1–2 MB/s.

## 5. Labelling

### 5.1 Label clusters, not scenes and not CAD objects

The annotation unit is **one DBSCAN cluster**, produced by
`pcml.data.clustering.cluster_scene` with the *same* `eps_m` and `min_samples`
used at inference, after ground removal. Labelling anything else — hand-drawn
boxes, whole scenes, downloaded CAD models — creates a train/inference mismatch
that augmentation does not close.

### 5.2 The shortcut that makes this tractable

Most classes here are **static** (trees, poles, walls, nets, parked vehicles),
and during a hover the sensor frame is effectively fixed. So:

> Label each static object **once per recording**, by its position, and
> propagate that label to every frame of the recording automatically by matching
> cluster centroids.

A 60-second hover is 600 frames. Labelling 8 objects once gives ~4,800 labelled
crops from ~8 human decisions. Only `person` needs per-frame attention, and only
while moving.

This is the difference between a feasible afternoon and an infeasible month, and
it is worth structuring the recordings around: **long static hovers, few
objects, known positions.**

### 5.3 Pseudo-labels stay labelled as pseudo-labels

Propagated labels are recorded with their provenance (`propagated` vs
`annotated`). They are not silently treated as ground truth: the held-out test
set uses `annotated` labels only, so that the final number is not partly a
measure of the propagation heuristic.

## 6. What to send first

In priority order. Each unblocks real work:

1. **The class list** — confirm, cut or extend §2. Unblocks `label_map.yaml`,
   the loss, the head width, and all of training.
2. **The range-calibration recording** (§4.1) plus its distances. Unblocks the
   empirical density curve and converts `max_clusters_per_frame` and
   `min_cluster_points` from provisional to measured.
3. **One ordinary 60-second hover bag** over the field. Unblocks the loader, the
   clustering parameter tuning, and the labelling tool — all of which can be
   built and validated against a single real recording before bulk collection
   starts.
4. **Then** bulk collection per §4.3, once (2) and (3) have confirmed the
   altitude and mount angle are right.

Doing (2) and (3) before (4) is the whole point: it is far cheaper to discover
that the mount angle is wrong on one recording than on forty.

## 7. Open questions

| # | Question | Why it matters |
|---|---|---|
| Q1 | Intended flight altitude and mount angle? | §1 — decides whether `person` is in scope at all |
| Q2 | Hover or moving during the real demo? | Decides whether accumulation and motion compensation are in scope |
| Q3 | Does anything downstream need the ball, stumps or line markings? | If yes, the Mid-70 cannot do it and the plan changes |
| Q4 | Is a second location available for a domain-shift set? | Without one, generalisation can be claimed only across sessions, not across places |
| Q5 | Who labels, and how much time do they have? | Decides how hard to lean on §5.2 propagation |
