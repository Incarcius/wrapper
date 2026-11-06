# Domain Roadmap — field PoC now, agriculture / road quality later

**Recorded:** 15 Oct 2026, after the owner clarified that field classification is
a **proof of concept**, and that the intended deployment domains are
**agriculture** and **road quality**.

This changes what is worth protecting. The six field classes are disposable; the
pipeline is not. Below: what transfers, what does not, and an honest verdict on
whether the Mid-70 can do each job at all.

---

## 1. What transfers unchanged

Everything except one stage. In pipeline order:

| Stage | Transfers? | Why |
|---|---|---|
| Ingestion, ROI crop, outlier removal | ✅ | Sensor-level, domain-blind |
| Ground-plane fitting | ✅ **and becomes more important** | §3 |
| **Region proposal** | ❌ **the one that breaks** | §2 |
| Normalise + FPS to fixed size | ✅ | Operates on a region, whatever produced it |
| Aux features | ✅ **and become the point** | §3 |
| Batching, adapter, layout | ✅ | Shape-level |
| PointNet, training loop, losses, metrics | ✅ | `K` comes from a label map; the architecture is indifferent |
| Distillation, pruning, quantisation, export | ✅ | Operate on the model, not the task |
| Jetson benchmarking, streaming, ROS 2 output | ✅ | Deployment-level |

So a domain change costs: a new label map, a new region-proposal config line, a
new dataset, and a retrain. It does **not** cost a rewrite. That is worth stating
plainly because it is the justification for everything built so far.

## 2. The stage that breaks, and the fix already made

Per-object classification assumes **discrete, separable objects**. A field has
those. Neither later domain does:

- **Agriculture** — a crop field is a *continuous surface*. "Cluster the wheat
  into objects" is not a meaningful operation; DBSCAN would carve arbitrary
  blobs out of a continuum and the labels would describe the radius, not the
  crop.
- **Road quality** — likewise continuous. A pothole is a property of a patch of
  surface, not an instance to be separated from its neighbours.

**Fix, implemented today:** region proposal is now behind one interface in
[`regions.py`](../src/pcml/data/regions.py), with two implementations:

| Method | Produces | Domain |
|---|---|---|
| `dbscan_instances` | one region per DBSCAN cluster | field PoC |
| `ground_tiles` | one region per fixed ground tile (`tile_size_m`, optional overlap) | agriculture, road quality |

Both return the **same** region records, so `prepare_cluster_batch`, the adapter
and the model consume either without modification —
[`test_regions.py::TestProposersAreInterchangeable`](../tests/test_regions.py)
asserts exactly that. 17 new tests, 103 passing in total.

Cost of adding this now: one file. Cost of retrofitting it after a training
pipeline, a labelling tool and a deployment adapter had all hard-coded
clustering: considerably more. That is the only reason it was done today rather
than when the pivot arrives.

## 3. The ground inverts from noise to signal

For object classification the ground is an obstacle — it bridges every object
into one cluster, so it is removed (see
[`data_collection_plan.md`](data_collection_plan.md) §3). For both later domains
**the ground surface is the measurement**:

- Agriculture: crop height, canopy density, row spacing — all defined relative
  to the soil surface.
- Road quality: rutting, deformation and roughness are deviations *of* the
  surface from a fitted plane.

**Fix, implemented today:** `split_ground()` returns ground points, non-ground
points **and** the fitted plane, with `remove_ground()` kept as a thin wrapper
for the PoC path. Discarding ground points in preprocessing would have been the
single hardest thing to undo later, because the information is gone from the
processed dataset and only recoverable by reprocessing raw recordings — which is
also why
[`data_collection_plan.md`](data_collection_plan.md) §4.5 insists on archiving
raw bags.

The aux-feature vector also ages well by accident: `height_above_ground_m` is
*literally crop height* in agriculture and *surface deviation* in road quality.
It was added for scale disambiguation; it turns out to be the primary variable
in both later domains.

## 4. Sensor suitability — the honest verdict per domain

Derived from the same geometry as
[`sensor_feasibility.py`](../scripts/sensor_feasibility.py), plus the Mid-70's
published range accuracy of **±2 cm**.

### Field PoC — ✅ viable

Established. Six classes, all resolvable at a safe altitude, with the caveats on
altitude and mount angle in the collection plan.

### Agriculture — ✅ viable, for geometry only

**What works.** Canopy height, canopy density, row structure, biomass proxies,
gap and bare-patch detection, plant counting for larger crops. The
non-repetitive rosette is an advantage here: over a few seconds of hover it fills
in a denser, more uniform sample than a spinning LiDAR of similar cost, and
successive passes probe slightly different paths through the canopy.

**What does not work.** Anything spectral. Crop health, disease, nitrogen status
and weed-vs-crop discrimination are *reflectance* problems, and the Mid-70 gives
a single uncalibrated intensity at one wavelength (905 nm). No NDVI, no
vegetation indices. If "agriculture" means crop health, this is a multispectral
camera job and the LiDAR is at best a complement providing structure.

**One real limitation.** The Mid-70 is single-return. Full-waveform or
multi-return LiDAR sees partway into a canopy and can separate top-of-canopy
from ground beneath it in one shot; a single-return sensor mostly sees the top
surface. Ground-under-canopy — needed for true crop height on dense crops —
depends on beams finding gaps, which works for row crops and open canopies and
degrades on closed ones.

**Verdict:** worth pursuing, with the scope stated as *structural* agriculture.
The tiling proposer plus `height_above_ground` is most of the machinery already.

### Road quality — ⚠️ partially viable, and the limits are hard

This is where I would push back before committing.

| Defect | Typical magnitude | Mid-70 verdict |
|---|---|---|
| Potholes | 20–100 mm deep, 100–500 mm wide | ✅ detectable |
| Rutting | 10–25 mm deep, ~1 m wide | ⚠️ marginal — comparable to the ±2 cm range noise, needs heavy spatial averaging |
| Shoving / deformation | 10–50 mm over metres | ✅ detectable, averaging helps |
| Roughness / IRI proxy | statistical | ✅ with enough density |
| **Cracking** | **1–5 mm wide** | ❌ **not detectable** |
| Ravelling, bleeding, surface texture | sub-mm to mm | ❌ not detectable |

Two independent walls, and both bite:

1. **Range noise.** At ±2 cm (1σ), a 10 mm rut is *below the noise floor of a
   single measurement*. It is recoverable only by averaging many points over a
   patch — which is what the tiling proposer does, and why tiles rather than
   points are the right unit — but it means per-point defect detection is not
   on the table, and the achievable precision scales as ~σ/√n per tile.
2. **Spatial resolution.** Even at 5 m altitude, nadir ground density is ~256
   points/m², i.e. ~6 cm mean spacing. A 3 mm crack lies between samples. No
   amount of processing recovers a feature finer than the sample spacing.

**Verdict:** the Mid-70 can support *geometric* road condition — potholes,
deformation, a roughness index — and cannot do *crack detection*, which is the
most commonly requested road-quality output and is a camera-plus-vision job. If
road quality is a serious target, settle which of those two is meant before
building anything, because they need different sensors.

## 5. What this changes about the PoC — and what it does not

**Changes:**

- Don't over-invest in the field dataset. Collect enough to prove the pipeline
  end to end (the six classes) and no more. The durable asset is the pipeline
  and the tooling, not the field labels.
- While you are out recording anyway, spend a few extra minutes capturing a
  **road surface** and **any crop or planted area** available. One extra flight
  now gives real data in both target domains to sanity-check §4's verdicts
  against, long before anyone commits to them. Low cost, high option value.
- Keep archiving raw bags. Non-negotiable now: the later domains will want to
  re-extract tiles from recordings made for a different purpose.

**Does not change:**

- The architecture. PointNet on fixed-size regions was chosen for TensorRT
  exportability and for cost that scales with region count — both of which hold
  for tiles exactly as for objects. Tiling actually makes the batching argument
  *stronger*, since a tiled frame yields many more regions per frame than an
  object-clustered one, and the cap in `max_regions` binds harder.
- The task formulation for the PoC. Per-object classification remains right for
  a field, and it is what the two running sessions are building.
- The decision to keep `K` out of the code. That is what makes a domain change a
  config change.

## 6. Open questions this raises

| # | Question | Why it matters now |
|---|---|---|
| D1 | For road quality: crack detection, or geometric condition? | Decides whether the Mid-70 is the right sensor at all (§4) |
| D2 | For agriculture: structure (height/density) or health (spectral)? | Same — spectral needs a different sensor |
| D3 | Will the later domains be classification, or **regression**? | Crop height and rut depth are continuous quantities. The backbone transfers, but the head and loss change from cross-entropy to a regression objective. Cheap if anticipated, awkward if not |
| D4 | Is the PoC's success criterion a working pipeline, or a specific accuracy? | Decides how much field data is enough |

D3 is the one worth thinking about early: if the real deliverable is "how tall
is the crop in this tile" rather than "what class is this tile", that is a
different head on the same body — and knowing it before the training loop is
written saves rewriting it.
