# Data

**Status: class list confirmed; no point clouds yet.**

Code, tests and the architecture decision are in place and verified; none of it
can produce an accuracy number until recordings arrive.

## What is needed, in priority order

### 1. The class list — ✅ DONE (15 Oct 2026)

[`label_map.yaml`](label_map.yaml) holds the confirmed 6-class vocabulary for the
college cricket/football field: `person`, `vegetation`, `pole`, `structure`,
`net_fence`, `vehicle`, with `-1`/`unknown` reserved as the reject outcome.
`K = 6` is derived from that file; no model config contains a class count.
Validated: ids contiguous from 0, every class carries a definition and an
annotation policy, and `pointnet_deploy` builds at 249,542 parameters.

`label_map.template.yaml` is kept as the blank form for any future vocabulary.

### 2. The point clouds — the remaining blocker

Collection protocol, including the flight constraints the sensor imposes:
[`../docs/data_collection_plan.md`](../docs/data_collection_plan.md).

**Record raw ROS 2 bags of `/livox/lidar`** and do not preprocess before saving
(§4.5): every preprocessing parameter here is one we will want to re-run against
the same bytes, and discarding raw points discards every future ablation.

Readers that exist and are tested ([`../src/pcml/data/io.py`](../src/pcml/data/io.py)):

| Format | Notes |
|---|---|
| ROS 2 `.db3` / rosbag2 directory | The target format. Read **without rclpy**; each field's offset and datatype come from the message header, so Livox's extra `intensity`/`tag`/`line` fields are handled rather than mis-sliced |
| `.npy` / `.npz` | One frame, or packed `points` + `offsets` + `stamps` |
| `.pcd` | `ascii` and `binary` via NumPy. `binary_compressed` is refused, not mis-parsed |
| `.ply` | `ascii` and both binary byte orders |
| directory of frame files | Sorted by filename; stamps are flagged `synthetic`, never passed off as sensor stamps |

`.las`/`.laz` and Livox `.lvx` are **not** implemented and raise an error naming
the alternative. No loader is written speculatively.

For each recording, these are needed and cannot be reconstructed later. Write
them into a `<recording>.meta.json` sidecar **at capture time** — generate a
blank one with
`python scripts/label_clusters.py --write-meta-template data/raw/session_01.meta.json`:

| Field | Why it matters |
|---|---|
| Sensor | Mid-70 or something else. A different sensor is a different point distribution |
| Session / flight id | Splits are by session. Random frame splits leak near-duplicate frames and inflate validation scores |
| Location / environment | The domain-shift evaluation set is built from held-out locations |
| Stationary or moving, and airspeed | At 5 m/s a 100 ms frame smears by 0.5 m. Smearing scales with speed |
| Mounting and orientation | Clusters arrive in the sensor frame; tilt distribution must match at training time |
| Date / conditions | Rain and dust change the noise floor |

### 3. The labels — the tool is built and waiting

Labels are **one class per object cluster**, not per point and not per scene. The
annotation unit is a DBSCAN cluster produced with the same `eps_m` /
`min_samples` used at inference time, after ground removal. Labelling anything
else — hand-drawn boxes, complete CAD objects, whole scenes — creates a
train/inference mismatch that no amount of augmentation closes.

[`scripts/label_clusters.py`](../scripts/label_clusters.py) implements this.
Each static object is labelled **once per recording** and propagated to every
frame by centroid matching (§5.2), so a 60-second hover over 8 objects costs 8
decisions rather than 4,800. Every label records its provenance, `annotated` or
`propagated`, and the test split uses `annotated` only (§5.3) so the final number
is not partly a measure of the propagation heuristic. The tool is text-first and
works with no display, and it saves after every decision, so it is resumable.

Labels live in `<session>_labels.json`, **separate from the crops**. Crops are
regenerable from (raw recording + `preprocess.yaml` + seed); labels are human
work and are not regenerable at all. Keeping them apart means re-extracting crops
with a different `eps_m` cannot destroy annotation — the labels re-match by
position instead.

## The transfer risk, restated

A ModelNet-style CAD object is watertight, complete and uniformly sampled. A
Mid-70 cluster is a single-view, partial, range-dependent, sparse shell, sometimes
merged with the floor. **Training on CAD data and reporting the result as
real-world accuracy would be wrong**, and the reports in this project will not do
it: any CAD-trained number is labelled as CAD-domain performance.

If only CAD data is available, the pipeline still gets built and validated on it —
but the headline metric stays `pending` until real labelled clusters exist.

## Layout once populated

```
data/
  label_map.yaml                       the class vocabulary (versioned with checkpoints)
  raw/
    session_01_hover_12m/              a rosbag2 directory, untouched
    session_01_hover_12m.meta.json     the §4.2 capture metadata
  processed/
    session_01_hover_12m_crops.npz     raw metric cluster points, packed
    session_01_hover_12m_crops.json    the crop index: geometry, stamps, provenance
    session_01_hover_12m_labels.json   the labels, with annotated/propagated
  manifests/
    train.json                         crop lists, split BY SESSION
    val.json
    test.json                          held out; annotated labels only
    splits.json                        the session assignment, and what was dropped
```

`raw/` and `processed/` are not version-controlled; the manifests and the label
map are. **`processed/*_labels.json` is the exception** — it is human work, not
regenerable, so back it up even though it lives beside generated files.

The dataset pipeline is versioned independently of the trained model: each crop
index records the sha256 of the raw recording, the sha256 of `preprocess.yaml`
and the seed, and each manifest records the label map's sha256. A checkpoint
therefore always names the exact dataset and vocabulary it was trained on, and a
`Dataset` refuses a manifest written against a different one rather than
silently training against shifted class ids.

## The transfer risk, and what the synthetic validation does not cover

The toolchain is validated end to end on a **geometric simulation** of the field,
not a sensor model: no incidence-angle falloff, no reflectivity variation, none
of the rosette's within-frame non-uniformity. That is enough to prove the shapes,
contracts and file formats line up. It proves nothing about accuracy, and no
accuracy number exists in this repository.
