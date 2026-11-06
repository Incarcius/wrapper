# Stage 1 — Measured Baseline

**Date:** 15 Oct 2026. **Reproduce:** `python scripts/model_summary.py`
**Raw output:** [`model_summary.json`](model_summary.json)

Everything in this file was executed. Nothing is estimated except where the word
"estimate" appears. Jetson figures are **pending** — not predicted.

---

## Environment used

| | |
|---|---|
| Host | WSL2 Ubuntu 24.04 on Windows 11, Intel Core 5 120U, 23.6 GB RAM |
| Python | 3.12.3 (venv at `~/venvs/pcml`) |
| torch | 2.14.1+**cpu** — `cuda_available: False` |
| Threads | 6 |
| numpy / sklearn | 2.5.4 / 1.9.1 |
| open3d | installed but **not importable** (`libusb-1.0.so.0` missing) |

> **This is not the deployment target.** An x86 CPU has a completely different
> performance profile from an Orin Nano's GPU. These numbers are useful for
> *relative* architecture comparison and for parameter/size budgets, and for
> nothing else.

## Test suites

| Suite | Result | When |
|---|---|---|
| `wrapper/` (unmodified, baseline) | **10 passed** in 2.16 s | 15 Oct 2026 |
| `pointcloud-ml/` at Stage 1–2 | **67 passed** in 4.75 s | 15 Oct 2026 |
| `pointcloud-ml/` current | **334 passed** in 44.1 s | 4 Nov 2026 |

The wrapper baseline was run before any change, so later regressions are
attributable. The suite includes an executable regression test for the
wrapper's C1 defect (`test_c1_regression_voxel_in_metres_keeps_unique_points`),
which asserts the duplicate fraction stays under 2%; the wrapper's own ordering
produces ~88%.

**4 Nov 2026 — two intermittent failures resolved, and one documented claim
corrected.** `test_crops.py` was failing because the simulated tree in
`tests/test_ground.py`'s `field_scene` had 500 points across a 3 × 3 × 4 m
volume: a 0.416 m mean spacing against the 0.5 m DBSCAN radius, so its cluster
count was seed-dependent ([2, 3, 4, 5] over 20 seeds). The pipeline was correct;
the fixture was handing it an object at the edge of detectability. Raised to
1500 points (0.288 m, 0.58× the radius), which is stable at 3 clusters over 20
seeds and all tilt variants, and is also the ceiling allowed by
`test_regions`' `ground > 3 × non_ground` invariant — 2000 breaks it.
`TestFixtureItself` now pins both bounds.

Fixing it exposed that the claim *"sparse ground fragments into object-sized
false positives that pass the size filter"* was **wrong**, and that its test had
been passing because the sparse tree split in two rather than because grass
fragments survived. Measured correction: there are three regimes, and false
objects belong to the middle one — see `docs/data_collection_plan.md` §3.

## Measured: parameters, size, CPU latency

Forward pass only, `N=1024` points per cluster, 50 repeats after 10 warm-up
passes, `torch.inference_mode()`. "Clusters" is `M`, the batch dimension — one
forward pass classifies a whole frame.

| Config | Params | Checkpoint (fp32) | M=1 median | M=4 | M=8 | M=16 | per-cluster @ M=16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pointnet_student` `[32,32,64,128]` | **37,349** | 0.16 MB | 0.61 ms | 1.67 ms | 2.90 ms | **6.00 ms** | 0.38 ms |
| `pointnet_deploy` `[64,64,64,128,512]` | **249,413** | 0.98 MB | 2.79 ms | 11.15 ms | 20.68 ms | **40.23 ms** | 2.51 ms |
| `pointnet_teacher` `[64,64,128,256,1024]` +T-Nets | **1,983,502** | 7.64 MB | 9.18 ms | 37.20 ms | 85.74 ms | **178.16 ms** | 11.13 ms |

p95 values: student 7.06 ms, deploy 46.20 ms, teacher 198.40 ms (all at M=16).

### What these numbers establish

1. **Cost is linear in cluster count, as predicted.** Per-cluster latency is
   flat across `M` (deploy: 2.79 → 2.51 ms), so the single-batch design behaves
   as intended and `M` is the variable that decides whether the frame budget is
   met. This is the measurement that justifies capping `max_clusters_per_frame`.
2. **Batching is worth real time at small `M`.** The student's per-cluster cost
   falls from 0.61 ms at M=1 to 0.38 ms at M=16 — a 38% saving that `M`
   individual calls would have thrown away on dispatch overhead.
3. **The teacher is not deployable, and now that is a measurement.** 178 ms at
   M=16 against a 100 ms frame budget, on a machine with a far higher
   single-thread budget than an Orin Nano, and before any preprocessing. It
   earns its keep as a distillation teacher, not as the shipped model.
4. **The deployment model is small enough that checkpoint size is a non-issue.**
   0.98 MB fp32, ~0.49 MB fp16.
5. **A 6.7× parameter reduction (deploy → student) bought a 6.7× latency
   reduction** at M=16. Unusual — parameter count and latency often decouple
   badly — and it holds here because PointNet is a stack of dense `Conv1d`s with
   no neighbourhood search, so FLOPs track width directly. This is exactly the
   property that makes structured pruning worth running later (Phase Six), since
   narrowing channels should translate to real speed rather than nominal sparsity.

### What these numbers do *not* establish

- Nothing about accuracy. **No model has been trained.** There is no data.
- Nothing about Jetson latency, memory, power or thermals.
- Nothing about end-to-end pipeline latency: preprocessing, DBSCAN and
  postprocessing are not in these figures. The wrapper's own (simulated) log line
  showed ~37 ms for preprocessing alone, and DBSCAN over a full frame is on top.
- Nothing about DGCNN or PointNet++ measured here — they are not implemented yet
  (Stage 4), so their costs remain a literature/analysis argument, not a
  measurement from this project.

## Phase Two comparison matrix (skeleton)

Filled only where measured. `pending` means not yet measured; no cell is
estimated.

| Candidate | Params | Ckpt MB | Val macro-F1 | mIoU | p95 model latency (x86 CPU, M=16) | p95 model latency (Jetson) | p95 pipeline latency | Peak RAM | TRT-exportable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PointNet (deploy) | 249,413 | 0.98 | pending — no data | n/a | 46.20 ms | pending | pending | pending | **ONNX verified**: 6 op types, all allowlisted; parity 3.4e-8. TRT build pending |
| PointNet (student) | 37,349 | 0.16 | pending — no data | n/a | 7.06 ms | pending | pending | pending | **ONNX verified**: 6 op types, all allowlisted; parity 6.0e-8. TRT build pending |
| PointNet (teacher) | 1,983,502 | 7.64 | pending — no data | n/a | 198.40 ms | pending | pending | pending | **ONNX verified**: 9 op types (+T-Net `MatMul`/`Reshape`), all allowlisted; parity 3.4e-8. TRT build pending |
| PointNet++ SSG | not implemented | — | pending | n/a | pending | pending | pending | pending | needs custom-op rewrite |
| DGCNN | not implemented | — | pending | n/a | pending | pending | pending | pending | exportable, costly |

## Measured: ONNX export, parity and onnxruntime CPU latency

**Date:** 15 Oct 2026. **Reproduce:** `python scripts/export_model.py`
**Raw output:** [`export_parity.json`](export_parity.json)
**Tests:** `tests/test_export_parity.py` (59 tests)

> ### What this section is NOT
> **It is not a TensorRT result.** TensorRT is never invoked. There is no NVIDIA
> GPU on this laptop (E2) and no Jetson in reach (E5), so no engine can be built
> here. Every figure below is **x86 CPU onnxruntime**.
>
> What is verified is narrower than "TensorRT-ready", and worth stating exactly:
> the exported graph contains **only ops from a TensorRT-native allowlist**, and
> it is **numerically faithful to PyTorch** under onnxruntime. That retires the
> "needs no custom plugins" half of the §9 argument. The `trtexec` build, fp16/int8
> numerics and device latency all remain pending on hardware.
>
> Weights are **randomly initialised** — no model is trained. Parity is a property
> of the graph, so it is fully meaningful untrained; the class outputs are not
> predictions and nothing here says anything about accuracy.

`K=6`, from the now-confirmed `data/label_map.yaml`. (The torch table above was
measured at the `K=5` placeholder, hence the small parameter-count differences:
deploy is 249,542 at `K=6` vs 249,413 at `K=5`.)

### Export: all three configs, dynamic batch axis

| Config | Graph I/O | Dynamic `M`? | Distinct op types |
|---|---|---|---:|
| `pointnet_deploy` | `points["M",3,1024] -> logits["M",6]` | **yes** | 6 |
| `pointnet_student` | `points["M",3,1024] -> logits["M",6]` | **yes** | 6 |
| `pointnet_teacher` | `points["M",3,1024] -> logits["M",6]` | **yes** | 9 |
| deploy + `aux_dim=6` | `points["M",3,1024], aux["M",6] -> logits["M",6]` | **yes** | 7 |

The batch axis is symbolic and the point axis is **fixed at 1024**. That split is
the design: `M` is the per-frame region count and varies, so a fixed batch would
force padding every frame to `max_clusters_per_frame` or re-exporting per `M`;
`N` is guaranteed by `resample_to_fixed_size`, and a fixed axis is what lets
TensorRT select kernels for it.

One session served `M` ∈ {1, 2, 3, 4, 16, 37} for every config — including
`M=1` and a non-power-of-two above the cap. A control test confirms
`dynamic_batch=False` really does produce a fixed dim 0 that rejects a different
`M`, so the dynamic-axis assertions are not passing vacuously.

### Op-type inventory vs the allowlist

Allowlist with per-op justification: `pcml.compression.export.EXPECTED_OPS`.
**No unexpected op appeared in any config.**

| Op | Where from | Configs |
|---|---|---|
| `Conv` | `Conv1d(kernel_size=1)` shared MLP | all |
| `BatchNormalization` | `BatchNorm1d`, eval mode | all |
| `Relu` | `ReLU` | all |
| `ReduceMax` | `x.max(dim=2)` symmetric aggregation | all |
| `Gemm` | `Linear` (head, T-Net head) | all |
| `Identity` | `Dropout` in eval mode | all |
| `MatMul` | `torch.bmm` applying a T-Net matrix | teacher |
| `Reshape` | `TNet.view(-1,k,k)` | teacher |
| `Constant` | the `Reshape` target shape | teacher |
| `Concat` | `cat([embedding, aux])` | `aux_dim>0` only |

Three of these were **not** in the op list the §9 claim named, so they are called
out rather than quietly absorbed. None needs a plugin: `Identity` is a no-op
TensorRT elides, `Constant` is a frozen initializer rather than a runtime op, and
`Concat` is natively supported. `MatMul`/`Reshape` confirm the teacher's T-Net
`bmm` exports on a dynamic batch axis — the one op here with a plausible failure
mode, and it has an explicit test.

Also asserted absent, as the positive form of the architecture argument: `TopK`,
`GatherND`, `ScatterND`, `NonZero`, `GridSample`, `Loop`, `If`, `Scan` — the ops a
DGCNN kNN graph or a PointNet++ ball query would need, and the ones that cost
plugins or TensorRT fallbacks.

### Parity: PyTorch vs onnxruntime, fp32, one session

Identical inputs by construction: regions come from the NumPy FPS path
(`use_open3d=False`), which is deterministic given an explicit `Generator`.

| Config | max abs logit diff | mean abs | vs tolerance 1e-5 |
|---|---:|---:|---|
| `pointnet_deploy` | 3.35e-8 | 8.1e-9 | **pass**, 298x headroom |
| `pointnet_teacher` | 3.35e-8 | 8.9e-9 | **pass**, 298x headroom |
| `pointnet_student` | 5.96e-8 | 1.5e-8 | **pass**, 168x headroom |
| deploy + `aux_dim=6` | 2.98e-8 | 6.9e-9 | **pass** |

Worst case over all configs and all `M`: **5.96e-8**, against an asserted bound
of **1e-5**. The bound is deliberately ~170x above the measured noise floor:
fp32 eps is ~1.2e-7 and a logit accumulates a 1024-point reduction over a Conv1d
stack, so a few ulps of divergence are legitimate wherever the runtimes order an
accumulation differently. The test is a tripwire for a *structurally* wrong graph
— BatchNorm folded with wrong statistics, a reduction over the wrong axis, a
dropped T-Net matmul — which move logits by O(0.1)–O(10). A bound pinned at the
noise floor would flake on a different BLAS or thread count and catch nothing more.

### Measured: onnxruntime CPU latency (x86, **not** Jetson)

Same protocol as the torch table: `N=1024`, 50 repeats after 10 warm-up,
`CPUExecutionProvider`.

| Config | M=1 | M=4 | M=8 | M=16 | p95 @ M=16 | per-region @ M=16 |
|---|---:|---:|---:|---:|---:|---:|
| `pointnet_student` | 0.23 ms | 0.81 ms | 1.79 ms | **2.73 ms** | 3.57 ms | 0.17 ms |
| `pointnet_deploy` | 1.13 ms | 5.29 ms | 9.84 ms | **17.21 ms** | 19.00 ms | 1.08 ms |
| `pointnet_teacher` | 5.97 ms | 29.88 ms | 46.95 ms | **87.45 ms** | 98.30 ms | 5.47 ms |

**onnxruntime is ~2.2x faster than eager PyTorch on the same CPU** at M=16
(deploy 17.21 vs 40.23 ms; teacher 87.45 vs 178.16; student 2.73 vs 6.00). The
graph is the same computation, so the gap is runtime overhead eager torch pays
per op and ORT removes by fusing and planning ahead — which is an argument for
treating ORT, not eager torch, as the floor for any CPU fallback path.

Two things this does **not** change:

1. **The teacher is still not deployable.** 87 ms at M=16 fits inside a 100 ms
   budget only on an x86 CPU with nothing else running, before preprocessing and
   before DBSCAN (111 ms median on its own, per the wrapper integration notes).
   The conclusion from the torch table stands; the margin is just less absurd.
2. **Nothing here is a Jetson number.** The ratio between these runtimes says
   nothing about the ratio on an Orin Nano's GPU under TensorRT.

### Findings about the architecture and the toolchain

| # | Finding | Status |
|---|---|---|
| X1 | torch 2.14 defaults to `dynamo=True`, whose exporter needs `onnxscript` — **not installed here**, so the default path raises `ModuleNotFoundError`. `pcml.compression.export` passes `dynamo=False` explicitly and uses the legacy TorchScript exporter | Worked around and recorded. The legacy exporter is deprecated and will be removed; `pip install onnxscript` is the whole migration, and it should be done before the graph is trusted on a newer torch |
| X2 | The legacy exporter's `DeprecationWarning` is attributed to the **calling** frame, so the repo's `filterwarnings = ["error::DeprecationWarning:pcml.*"]` would turn export into a hard error. Warnings are captured and returned on `ExportResult.warnings_raised` instead of being suppressed | Handled; the CLI prints them |
| X3 | `PointNetClassifier.encode` (`pointnet.py:173`) and `forward` (`pointnet.py:206`) raise `TracerWarning`: their input-validation comparisons touch traced tensor shapes, so the checks are **baked as constants** and do not run in the exported graph | **Benign but worth knowing.** The comparisons only guard a `raise`, so no output is affected and parity is unaffected. The consequence is that ONNX/TensorRT inference has *no* shape validation — the deploy-side caller owns it. `pcml/models/*` was not modified (another session owns it); flagged rather than changed |
| X4 | `do_constant_folding=True` does **not** fold `BatchNormalization` into the preceding `Conv` | Expected; TensorRT performs Conv+BN fusion at engine-build time. Worth re-checking once `trtexec` is reachable, since the ORT numbers above do not benefit from it |
| X5 | At opset ≥ 18 `ReduceMax` takes its axes as an input tensor, adding a `Constant` node. Opset 17 keeps it an attribute | `DEFAULT_OPSET = 17`, chosen for the smaller graph. Opsets 13 and 17 give an identical inventory |

### What is still unverified about export

- The TensorRT engine build (`trtexec`), on the device. **This is the remaining half of the §9 claim.**
- fp16 and int8 TensorRT numerics. The parity above is fp32 only, and fp16 will
  not hold a 1e-5 bound — it needs its own tolerance, derived on the device.
- Jetson latency, memory, power, thermals.
- Anything about accuracy. No data, no training.

## Known environment limitations

| # | Limitation | Impact | Resolution |
|---|---|---|---|
| E1 | Windows Smart App Control enforcing (`VerifiedAndReputablePolicyState = 1`) blocks NumPy's compiled DLL under the Windows Python 3.11 install, and blocks the agent's Git-Bash tool | No Python ML work possible under Windows Python | Worked around via WSL2. SAC cannot be disabled without reinstalling Windows |
| E2 | No NVIDIA GPU on the dev laptop | CPU-only training; slow, and no CUDA-path validation | Cloud GPU (Colab/Kaggle) for training runs; Jetson for deployment validation |
| E3 | `open3d` imports fail: `libusb-1.0.so.0` missing | Falls back to scikit-learn DBSCAN and the NumPy FPS reference path. Correctness unaffected (both paths are tested); speed is lower | `sudo apt install -y libusb-1.0-0 libgl1 libgomp1` in WSL — needs the user's password |
| E4 | venv is Python 3.12; JetPack 6 ships Python 3.10 | Not version parity | Pin via `requirements-dev-lock.txt`; re-verify on device. ONNX export is version-independent |
| E5 | No Jetson Orin Nano accessible from this session | Every on-device figure is pending | Requires the physical device |

## Criteria to advance to Stage 3 (data pipeline)

1. Class list and semantic definitions supplied → `label_map.yaml` written.
2. Labelled data located, with format, volume and provenance recorded.
3. Dataset split manifests generated **by recording session**, not by frame.
4. Cluster-count and cluster-size distributions profiled on real recordings, so
   `max_clusters_per_frame` and `min_cluster_points` stop being provisional.

Items 1 and 2 are owner-blocked. Items 3 and 4 are mechanical once 2 lands.
