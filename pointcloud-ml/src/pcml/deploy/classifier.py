"""The deployed classifier: cleaned scene in, per-cluster predictions out.

This implements the interface specified in ``docs/wrapper_audit.md`` Â§5 and the
output contract in ``docs/task_definition.md`` Â§5. It is the one class the
wrapper constructs; everything about architecture, layout, device and label
vocabulary is config behind it.

Two entry points, one implementation:

* :meth:`ClusterClassifier.classify_frame` is the published contract â€” a cleaned
  ``(N, 3)`` scene in metres, and it owns clustering, resampling, normalisation,
  inference and post-processing. Any consumer that is not the wrapper calls this.
* :meth:`ClusterClassifier.classify_prepared` takes the output of
  ``pcml.data.clustering.prepare_cluster_batch`` instead. The wrapper's
  ``PointCloudPipeline.process`` already produces exactly that, so the live path
  uses it and DBSCAN runs once per frame rather than twice. ``classify_frame``
  is a thin front-end over it, so the two can never drift.

The four invariants from the audit are implemented here, not assumed:

1. **Never raises on malformed input.** Every public method catches and reports
   through ``FramePrediction.diagnostic``. A perception module must not be able
   to take the flight software down.
2. **Bounded memory and time.** The input tensor is allocated once at
   ``(max_clusters, 3, num_points)`` and sliced; at most ``max_clusters``
   clusters are classified per frame.
3. **Frames, units and class ids are never silently changed.** Centroids and
   extents come back in the sensor frame, in metres, exactly as they went in.
4. **Deterministic given a seed.** The only stochastic element is resampling,
   which uses a generator owned by this object and seeded from config.

PLACEHOLDER STATUS, as of 15 Oct 2026.

* The class list (``task_definition.md`` P1) is **resolved**: ``data/label_map.yaml``
  is owner-confirmed, K = 6, and ``K`` is read from it and from nowhere else.
* The labelled data (P2) is **still open**, so there is no training run and no
  checkpoint. With ``checkpoint: null`` this class builds a randomly-initialised
  model so the end-to-end path stays runnable and measurable.

Either a null checkpoint or an ``example: true`` label map sets
``is_placeholder=True`` on every ``FramePrediction``. Those outputs measure
latency and plumbing; they are not predictions and must never be reported as
accuracy.

A real vocabulary over random weights is **more** misleading than a synthetic
one, not less — ``person`` reads like a result when it only means the largest
logit landed at index 0 — which is why the flag survives P1 being resolved. With
no label map at all, :func:`placeholder_label_map` supplies classes named
``PLACEHOLDER_k`` instead, so the output is self-identifying.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..data.clustering import Cluster, cluster_scene, finite_points_only, prepare_cluster_batch
from ..data.ground import remove_ground
from ..data.label_mapping import ClassEntry, LabelMap, load_label_map
from ..data.regions import build_proposer
from ..data.sampling import SamplingStats
from ..models.model_factory import build_model
from .adapter import LAYOUTS, to_model_input

logger = logging.getLogger(__name__)

__all__ = [
    "ClusterPrediction",
    "FramePrediction",
    "PreparedFrame",
    "ClusterClassifier",
    "placeholder_label_map",
    "PLACEHOLDER_CLASS_PREFIX",
]

#: Class names are prefixed with this when no real label map exists, so a
#: placeholder prediction is self-identifying wherever it is printed or logged.
PLACEHOLDER_CLASS_PREFIX = "PLACEHOLDER_"


def placeholder_label_map(num_classes: int) -> LabelMap:
    """A structural stand-in vocabulary, for use until the real class list exists.

    The class list is owner-blocked (``docs/task_definition.md`` P1), and a
    model head cannot be built without a ``K``. This supplies one so that the
    end-to-end path is runnable and its latency measurable.

    The names are deliberately not plausible object categories: a placeholder
    that reads like a vocabulary is how a plumbing test ends up quoted as a
    result. ``is_example`` is set, which is the flag every consumer in this
    repository already checks before letting a number be reported.
    """
    if num_classes < 2:
        raise ValueError(f"placeholder_num_classes must be >= 2, got {num_classes}")
    return LabelMap(
        classes=tuple(
            ClassEntry(
                class_id=i,
                name=f"{PLACEHOLDER_CLASS_PREFIX}{i}",
                definition=(
                    "Structural placeholder, not a class. The class list has not "
                    "been supplied (docs/task_definition.md P1)."
                ),
            )
            for i in range(num_classes)
        ),
        is_example=True,
    )


# ----------------------------------------------------------------------------
# Output contract (docs/task_definition.md Â§5)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterPrediction:
    """One classified cluster. Geometry is metric; the class label is not."""

    cluster_id: int  # index within this frame only; not a track id
    class_id: int  # index into label_map.yaml; -1 == unknown/rejected
    class_name: str  # resolved from the label map, never from a literal
    confidence: float  # softmax max in [0, 1]; not evidence of correctness
    n_points_raw: int  # points in the cluster BEFORE resampling
    centroid_m: tuple[float, float, float]  # sensor frame, metres
    extent_m: tuple[float, float, float]  # axis-aligned, sensor frame, metres


@dataclass(frozen=True)
class FramePrediction:
    """One frame's worth of predictions, plus the diagnostics to trust them by.

    ``n_clusters_found`` vs ``n_clusters_classified`` is deliberate: objects
    dropped by the ``max_clusters`` cap or the minimum-size filter must be
    visible in the output rather than inferred from its absence.

    Two fields extend the dataclass in ``task_definition.md`` Â§5:

    * ``diagnostic`` â€” required by invariant 1. A method that may not raise
      needs somewhere to put the reason it produced nothing.
    * ``is_placeholder`` â€” true whenever the weights or the vocabulary are
      stand-ins. It exists so that no consumer can mistake a plumbing test for
      a measurement.
    """

    stamp: float  # sensor header stamp, seconds
    predictions: list[ClusterPrediction]
    n_clusters_found: int  # before the max_clusters cap and size filter
    n_clusters_classified: int  # after cap and size filter
    latencies_ms: dict[str, float]  # per-stage; extends the wrapper's StageTimings
    diagnostic: str = ""
    is_placeholder: bool = False


@dataclass
class PreparedFrame:
    """A frame that has been clustered and resampled but not yet classified.

    The canonical carrier between a preprocessing front-end and
    :meth:`ClusterClassifier.classify_prepared`. The wrapper's pipeline returns
    its own equivalent object; these fields are the contract.
    """

    clusters: list[Cluster]
    batch: np.ndarray  # (M, num_points, 3) float32, per-cluster normalised
    aux: Optional[np.ndarray] = None  # (M, 6) float32, from the raw metric clusters
    stats: Optional[list[SamplingStats]] = None
    n_clusters_found: int = 0
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------------
def _resolve(base: Path, value: Any) -> Optional[Path]:
    """Resolve a config path relative to the config file that named it."""
    if value is None:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else (base / p).resolve()


def _require_positive_int(value: Any, name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if out <= 0:
        raise ValueError(f"{name} must be > 0, got {out}")
    return out


class ClusterClassifier:
    """Owns the model, the label map and the preprocessing config.

    Constructed once at startup; :meth:`classify_frame` is called per frame and
    allocates nothing on the steady-state path.

    Args:
        config_path: path to a deploy config (see
            ``configs/deploy_placeholder.yaml`` for the annotated schema). All
            paths inside it are resolved relative to its own directory, so a
            config plus its checkpoint and label map move as one unit.

    Raises:
        FileNotFoundError, ValueError, KeyError: from the constructor only.
        Startup is allowed â€” and expected â€” to fail loudly on a bad config; it
        is the *per-frame* path that may never raise.
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        config_path = Path(config_path)
        with open(config_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        base = config_path.parent.resolve()
        self.config_path = config_path

        # --- preprocessing: shared with training, loaded from one file -------
        pre_path = _resolve(base, cfg.get("preprocess"))
        if pre_path is None:
            raise KeyError(
                f"{config_path}: 'preprocess' is required â€” it is the file whose hash "
                "is recorded next to the checkpoint. A checkpoint evaluated with one "
                "preprocess.yaml and deployed with another is a silent accuracy bug."
            )
        with open(pre_path) as f:
            pre: dict[str, Any] = yaml.safe_load(f) or {}
        self.preprocess_path = pre_path
        self.preprocess = pre

        self.seed = int(pre.get("seed", 0))
        self._rng = np.random.default_rng(self.seed)

        # How the scene is divided into things to classify. ``region_proposal``
        # is the current section and is pluggable (DBSCAN instances for discrete
        # objects, tiles for continuous surfaces); ``clustering`` is the earlier
        # name, still read so an older preprocess.yaml keeps working. Reading
        # whichever exists matters: silently defaulting while the file says
        # otherwise is the failure this project keeps finding in its own code.
        region_cfg = pre.get("region_proposal") or {}
        legacy_cfg = pre.get("clustering") or {}
        # The whole-scene debug switch stayed in the legacy section.
        self.clustering_enabled = bool(legacy_cfg.get("enabled", True))

        if region_cfg:
            self.region_method = str(region_cfg.get("method", "dbscan_instances"))
            self.min_region_points = int(region_cfg.get("min_region_points", 50))
            self.region_cap = int(region_cfg.get("max_regions", 16))
            self._propose = build_proposer(region_cfg)
        else:
            self.region_method = "dbscan_instances"
            self.min_region_points = int(legacy_cfg.get("min_cluster_points", 50))
            self.region_cap = int(legacy_cfg.get("max_clusters_per_frame", 16))
            eps_m = float(legacy_cfg.get("eps_m", 0.5))
            min_samples = int(legacy_cfg.get("min_samples", 10))
            use_open3d = bool(legacy_cfg.get("use_open3d", True))
            self._propose = lambda pts: cluster_scene(
                pts,
                eps_m=eps_m,
                min_samples=min_samples,
                min_cluster_points=self.min_region_points,
                max_clusters=self.region_cap,
                use_open3d=use_open3d,
            )

        smp = pre.get("sampling") or {}
        self.num_points = _require_positive_int(smp.get("num_points", 1024), "sampling.num_points")
        voxel = smp.get("voxel_size_m")
        self.voxel_size_m = None if voxel is None else float(voxel)
        if self.voxel_size_m is not None and self.voxel_size_m <= 0:
            raise ValueError(f"sampling.voxel_size_m must be > 0 or null, got {voxel}")
        self.use_open3d_sampling = bool(smp.get("use_open3d", True))

        norm = pre.get("normalization") or {}
        self.center = bool(norm.get("center", True))
        self.scale_to_unit_sphere = bool(norm.get("scale_to_unit_sphere", True))

        self.use_aux = bool((pre.get("features") or {}).get("use_aux", False))

        # Ground removal, which on a field is a prerequisite rather than an
        # ablation: at eps_m = 0.5 a mown surface connects every object standing
        # on it to every other, so clustering without it returns one component
        # holding most of the frame (measured in tests/test_ground.py).
        ground = pre.get("ground_removal") or {}
        self.ground_removal_enabled = bool(ground.get("enabled", False))
        method = str(ground.get("method", "ransac_plane"))
        if self.ground_removal_enabled and method != "ransac_plane":
            raise ValueError(
                f"{pre_path}: ground_removal.method {method!r} is not implemented; "
                "only 'ransac_plane' exists (pcml.data.ground). Refusing rather than "
                "silently running a different method than the config names."
            )
        self.ground_method = method
        self.ground_params = {
            "distance_threshold_m": float(ground.get("distance_threshold_m", 0.15)),
            "num_iterations": int(ground.get("num_iterations", 200)),
            "max_tilt_deg": float(ground.get("max_tilt_deg", 30.0)),
            "min_inlier_fraction": float(ground.get("min_inlier_fraction", 0.15)),
            "min_height_m": float(ground.get("min_height_m", 0.0)),
        }

        # --- label map: the source of K --------------------------------------
        lm_path = _resolve(base, cfg.get("label_map"))
        if lm_path is not None:
            self.label_map = load_label_map(lm_path)
        else:
            self.label_map = placeholder_label_map(
                _require_positive_int(
                    cfg.get("placeholder_num_classes", 4), "placeholder_num_classes"
                )
            )
        self.label_map_path = lm_path

        # --- deployment knobs -------------------------------------------------
        dep = cfg.get("deployment") or {}
        self.layout = str(dep.get("input_layout", "bcn"))
        if self.layout not in LAYOUTS:
            raise ValueError(
                f"deployment.input_layout must be one of {LAYOUTS}, got {self.layout!r}"
            )
        self.max_clusters = _require_positive_int(
            dep.get("max_clusters", self.region_cap), "deployment.max_clusters"
        )
        if self.region_cap != self.max_clusters:
            raise ValueError(
                "deployment.max_clusters must equal the preprocessing region cap "
                f"({self.region_cap} vs {self.max_clusters}); the preallocated batch "
                "shape is derived from it, so a mismatch would silently drop regions. "
                "The cap is region_proposal.max_regions, or clustering."
                "max_clusters_per_frame in an older preprocess.yaml."
            )
        self.warmup_frames = int(dep.get("warmup_frames", 10))
        self.device = self._select_device(str(dep.get("device", "cpu")))
        threshold = dep.get("reject_threshold")
        self.reject_threshold = (
            self.label_map.confidence_threshold if threshold is None else float(threshold)
        )
        if not 0.0 <= self.reject_threshold <= 1.0:
            raise ValueError(
                f"deployment.reject_threshold must be in [0, 1], got {self.reject_threshold}"
            )

        # --- model -------------------------------------------------------------
        model_cfg = self._load_model_config(cfg, base)
        aux_dim = int(model_cfg.get("aux_dim", 0) or 0)
        if self.use_aux and aux_dim == 0:
            raise ValueError(
                "preprocess features.use_aux is true but the model was configured with "
                "aux_dim: 0. The aux feature vector changes the model's input signature, "
                "so the two must agree."
            )
        if aux_dim and not self.use_aux:
            raise ValueError(
                f"model aux_dim is {aux_dim} but preprocess features.use_aux is false; "
                "the model would be fed nothing for those inputs."
            )
        self.aux_dim = aux_dim
        # Seed the weight initialisation from the same seed as the sampling.
        # With a real checkpoint this is immaterial â€” the weights are overwritten
        # â€” but on the placeholder path it is what makes invariant 4 hold: an
        # unseeded random init gives different logits on every process start, so
        # nothing about the end-to-end path could be compared run to run. Forked
        # so that constructing a classifier does not perturb the caller's global
        # torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = build_model(model_cfg, num_classes=self.label_map.num_classes)

        ckpt_path = _resolve(base, cfg.get("checkpoint"))
        self.checkpoint_path = ckpt_path
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)
        else:
            logger.warning(
                "No checkpoint configured (%s: checkpoint: null): the model is "
                "RANDOMLY INITIALISED. This path exists to measure latency and "
                "exercise plumbing while P1/P2 are owner-blocked. Its class "
                "outputs are meaningless and must not be reported as accuracy.",
                config_path,
            )
        if lm_path is None:
            logger.warning(
                "No label map configured (%s: label_map: null): using a PLACEHOLDER "
                "vocabulary of %d structural classes. These are not object "
                "categories (task_definition.md P1).",
                config_path,
                self.label_map.num_classes,
            )
        elif self.label_map.is_example:
            logger.warning(
                "Label map %s declares example: true — it is a proposal pending owner "
                "confirmation, not a vocabulary to report numbers against.",
                lm_path,
            )

        # An example vocabulary counts as a placeholder for the same reason a
        # missing checkpoint does: nothing downstream may treat the output as a
        # measurement.
        self.is_placeholder = ckpt_path is None or self.label_map.is_example

        # eval() before any forward pass: BatchNorm must use running statistics,
        # and dropout must be off. In train() mode a single-cluster frame would
        # also hit BatchNorm's batch-of-1 variance error.
        self.model.eval()
        self.model.to(self.device)

        # --- preallocated input (invariant 2) ---------------------------------
        self._buf_shape = (
            (self.max_clusters, 3, self.num_points)
            if self.layout == "bcn"
            else (self.max_clusters, self.num_points, 3)
        )
        self._batch_buf = torch.zeros(self._buf_shape, dtype=torch.float32, device=self.device)
        self._aux_buf = (
            torch.zeros((self.max_clusters, self.aux_dim), dtype=torch.float32, device=self.device)
            if self.aux_dim
            else None
        )
        self._warmed_up = False

    # ------------------------------------------------------------------
    @staticmethod
    def _select_device(requested: str) -> torch.device:
        want = requested.strip().lower()
        if want in ("auto", ""):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if want.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "deployment.device requested %r but torch.cuda.is_available() is False; "
                "falling back to CPU. On the Jetson this means the JetPack torch wheel "
                "is not the one being used â€” a PyPI aarch64 wheel is CPU-only.",
                requested,
            )
            return torch.device("cpu")
        return torch.device(want)

    @staticmethod
    def _load_model_config(cfg: dict[str, Any], base: Path) -> dict[str, Any]:
        """Model architecture from an inline block or a referenced config file."""
        inline = cfg.get("model")
        if isinstance(inline, dict):
            return dict(inline)
        mc_path = _resolve(base, cfg.get("model_config"))
        if mc_path is None:
            raise KeyError(
                "deploy config needs either an inline 'model:' block or a "
                "'model_config:' path (e.g. configs/models/pointnet_deploy.yaml)"
            )
        with open(mc_path) as f:
            doc = yaml.safe_load(f) or {}
        if "model" not in doc:
            raise KeyError(f"{mc_path}: no 'model:' section")
        return dict(doc["model"])

    def _load_checkpoint(self, path: Path) -> None:
        """Load weights strictly: a partial load is an accuracy bug, not a warning."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=True)
        logger.info("Loaded checkpoint %s", path)

    # ------------------------------------------------------------------
    def warmup(self, n_frames: Optional[int] = None) -> None:
        """Run synthetic frames to pay one-off kernel and context costs.

        Mandatory on Jetson: the first CUDA call can take seconds and would
        otherwise be recorded as a latency spike on a real frame. Uses the
        full-size preallocated batch, so the autotuned shape is the shape the
        steady-state path will use.
        """
        n = self.warmup_frames if n_frames is None else int(n_frames)
        if n <= 0:
            self._warmed_up = True
            return
        rng = np.random.default_rng(self.seed)
        synthetic = rng.standard_normal((self.max_clusters, self.num_points, 3)).astype(np.float32)
        tensor = to_model_input(synthetic, layout=self.layout, device=self.device)
        aux = (
            torch.zeros((self.max_clusters, self.aux_dim), dtype=torch.float32, device=self.device)
            if self.aux_dim
            else None
        )
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n):
                self.model(tensor, aux) if aux is not None else self.model(tensor)
        if self.device.type == "cuda":  # pragma: no cover - no CUDA in this environment
            torch.cuda.synchronize()
        self._warmed_up = True
        logger.info(
            "Warmed up %d frames at batch %d x %d points on %s in %.0f ms",
            n,
            self.max_clusters,
            self.num_points,
            self.device,
            (time.perf_counter() - t0) * 1e3,
        )

    # ------------------------------------------------------------------
    def classify_frame(self, points: np.ndarray, stamp: float) -> FramePrediction:
        """Cleaned scene in metres -> per-cluster predictions.

        Args:
            points: ``(N, >=3)`` float array, sensor frame, metres, after ROI
                crop and outlier removal. Empty, all-NaN and single-point inputs
                are valid inputs, not errors.
            stamp: sensor header stamp, seconds. Echoed back unchanged.

        Returns:
            A :class:`FramePrediction`. Never raises: a malformed input yields an
            empty prediction list and a populated ``diagnostic`` (invariant 1).
        """
        latencies: dict[str, float] = {}
        try:
            t = time.perf_counter()
            pts = np.asarray(points)
            if pts.ndim != 2 or pts.shape[1] < 3:
                return self._empty(
                    stamp,
                    f"expected (N,>=3) points, got shape {tuple(pts.shape)}",
                    latencies,
                )
            pts = finite_points_only(np.ascontiguousarray(pts[:, :3], dtype=np.float32))
            latencies["validate"] = (time.perf_counter() - t) * 1e3
            if pts.shape[0] == 0:
                return self._empty(stamp, "no finite points in frame", latencies)

            # Ground removal before region proposal, matching the wrapper's
            # pipeline stage order exactly. The two entry points must agree here:
            # if one removed the ground and the other did not, the same frame
            # would cluster differently depending on which was called.
            t = time.perf_counter()
            ground_z_m: Optional[float] = None
            if self.ground_removal_enabled:
                pts, plane = remove_ground(pts, rng=self._rng, **self.ground_params)
                if plane is None:
                    logger.warning(
                        "no ground plane found (min_inlier_fraction=%.2f, "
                        "max_tilt_deg=%.0f); the frame is clustered with the ground "
                        "still in it and is likely to merge objects.",
                        self.ground_params["min_inlier_fraction"],
                        self.ground_params["max_tilt_deg"],
                    )
                else:
                    z = plane.ground_z_at_origin()
                    # A plane parallel to the z axis has no height on it; fall
                    # back to the min-z proxy rather than poisoning the aux
                    # feature with a NaN.
                    ground_z_m = float(z) if np.isfinite(z) else None
            latencies["ground_removal"] = (time.perf_counter() - t) * 1e3
            if pts.shape[0] == 0:
                return self._empty(
                    stamp, "every point was ground; nothing left to classify", latencies
                )

            t = time.perf_counter()
            if self.clustering_enabled:
                clusters, n_found = self._propose(pts)
            else:
                # Whole-scene mode: debugging only. One "cluster" covering the
                # entire crop, which is exactly the C2 defect â€” kept available
                # so the old behaviour can be reproduced deliberately.
                clusters, n_found = [Cluster.from_points(pts)], 1
            latencies["clustering"] = (time.perf_counter() - t) * 1e3
            if not clusters:
                return self._empty(
                    stamp,
                    f"no region passed min_region_points={self.min_region_points} "
                    f"(method={self.region_method})",
                    latencies,
                    n_clusters_found=n_found,
                )

            t = time.perf_counter()
            batch, aux, stats = prepare_cluster_batch(
                clusters,
                num_points=self.num_points,
                rng=self._rng,
                voxel_size_m=self.voxel_size_m,
                center=self.center,
                scale_to_unit_sphere=self.scale_to_unit_sphere,
                # A measured ground height, where one was fitted, instead of the
                # minimum-z guess prepare_cluster_batch falls back to. This is
                # what makes the height_above_ground_m aux feature meaningful.
                ground_z_m=ground_z_m,
                use_open3d=self.use_open3d_sampling,
            )
            latencies["cluster_prep"] = (time.perf_counter() - t) * 1e3

            return self.classify_prepared(
                clusters,
                batch,
                aux=aux if self.aux_dim else None,
                stamp=stamp,
                n_clusters_found=n_found,
                stats=stats,
                stage_latencies_ms=latencies,
            )
        except Exception as exc:  # noqa: BLE001 - invariant 1 is the whole point
            logger.exception("classify_frame failed; returning an empty prediction")
            return self._empty(stamp, f"{type(exc).__name__}: {exc}", latencies)

    # ------------------------------------------------------------------
    def classify_prepared(
        self,
        clusters: Sequence[Cluster],
        batch: np.ndarray,
        aux: Optional[np.ndarray] = None,
        stamp: float = 0.0,
        *,
        n_clusters_found: Optional[int] = None,
        stats: Optional[Sequence[SamplingStats]] = None,
        stage_latencies_ms: Optional[dict[str, float]] = None,
    ) -> FramePrediction:
        """Classify an already-clustered, already-resampled frame.

        This is the path the wrapper uses, because its pipeline has done the
        clustering and resampling already; re-doing DBSCAN here would double the
        most expensive preprocessing stage on every frame.

        This method deliberately performs **no ground removal**, unlike
        :meth:`classify_frame`. By the time clusters exist the ground is already
        gone — the wrapper's pipeline removes it before clustering, as this class
        does — and re-running a RANSAC fit on individual clusters would fit
        planes to the objects themselves. Callers that assemble a batch by some
        other route own that stage.

        Args:
            clusters: the raw metric clusters, parallel to ``batch``. Their
                geometry is what is reported back, so it stays in metres.
            batch: ``(M, num_points, 3)`` float32, per-cluster normalised.
            aux: ``(M, aux_dim)`` float32, required iff the model has ``aux_dim > 0``.
            stamp: sensor header stamp, echoed unchanged.
            n_clusters_found: clusters found before the cap and size filter.
                Defaults to ``len(clusters)``, which understates the count when
                the caller already capped.
            stats: per-cluster sampling diagnostics. Entries with
                ``mode == "empty"`` are all-zero padding and are **not**
                classified (finding I4).
            stage_latencies_ms: upstream per-stage timings to merge into the
                result, so one frame has one latency record.

        Returns:
            A :class:`FramePrediction`. Never raises.
        """
        latencies: dict[str, float] = dict(stage_latencies_ms or {})
        found = len(clusters) if n_clusters_found is None else int(n_clusters_found)
        try:
            batch = np.asarray(batch, dtype=np.float32)
            if batch.ndim != 3 or batch.shape[0] != len(clusters):
                return self._empty(
                    stamp,
                    f"batch shape {tuple(batch.shape)} does not match "
                    f"{len(clusters)} clusters",
                    latencies,
                    n_clusters_found=found,
                )

            # I4: an empty cloud resamples to zeros. Classifying zeros produces a
            # confident label for nothing at all, which is worse than no output.
            keep = [
                i
                for i in range(len(clusters))
                if stats is None or i >= len(stats) or stats[i].mode != "empty"
            ]
            # Invariant 2: the cap is enforced here too, not only in cluster_scene,
            # because classify_prepared accepts batches from any front-end.
            if len(keep) > self.max_clusters:
                keep = keep[: self.max_clusters]
            m = len(keep)
            if m == 0:
                return self._empty(
                    stamp,
                    "no non-empty cluster to classify",
                    latencies,
                    n_clusters_found=found,
                )
            if batch.shape[1] != self.num_points or batch.shape[2] != 3:
                return self._empty(
                    stamp,
                    f"expected (M, {self.num_points}, 3) batch, got {tuple(batch.shape)}",
                    latencies,
                    n_clusters_found=found,
                )

            t = time.perf_counter()
            view = self._fill_buffer(batch[keep])
            aux_view = None
            if self.aux_dim:
                if aux is None:
                    return self._empty(
                        stamp,
                        f"model needs aux_dim={self.aux_dim} features but aux is None",
                        latencies,
                        n_clusters_found=found,
                    )
                aux_arr = np.asarray(aux, dtype=np.float32)[keep]
                assert self._aux_buf is not None
                self._aux_buf[:m].copy_(torch.from_numpy(np.ascontiguousarray(aux_arr)))
                aux_view = self._aux_buf[:m]
            latencies["batching"] = (time.perf_counter() - t) * 1e3

            t = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(view, aux_view) if aux_view is not None else self.model(view)
                if self.device.type == "cuda":  # pragma: no cover - no CUDA here
                    torch.cuda.synchronize()
            latencies["inference"] = (time.perf_counter() - t) * 1e3

            t = time.perf_counter()
            probs = F.softmax(logits.float(), dim=1)
            conf_t, cls_t = probs.max(dim=1)
            confidences = conf_t.cpu().numpy()
            class_ids = cls_t.cpu().numpy()

            predictions: list[ClusterPrediction] = []
            for slot, cluster_index in enumerate(keep):
                cluster = clusters[cluster_index]
                conf = float(confidences[slot])
                cid = int(class_ids[slot])
                # Open-set handling: a closed softmax labels anything confidently.
                if conf < self.reject_threshold:
                    cid = self.label_map.reserved_class_id
                predictions.append(
                    ClusterPrediction(
                        cluster_id=int(cluster_index),
                        class_id=cid,
                        class_name=self.label_map.name_for(cid),
                        confidence=conf,
                        n_points_raw=int(cluster.n_points_raw),
                        centroid_m=cluster.centroid_m,
                        extent_m=cluster.extent_m,
                    )
                )
            latencies["postprocess"] = (time.perf_counter() - t) * 1e3

            return FramePrediction(
                stamp=float(stamp),
                predictions=predictions,
                n_clusters_found=found,
                n_clusters_classified=m,
                latencies_ms=latencies,
                diagnostic=(
                    ""
                    if self._warmed_up
                    else "warmup() has not been called: this frame's latency includes "
                    "one-off kernel and context costs and is not representative"
                ),
                is_placeholder=self.is_placeholder,
            )
        except Exception as exc:  # noqa: BLE001 - invariant 1
            logger.exception("classify_prepared failed; returning an empty prediction")
            return self._empty(
                stamp,
                f"{type(exc).__name__}: {exc}",
                latencies,
                n_clusters_found=found,
            )

    # ------------------------------------------------------------------
    def classify_prepared_frame(self, prepared: PreparedFrame, stamp: float) -> FramePrediction:
        """:meth:`classify_prepared` for callers that carry a :class:`PreparedFrame`."""
        return self.classify_prepared(
            prepared.clusters,
            prepared.batch,
            aux=prepared.aux,
            stamp=stamp,
            n_clusters_found=prepared.n_clusters_found,
            stats=prepared.stats,
            stage_latencies_ms=prepared.stage_latencies_ms,
        )

    # ------------------------------------------------------------------
    def _fill_buffer(self, batch: np.ndarray) -> torch.Tensor:
        """Copy ``(m, num_points, 3)`` into the preallocated buffer, laid out.

        This is the allocation-free equivalent of
        :func:`pcml.deploy.adapter.to_model_input`: same layout, same dtype,
        same contiguity, but writing into the tensor allocated in ``__init__``
        instead of making a new one every frame (invariant 2). A test asserts
        the two agree exactly, so the layout rule still lives in one place.
        """
        m = batch.shape[0]
        src = torch.from_numpy(np.ascontiguousarray(batch))
        if self.layout == "bcn":
            src = src.transpose(1, 2)
        self._batch_buf[:m].copy_(src)
        return self._batch_buf[:m]

    def _empty(
        self,
        stamp: float,
        diagnostic: str,
        latencies: dict[str, float],
        n_clusters_found: int = 0,
    ) -> FramePrediction:
        return FramePrediction(
            stamp=float(stamp),
            predictions=[],
            n_clusters_found=int(n_clusters_found),
            n_clusters_classified=0,
            latencies_ms=dict(latencies),
            diagnostic=diagnostic,
            is_placeholder=self.is_placeholder,
        )
