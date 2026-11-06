"""
PointCloudPipeline
-------------------
Turns a raw Livox Mid-70 frame (Nx3 or Nx4 numpy array) into a batch of
model-ready object clusters: ``(M, target_points, 3)`` normalised per cluster,
plus each cluster's geometry in metres.

Designed for real-time use on a Jetson Orin Nano:
- Every stage is independently timed and toggleable via config.
- Heavy stages (outlier removal, FPS) use Open3D's C++ implementations
  rather than pure-Python/sklearn, which are too slow for live frames.
- DBSCAN (as used in the reference paper) is supported but only recommended
  after a voxel pre-downsample, since raw DBSCAN over a full Mid-70 frame
  will not hit real-time budgets on the Orin Nano.

What changed from the first version, and why (see
``pointcloud-ml/docs/wrapper_audit.md``):

* **C1** — the voxel pre-pass and FPS now run *before* normalisation, so
  ``downsampling.voxel_size_m`` is a length in metres rather than a fraction of
  a unit sphere. Both are delegated to
  ``pcml.data.sampling.resample_to_fixed_size``, which is the same code the
  training path uses; a second implementation here is how train/inference skew
  gets in.
* **C2** — normalisation and FPS are scoped to each cluster, not to the whole
  ±10 m scene. A new clustering stage sits between outlier removal and sampling,
  and ``process()`` returns M clusters rather than one 1024-point blob.
  ``clustering.enabled: false`` restores the whole-scene behaviour for
  debugging, which is the only thing it is good for.
* **A1** — sampling draws from a ``numpy.random.Generator`` owned by this
  object and seeded from config, not from NumPy's global RNG. After C2 most
  clusters are smaller than 1024 points, so the random-padding path is the
  common case and an unseeded one makes every run differ.
* **A2** — no float64 upcast. The stages downstream of normalisation were
  running at double the memory traffic on a bandwidth-bound device.
* **A6** — non-finite rows are dropped immediately after ingestion. ``skip_nans``
  protects the ROS 2 path only; a frame from a file or a recording can carry
  NaNs, and one NaN poisons a centroid and therefore the whole cloud.
* **A4** — ``StageTimings`` is extensible and carries the cluster count.
* **I6** — the config is validated and defaulted once in ``__init__``; the
  per-frame path does no ``cfg[...]`` lookups that could raise ``KeyError`` on
  frame 3000.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import numpy as np

from pcml.data.clustering import Cluster, cluster_scene, finite_points_only, prepare_cluster_batch
from pcml.data.ground import GroundPlane, remove_ground
from pcml.data.sampling import SamplingStats

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False

try:
    from sklearn.cluster import DBSCAN

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

logger = logging.getLogger(__name__)

__all__ = ["ConfigError", "FrameClusters", "PointCloudPipeline", "StageTimings"]


class ConfigError(ValueError):
    """Raised at construction time for an invalid pipeline config.

    Deliberately a startup error. The alternative — the previous behaviour —
    was a ``KeyError`` from a bare ``cfg[...]`` on the per-frame path, which
    surfaces as a crashed perception node mid-flight (finding I6).
    """


@dataclass
class StageTimings:
    """Per-stage latencies in **seconds**, plus the frame's cluster count.

    Extensible, which the five fixed float fields of the first version were not
    (finding A4): stages that live outside this module — inference,
    post-processing, publication — are recorded through ``__setitem__``, and
    anything not named here lands in ``extra`` rather than being dropped.

    ``normalization`` and ``downsampling`` from the first version are replaced by
    the single ``cluster_prep``: per-cluster resampling and normalisation are one
    fused loop inside ``prepare_cluster_batch``, and reporting two numbers for
    one measurement would be a fiction.
    """

    roi_filter: float = 0.0
    finite_filter: float = 0.0
    outlier_removal: float = 0.0
    ground_removal: float = 0.0
    clustering: float = 0.0
    cluster_prep: float = 0.0
    inference: float = 0.0
    postprocess: float = 0.0
    total: float = 0.0
    n_clusters: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    _RESERVED = ("extra",)

    def __setitem__(self, name: str, seconds: float) -> None:
        if name in self._RESERVED:
            raise KeyError(f"{name!r} is not a timing field")
        if any(f.name == name for f in fields(self)):
            setattr(self, name, float(seconds))
        else:
            self.extra[name] = float(seconds)

    def __getitem__(self, name: str) -> float:
        if name in self._RESERVED:
            raise KeyError(f"{name!r} is not a timing field")
        if any(f.name == name for f in fields(self)):
            return float(getattr(self, name))
        return self.extra[name]

    def as_ms(self) -> dict[str, float]:
        """Every timing in milliseconds, for ``FramePrediction.latencies_ms``.

        ``n_clusters`` is a count, not a duration, so it is not included here;
        it travels as ``FramePrediction.n_clusters_classified``.
        """
        out = {
            f.name: float(getattr(self, f.name)) * 1e3
            for f in fields(self)
            if f.name not in self._RESERVED and f.name != "n_clusters"
        }
        out.update({k: v * 1e3 for k, v in self.extra.items()})
        return out


@dataclass
class FrameClusters:
    """One frame, clustered and resampled: what ``process()`` now returns.

    Structurally the same contract as ``pcml.deploy.classifier.PreparedFrame``;
    ``ClusterClassifier.classify_prepared`` consumes these fields directly.

    ``batch`` is normalised (unit sphere, per cluster) and has lost absolute
    scale by construction. ``clusters`` and ``aux`` are the metric record: the
    geometry reported downstream comes from them, in the sensor frame, in
    metres.
    """

    batch: np.ndarray  # (M, target_points, 3) float32, per-cluster normalised
    aux: np.ndarray  # (M, 6) float32, from the raw metric clusters
    clusters: list[Cluster]
    stats: list[SamplingStats]
    n_clusters_found: int  # before the max_clusters cap and min-size filter
    n_points_in: int  # after the (N,>=3) check, before any filtering
    n_points_clean: int  # points handed to the clustering stage
    whole_scene: bool  # True when clustering.enabled is false (debug mode)
    #: The fitted ground plane, or None when ground removal is off or the fit was
    #: rejected. None with ground removal enabled is worth logging: it means the
    #: frame was clustered with the ground still in it.
    ground_plane: Optional[GroundPlane] = None
    n_points_non_ground: int = 0  # after ground removal, before clustering

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def stage_latencies_ms(self) -> dict[str, float]:  # PreparedFrame compatibility
        return {}


# ---------------------------------------------------------------------------
# Config schema (finding I6)
#
# Validated and defaulted once, at construction. Each entry is
# (default, validator-or-None); a validator returns the coerced value or raises
# ConfigError. Unknown keys are rejected rather than ignored, because a config
# key that is silently dropped is how a tuning session ends up untraceable.
# ---------------------------------------------------------------------------
def _enum(name: str, allowed: tuple[str, ...]):
    def check(value: Any) -> str:
        if value not in allowed:
            raise ConfigError(f"{name} must be one of {allowed}, got {value!r}")
        return str(value)

    return check


def _number(name: str, *, minimum: Optional[float] = None, allow_none: bool = False):
    def check(value: Any):
        if value is None:
            if allow_none:
                return None
            raise ConfigError(f"{name} must be a number, got null")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name} must be a number, got {value!r}")
        if minimum is not None and value < minimum:
            raise ConfigError(f"{name} must be >= {minimum}, got {value}")
        return value

    return check


def _positive_int(name: str):
    def check(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{name} must be an integer, got {value!r}")
        if value <= 0:
            raise ConfigError(f"{name} must be > 0, got {value}")
        return value

    return check


def _output_format(value: Any) -> str:
    """``output.format`` now has exactly one valid value.

    Tensor creation, layout and device placement moved to
    ``pcml.deploy.adapter.to_model_input`` (finding I5), which is also where the
    classifier's preallocated batch is filled. ``process()`` therefore always
    returns NumPy, and ``format: torch`` is rejected loudly rather than ignored
    silently.
    """
    if value == "numpy":
        return "numpy"
    if value == "torch":
        raise ConfigError(
            "output.format: 'torch' is no longer supported here. The tensor, its "
            "layout and its device are the classifier adapter's responsibility "
            "(finding I5) — set model.input_layout in the classifier's deploy "
            "config instead, and leave output.format: numpy."
        )
    raise ConfigError(f"output.format must be 'numpy', got {value!r}")


def _downsampling_method(value: Any) -> str:
    """``open3d_fps`` or ``numpy_fps``; ``random`` is rejected with a reason.

    The first version offered ``random`` as a cheaper, lower-quality substitute
    for FPS over the whole scene. Per-cluster resampling removed its reason to
    exist — a cluster is a few thousand points at most, and the metric voxel
    pre-pass already bounds FPS cost — and the sampler this pipeline now shares
    with the training path has no random-subsample mode. Rather than quietly
    reinterpret an existing config value as something else, it is rejected with
    the replacement named.
    """
    if value in ("open3d_fps", "numpy_fps"):
        return str(value)
    if value == "random":
        raise ConfigError(
            "downsampling.method: 'random' has been removed. Per-cluster resampling "
            "uses pcml.data.sampling.resample_to_fixed_size, which is shared with the "
            "training path and has no random-subsample mode; a second implementation "
            "here is how train/inference skew gets in. Use 'open3d_fps', or "
            "'numpy_fps' for the reproducible reference sampler."
        )
    raise ConfigError(
        f"downsampling.method must be 'open3d_fps' or 'numpy_fps', got {value!r}"
    )


def _boolean(name: str):
    def check(value: Any) -> bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{name} must be true or false, got {value!r}")
        return value

    return check


_SCHEMA: dict[str, dict[str, tuple[Any, Any]]] = {
    "roi_filter": {
        "enabled": (True, _boolean("roi_filter.enabled")),
        "mode": ("box", _enum("roi_filter.mode", ("box", "range", "none"))),
        "x_min": (-10.0, _number("roi_filter.x_min")),
        "x_max": (10.0, _number("roi_filter.x_max")),
        "y_min": (-10.0, _number("roi_filter.y_min")),
        "y_max": (10.0, _number("roi_filter.y_max")),
        "z_min": (-2.0, _number("roi_filter.z_min")),
        "z_max": (3.0, _number("roi_filter.z_max")),
        "max_range_m": (40.0, _number("roi_filter.max_range_m", minimum=0.0)),
    },
    "outlier_removal": {
        "method": (
            "open3d_statistical",
            _enum(
                "outlier_removal.method",
                ("open3d_statistical", "open3d_radius", "dbscan", "none"),
            ),
        ),
        "nb_neighbors": (16, _positive_int("outlier_removal.nb_neighbors")),
        "std_ratio": (1.5, _number("outlier_removal.std_ratio", minimum=0.0)),
        "radius_nb_points": (6, _positive_int("outlier_removal.radius_nb_points")),
        "radius": (0.6, _number("outlier_removal.radius", minimum=0.0)),
        "dbscan_eps": (0.5, _number("outlier_removal.dbscan_eps", minimum=0.0)),
        "dbscan_min_points": (10, _positive_int("outlier_removal.dbscan_min_points")),
        "pre_voxel_downsample_for_dbscan": (
            0.15,
            _number("outlier_removal.pre_voxel_downsample_for_dbscan", minimum=0.0, allow_none=True),
        ),
    },
    "ground_removal": {
        # Mandatory on a sports field, not an ablation: a mown field is a
        # continuous sheet of returns, and at eps_m = 0.5 every object standing on
        # it is connected to the grass and therefore to every other object, so
        # DBSCAN collapses the scene into one component. Measured in
        # pointcloud-ml/tests/test_ground.py.
        "enabled": (True, _boolean("ground_removal.enabled")),
        "method": ("ransac_plane", _enum("ground_removal.method", ("ransac_plane",))),
        "distance_threshold_m": (
            0.15,
            _number("ground_removal.distance_threshold_m", minimum=1e-6),
        ),
        "num_iterations": (200, _positive_int("ground_removal.num_iterations")),
        "max_tilt_deg": (30.0, _number("ground_removal.max_tilt_deg", minimum=0.0)),
        "min_inlier_fraction": (
            0.15,
            _number("ground_removal.min_inlier_fraction", minimum=1e-6),
        ),
        "min_height_m": (0.15, _number("ground_removal.min_height_m", minimum=0.0)),
    },
    "clustering": {
        "enabled": (True, _boolean("clustering.enabled")),
        "eps_m": (0.5, _number("clustering.eps_m", minimum=1e-6)),
        "min_samples": (10, _positive_int("clustering.min_samples")),
        "min_cluster_points": (50, _positive_int("clustering.min_cluster_points")),
        "max_clusters_per_frame": (16, _positive_int("clustering.max_clusters_per_frame")),
        "use_open3d": (True, _boolean("clustering.use_open3d")),
    },
    "normalization": {
        "center": (True, _boolean("normalization.center")),
        "scale_to_unit_sphere": (True, _boolean("normalization.scale_to_unit_sphere")),
    },
    "downsampling": {
        "voxel_size_m": (
            0.05,
            _number("downsampling.voxel_size_m", minimum=1e-6, allow_none=True),
        ),
        "target_points": (1024, _positive_int("downsampling.target_points")),
        "method": ("open3d_fps", _downsampling_method),
    },
    "output": {
        "format": ("numpy", _output_format),
        "dtype": ("float32", _enum("output.dtype", ("float32",))),
    },
    "logging": {
        "log_stage_timings": (True, _boolean("logging.log_stage_timings")),
        "log_every_n_frames": (30, _positive_int("logging.log_every_n_frames")),
    },
}

#: Sections the pipeline does not read. Present in the config file for other
#: components (ingestion for LivoxInterface, model for the classifier), so they
#: are tolerated rather than rejected as unknown.
_FOREIGN_SECTIONS = ("ingestion", "model")

#: Top-level scalars the pipeline reads.
_TOP_LEVEL = {"seed": (0, _number("seed", minimum=0))}


def validate_config(config: dict) -> dict:
    """Return a fully-defaulted copy of ``config``, or raise :class:`ConfigError`.

    Called once from ``__init__``, so the per-frame path can index the result
    without guards. Validation covers only the sections this module reads;
    ``ingestion`` and ``model`` are passed through untouched for their owners.
    """
    if not isinstance(config, dict):
        raise ConfigError(f"config must be a dict, got {type(config).__name__}")

    unknown = set(config) - set(_SCHEMA) - set(_FOREIGN_SECTIONS) - set(_TOP_LEVEL)
    if unknown:
        raise ConfigError(
            f"unknown config section(s): {sorted(unknown)}. Known sections: "
            f"{sorted(set(_SCHEMA) | set(_FOREIGN_SECTIONS))}"
        )

    out: dict[str, Any] = {}
    for key, (default, check) in _TOP_LEVEL.items():
        out[key] = check(config[key]) if key in config else default

    for section, keys in _SCHEMA.items():
        given = config.get(section) or {}
        if not isinstance(given, dict):
            raise ConfigError(f"config section {section!r} must be a mapping")
        stray = set(given) - set(keys)
        if stray:
            raise ConfigError(
                f"unknown key(s) in {section!r}: {sorted(stray)}. Known keys: {sorted(keys)}"
            )
        resolved: dict[str, Any] = {}
        for key, (default, check) in keys.items():
            resolved[key] = check(given[key]) if key in given else default
        out[section] = resolved

    for section in _FOREIGN_SECTIONS:
        if section in config:
            out[section] = config[section]

    if out["ground_removal"]["min_inlier_fraction"] > 1.0:
        raise ConfigError(
            "ground_removal.min_inlier_fraction must be in (0, 1], got "
            f"{out['ground_removal']['min_inlier_fraction']}"
        )

    roi = out["roi_filter"]
    for axis in ("x", "y", "z"):
        if roi[f"{axis}_min"] > roi[f"{axis}_max"]:
            raise ConfigError(
                f"roi_filter.{axis}_min ({roi[f'{axis}_min']}) is greater than "
                f"roi_filter.{axis}_max ({roi[f'{axis}_max']}): the box is empty"
            )
    return out


class PointCloudPipeline:
    def __init__(self, config: dict):
        """
        config: parsed dict from pipeline_config.yaml (see config/pipeline_config.yaml).
        Validated and defaulted here, once, so ``process()`` never indexes raw config.
        """
        self.cfg = validate_config(config)
        self._frame_count = 0

        # A1: one generator, owned here, seeded from config. The seed belongs in
        # every experiment manifest; without it a cluster that needs padding
        # yields a different model input on every run.
        self.seed = int(self.cfg["seed"])
        self.rng = np.random.default_rng(self.seed)

        outlier = self.cfg["outlier_removal"]
        if outlier["method"].startswith("open3d") and not _HAS_OPEN3D:
            raise ConfigError(
                "outlier_removal.method requires open3d but it is not importable. "
                "pip install open3d, or switch to method: dbscan / none."
            )
        if outlier["method"] == "dbscan" and not _HAS_SKLEARN:
            raise ConfigError("outlier_removal.method 'dbscan' requires scikit-learn.")

        clustering = self.cfg["clustering"]
        if clustering["enabled"] and not (_HAS_OPEN3D or _HAS_SKLEARN):
            raise ConfigError(
                "clustering.enabled requires open3d or scikit-learn for DBSCAN. "
                "Install one, or set clustering.enabled: false — which is "
                "whole-scene mode and is for debugging only (finding C2)."
            )
        if self.cfg["downsampling"]["method"] == "open3d_fps" and not _HAS_OPEN3D:
            # Not fatal: pcml's sampler has a reference NumPy FPS. Say so rather
            # than silently delivering a different sampling than was configured.
            logger.warning(
                "downsampling.method is 'open3d_fps' but open3d is not importable; "
                "using pcml's reference NumPy FPS instead. Same semantics, slower, "
                "and reproducible across Open3D versions."
            )
        self._use_open3d_fps = self.cfg["downsampling"]["method"] == "open3d_fps" and _HAS_OPEN3D

        if not clustering["enabled"]:
            logger.warning(
                "clustering.enabled is false: the whole ROI is normalised as one "
                "sample. That is finding C2 and it is not what the classifier is "
                "trained on — debugging only."
            )

    # ------------------------------------------------------------------
    # Stage 1: ROI filter (cheapest — run first to shrink point count ASAP)
    # ------------------------------------------------------------------
    def roi_filter(self, points: np.ndarray) -> np.ndarray:
        cfg = self.cfg["roi_filter"]
        if not cfg["enabled"]:
            return points

        if cfg["mode"] == "box":
            mask = (
                (points[:, 0] >= cfg["x_min"]) & (points[:, 0] <= cfg["x_max"]) &
                (points[:, 1] >= cfg["y_min"]) & (points[:, 1] <= cfg["y_max"]) &
                (points[:, 2] >= cfg["z_min"]) & (points[:, 2] <= cfg["z_max"])
            )
            return points[mask]

        if cfg["mode"] == "range":
            # NOTE (finding A3): this is a sphere centred on the sensor, not a
            # cylinder. Left as-is deliberately — changing it would change the
            # meaning of an existing key.
            dist = np.linalg.norm(points[:, :3], axis=1)
            return points[dist <= cfg["max_range_m"]]

        return points  # mode == "none"

    # ------------------------------------------------------------------
    # Stage 2: Outlier / noise removal
    # ------------------------------------------------------------------
    def remove_outliers(self, points: np.ndarray) -> np.ndarray:
        cfg = self.cfg["outlier_removal"]
        method = cfg["method"]

        if method == "none" or len(points) == 0:
            return points

        if method == "open3d_statistical":
            pcd = self._to_o3d(points)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=cfg["nb_neighbors"], std_ratio=cfg["std_ratio"]
            )
            return np.asarray(pcd.points, dtype=np.float32)

        if method == "open3d_radius":
            pcd = self._to_o3d(points)
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=cfg["radius_nb_points"], radius=cfg["radius"]
            )
            return np.asarray(pcd.points, dtype=np.float32)

        if method == "dbscan":
            return self.remove_outliers_dbscan(
                points,
                eps=cfg["dbscan_eps"],
                min_pts=cfg["dbscan_min_points"],
                pre_voxel=cfg["pre_voxel_downsample_for_dbscan"],
            )

        raise ConfigError(f"Unknown outlier_removal.method: {method}")

    def remove_outliers_dbscan(
        self,
        points: np.ndarray,
        eps: float = 0.5,
        min_pts: int = 10,
        pre_voxel: Optional[float] = None,
    ) -> np.ndarray:
        """DBSCAN as used in the reference paper (eps=0.5, minPts=10).

        Noise rejection only: the labels are discarded here, because instance
        separation is the job of the clustering stage below. ``pre_voxel`` bounds
        the point count before DBSCAN so it stays usable at frame rate."""
        if len(points) == 0:
            return points

        work_points = points
        if pre_voxel and _HAS_OPEN3D:
            pcd = self._to_o3d(points)
            pcd = pcd.voxel_down_sample(voxel_size=pre_voxel)
            work_points = np.asarray(pcd.points, dtype=np.float32)

        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for DBSCAN outlier removal.")

        labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(work_points[:, :3])
        return work_points[labels != -1]

    # ------------------------------------------------------------------
    # Stage 3: ground-plane removal — a prerequisite, not an ablation
    # ------------------------------------------------------------------
    def remove_ground_plane(self, points: np.ndarray) -> tuple[np.ndarray, Optional[GroundPlane]]:
        """Drop the ground so that objects standing on it become separable.

        This stage has to sit *before* clustering and it is not optional on a
        field: at an 0.5 m DBSCAN radius a mown surface connects every object
        standing on it to every other, so leaving it in returns one cluster
        holding most of the frame.

        Delegated to ``pcml.data.ground.remove_ground``, with this pipeline's
        seeded generator passed in so the RANSAC draw is reproducible (finding
        A1 applies to the plane fit as much as to the sampling).

        Returns:
            ``(non_ground_points, plane)``. When no acceptable plane is found the
            points come back **unchanged** and ``plane`` is ``None``: deleting a
            wrongly-fitted "ground" would delete real objects, so the stage
            declines rather than guesses. That outcome is logged, because a frame
            clustered with the ground still in it will cluster badly and the
            reason should not have to be inferred.
        """
        cfg = self.cfg["ground_removal"]
        if not cfg["enabled"] or len(points) == 0:
            return points, None

        non_ground, plane = remove_ground(
            points,
            distance_threshold_m=cfg["distance_threshold_m"],
            num_iterations=cfg["num_iterations"],
            max_tilt_deg=cfg["max_tilt_deg"],
            min_inlier_fraction=cfg["min_inlier_fraction"],
            min_height_m=cfg["min_height_m"],
            rng=self.rng,
        )
        if plane is None:
            logger.warning(
                "frame %d: no ground plane found (min_inlier_fraction=%.2f, "
                "max_tilt_deg=%.0f); clustering will run with the ground still in "
                "the frame and is likely to merge objects.",
                self._frame_count,
                cfg["min_inlier_fraction"],
                cfg["max_tilt_deg"],
            )
        return non_ground, plane

    # ------------------------------------------------------------------
    # Stage 4: clustering — scene to object instances (finding C2)
    # ------------------------------------------------------------------
    def cluster(self, points: np.ndarray) -> tuple[list[Cluster], int]:
        """Split the cleaned scene into candidate objects, in metres.

        Delegated to ``pcml.data.clustering.cluster_scene`` so the inference path
        and the training path split scenes identically. Returns
        ``(clusters, n_found)``; ``n_found`` counts clusters before the cap and
        the minimum-size filter, because objects dropped silently are the
        failure mode worth logging.
        """
        cfg = self.cfg["clustering"]
        if len(points) == 0:
            return [], 0
        if not cfg["enabled"]:
            # Whole-scene debug mode: one "cluster" spanning the whole ROI.
            return [Cluster.from_points(points)], 1
        return cluster_scene(
            points,
            eps_m=cfg["eps_m"],
            min_samples=cfg["min_samples"],
            min_cluster_points=cfg["min_cluster_points"],
            max_clusters=cfg["max_clusters_per_frame"],
            use_open3d=cfg["use_open3d"] and _HAS_OPEN3D,
        )

    # ------------------------------------------------------------------
    # Stage 4: per-cluster resample (metres) then normalise (findings C1, C2, A2)
    # ------------------------------------------------------------------
    def prepare_clusters(
        self, clusters: list[Cluster], ground_z_m: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray, list[SamplingStats]]:
        """Voxel + FPS in metres, then per-cluster unit-sphere normalisation.

        Order matters and is the C1 fix: ``downsampling.voxel_size_m`` is a
        length in metres applied to the raw cluster, and normalisation is the
        last thing that happens before the network. Everything stays float32
        (finding A2).
        """
        down = self.cfg["downsampling"]
        norm = self.cfg["normalization"]
        return prepare_cluster_batch(
            clusters,
            num_points=down["target_points"],
            rng=self.rng,
            voxel_size_m=down["voxel_size_m"],
            center=norm["center"],
            scale_to_unit_sphere=norm["scale_to_unit_sphere"],
            # With a fitted plane this is a measured ground height rather than
            # the minimum-z guess prepare_cluster_batch falls back to, so the
            # height_above_ground_m aux feature becomes meaningful.
            ground_z_m=ground_z_m,
            use_open3d=self._use_open3d_fps,
        )

    # ------------------------------------------------------------------
    def normalize(self, points: np.ndarray) -> np.ndarray:
        """Centre and unit-sphere-scale one cloud, in float32 (finding A2).

        Retained as a single-cloud utility; the per-frame path normalises *per
        cluster* inside :meth:`prepare_clusters`, which is the C2 fix. Calling
        this on a whole scene is the defect, not the API.
        """
        cfg = self.cfg["normalization"]
        if len(points) == 0:
            return points

        pts = np.array(points[:, :3], dtype=np.float32, copy=True)

        if cfg["center"]:
            pts -= pts.mean(axis=0)

        if cfg["scale_to_unit_sphere"]:
            max_dist = float(np.linalg.norm(pts, axis=1).max())
            if max_dist > 1e-9:
                pts /= max_dist

        return pts

    def farthest_point_sampling(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """Bring one cloud to exactly ``num_points``, using the seeded generator.

        Thin wrapper over ``pcml.data.sampling.resample_to_fixed_size`` — kept so
        existing callers and tests keep working — but note that it no longer uses
        the global NumPy RNG (finding A1), and that the voxel pre-pass is applied
        by the caller in metres (finding C1).
        """
        from pcml.data.sampling import resample_to_fixed_size

        out, _ = resample_to_fixed_size(
            np.asarray(points, dtype=np.float32),
            num_points,
            self.rng,
            voxel_size_m=None,
            use_open3d=self._use_open3d_fps,
        )
        return out

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process(self, raw_point_cloud: np.ndarray) -> tuple[FrameClusters, StageTimings]:
        """
        raw_point_cloud: (N, 3) or (N, 4+) numpy array (xyz[, intensity, ...])

        Returns ``(FrameClusters, StageTimings)``. ``FrameClusters.batch`` is
        ``(M, target_points, 3)`` float32 normalised per cluster, and
        ``FrameClusters.clusters`` carries each cluster's centroid and extent in
        sensor-frame metres. ``M`` can be 0 — an empty frame is a normal
        outcome, not an error (finding I4 is handled by the caller checking
        ``stats[i].mode == "empty"`` or ``M == 0``).
        """
        t0 = time.perf_counter()
        timings = StageTimings()

        pts = np.asarray(raw_point_cloud, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Expected (N,>=3) array, got shape {pts.shape}")
        pts = pts[:, :3]
        n_points_in = int(pts.shape[0])

        # A6: drop NaN/Inf rows now. One non-finite row poisons a centroid, and
        # from there the whole normalised cloud, with no error anywhere.
        t = time.perf_counter()
        pts = finite_points_only(pts)
        timings["finite_filter"] = time.perf_counter() - t
        if pts.shape[0] != n_points_in:
            logger.debug(
                "dropped %d non-finite point(s) from frame %d",
                n_points_in - pts.shape[0],
                self._frame_count,
            )

        t = time.perf_counter()
        pts = self.roi_filter(pts)
        timings["roi_filter"] = time.perf_counter() - t

        t = time.perf_counter()
        pts = self.remove_outliers(pts)
        timings["outlier_removal"] = time.perf_counter() - t
        n_points_clean = int(len(pts))

        t = time.perf_counter()
        pts, ground_plane = self.remove_ground_plane(pts)
        timings["ground_removal"] = time.perf_counter() - t
        n_points_non_ground = int(len(pts))

        t = time.perf_counter()
        clusters, n_found = self.cluster(pts)
        timings["clustering"] = time.perf_counter() - t

        t = time.perf_counter()
        ground_z_m = None if ground_plane is None else ground_plane.ground_z_at_origin()
        if ground_z_m is not None and not np.isfinite(ground_z_m):
            # A plane parallel to the z axis has no height on it. Fall back to
            # the min-z proxy rather than poisoning the aux feature with a NaN.
            ground_z_m = None
        batch, aux, stats = self.prepare_clusters(clusters, ground_z_m=ground_z_m)
        timings["cluster_prep"] = time.perf_counter() - t

        timings.n_clusters = len(clusters)
        timings["total"] = time.perf_counter() - t0

        result = FrameClusters(
            batch=batch,
            aux=aux,
            clusters=clusters,
            stats=stats,
            n_clusters_found=n_found,
            n_points_in=n_points_in,
            n_points_clean=n_points_clean,
            whole_scene=not self.cfg["clustering"]["enabled"],
            ground_plane=ground_plane,
            n_points_non_ground=n_points_non_ground,
        )

        self._frame_count += 1
        self.maybe_log_timings(timings)
        return result, timings

    # ------------------------------------------------------------------
    def maybe_log_timings(self, timings: StageTimings) -> None:
        """Emit the periodic per-stage log line, including ``M`` (finding A4).

        Public so the CLI can call it once *after* inference and publication
        have been recorded into the same ``StageTimings``, rather than logging a
        frame's preprocessing and its inference on two separate lines.
        """
        log_cfg = self.cfg["logging"]
        if not log_cfg["log_stage_timings"]:
            return
        if self._frame_count % log_cfg["log_every_n_frames"] != 0:
            return
        extra = " ".join(f"{k}={v * 1e3:.1f}ms" for k, v in timings.extra.items())
        logger.info(
            "frame=%d M=%d roi=%.1fms finite=%.1fms outlier=%.1fms ground=%.1fms "
            "cluster=%.1fms prep=%.1fms infer=%.1fms post=%.1fms total=%.1fms%s",
            self._frame_count,
            timings.n_clusters,
            timings.roi_filter * 1e3,
            timings.finite_filter * 1e3,
            timings.outlier_removal * 1e3,
            timings.ground_removal * 1e3,
            timings.clustering * 1e3,
            timings.cluster_prep * 1e3,
            timings.inference * 1e3,
            timings.postprocess * 1e3,
            timings.total * 1e3,
            f" {extra}" if extra else "",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _to_o3d(points: np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        return pcd
