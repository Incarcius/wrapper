"""Typed access to ``configs/preprocess.yaml``.

Why this exists rather than passing a raw dict around: the file is part of the
model artifact. A checkpoint evaluated with one ``preprocess.yaml`` and deployed
with another is a silent accuracy bug, so every consumer needs (a) the values and
(b) the **hash of the exact bytes** they came from. :attr:`PreprocessConfig.sha256`
is what the crop index and the training manifests record, which is what makes
"reproducible from raw file + preprocess.yaml + seed" a checkable claim rather
than a hope.

Unknown keys are rejected. A typo like ``jitter_sigma`` instead of
``jitter_sigma_m`` would otherwise silently disable an augmentation and show up
only as an unexplained accuracy difference.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

import yaml

from .regions import PROPOSERS

__all__ = [
    "ConfigError",
    "RegionProposalConfig",
    "ClusteringConfig",
    "SamplingConfig",
    "NormalizationConfig",
    "AugmentationConfig",
    "GroundRemovalConfig",
    "PreprocessConfig",
    "load_preprocess_config",
]


class ConfigError(ValueError):
    """A preprocessing config that would silently change results."""


def _check_keys(section: str, got: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(got) - allowed)
    if unknown:
        raise ConfigError(
            f"{section}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}. "
            "Unknown keys are rejected because a typo would silently disable a "
            "stage and show up only as an unexplained accuracy difference."
        )


def _section(doc: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = doc.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"`{name}` must be a mapping, got {type(value).__name__}")
    return value


#: Every parameter any registered proposer takes. The union, not the
#: per-method set, because one config file deliberately carries both domains'
#: settings (see ``region_proposal`` in ``preprocess.yaml``) and each proposer
#: ignores the keys that are not its own. Still a closed set, so a typo is
#: caught rather than silently ignored by ``**_ignored``.
_REGION_PROPOSAL_KEYS = {
    "method",
    # dbscan_instances
    "eps_m",
    "min_samples",
    "min_region_points",
    "max_regions",
    "use_open3d",
    # ground_tiles
    "tile_size_m",
    "overlap_factor",
}


@dataclass(frozen=True)
class RegionProposalConfig:
    """How a scene is divided into things to classify.

    Thin on purpose: the parameters are passed straight to
    :func:`pcml.data.regions.build_proposer`, so this class validates and
    carries them rather than re-describing each proposer's signature. The one
    stage that does not survive a change of domain lives in exactly one place.
    """

    method: str = "dbscan_instances"
    params: Mapping[str, Any] = dc_field(default_factory=dict)

    def as_proposer_config(self) -> dict[str, Any]:
        """The dict :func:`pcml.data.regions.build_proposer` consumes."""
        return {"method": self.method, **dict(self.params)}

    @property
    def is_instance_clustering(self) -> bool:
        """True for the discrete-object case, where the ground is noise."""
        return self.method == "dbscan_instances"

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "RegionProposalConfig":
        _check_keys("region_proposal", d, _REGION_PROPOSAL_KEYS)
        method = str(d.get("method", "dbscan_instances")).lower()
        if method not in PROPOSERS:
            raise ConfigError(
                f"region_proposal.method {method!r} is not registered; available: "
                f"{sorted(PROPOSERS)} (pcml.data.regions.PROPOSERS)"
            )
        params = {k: v for k, v in d.items() if k != "method"}
        for key in ("eps_m", "tile_size_m"):
            if key in params and float(params[key]) <= 0:
                raise ConfigError(f"region_proposal.{key} must be > 0, got {params[key]}")
        for key in ("min_samples", "min_region_points", "overlap_factor"):
            if key in params and int(params[key]) < 1:
                raise ConfigError(f"region_proposal.{key} must be >= 1, got {params[key]}")
        if params.get("max_regions") is not None and "max_regions" in params:
            if int(params["max_regions"]) < 1:
                raise ConfigError(
                    f"region_proposal.max_regions must be >= 1 or null, got "
                    f"{params['max_regions']}"
                )
        return RegionProposalConfig(
            method=method, params=MappingProxyType(dict(params))
        )


@dataclass(frozen=True)
class ClusteringConfig:
    """DBSCAN parameters, as the rest of the codebase already names them.

    Superseded on disk by ``region_proposal``, which is where the values now
    live. This view is derived from it (``min_region_points`` ->
    ``min_cluster_points``, ``max_regions`` -> ``max_clusters_per_frame``) so
    that code and configs written against the older names keep working, and so
    that there is one answer to "what DBSCAN radius was this crop set built
    with" rather than two places to check.
    """

    enabled: bool = True
    eps_m: float = 0.5
    min_samples: int = 10
    min_cluster_points: int = 50
    max_clusters_per_frame: Optional[int] = 16
    use_open3d: bool = True

    @staticmethod
    def from_dict(
        d: Mapping[str, Any], region_proposal: Optional[Mapping[str, Any]] = None
    ) -> "ClusteringConfig":
        _check_keys(
            "clustering",
            d,
            {
                "enabled",
                "eps_m",
                "min_samples",
                "min_cluster_points",
                "max_clusters_per_frame",
                "use_open3d",
            },
        )
        rp = dict(region_proposal or {})

        def pick(rp_key: str, legacy_key: str, default: Any) -> Any:
            """region_proposal wins; the legacy `clustering` key is the fallback."""
            if rp_key in rp:
                return rp[rp_key]
            return d.get(legacy_key, default)

        cap = pick("max_regions", "max_clusters_per_frame", 16)
        cfg = ClusteringConfig(
            enabled=bool(d.get("enabled", True)),
            eps_m=float(pick("eps_m", "eps_m", 0.5)),
            min_samples=int(pick("min_samples", "min_samples", 10)),
            min_cluster_points=int(pick("min_region_points", "min_cluster_points", 50)),
            max_clusters_per_frame=None if cap is None else int(cap),
            use_open3d=bool(pick("use_open3d", "use_open3d", True)),
        )
        if cfg.eps_m <= 0:
            raise ConfigError(f"clustering.eps_m must be > 0, got {cfg.eps_m}")
        if cfg.min_samples < 1:
            raise ConfigError(f"clustering.min_samples must be >= 1, got {cfg.min_samples}")
        if cfg.min_cluster_points < 1:
            raise ConfigError(
                f"clustering.min_cluster_points must be >= 1, got {cfg.min_cluster_points}"
            )
        if cfg.max_clusters_per_frame is not None and cfg.max_clusters_per_frame < 1:
            raise ConfigError(
                "clustering.max_clusters_per_frame must be >= 1 or null, got "
                f"{cfg.max_clusters_per_frame}"
            )
        return cfg


@dataclass(frozen=True)
class SamplingConfig:
    num_points: int = 1024
    voxel_size_m: Optional[float] = 0.05
    use_open3d: bool = True

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SamplingConfig":
        _check_keys("sampling", d, {"num_points", "voxel_size_m", "use_open3d"})
        voxel = d.get("voxel_size_m", 0.05)
        cfg = SamplingConfig(
            num_points=int(d.get("num_points", 1024)),
            voxel_size_m=None if voxel is None else float(voxel),
            use_open3d=bool(d.get("use_open3d", True)),
        )
        if cfg.num_points <= 0:
            raise ConfigError(f"sampling.num_points must be > 0, got {cfg.num_points}")
        if cfg.voxel_size_m is not None and cfg.voxel_size_m <= 0:
            raise ConfigError(
                f"sampling.voxel_size_m must be > 0 or null, got {cfg.voxel_size_m}"
            )
        return cfg


@dataclass(frozen=True)
class NormalizationConfig:
    center: bool = True
    scale_to_unit_sphere: bool = True

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "NormalizationConfig":
        _check_keys("normalization", d, {"center", "scale_to_unit_sphere"})
        return NormalizationConfig(
            center=bool(d.get("center", True)),
            scale_to_unit_sphere=bool(d.get("scale_to_unit_sphere", True)),
        )


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-time augmentation, applied to the **raw metric** cluster.

    Applied before normalisation and before resampling, so that it perturbs the
    cluster the way the sensor would have: a dropped point is a point the sensor
    did not return, and jitter is in metres of ranging noise. Applying these
    after unit-sphere normalisation would make ``jitter_sigma_m`` mean "a
    fraction of the object's own size", which is a different and wrong thing.
    """

    yaw_degrees: float = 180.0
    pitch_roll_degrees: float = 10.0
    translation_m: float = 0.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    point_dropout_max: float = 0.25
    jitter_sigma_m: float = 0.01
    jitter_clip_m: float = 0.03
    occlusion_dropout_max: float = 0.0

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "AugmentationConfig":
        _check_keys(
            "augmentation",
            d,
            {
                "yaw_degrees",
                "pitch_roll_degrees",
                "translation_m",
                "scale_range",
                "point_dropout_max",
                "jitter_sigma_m",
                "jitter_clip_m",
                "occlusion_dropout_max",
            },
        )
        raw_scale = d.get("scale_range", (0.9, 1.1))
        if raw_scale is None:
            scale = (1.0, 1.0)
        else:
            if len(raw_scale) != 2:
                raise ConfigError(
                    f"augmentation.scale_range must be [min, max], got {raw_scale!r}"
                )
            scale = (float(raw_scale[0]), float(raw_scale[1]))
        cfg = AugmentationConfig(
            yaw_degrees=float(d.get("yaw_degrees", 180.0)),
            pitch_roll_degrees=float(d.get("pitch_roll_degrees", 10.0)),
            translation_m=float(d.get("translation_m", 0.0)),
            scale_range=scale,
            point_dropout_max=float(d.get("point_dropout_max", 0.25)),
            jitter_sigma_m=float(d.get("jitter_sigma_m", 0.01)),
            jitter_clip_m=float(d.get("jitter_clip_m", 0.03)),
            occlusion_dropout_max=float(d.get("occlusion_dropout_max", 0.0)),
        )
        if cfg.scale_range[0] <= 0 or cfg.scale_range[0] > cfg.scale_range[1]:
            raise ConfigError(
                f"augmentation.scale_range must satisfy 0 < min <= max, got {cfg.scale_range}"
            )
        if not 0.0 <= cfg.point_dropout_max < 1.0:
            raise ConfigError(
                "augmentation.point_dropout_max must be in [0, 1), got "
                f"{cfg.point_dropout_max}"
            )
        if not 0.0 <= cfg.occlusion_dropout_max < 1.0:
            raise ConfigError(
                "augmentation.occlusion_dropout_max must be in [0, 1), got "
                f"{cfg.occlusion_dropout_max}"
            )
        if cfg.jitter_sigma_m < 0 or cfg.jitter_clip_m < 0:
            raise ConfigError("augmentation jitter parameters must be >= 0")
        for name in ("yaw_degrees", "pitch_roll_degrees", "translation_m"):
            if getattr(cfg, name) < 0:
                raise ConfigError(f"augmentation.{name} must be >= 0")
        return cfg

    @property
    def is_identity(self) -> bool:
        """True when no enabled augmentation would change a cluster."""
        return (
            self.yaw_degrees == 0.0
            and self.pitch_roll_degrees == 0.0
            and self.translation_m == 0.0
            and self.scale_range == (1.0, 1.0)
            and self.point_dropout_max == 0.0
            and self.jitter_sigma_m == 0.0
            and self.occlusion_dropout_max == 0.0
        )


@dataclass(frozen=True)
class GroundRemovalConfig:
    enabled: bool = True
    method: str = "ransac_plane"
    distance_threshold_m: float = 0.15
    num_iterations: int = 200
    max_tilt_deg: float = 30.0
    min_inlier_fraction: float = 0.15
    min_height_m: float = 0.15

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "GroundRemovalConfig":
        _check_keys(
            "ground_removal",
            d,
            {
                "enabled",
                "method",
                "distance_threshold_m",
                "num_iterations",
                "max_tilt_deg",
                "min_inlier_fraction",
                "min_height_m",
            },
        )
        cfg = GroundRemovalConfig(
            enabled=bool(d.get("enabled", True)),
            method=str(d.get("method", "ransac_plane")),
            distance_threshold_m=float(d.get("distance_threshold_m", 0.15)),
            num_iterations=int(d.get("num_iterations", 200)),
            max_tilt_deg=float(d.get("max_tilt_deg", 30.0)),
            min_inlier_fraction=float(d.get("min_inlier_fraction", 0.15)),
            min_height_m=float(d.get("min_height_m", 0.15)),
        )
        if cfg.method != "ransac_plane":
            raise ConfigError(
                f"ground_removal.method {cfg.method!r} is not implemented; only "
                "'ransac_plane' exists (pcml.data.ground.remove_ground)"
            )
        return cfg


@dataclass(frozen=True)
class PreprocessConfig:
    """The whole of ``preprocess.yaml``, plus the hash of the bytes it came from."""

    seed: int = 0
    region_proposal: RegionProposalConfig = RegionProposalConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    sampling: SamplingConfig = SamplingConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    use_aux: bool = False
    augmentation: AugmentationConfig = AugmentationConfig()
    ground_removal: GroundRemovalConfig = GroundRemovalConfig()
    accumulation_enabled: bool = False
    accumulation_n_frames: int = 1
    path: Optional[Path] = None
    sha256: Optional[str] = None

    def with_seed(self, seed: int) -> "PreprocessConfig":
        """A copy with a different seed, for seed-sensitivity checks.

        ``sha256`` is deliberately **kept**: it identifies the file, and the seed
        is recorded separately in every index and manifest. A crop set is
        reproducible from (raw file, preprocess.yaml, seed), so both halves are
        stored, not conflated.
        """
        return replace(self, seed=int(seed))


def load_preprocess_config(
    path: Union[str, Path] = "configs/preprocess.yaml",
) -> PreprocessConfig:
    """Load and validate ``preprocess.yaml``.

    Raises:
        FileNotFoundError: if the file is missing. There is no built-in default,
            on purpose: the values are part of the model artifact and must come
            from a file that can be hashed and version-controlled.
        ConfigError: on an unknown key or an out-of-range value.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no preprocessing config at {p}. It is part of the model artifact and "
            "has no code default; copy configs/preprocess.yaml."
        )

    raw_bytes = p.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    try:
        doc = yaml.safe_load(raw_bytes.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p} is not valid YAML: {exc}") from exc
    if doc is None:
        raise ConfigError(f"{p} is empty")
    if not isinstance(doc, Mapping):
        raise ConfigError(f"{p} must contain a mapping at the top level")

    _check_keys(
        str(p),
        doc,
        {
            "seed",
            "region_proposal",
            "clustering",
            "sampling",
            "normalization",
            "features",
            "augmentation",
            "ground_removal",
            "accumulation",
        },
    )

    features = _section(doc, "features")
    _check_keys("features", features, {"use_aux"})
    accumulation = _section(doc, "accumulation")
    _check_keys("accumulation", accumulation, {"enabled", "n_frames"})

    n_frames = int(accumulation.get("n_frames", 1))
    if n_frames < 1:
        raise ConfigError(f"accumulation.n_frames must be >= 1, got {n_frames}")

    region_proposal = RegionProposalConfig.from_dict(_section(doc, "region_proposal"))

    return PreprocessConfig(
        seed=int(doc.get("seed", 0)),
        region_proposal=region_proposal,
        clustering=ClusteringConfig.from_dict(
            _section(doc, "clustering"), region_proposal=region_proposal.params
        ),
        sampling=SamplingConfig.from_dict(_section(doc, "sampling")),
        normalization=NormalizationConfig.from_dict(_section(doc, "normalization")),
        use_aux=bool(features.get("use_aux", False)),
        augmentation=AugmentationConfig.from_dict(_section(doc, "augmentation")),
        ground_removal=GroundRemovalConfig.from_dict(_section(doc, "ground_removal")),
        accumulation_enabled=bool(accumulation.get("enabled", False)),
        accumulation_n_frames=n_frames,
        path=p,
        sha256=sha,
    )
