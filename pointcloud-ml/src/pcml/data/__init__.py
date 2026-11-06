"""Data pipeline: ingestion, ground handling, region proposal, crops, labels, splits.

``pcml.data.dataset`` is deliberately **not** imported here. It is the only module
in this package that needs torch, and ``pcml.deploy`` runs on the drone with
numpy as its sole dependency besides the inference runtime — importing
``pcml.data`` must not pull torch in. Import it explicitly:

    from pcml.data.dataset import ClusterCropDataset
"""

from .clustering import (
    AUX_FEATURE_NAMES,
    Cluster,
    cluster_scene,
    finite_points_only,
    prepare_cluster_batch,
)
from .config import (
    AugmentationConfig,
    ConfigError,
    PreprocessConfig,
    RegionProposalConfig,
    load_preprocess_config,
)
from .crops import (
    CropError,
    CropIndex,
    CropRecord,
    crop_geometry_summary,
    extract_many,
    extract_session_crops,
    find_crop_indices,
    load_crop_index,
)
from .ground import GroundPlane, fit_ground_plane, remove_ground, split_ground
from .io import (
    DEFAULT_LIVOX_TOPIC,
    Recording,
    RecordingError,
    RecordingMeta,
    decode_pointcloud2_cdr,
    list_bag_topics,
    open_recording,
    read_npy_frames,
    read_pcd,
    read_ply,
    read_pointcloud2_bag,
)
from .label_mapping import ClassEntry, LabelMap, LabelMapError, load_label_map
from .labels import (
    KNOWN_ATTRIBUTES,
    PROVENANCE_ANNOTATED,
    PROVENANCE_PROPAGATED,
    CropLabel,
    LabelledObject,
    LabelStore,
    LabelStoreError,
    ascii_projection,
    describe_crop,
    validate_attributes,
)
from .regions import (
    PROPOSERS,
    build_proposer,
    propose_dbscan_instances,
    propose_ground_tiles,
)
from .sampling import (
    SamplingStats,
    farthest_point_sample,
    resample_to_fixed_size,
    voxel_downsample,
)
from .splits import (
    SplitAssignment,
    SplitError,
    assign_sessions_to_splits,
    class_distribution_report,
    format_class_distribution_report,
    write_manifests,
)

__all__ = [
    # clustering / sampling / region proposal
    "AUX_FEATURE_NAMES",
    "PROPOSERS",
    "Cluster",
    "SamplingStats",
    "build_proposer",
    "cluster_scene",
    "farthest_point_sample",
    "finite_points_only",
    "prepare_cluster_batch",
    "propose_dbscan_instances",
    "propose_ground_tiles",
    "resample_to_fixed_size",
    "voxel_downsample",
    # ground
    "GroundPlane",
    "fit_ground_plane",
    "remove_ground",
    "split_ground",
    # preprocessing config
    "AugmentationConfig",
    "ConfigError",
    "PreprocessConfig",
    "load_preprocess_config",
    # ingestion
    "DEFAULT_LIVOX_TOPIC",
    "Recording",
    "RecordingError",
    "RecordingMeta",
    "decode_pointcloud2_cdr",
    "list_bag_topics",
    "open_recording",
    "read_npy_frames",
    "read_pcd",
    "read_ply",
    "read_pointcloud2_bag",
    # cluster crops
    "CropError",
    "CropIndex",
    "CropRecord",
    "crop_geometry_summary",
    "extract_many",
    "extract_session_crops",
    "find_crop_indices",
    "load_crop_index",
    # labels and propagation
    "KNOWN_ATTRIBUTES",
    "PROVENANCE_ANNOTATED",
    "PROVENANCE_PROPAGATED",
    "CropLabel",
    "LabelStore",
    "LabelStoreError",
    "LabelledObject",
    "ascii_projection",
    "describe_crop",
    "validate_attributes",
    # region proposal config
    "RegionProposalConfig",
    # class vocabulary
    "ClassEntry",
    "LabelMap",
    "LabelMapError",
    "load_label_map",
    # splits and manifests
    "SplitAssignment",
    "SplitError",
    "assign_sessions_to_splits",
    "class_distribution_report",
    "format_class_distribution_report",
    "write_manifests",
]
