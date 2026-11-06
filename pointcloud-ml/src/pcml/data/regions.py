"""Region proposal: how a scene is divided into things to classify.

This is the one stage that does **not** survive a change of domain, so it is
isolated behind a single interface.

The field proof-of-concept classifies discrete objects, so regions come from
DBSCAN instance clustering. The intended later domains do not have discrete
objects at all:

* **Agriculture** — a crop field is a continuous surface. "Cluster the wheat
  into objects" is meaningless; what is wanted is a label (or a measurement)
  per patch of ground.
* **Road quality** — likewise continuous. Defects are properties of a surface
  patch, not instances to be separated.

For those, regions are **fixed spatial tiles**: a grid over the ground plane.
Everything downstream — normalisation, FPS, the aux features, the adapter, the
model, training, export, the Jetson work — is unchanged, because a proposer
returns the same :class:`~pcml.data.clustering.Cluster` records either way. That
is the whole point of putting the seam here: the domain pivot costs a config
line and a retrain, not a rewrite.

``Cluster`` is reused rather than renamed to ``Region`` to avoid churning code
that already depends on it; read it as "a region of points with its metric
geometry", which is all any consumer uses it for.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .clustering import Cluster, cluster_scene, finite_points_only
from .ground import GroundPlane

__all__ = [
    "PROPOSERS",
    "build_proposer",
    "propose_dbscan_instances",
    "propose_ground_tiles",
]


def propose_dbscan_instances(
    points: np.ndarray,
    eps_m: float = 0.5,
    min_samples: int = 10,
    min_region_points: int = 50,
    max_regions: Optional[int] = 16,
    use_open3d: bool = True,
    **_ignored: Any,
) -> tuple[list[Cluster], int]:
    """Discrete-object proposer: one region per DBSCAN instance.

    Correct when the scene contains separable objects standing clear of a
    removed ground plane. Fails by construction on continuous surfaces, where
    the "objects" are whatever the radius happens to carve out.
    """
    return cluster_scene(
        points,
        eps_m=eps_m,
        min_samples=min_samples,
        min_cluster_points=min_region_points,
        max_clusters=max_regions,
        use_open3d=use_open3d,
    )


def _tile_regions(
    xyz: np.ndarray,
    tile_size_m: float,
    min_region_points: int,
    offset: tuple[float, float],
) -> list[Cluster]:
    """Non-overlapping tiles on a grid shifted by ``offset``."""
    shifted = xyz[:, :2] - np.asarray(offset, dtype=np.float64)
    keys = np.floor(shifted / tile_size_m).astype(np.int64)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()

    regions: list[Cluster] = []
    # Only visit tiles that can possibly qualify.
    for tile_index in np.flatnonzero(counts >= min_region_points):
        member = xyz[inverse == tile_index]
        regions.append(Cluster.from_points(member))
    return regions


def propose_ground_tiles(
    points: np.ndarray,
    tile_size_m: float = 2.0,
    min_region_points: int = 50,
    max_regions: Optional[int] = 64,
    overlap_factor: int = 1,
    plane: Optional[GroundPlane] = None,
    **_ignored: Any,
) -> tuple[list[Cluster], int]:
    """Continuous-surface proposer: one region per fixed ground tile.

    The grid is laid out in the sensor's x-y plane. When a fitted ``plane`` is
    supplied the tiling is unchanged, but it is recorded that the caller has a
    ground reference — the plane matters to the *features* (height above ground,
    deviation from the surface), not to the tiling.

    Args:
        points: ``(n, >=3)`` array in metres. For surface work this is the
            cloud **with** the ground retained — the ground is the signal here,
            not noise.
        tile_size_m: edge length of a tile. The scale at which labels are
            defined: a 2 m tile asks "what is this 2 m patch", a 0.5 m tile asks
            a much finer question and needs proportionally denser data.
        min_region_points: tiles with fewer points are not emitted. At 1/R^2
            density the far edge of the view will fall below this, and that is
            the honest outcome — a sparse tile carries no surface information.
        max_regions: cap, largest-first, to bound latency. Tiling a wide view
            at a small tile size produces far more regions than DBSCAN does, so
            this cap binds much harder than in the object case.
        overlap_factor: 1 = non-overlapping. k > 1 adds ``k^2 - 1`` shifted
            grids (offsets of ``tile/k``), so a feature straddling a tile
            boundary still lands whole inside some tile. Cost scales as
            ``k^2``.

    Returns:
        ``(regions, n_found)`` with ``regions`` sorted by descending point count
        and capped, and ``n_found`` the count before the cap.
    """
    if tile_size_m <= 0:
        raise ValueError(f"tile_size_m must be > 0, got {tile_size_m}")
    if overlap_factor < 1:
        raise ValueError(f"overlap_factor must be >= 1, got {overlap_factor}")

    pts = finite_points_only(np.asarray(points, dtype=np.float32))
    if pts.shape[0] == 0:
        return [], 0

    xyz = pts[:, :3]
    step = tile_size_m / overlap_factor

    regions: list[Cluster] = []
    for i in range(overlap_factor):
        for j in range(overlap_factor):
            regions.extend(
                _tile_regions(
                    xyz, tile_size_m, min_region_points, (i * step, j * step)
                )
            )

    n_found = len(regions)
    regions.sort(key=lambda r: r.n_points_raw, reverse=True)
    if max_regions is not None:
        regions = regions[:max_regions]
    return regions, n_found


#: name -> proposer. The domain pivot is a change of this key in config.
PROPOSERS: dict[str, Callable[..., tuple[list[Cluster], int]]] = {
    "dbscan_instances": propose_dbscan_instances,
    "ground_tiles": propose_ground_tiles,
}


def build_proposer(config: dict[str, Any]) -> Callable[[np.ndarray], tuple[list[Cluster], int]]:
    """Resolve a proposer from the ``region_proposal`` config section.

    Args:
        config: must contain ``method``; remaining keys are passed through as
            keyword arguments. Unknown keys are tolerated by the proposers
            (``**_ignored``) so that one config file can carry the settings for
            several methods without each one rejecting the others' keys.

    Raises:
        KeyError: if ``method`` is absent.
        ValueError: if the method is not registered.
    """
    if "method" not in config:
        raise KeyError("region_proposal config needs a 'method' key")

    method = str(config["method"]).lower()
    if method not in PROPOSERS:
        raise ValueError(
            f"unknown region proposal method {method!r}; "
            f"available: {sorted(PROPOSERS)}"
        )

    proposer = PROPOSERS[method]
    kwargs = {k: v for k, v in config.items() if k != "method"}

    def _run(points: np.ndarray) -> tuple[list[Cluster], int]:
        return proposer(points, **kwargs)

    return _run
