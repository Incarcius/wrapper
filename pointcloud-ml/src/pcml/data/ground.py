"""Ground-plane estimation and removal.

On a sports field this is **not** an optional preprocessing nicety, it is a
prerequisite for the whole cluster-then-classify approach. Measured in
``tests/test_ground.py`` against a 0.5 m DBSCAN radius, ground density puts the
scene into one of three regimes, and two of them are unusable:

* **Dense ground** (≲0.25 m spacing; Mid-70 nadir at ≲15 m altitude) — every
  object standing on the grass is connected to it, and therefore to every other
  object. Clustering collapses the whole scene into **one** field-sized
  component.
* **Transitional** (~0.3–0.35 m spacing; roughly 20–30 m altitude) — the
  dangerous band. The ground fragments, and the fragments are *large enough to
  clear the minimum-points floor*, so they are handed on as though they were
  real objects. Measured: 6–13 clusters kept, of which 3–10 are ground debris.
* **Very sparse** (≳0.45 m spacing) — the ground still fragments heavily (30+
  components) but every fragment falls below the floor, so the filter absorbs
  them and only real objects survive. The one regime that degrades gracefully.

Removing the ground first is what makes instances separable in all three.

The flip side is that a playing field is close to the ideal case for RANSAC plane
fitting: genuinely planar, overwhelmingly the largest surface in view, and
roughly perpendicular to the sensor's nominal down direction. So the cheapest
possible method is also the right one here.

A NumPy RANSAC is the reference implementation rather than Open3D's
``segment_plane`` because it is deterministic given a seeded generator, which
the export-parity and reproducibility requirements need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["GroundPlane", "fit_ground_plane", "remove_ground", "split_ground"]


@dataclass(frozen=True)
class GroundPlane:
    """A fitted plane ``normal . p + offset = 0``, with the normal pointing up.

    "Up" means ``normal[2] > 0``, so :meth:`height_above` is positive for points
    above the ground regardless of which three points RANSAC happened to draw.
    """

    normal: np.ndarray  # (3,) unit vector, normal[2] > 0
    offset: float  # d in n.p + d = 0
    n_inliers: int
    inlier_fraction: float

    @property
    def tilt_deg(self) -> float:
        """Angle between the plane normal and the sensor's +z axis, in degrees.

        With a nadir-mounted sensor on a level drone over a level field this is
        near 0. A large value means drone attitude, a sloped field, or a bad fit.
        """
        return math.degrees(math.acos(min(1.0, abs(float(self.normal[2])))))

    def height_above(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from the plane, in metres. Positive = above ground."""
        return points[:, :3] @ self.normal + self.offset

    def ground_z_at_origin(self) -> float:
        """Plane height on the sensor's z axis, for the aux ground reference."""
        if abs(self.normal[2]) < 1e-9:
            return float("nan")
        return float(-self.offset / self.normal[2])


def _plane_from_three(p: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    """Plane through three points, or None if they are collinear."""
    v1 = p[1] - p[0]
    v2 = p[2] - p[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal = normal / norm
    if normal[2] < 0:  # orient upward so height_above is signed consistently
        normal = -normal
    return normal, float(-normal @ p[0])


def fit_ground_plane(
    points: np.ndarray,
    distance_threshold_m: float = 0.15,
    num_iterations: int = 200,
    max_tilt_deg: float = 30.0,
    min_inlier_fraction: float = 0.15,
    max_fit_points: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Optional[GroundPlane]:
    """RANSAC-fit the dominant near-horizontal plane.

    Args:
        points: ``(n, >=3)`` array in metres, sensor frame.
        distance_threshold_m: inlier band half-thickness. 0.15 m suits mown
            grass plus sensor noise; too small and the fit chases noise, too
            large and it eats the bottom of every object standing on the ground.
        num_iterations: RANSAC trials.
        max_tilt_deg: reject candidate planes whose normal is further than this
            from the sensor's +z axis. This is what stops the fit latching onto a
            large wall or a practice net, which are also planar and can be large
            in view. 30 deg allows for real drone attitude and a sloped field.
        min_inlier_fraction: if the best plane explains less than this fraction
            of points, report **no plane** rather than a bad one. Removing a
            wrongly-fitted "ground" would delete real objects, so failing loudly
            is the safer default.
        max_fit_points: subsample cap for the fitting stage. Inlier counting is
            done on the subsample; the returned plane is then applied to the full
            cloud by the caller. Keeps the cost bounded on dense frames.
        rng: seeded generator. Required for reproducibility; defaults to a
            fixed seed rather than system entropy, so a forgotten argument
            cannot silently make results unrepeatable.

    Returns:
        The fitted :class:`GroundPlane`, or ``None`` if no acceptable plane was
        found.
    """
    if distance_threshold_m <= 0:
        raise ValueError(f"distance_threshold_m must be > 0, got {distance_threshold_m}")
    if not 0.0 < min_inlier_fraction <= 1.0:
        raise ValueError(
            f"min_inlier_fraction must be in (0, 1], got {min_inlier_fraction}"
        )

    if rng is None:
        rng = np.random.default_rng(0)

    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    n = xyz.shape[0]
    if n < 3:
        return None

    if n > max_fit_points:
        fit_set = xyz[rng.choice(n, size=max_fit_points, replace=False)]
    else:
        fit_set = xyz

    cos_limit = math.cos(math.radians(max_tilt_deg))
    best: Optional[tuple[np.ndarray, float, int]] = None

    for _ in range(num_iterations):
        trio = fit_set[rng.choice(fit_set.shape[0], size=3, replace=False)]
        plane = _plane_from_three(trio)
        if plane is None:
            continue
        normal, offset = plane
        # Horizontality gate, applied before the expensive inlier count.
        if normal[2] < cos_limit:
            continue

        dist = np.abs(fit_set @ normal + offset)
        n_inliers = int(np.count_nonzero(dist <= distance_threshold_m))
        if best is None or n_inliers > best[2]:
            best = (normal, offset, n_inliers)

    if best is None:
        return None

    normal, offset, n_inliers = best
    fraction = n_inliers / fit_set.shape[0]
    if fraction < min_inlier_fraction:
        return None

    # Refit by least squares on the inliers: the three-point plane is only as
    # good as the three points drawn, and the refit typically halves the
    # residual, which matters because the threshold then sits symmetrically.
    inliers = fit_set[np.abs(fit_set @ normal + offset) <= distance_threshold_m]
    centroid = inliers.mean(axis=0)
    centred = inliers - centroid
    # Smallest singular vector of the centred inliers is the plane normal.
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    refined = vh[-1]
    if refined[2] < 0:
        refined = -refined
    if refined[2] >= cos_limit:
        normal = refined
        offset = float(-refined @ centroid)
        n_inliers = int(
            np.count_nonzero(np.abs(fit_set @ normal + offset) <= distance_threshold_m)
        )
        fraction = n_inliers / fit_set.shape[0]

    return GroundPlane(
        normal=np.asarray(normal, dtype=np.float64),
        offset=float(offset),
        n_inliers=n_inliers,
        inlier_fraction=float(fraction),
    )


def split_ground(
    points: np.ndarray,
    distance_threshold_m: float = 0.15,
    num_iterations: int = 200,
    max_tilt_deg: float = 30.0,
    min_inlier_fraction: float = 0.15,
    min_height_m: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[GroundPlane]]:
    """Separate ground from non-ground, keeping **both**.

    Prefer this over :func:`remove_ground` in anything that might outlive the
    object-classification task. For discrete objects the ground is noise; for
    the intended later domains it is the signal itself — crop height and canopy
    structure in agriculture, rutting and surface deviation in road quality are
    all measured *on* the ground surface. Throwing those points away at the
    preprocessing stage would be the single hardest thing to undo later.

    Args:
        min_height_m: clearance above the inlier band for the non-ground set.
            Raising it trims the skirt of ground points that survive at the base
            of each object, which is what otherwise bridges two adjacent
            clusters; the cost is losing the lowest part of short objects. It
            creates a deliberate gap: points between the band and the clearance
            belong to neither returned set.
        (others as :func:`fit_ground_plane`)

    Returns:
        ``(ground_points, non_ground_points, plane)``. If no plane is found,
        ground is empty, non-ground is the **unchanged** input, and ``plane`` is
        ``None`` — classifying points as ground on the strength of a rejected
        fit would be worse than doing nothing.
    """
    pts = np.ascontiguousarray(points, dtype=np.float32)
    empty = np.zeros((0, 3), dtype=np.float32)
    if pts.shape[0] == 0:
        return empty, pts, None

    plane = fit_ground_plane(
        pts,
        distance_threshold_m=distance_threshold_m,
        num_iterations=num_iterations,
        max_tilt_deg=max_tilt_deg,
        min_inlier_fraction=min_inlier_fraction,
        rng=rng,
    )
    if plane is None:
        return empty, pts, None

    height = plane.height_above(pts[:, :3].astype(np.float64))
    is_ground = np.abs(height) <= distance_threshold_m
    is_above = height > (distance_threshold_m + min_height_m)
    return pts[is_ground], pts[is_above], plane


def remove_ground(
    points: np.ndarray,
    distance_threshold_m: float = 0.15,
    num_iterations: int = 200,
    max_tilt_deg: float = 30.0,
    min_inlier_fraction: float = 0.15,
    min_height_m: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, Optional[GroundPlane]]:
    """Drop ground points, returning the rest plus the fitted plane.

    Convenience wrapper over :func:`split_ground` for the object-classification
    path, where the ground genuinely is noise.

    Returns:
        ``(non_ground_points, plane)``. If no plane is found the input is
        returned **unchanged** with ``plane=None``.
    """
    _, non_ground, plane = split_ground(
        points,
        distance_threshold_m=distance_threshold_m,
        num_iterations=num_iterations,
        max_tilt_deg=max_tilt_deg,
        min_inlier_fraction=min_inlier_fraction,
        min_height_m=min_height_m,
        rng=rng,
    )
    return non_ground, plane
