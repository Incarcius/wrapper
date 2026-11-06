"""Frame readers: a recording on disk -> ``(points_xyz_float32, stamp)`` per frame.

Every reader yields the same thing, so nothing downstream knows or cares what the
recording was stored as:

* ``.npy`` / ``.npz`` — always available, pure NumPy. The format used for
  synthetic recordings and for anything exported from another tool.
* ``.pcd`` / ``.ply`` — via Open3D when it imports, and via a NumPy parser when
  it does not. **In this environment Open3D is installed but does not import**
  (missing ``libusb-1.0.so.0``), so the NumPy path is the one that actually runs
  and is therefore the one the tests exercise.
* ROS 2 ``.db3`` bags — ``sensor_msgs/msg/PointCloud2`` read **without rclpy**,
  which is not installed and will not be on the dev laptop. The sqlite3 tables
  are read directly and the CDR payload is deserialised here.
* a directory — a rosbag2 directory (``metadata.yaml`` + ``*.db3``), or a
  directory of per-frame files.

## Why the bag parser reads field offsets instead of assuming x,y,z

A Livox ``PointCloud2`` is not three float32s per point. ``livox_ros_driver2``
publishes xyz plus intensity/reflectivity, ``tag`` and ``line``, and depending on
the driver version and its ``xfer_format`` the field order, the per-point stride
and the presence of a per-point timestamp all change. Slicing the first 12 bytes
of each point would appear to work — it would produce plausible-looking garbage
whenever the layout differs. So the offsets, datatypes and ``point_step`` come
from the message header, every time, and a missing ``x``/``y``/``z`` field is a
loud error rather than a silent reinterpretation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from dataclasses import dataclass
from dataclasses import field as dc_field  # `field` is taken: RecordingMeta has one
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np

try:  # pragma: no cover - import guard
    import yaml

    _HAS_YAML = True
except ImportError:  # pragma: no cover - import guard
    _HAS_YAML = False

try:  # pragma: no cover - import guard; Open3D is installed but unimportable here
    import open3d as o3d

    _HAS_OPEN3D = True
except Exception:  # noqa: BLE001 - Open3D raises ImportError *or* OSError
    _HAS_OPEN3D = False


__all__ = [
    "RecordingError",
    "RecordingMeta",
    "Recording",
    "open_recording",
    "iter_recordings",
    "DEFAULT_LIVOX_TOPIC",
    "NOMINAL_FRAME_PERIOD_S",
    "read_npy_frames",
    "read_pcd",
    "read_ply",
    "read_pointcloud2_bag",
    "decode_pointcloud2_cdr",
    "list_bag_topics",
    "sha256_of",
]

#: The topic the collection plan records (``ros2 bag record /livox/lidar``).
DEFAULT_LIVOX_TOPIC = "/livox/lidar"

#: Used **only** to synthesise stamps for formats that carry none (a directory of
#: ``.npy`` frames, say). Flagged as synthetic on the :class:`Recording` so that
#: nothing downstream mistakes it for a sensor stamp.
NOMINAL_FRAME_PERIOD_S = 0.1

_POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"

#: ``sensor_msgs/PointField`` datatype enum -> NumPy type code.
_POINTFIELD_DTYPES: dict[int, str] = {
    1: "i1",  # INT8
    2: "u1",  # UINT8
    3: "i2",  # INT16
    4: "u2",  # UINT16
    5: "i4",  # INT32
    6: "u4",  # UINT32
    7: "f4",  # FLOAT32
    8: "f8",  # FLOAT64
}


class RecordingError(RuntimeError):
    """A recording that cannot be read, or that would be read wrongly."""


def sha256_of(path: Union[str, Path], chunk_bytes: int = 1 << 20) -> str:
    """Hash a file's bytes, streamed so a multi-GB bag does not land in memory.

    Recorded in the crop index: it is what makes "reproducible from the raw file"
    verifiable rather than assumed, and it catches a recording that was
    re-exported or truncated between extraction runs.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


# ======================================================================
# Per-session metadata (docs/data_collection_plan.md §4.2)
# ======================================================================
#: Fields from §4.2 that cannot be reconstructed after the flight. Absent ones are
#: reported by :meth:`RecordingMeta.missing_fields` rather than silently defaulted.
_REQUIRED_META_FIELDS: tuple[str, ...] = (
    "session_id",
    "date",
    "altitude_m",
    "mount_angle_deg",
    "motion",
    "field",
)


@dataclass(frozen=True)
class RecordingMeta:
    """The per-session capture metadata of ``data_collection_plan.md`` §4.2.

    Every field here is something that **cannot be reconstructed later** from the
    points. Altitude and mount angle decide whether ``person`` was even
    resolvable in this session (§1); ``motion`` and ``airspeed_mps`` decide
    whether the frames are smeared; ``session_id`` is what the splits are made by
    (``docs/task_definition.md`` §7), so a recording without one cannot be put in
    a manifest at all.

    Loaded from a JSON or YAML sidecar next to the recording. Missing fields are
    allowed — a real capture session will sometimes forget one — but they are
    *reported*, never defaulted to something plausible.
    """

    session_id: str
    date: Optional[str] = None
    time: Optional[str] = None
    sensor: str = "livox_mid70"
    altitude_m: Optional[float] = None
    mount_angle_deg: Optional[float] = None  # degrees below horizontal
    motion: Optional[str] = None  # "hover" | "moving"
    airspeed_mps: Optional[float] = None
    # NOTE: this attribute is named `field` to match the §4.2 field name verbatim
    # ("which field (cricket / football)"), which means `dataclasses.field` is
    # shadowed for the rest of the class body — hence the `dc_field` alias below.
    field: Optional[str] = None  # "cricket" | "football"
    field_part: Optional[str] = None
    in_view: Optional[str] = None
    weather: Optional[str] = None
    grass_wet: Optional[bool] = None
    staged: Optional[Sequence[Mapping[str, Any]]] = None
    notes: Optional[str] = None
    extra: Mapping[str, Any] = dc_field(default_factory=dict)
    path: Optional[Path] = None

    # ------------------------------------------------------------------
    @staticmethod
    def template(session_id: str = "session_01") -> dict[str, Any]:
        """A blank §4.2 sidecar, to be filled in **at capture time**.

        Written by ``scripts/label_clusters.py --write-meta-template`` so that the
        file exists before the flight rather than being reconstructed from memory
        afterwards, which is the failure §4.2 is warning about.
        """
        return {
            "session_id": session_id,
            "date": None,  # ISO date, e.g. "2026-10-20"
            "time": None,  # local clock time, e.g. "16:40"
            "sensor": "livox_mid70",
            "altitude_m": None,  # metres above ground
            "mount_angle_deg": None,  # degrees below horizontal; 30-45 per §1
            "motion": None,  # "hover" or "moving"
            "airspeed_mps": None,  # 0 for a hover
            "field": None,  # "cricket" or "football"
            "field_part": None,  # which part of it
            "in_view": None,  # what is in view, and roughly where
            "weather": None,  # dry / wet grass / dust
            "grass_wet": None,
            "staged": [],  # [{"what": "person", "distance_m": 10.0, "pose": "standing"}]
            "notes": None,
        }

    @staticmethod
    def from_dict(doc: Mapping[str, Any], path: Optional[Path] = None) -> "RecordingMeta":
        known = {
            "session_id",
            "date",
            "time",
            "sensor",
            "altitude_m",
            "mount_angle_deg",
            "motion",
            "airspeed_mps",
            "field",
            "field_part",
            "in_view",
            "weather",
            "grass_wet",
            "staged",
            "notes",
        }
        session_id = doc.get("session_id")
        if not session_id or not isinstance(session_id, str):
            raise RecordingError(
                f"session metadata {path or ''} has no usable `session_id`. Splits are "
                "by session (docs/task_definition.md §7), so a recording without a "
                "session id cannot be put in a manifest."
            )

        def num(key: str) -> Optional[float]:
            v = doc.get(key)
            return None if v is None else float(v)

        motion = doc.get("motion")
        if motion is not None:
            motion = str(motion).lower()
            if motion not in ("hover", "moving"):
                raise RecordingError(
                    f"session {session_id}: motion must be 'hover' or 'moving', got "
                    f"{motion!r}. It decides whether frames are smeared and whether "
                    "accumulation is valid (docs/data_collection_plan.md §1.3)."
                )

        return RecordingMeta(
            session_id=session_id,
            date=doc.get("date"),
            time=doc.get("time"),
            sensor=str(doc.get("sensor", "livox_mid70")),
            altitude_m=num("altitude_m"),
            mount_angle_deg=num("mount_angle_deg"),
            motion=motion,
            airspeed_mps=num("airspeed_mps"),
            field=doc.get("field"),
            field_part=doc.get("field_part"),
            in_view=doc.get("in_view"),
            weather=doc.get("weather"),
            grass_wet=doc.get("grass_wet"),
            staged=doc.get("staged"),
            notes=doc.get("notes"),
            extra={k: v for k, v in doc.items() if k not in known},
            path=path,
        )

    @staticmethod
    def from_file(path: Union[str, Path]) -> "RecordingMeta":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            if not _HAS_YAML:  # pragma: no cover - pyyaml is a hard dependency
                raise RecordingError("reading a YAML sidecar needs pyyaml")
            doc = yaml.safe_load(text)
        else:
            doc = json.loads(text)
        if not isinstance(doc, Mapping):
            raise RecordingError(f"{p} must contain a mapping at the top level")
        return RecordingMeta.from_dict(doc, path=p)

    @staticmethod
    def find_for(recording_path: Union[str, Path]) -> Optional["RecordingMeta"]:
        """Look for a sidecar beside a recording.

        Tried, in order: ``<name>.meta.json``, ``<name>.meta.yaml``,
        ``<name>.meta.yml``, and — for a directory recording — ``meta.json`` /
        ``session.yaml`` inside it.
        """
        p = Path(recording_path)
        candidates: list[Path] = []
        for suffix in (".meta.json", ".meta.yaml", ".meta.yml"):
            candidates.append(p.with_name(p.name + suffix))
            if p.suffix:
                candidates.append(p.with_suffix(suffix))
        if p.is_dir():
            candidates += [
                p / "meta.json",
                p / "meta.yaml",
                p / "session.json",
                p / "session.yaml",
            ]
        for c in candidates:
            if c.is_file():
                return RecordingMeta.from_file(c)
        return None

    # ------------------------------------------------------------------
    def missing_fields(self) -> tuple[str, ...]:
        """§4.2 fields that were not supplied. Reported, never guessed."""
        return tuple(f for f in _REQUIRED_META_FIELDS if getattr(self, f, None) is None)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "session_id": self.session_id,
            "date": self.date,
            "time": self.time,
            "sensor": self.sensor,
            "altitude_m": self.altitude_m,
            "mount_angle_deg": self.mount_angle_deg,
            "motion": self.motion,
            "airspeed_mps": self.airspeed_mps,
            "field": self.field,
            "field_part": self.field_part,
            "in_view": self.in_view,
            "weather": self.weather,
            "grass_wet": self.grass_wet,
            "staged": list(self.staged) if self.staged is not None else None,
            "notes": self.notes,
        }
        out.update(dict(self.extra))
        return out


# ======================================================================
# .npy / .npz
# ======================================================================
def _as_xyz(array: np.ndarray, where: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RecordingError(
            f"{where}: expected a (n, >=3) point array, got shape {arr.shape}. "
            "Columns beyond the first three are dropped (xyz only, per "
            "docs/task_definition.md §4)."
        )
    return np.ascontiguousarray(arr[:, :3], dtype=np.float32)


def read_npy_frames(path: Union[str, Path]) -> Iterator[tuple[np.ndarray, Optional[float]]]:
    """Frames from a ``.npy`` or ``.npz`` file.

    Conventions, in the order they are tried:

    * ``.npy`` — one ``(n, >=3)`` array, one frame, no stamp.
    * ``.npz`` with ``points`` ``(total, 3)`` **and** ``offsets`` ``(F+1,)`` —
      ``F`` frames, concatenated. Optional ``stamps`` ``(F,)``. This is the
      packed form the crop store also uses.
    * ``.npz`` with ``points`` only — one frame, optional scalar ``stamp``.
    * any other ``.npz`` — every 2-D array, in sorted key order, is one frame.

    Yields ``(points, stamp_or_None)``. ``None`` means the file carried no stamp;
    the caller decides what to do about that rather than having a fabricated
    value pushed on it.
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        yield _as_xyz(np.load(p, allow_pickle=False), str(p)), None
        return

    with np.load(p, allow_pickle=False) as npz:
        keys = list(npz.files)
        if "points" in keys and "offsets" in keys:
            points = np.asarray(npz["points"])
            offsets = np.asarray(npz["offsets"]).astype(np.int64).ravel()
            stamps = np.asarray(npz["stamps"]).ravel() if "stamps" in keys else None
            n_frames = offsets.shape[0] - 1
            if n_frames < 0:
                raise RecordingError(f"{p}: `offsets` must have at least one entry")
            if stamps is not None and stamps.shape[0] != n_frames:
                raise RecordingError(
                    f"{p}: `stamps` has {stamps.shape[0]} entries but `offsets` "
                    f"describes {n_frames} frames"
                )
            for i in range(n_frames):
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                if not 0 <= lo <= hi <= points.shape[0]:
                    raise RecordingError(
                        f"{p}: frame {i} offsets ({lo}, {hi}) are outside `points` "
                        f"of length {points.shape[0]}"
                    )
                stamp = float(stamps[i]) if stamps is not None else None
                yield _as_xyz(points[lo:hi], f"{p}[frame {i}]"), stamp
            return

        if "points" in keys:
            stamp = float(npz["stamp"]) if "stamp" in keys else None
            yield _as_xyz(npz["points"], str(p)), stamp
            return

        frame_keys = [k for k in sorted(keys) if np.asarray(npz[k]).ndim == 2]
        if not frame_keys:
            raise RecordingError(
                f"{p}: no 2-D point arrays found. Expected `points` (+ optional "
                f"`offsets`/`stamps`), or one 2-D array per frame; got keys {keys}"
            )
        for k in frame_keys:
            yield _as_xyz(npz[k], f"{p}[{k}]"), None


# ======================================================================
# .pcd
# ======================================================================
_PCD_TYPE_CODES = {"F": "f", "I": "i", "U": "u"}


def _parse_pcd_header(fh) -> tuple[dict[str, list[str]], int]:
    """Read a PCD header, returning the fields and the byte offset of the data."""
    header: dict[str, list[str]] = {}
    while True:
        raw = fh.readline()
        if not raw:
            raise RecordingError("PCD ended before its DATA line")
        line = raw.decode("ascii", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        header[key] = parts[1:]
        if key == "DATA":
            return header, fh.tell()


def read_pcd(path: Union[str, Path], use_open3d: bool = True) -> np.ndarray:
    """Read one frame from a ``.pcd``.

    Open3D is preferred when it imports, because it handles every PCD variant.
    The NumPy fallback handles ``DATA ascii`` and ``DATA binary``, which is what
    ``pcl_ros`` and ``ros2 bag`` exports actually produce. ``binary_compressed``
    is refused with a message rather than mis-parsed.
    """
    p = Path(path)
    if use_open3d and _HAS_OPEN3D:
        pcd = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(pcd.points)
        if pts.size:
            return np.ascontiguousarray(pts[:, :3], dtype=np.float32)
        # Fall through: Open3D returns an empty cloud on formats it declined,
        # which is indistinguishable from a genuinely empty frame. Re-read with
        # the NumPy parser so a parse failure is an error, not an empty frame.

    with open(p, "rb") as fh:
        header, data_offset = _parse_pcd_header(fh)

        data_mode = (header.get("DATA") or ["ascii"])[0].lower()
        names = header.get("FIELDS")
        if not names:
            raise RecordingError(f"{p}: PCD header has no FIELDS line")
        sizes = [int(s) for s in header.get("SIZE", [])]
        types = [t.upper() for t in header.get("TYPE", [])]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(names))]
        if not (len(names) == len(sizes) == len(types) == len(counts)):
            raise RecordingError(
                f"{p}: PCD FIELDS/SIZE/TYPE/COUNT lengths disagree "
                f"({len(names)}/{len(sizes)}/{len(types)}/{len(counts)})"
            )
        missing = [a for a in ("x", "y", "z") if a not in names]
        if missing:
            raise RecordingError(f"{p}: PCD has no {missing} field(s); FIELDS={names}")

        if "POINTS" in header:
            n_points = int(header["POINTS"][0])
        else:
            width = int(header.get("WIDTH", ["0"])[0])
            height = int(header.get("HEIGHT", ["1"])[0])
            n_points = width * height

        if data_mode == "ascii":
            fh.seek(data_offset)
            flat = np.loadtxt(fh, dtype=np.float64, ndmin=2)
            if flat.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
            expected_cols = sum(counts)
            if flat.shape[1] != expected_cols:
                raise RecordingError(
                    f"{p}: ascii PCD has {flat.shape[1]} columns but the header "
                    f"describes {expected_cols}"
                )
            # Column index of each field, accounting for COUNT > 1 fields.
            col_of: dict[str, int] = {}
            col = 0
            for name, count in zip(names, counts):
                col_of[name] = col
                col += count
            xyz = flat[:, [col_of["x"], col_of["y"], col_of["z"]]]
            return np.ascontiguousarray(xyz, dtype=np.float32)

        if data_mode == "binary":
            dtype_fields = []
            for name, size, type_code, count in zip(names, sizes, types, counts):
                if type_code not in _PCD_TYPE_CODES:
                    raise RecordingError(f"{p}: unsupported PCD TYPE {type_code!r}")
                base = f"<{_PCD_TYPE_CODES[type_code]}{size}"
                dtype_fields.append((name, base, (count,)) if count > 1 else (name, base))
            dtype = np.dtype(dtype_fields)
            fh.seek(data_offset)
            buf = fh.read(n_points * dtype.itemsize)
            if len(buf) < n_points * dtype.itemsize:
                raise RecordingError(
                    f"{p}: binary PCD is truncated — header claims {n_points} points "
                    f"({n_points * dtype.itemsize} bytes), file holds {len(buf)}"
                )
            arr = np.frombuffer(buf, dtype=dtype, count=n_points)
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
            return np.ascontiguousarray(xyz, dtype=np.float32)

        raise RecordingError(
            f"{p}: PCD DATA mode {data_mode!r} is not supported by the NumPy "
            "fallback (only 'ascii' and 'binary'). Open3D would handle it, but it "
            "does not import in this environment. Re-export the file as ascii or "
            "binary PCD, or as .npy."
        )


# ======================================================================
# .ply
# ======================================================================
_PLY_TYPE_CODES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_ply(path: Union[str, Path], use_open3d: bool = True) -> np.ndarray:
    """Read the vertex element of a ``.ply`` as xyz.

    As with PCD: Open3D first, then a NumPy parser covering ``ascii`` and
    ``binary_little_endian``, which is what every tool in this stack writes.
    """
    p = Path(path)
    if use_open3d and _HAS_OPEN3D:
        pcd = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(pcd.points)
        if pts.size:
            return np.ascontiguousarray(pts[:, :3], dtype=np.float32)

    with open(p, "rb") as fh:
        magic = fh.readline().strip()
        if magic != b"ply":
            raise RecordingError(f"{p}: not a PLY file (first line {magic!r})")

        fmt: Optional[str] = None
        current_element: Optional[str] = None
        vertex_count = 0
        properties: list[tuple[str, str]] = []  # (name, numpy type code)

        while True:
            raw = fh.readline()
            if not raw:
                raise RecordingError(f"{p}: PLY ended before end_header")
            line = raw.decode("ascii", errors="replace").strip()
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if parts[1] == "list":
                    raise RecordingError(
                        f"{p}: list properties on the vertex element are not supported"
                    )
                type_name = parts[1].lower()
                if type_name not in _PLY_TYPE_CODES:
                    raise RecordingError(f"{p}: unsupported PLY property type {parts[1]!r}")
                properties.append((parts[2], _PLY_TYPE_CODES[type_name]))
            elif parts[0] == "end_header":
                break

        names = [n for n, _ in properties]
        missing = [a for a in ("x", "y", "z") if a not in names]
        if missing:
            raise RecordingError(f"{p}: PLY vertex element has no {missing}; got {names}")

        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                raw = fh.readline()
                if not raw:
                    raise RecordingError(
                        f"{p}: ascii PLY is truncated — header claims {vertex_count} "
                        "vertices"
                    )
                rows.append([float(v) for v in raw.split()[: len(properties)]])
            if not rows:
                return np.zeros((0, 3), dtype=np.float32)
            flat = np.asarray(rows, dtype=np.float64)
            idx = [names.index(a) for a in ("x", "y", "z")]
            return np.ascontiguousarray(flat[:, idx], dtype=np.float32)

        if fmt in ("binary_little_endian", "binary_big_endian"):
            prefix = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(n, prefix + t) for n, t in properties])
            buf = fh.read(vertex_count * dtype.itemsize)
            if len(buf) < vertex_count * dtype.itemsize:
                raise RecordingError(
                    f"{p}: binary PLY is truncated — header claims {vertex_count} "
                    f"vertices ({vertex_count * dtype.itemsize} bytes), file holds "
                    f"{len(buf)}"
                )
            arr = np.frombuffer(buf, dtype=dtype, count=vertex_count)
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
            return np.ascontiguousarray(xyz, dtype=np.float32)

        raise RecordingError(f"{p}: unsupported PLY format {fmt!r}")


# ======================================================================
# ROS 2 .db3 bags: CDR deserialisation without rclpy
# ======================================================================
class _CdrReader:
    """Minimal little/big-endian CDR reader for one serialised ROS 2 message.

    CDR aligns each primitive to its own size, measured from the start of the
    body — i.e. **after** the 4-byte encapsulation header. Getting that origin
    wrong shifts every field after the first string, which is why the origin is
    explicit here instead of implied.
    """

    __slots__ = ("buf", "pos", "origin", "_le")

    def __init__(self, buf: bytes) -> None:
        if len(buf) < 4:
            raise RecordingError(
                f"CDR payload is {len(buf)} bytes; too short to hold even the "
                "encapsulation header"
            )
        # Encapsulation: byte 0 is 0x00, byte 1 selects the scheme.
        #   0x00 CDR_BE, 0x01 CDR_LE, 0x02 PL_CDR_BE, 0x03 PL_CDR_LE
        scheme = buf[1]
        if scheme not in (0x00, 0x01, 0x02, 0x03):
            raise RecordingError(
                f"unknown CDR encapsulation 0x{buf[0]:02x}{buf[1]:02x}. This reader "
                "handles plain CDR as written by rmw_fastrtps / rmw_cyclonedds."
            )
        self._le = scheme in (0x01, 0x03)
        self.buf = buf
        self.origin = 4
        self.pos = 4

    @property
    def little_endian(self) -> bool:
        return self._le

    def _align(self, size: int) -> None:
        rem = (self.pos - self.origin) % size
        if rem:
            self.pos += size - rem

    def _take(self, size: int) -> bytes:
        end = self.pos + size
        if end > len(self.buf):
            raise RecordingError(
                f"CDR payload is truncated: needed {size} bytes at offset {self.pos}, "
                f"message is {len(self.buf)} bytes"
            )
        out = self.buf[self.pos : end]
        self.pos = end
        return out

    def _scalar(self, code: str, size: int) -> Any:
        self._align(size)
        return struct.unpack(("<" if self._le else ">") + code, self._take(size))[0]

    def u8(self) -> int:
        return self._take(1)[0]

    def boolean(self) -> bool:
        return self.u8() != 0

    def i32(self) -> int:
        return int(self._scalar("i", 4))

    def u32(self) -> int:
        return int(self._scalar("I", 4))

    def string(self) -> str:
        length = self.u32()  # includes the trailing NUL
        if length == 0:
            return ""
        raw = self._take(length)
        return raw[:-1].decode("utf-8", errors="replace")

    def byte_array(self, length: int) -> bytes:
        return self._take(length)


@dataclass(frozen=True)
class _PointField:
    name: str
    offset: int
    datatype: int
    count: int


def decode_pointcloud2_cdr(payload: bytes) -> tuple[np.ndarray, float, str]:
    """Deserialise one CDR ``sensor_msgs/msg/PointCloud2`` to xyz + stamp.

    The point of this function is that it reads the **layout from the message**:
    each field's byte offset, its datatype and the per-point ``point_step`` all
    come from the header, so a Livox cloud carrying ``intensity``/``tag``/``line``
    (or a per-point timestamp, or the fields in an unexpected order) is read
    correctly instead of being reinterpreted as three float32s.

    Args:
        payload: the raw ``messages.data`` blob from a rosbag2 sqlite table.

    Returns:
        ``(points (n, 3) float32, stamp_seconds, frame_id)``.

    Raises:
        RecordingError: if ``x``, ``y`` or ``z`` is missing, if a field has
            ``count != 1``, or if the blob is truncated.
    """
    r = _CdrReader(payload)

    # std_msgs/Header
    sec = r.i32()
    nanosec = r.u32()
    frame_id = r.string()

    height = r.u32()
    width = r.u32()

    n_fields = r.u32()
    fields: list[_PointField] = []
    for _ in range(n_fields):
        name = r.string()
        offset = r.u32()
        datatype = r.u8()
        count = r.u32()
        fields.append(_PointField(name=name, offset=offset, datatype=datatype, count=count))

    is_bigendian = r.boolean()
    point_step = r.u32()
    _row_step = r.u32()
    data_len = r.u32()
    data = r.byte_array(data_len)
    # `is_dense` follows; not needed — non-finite points are filtered downstream
    # by pcml.data.clustering.finite_points_only.

    by_name = {f.name: f for f in fields}
    missing = [a for a in ("x", "y", "z") if a not in by_name]
    if missing:
        raise RecordingError(
            f"PointCloud2 has no {missing} field(s). Fields present: "
            f"{[f.name for f in fields]}. This reader deliberately refuses to guess "
            "the layout rather than slice the first 12 bytes of each point."
        )

    if point_step == 0:
        raise RecordingError("PointCloud2 has point_step == 0")

    n_points = int(height) * int(width)
    if n_points * point_step > len(data):
        raise RecordingError(
            f"PointCloud2 data is truncated: height*width = {n_points} points at "
            f"point_step {point_step} needs {n_points * point_step} bytes, blob holds "
            f"{len(data)}"
        )

    prefix = ">" if is_bigendian else "<"
    names, formats, offsets = [], [], []
    for axis in ("x", "y", "z"):
        f = by_name[axis]
        if f.datatype not in _POINTFIELD_DTYPES:
            raise RecordingError(
                f"PointCloud2 field {axis!r} has unknown datatype {f.datatype}; "
                f"known: {sorted(_POINTFIELD_DTYPES)}"
            )
        if f.count != 1:
            raise RecordingError(
                f"PointCloud2 field {axis!r} has count={f.count}; only count==1 is "
                "meaningful for a coordinate"
            )
        if f.offset + np.dtype(_POINTFIELD_DTYPES[f.datatype]).itemsize > point_step:
            raise RecordingError(
                f"PointCloud2 field {axis!r} at offset {f.offset} does not fit inside "
                f"point_step {point_step}"
            )
        names.append(axis)
        formats.append(prefix + _POINTFIELD_DTYPES[f.datatype])
        offsets.append(int(f.offset))

    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": int(point_step),
    })
    arr = np.frombuffer(data, dtype=dtype, count=n_points)
    xyz = np.stack(
        [arr["x"].astype(np.float32), arr["y"].astype(np.float32), arr["z"].astype(np.float32)],
        axis=1,
    )

    stamp = float(sec) + float(nanosec) * 1e-9
    return np.ascontiguousarray(xyz, dtype=np.float32), stamp, frame_id


def _bag_db3_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.db3"))
    if not files:
        raise RecordingError(
            f"{path} contains no .db3 files. A rosbag2 recording is a directory "
            "holding metadata.yaml plus one or more .db3 files."
        )
    return files


def list_bag_topics(path: Union[str, Path]) -> dict[str, dict[str, Any]]:
    """Topics in a bag: ``{name: {"type": ..., "n_messages": ...}}``.

    Exists so that "the topic is not in this bag" can be reported with the list
    of topics that *are*, which is the difference between a two-minute fix and an
    afternoon.
    """
    out: dict[str, dict[str, Any]] = {}
    for db3 in _bag_db3_files(Path(path)):
        conn = sqlite3.connect(f"file:{db3.as_posix()}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT topics.name, topics.type, COUNT(messages.id) "
                "FROM topics LEFT JOIN messages ON messages.topic_id = topics.id "
                "GROUP BY topics.id"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise RecordingError(f"{db3} is not a readable rosbag2 sqlite file: {exc}") from exc
        finally:
            conn.close()
        for name, type_name, count in rows:
            entry = out.setdefault(name, {"type": type_name, "n_messages": 0})
            entry["n_messages"] += int(count)
    return out


def read_pointcloud2_bag(
    path: Union[str, Path],
    topic: str = DEFAULT_LIVOX_TOPIC,
    expected_type: Optional[str] = _POINTCLOUD2_TYPE,
) -> Iterator[tuple[np.ndarray, Optional[float]]]:
    """Frames from a ROS 2 bag's ``PointCloud2`` topic, without rclpy.

    rclpy and the ROS 2 message packages are not installed on the dev laptop and
    will not be, so the sqlite tables are read directly and the CDR payload is
    deserialised by :func:`decode_pointcloud2_cdr`.

    The stamp used is the **message header stamp**, not the bag's receive
    timestamp: the header stamp is the sensor's, which is what
    ``FramePrediction.stamp`` is contracted to carry
    (``docs/task_definition.md`` §5).

    Args:
        path: a ``.db3`` file, or a rosbag2 directory containing one or more.
        topic: topic name. Defaults to the one ``§4.5`` says to record.
        expected_type: refuse the topic unless the bag declares this type. Pass
            ``None`` to skip the check.

    Raises:
        RecordingError: if the topic is absent (the message lists the topics that
            are present), or declares an unexpected type.
    """
    p = Path(path)
    db3_files = _bag_db3_files(p)

    seen_any_topic = False
    for db3 in db3_files:
        conn = sqlite3.connect(f"file:{db3.as_posix()}?mode=ro", uri=True)
        try:
            try:
                topic_rows = conn.execute(
                    "SELECT id, name, type FROM topics WHERE name = ?", (topic,)
                ).fetchall()
            except sqlite3.DatabaseError as exc:
                raise RecordingError(
                    f"{db3} is not a readable rosbag2 sqlite file: {exc}"
                ) from exc

            if not topic_rows:
                continue
            seen_any_topic = True

            for topic_id, _name, type_name in topic_rows:
                if expected_type is not None and type_name != expected_type:
                    raise RecordingError(
                        f"{db3}: topic {topic!r} has type {type_name!r}, expected "
                        f"{expected_type!r}. Livox can also publish "
                        "livox_ros_driver2/msg/CustomMsg; that needs its own reader. "
                        "Record with xfer_format set to PointCloud2 "
                        "(docs/data_collection_plan.md §4.5)."
                    )
                cursor = conn.execute(
                    "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                    (topic_id,),
                )
                for (blob,) in cursor:
                    points, stamp, _frame_id = decode_pointcloud2_cdr(bytes(blob))
                    yield points, stamp
        finally:
            conn.close()

    if not seen_any_topic:
        available = list_bag_topics(p)
        raise RecordingError(
            f"topic {topic!r} is not in {p}. Topics present: "
            f"{ {k: v['type'] for k, v in available.items()} }. "
            "Pass topic=... to open_recording()."
        )


# ======================================================================
# Uniform dispatcher
# ======================================================================
_FRAME_FILE_SUFFIXES = (".npy", ".npz", ".pcd", ".ply")


@dataclass
class Recording:
    """A uniform handle on one recording session, whatever it is stored as.

    Iterating yields ``(points_xyz_float32, stamp_seconds)`` per frame, in
    recorded order.

    Attributes:
        path: what was opened.
        kind: ``"npy"``, ``"pcd"``, ``"ply"``, ``"bag"`` or ``"frame_dir"``.
        session_id: from the §4.2 sidecar if there is one, else derived from the
            filename. Splits are by session, so this is load-bearing.
        stamp_source: ``"sensor"`` when the format carries real header stamps,
            ``"synthetic"`` when they are index * :data:`NOMINAL_FRAME_PERIOD_S`.
            A synthetic stamp is a frame counter in disguise and must never be
            treated as a measurement; this flag is what lets a consumer tell.
        meta: the parsed sidecar, or ``None`` if there is none.
    """

    path: Path
    kind: str
    session_id: str
    stamp_source: str
    meta: Optional[RecordingMeta] = None
    topic: Optional[str] = None
    use_open3d: bool = True
    _frame_files: tuple[Path, ...] = ()

    # ------------------------------------------------------------------
    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        """Yield ``(points, stamp)`` per frame, in recorded order."""
        for index, (points, stamp) in enumerate(self._raw_frames()):
            if stamp is None:
                stamp = index * NOMINAL_FRAME_PERIOD_S
            yield points, float(stamp)

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        return self.frames()

    def _raw_frames(self) -> Iterator[tuple[np.ndarray, Optional[float]]]:
        if self.kind == "npy":
            yield from read_npy_frames(self.path)
        elif self.kind == "pcd":
            yield read_pcd(self.path, use_open3d=self.use_open3d), None
        elif self.kind == "ply":
            yield read_ply(self.path, use_open3d=self.use_open3d), None
        elif self.kind == "bag":
            yield from read_pointcloud2_bag(self.path, topic=self.topic or DEFAULT_LIVOX_TOPIC)
        elif self.kind == "frame_dir":
            for f in self._frame_files:
                suffix = f.suffix.lower()
                if suffix in (".npy", ".npz"):
                    yield from read_npy_frames(f)
                elif suffix == ".pcd":
                    yield read_pcd(f, use_open3d=self.use_open3d), None
                else:
                    yield read_ply(f, use_open3d=self.use_open3d), None
        else:  # pragma: no cover - guarded by open_recording
            raise RecordingError(f"unknown recording kind {self.kind!r}")

    def describe(self) -> dict[str, Any]:
        """Summary for logs and for the crop index header."""
        return {
            "path": str(self.path),
            "kind": self.kind,
            "session_id": self.session_id,
            "stamp_source": self.stamp_source,
            "topic": self.topic,
            "has_meta": self.meta is not None,
            "meta_missing_fields": list(self.meta.missing_fields()) if self.meta else None,
        }


def _derive_session_id(path: Path) -> str:
    """Fall back to the filename when there is no sidecar.

    ``session_01_hover_12m.db3`` -> ``session_01_hover_12m``, which is exactly
    the naming ``§4.5``'s ``ros2 bag record -o`` produces.
    """
    name = path.name if path.is_dir() else path.stem
    return name or path.name


def open_recording(
    path: Union[str, Path],
    topic: str = DEFAULT_LIVOX_TOPIC,
    session_id: Optional[str] = None,
    use_open3d: bool = True,
) -> Recording:
    """Open any supported recording and return a uniform :class:`Recording`.

    Dispatch is by suffix, with directories inspected: a directory holding
    ``*.db3`` is a rosbag2 recording, a directory holding frame files is a frame
    sequence in sorted filename order.

    Args:
        path: file or directory.
        topic: which topic to read, for bags.
        session_id: override. Otherwise taken from the §4.2 sidecar, else derived
            from the filename.
        use_open3d: try Open3D for ``.pcd``/``.ply``. Harmless when Open3D does
            not import — the NumPy parsers take over.

    Raises:
        FileNotFoundError: if the path does not exist.
        RecordingError: for an unsupported format, with a note on what to do.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no recording at {p}")

    meta = RecordingMeta.find_for(p)

    if p.is_dir():
        if list(p.glob("*.db3")):
            kind, stamp_source, frame_files = "bag", "sensor", ()
        else:
            frame_files = tuple(
                f for f in sorted(p.iterdir()) if f.suffix.lower() in _FRAME_FILE_SUFFIXES
            )
            if not frame_files:
                raise RecordingError(
                    f"{p} holds neither .db3 files nor frame files "
                    f"({', '.join(_FRAME_FILE_SUFFIXES)})"
                )
            kind, stamp_source = "frame_dir", "synthetic"
    else:
        suffix = p.suffix.lower()
        frame_files = ()
        if suffix in (".npy", ".npz"):
            # A packed .npz may carry real stamps; checked lazily below.
            kind, stamp_source = "npy", "synthetic"
            if suffix == ".npz":
                with np.load(p, allow_pickle=False) as npz:
                    if "stamps" in npz.files or "stamp" in npz.files:
                        stamp_source = "sensor"
        elif suffix == ".pcd":
            kind, stamp_source = "pcd", "synthetic"
        elif suffix == ".ply":
            kind, stamp_source = "ply", "synthetic"
        elif suffix == ".db3":
            kind, stamp_source = "bag", "sensor"
        elif suffix in (".las", ".laz", ".lvx", ".lvx2", ".bag"):
            raise RecordingError(
                f"{p}: {suffix} is not implemented. No loader is written "
                "speculatively (data/README.md); add one when a recording in this "
                "format actually arrives. For .lvx, convert with Livox Viewer or "
                "record PointCloud2 to a ROS 2 bag instead "
                "(docs/data_collection_plan.md §4.5). .bag is ROS 1, which this "
                "project does not use."
            )
        else:
            raise RecordingError(
                f"{p}: unsupported recording format {suffix!r}. Supported: "
                f"{', '.join(_FRAME_FILE_SUFFIXES)}, .db3, or a directory of either."
            )

    resolved_session = session_id or (meta.session_id if meta else None) or _derive_session_id(p)

    return Recording(
        path=p,
        kind=kind,
        session_id=resolved_session,
        stamp_source=stamp_source,
        meta=meta,
        topic=topic if kind == "bag" else None,
        use_open3d=use_open3d,
        _frame_files=frame_files,
    )


def iter_recordings(
    paths: Iterable[Union[str, Path]], **kwargs: Any
) -> Iterator[Recording]:
    """Open several recordings. Convenience for the extraction script."""
    for p in paths:
        yield open_recording(p, **kwargs)
