"""Synthetic recordings, so the ingestion path can be validated before real data.

No real recording exists yet (``data/README.md``), and the point of this task is
that the toolchain is ready the day the first bag lands. So the tests build
recordings here: multi-frame, multi-session, with objects at known positions, in
every format the readers claim to support — **including a genuine ROS 2 ``.db3``
bag with CDR-serialised ``PointCloud2`` messages**, written by
:func:`write_pointcloud2_bag` below.

That bag writer matters more than it looks. A hand-rolled CDR *reader* validated
against a hand-rolled CDR *writer* would only prove the two agree, so the writer
is deliberately built to be awkward in the ways a real Livox cloud is awkward:

* the point layout is ``x, y, z, intensity (f4), tag (u1), line (u1),
  timestamp (f8)`` with ``point_step = 26`` — not three tightly packed float32s;
* ``x``, ``y`` and ``z`` are **not** listed first in the ``fields`` array, so a
  reader that trusts field order rather than ``offset`` fails;
* the field offsets are not multiples of 4 for the trailing fields, so a reader
  that assumes natural alignment inside a point fails;
* the header carries a real ``stamp`` that differs from the bag's receive
  timestamp, so a reader that uses the wrong one produces visibly wrong stamps.

Every scene reuses ``field_scene`` from ``tests/test_ground.py``, which is the
simulated mown field whose density regime is the one measured in
``docs/data_collection_plan.md`` §3.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from test_ground import OBJECT_POSITIONS, field_scene

__all__ = [
    "OBJECT_POSITIONS",
    "field_scene",
    "synthetic_session_frames",
    "write_npz_session",
    "write_npy_frame_dir",
    "write_ascii_pcd",
    "write_binary_pcd",
    "write_ascii_ply",
    "write_pointcloud2_bag",
    "write_meta_sidecar",
    "LIVOX_POINT_STEP",
]

#: ``x, y, z, intensity`` as float32 at 0/4/8/12, then ``tag`` and ``line`` as
#: uint8 at 16/17, then a float64 per-point ``timestamp`` at 18. 26 bytes, and
#: the timestamp is deliberately not 8-byte aligned within the point.
LIVOX_POINT_STEP = 26


def synthetic_session_frames(
    n_frames: int = 4,
    extent_m: float = 8.0,
    density_per_m2: float = 25.0,
    jitter_m: float = 0.01,
    seed: int = 11,
    start_stamp: float = 1_700_000_000.0,
    period_s: float = 0.1,
) -> list[tuple[np.ndarray, float]]:
    """A hover over the simulated field: ``n_frames`` of the same three objects.

    Each frame re-draws the scene with a different seed and adds a little jitter,
    which mimics the Mid-70's non-repetitive rosette sampling different points of
    the same static objects — the property §5.2's propagation relies on, and the
    reason object centroids move slightly between frames instead of being
    bit-identical.
    """
    rng = np.random.default_rng(seed + 9999)
    frames: list[tuple[np.ndarray, float]] = []
    for i in range(n_frames):
        points, _mask = field_scene(
            extent_m=extent_m, density_per_m2=density_per_m2, seed=seed + i
        )
        if jitter_m > 0:
            points = points + rng.normal(0.0, jitter_m, points.shape).astype(np.float32)
        frames.append(
            (np.ascontiguousarray(points, dtype=np.float32), start_stamp + i * period_s)
        )
    return frames


# ======================================================================
# Simple formats
# ======================================================================
def write_npz_session(
    path: Path, frames: Sequence[tuple[np.ndarray, float]]
) -> Path:
    """Pack frames into one ``.npz`` with ``points`` / ``offsets`` / ``stamps``."""
    points = np.concatenate([f[0] for f in frames], axis=0).astype(np.float32)
    offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    for i, (frame, _stamp) in enumerate(frames):
        offsets[i + 1] = offsets[i] + frame.shape[0]
    stamps = np.array([f[1] for f in frames], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points, offsets=offsets, stamps=stamps)
    return path


def write_npy_frame_dir(directory: Path, frames: Sequence[tuple[np.ndarray, float]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i, (frame, _stamp) in enumerate(frames):
        np.save(directory / f"frame_{i:06d}.npy", frame)
    return directory


def write_ascii_pcd(path: Path, points: np.ndarray, with_intensity: bool = True) -> Path:
    """An ascii PCD, optionally with an extra ``intensity`` column.

    The extra column is the point: it moves ``z`` off being the last field and
    makes a column-count assumption fail.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    fields = ["x", "y", "z"] + (["intensity"] if with_intensity else [])
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            f"FIELDS {' '.join(fields)}",
            f"SIZE {' '.join(['4'] * len(fields))}",
            f"TYPE {' '.join(['F'] * len(fields))}",
            f"COUNT {' '.join(['1'] * len(fields))}",
            f"WIDTH {n}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {n}",
            "DATA ascii",
        ]
    )
    rows = []
    for i in range(n):
        values = [f"{pts[i, 0]:.6f}", f"{pts[i, 1]:.6f}", f"{pts[i, 2]:.6f}"]
        if with_intensity:
            values.append(f"{float(i % 255):.6f}")
        rows.append(" ".join(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="ascii")
    return path


def write_binary_pcd(path: Path, points: np.ndarray) -> Path:
    """A binary PCD with an ``intensity`` field after xyz."""
    pts = np.asarray(points, dtype=np.float32)
    n = pts.shape[0]
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z intensity",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            f"WIDTH {n}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {n}",
            "DATA binary",
        ]
    )
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    rows = np.zeros(n, dtype=dtype)
    rows["x"], rows["y"], rows["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rows["intensity"] = np.arange(n, dtype=np.float32) % 255.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write((header + "\n").encode("ascii"))
        fh.write(rows.tobytes())
    return path


def write_ascii_ply(path: Path, points: np.ndarray) -> Path:
    """An ascii PLY whose vertex element carries ``intensity`` after xyz."""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            "comment written by tests/synthetic_recordings.py",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar intensity",
            "end_header",
        ]
    )
    rows = [
        f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f} {i % 255}" for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="ascii")
    return path


def write_meta_sidecar(
    recording_path: Path,
    session_id: str,
    altitude_m: float = 12.0,
    mount_angle_deg: float = 35.0,
    motion: str = "hover",
    field: str = "cricket",
    **extra: object,
) -> Path:
    """Write a §4.2 metadata sidecar next to a recording."""
    doc = {
        "session_id": session_id,
        "date": "2026-10-20",
        "time": "16:40",
        "sensor": "livox_mid70",
        "altitude_m": altitude_m,
        "mount_angle_deg": mount_angle_deg,
        "motion": motion,
        "airspeed_mps": 0.0 if motion == "hover" else 4.0,
        "field": field,
        "field_part": "north edge",
        "in_view": "one person at ~5 m, a fence pole at ~5 m, a tree at ~6 m",
        "weather": "dry",
        "grass_wet": False,
        "staged": [{"what": "person", "distance_m": 5.0, "pose": "standing"}],
    }
    doc.update(extra)
    path = recording_path.with_name(recording_path.name + ".meta.json")
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


# ======================================================================
# A real ROS 2 bag: CDR serialisation + the rosbag2 sqlite3 schema
# ======================================================================
class _CdrWriter:
    """CDR encoder, little-endian, with alignment measured from the body start.

    Mirror image of ``pcml.data.io._CdrReader``. Kept in the tests rather than in
    the package because nothing in production writes bags — the drone's ROS 2
    stack does that.
    """

    def __init__(self) -> None:
        # Encapsulation header: 0x00 0x01 = CDR_LE, then two options bytes.
        self.buf = bytearray(b"\x00\x01\x00\x00")
        self.origin = 4

    def _align(self, size: int) -> None:
        rem = (len(self.buf) - self.origin) % size
        if rem:
            self.buf += b"\x00" * (size - rem)

    def u8(self, value: int) -> None:
        self.buf += struct.pack("<B", value & 0xFF)

    def boolean(self, value: bool) -> None:
        self.u8(1 if value else 0)

    def i32(self, value: int) -> None:
        self._align(4)
        self.buf += struct.pack("<i", value)

    def u32(self, value: int) -> None:
        self._align(4)
        self.buf += struct.pack("<I", value)

    def string(self, value: str) -> None:
        raw = value.encode("utf-8") + b"\x00"
        self.u32(len(raw))
        self.buf += raw

    def raw_bytes(self, value: bytes) -> None:
        self.buf += value

    def bytes(self) -> bytes:
        return bytes(self.buf)


def _livox_point_blob(points: np.ndarray) -> bytes:
    """Pack xyz into the 26-byte Livox-style point layout described above.

    Built with an explicit-offset structured dtype whose ``itemsize`` is 26, so
    the trailing ``timestamp`` genuinely sits at byte 18 of each point — not at
    the 24 that natural alignment would choose.
    """
    pts = np.asarray(points, dtype=np.float32)
    n = pts.shape[0]
    dtype = np.dtype(
        {
            "names": ["x", "y", "z", "intensity", "tag", "line", "timestamp"],
            "formats": ["<f4", "<f4", "<f4", "<f4", "u1", "u1", "<f8"],
            "offsets": [0, 4, 8, 12, 16, 17, 18],
            "itemsize": LIVOX_POINT_STEP,
        }
    )
    rows = np.zeros(n, dtype=dtype)
    rows["x"], rows["y"], rows["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rows["intensity"] = (np.arange(n, dtype=np.float32) % 255.0)
    rows["tag"] = 0
    rows["line"] = np.arange(n, dtype=np.uint8) % 6
    rows["timestamp"] = 1.0e-6 * np.arange(n, dtype=np.float64)
    assert dtype.itemsize == LIVOX_POINT_STEP
    return rows.tobytes()


def serialise_pointcloud2(
    points: np.ndarray,
    stamp: float,
    frame_id: str = "livox_frame",
    shuffle_fields: bool = True,
) -> bytes:
    """CDR-serialise one ``sensor_msgs/msg/PointCloud2``.

    Args:
        shuffle_fields: list the fields in the order
            ``intensity, tag, z, line, x, timestamp, y`` rather than
            ``x, y, z, ...``. Legal — ``fields`` is an unordered description and
            ``offset`` is authoritative — and it is what makes a reader that
            trusts the order produce garbage instead of coordinates.
    """
    w = _CdrWriter()

    sec = int(stamp)
    nanosec = int(round((stamp - sec) * 1e9))
    if nanosec >= 1_000_000_000:  # rounding at the top of a second
        sec += 1
        nanosec -= 1_000_000_000
    w.i32(sec)
    w.u32(nanosec)
    w.string(frame_id)

    n = int(np.asarray(points).shape[0])
    w.u32(1)  # height: unorganised cloud
    w.u32(n)  # width

    #                name,         offset, datatype, count
    fields = [
        ("x", 0, 7, 1),
        ("y", 4, 7, 1),
        ("z", 8, 7, 1),
        ("intensity", 12, 7, 1),
        ("tag", 16, 2, 1),
        ("line", 17, 2, 1),
        ("timestamp", 18, 8, 1),
    ]
    if shuffle_fields:
        order = ["intensity", "tag", "z", "line", "x", "timestamp", "y"]
        by_name = {f[0]: f for f in fields}
        fields = [by_name[name] for name in order]

    w.u32(len(fields))
    for name, offset, datatype, count in fields:
        w.string(name)
        w.u32(offset)
        w.u8(datatype)
        w.u32(count)

    blob = _livox_point_blob(points)
    w.boolean(False)  # is_bigendian
    w.u32(LIVOX_POINT_STEP)  # point_step
    w.u32(LIVOX_POINT_STEP * n)  # row_step
    w.u32(len(blob))  # data sequence length
    w.raw_bytes(blob)
    w.boolean(True)  # is_dense
    return w.bytes()


def write_pointcloud2_bag(
    db3_path: Path,
    frames: Sequence[tuple[np.ndarray, float]],
    topic: str = "/livox/lidar",
    extra_topics: Optional[Iterable[tuple[str, str]]] = (
        ("/tf_static", "tf2_msgs/msg/TFMessage"),
    ),
    receive_offset_s: float = 0.037,
    write_metadata_yaml: bool = True,
) -> Path:
    """Write a rosbag2 sqlite3 bag holding CDR ``PointCloud2`` messages.

    The bag's ``messages.timestamp`` is deliberately offset from the header
    ``stamp`` by ``receive_offset_s``, because they are different things: the
    header stamp is the sensor's and is what ``FramePrediction.stamp`` is
    contracted to carry (``docs/task_definition.md`` §5), while the bag timestamp
    is when the recorder received the message. A reader that returns the latter
    looks right and is wrong by tens of milliseconds.

    ``extra_topics`` adds topics with no messages, so that "topic not found"
    error paths have something to list.
    """
    db3_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db3_path))
    try:
        conn.executescript(
            """
            CREATE TABLE topics (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                serialization_format TEXT NOT NULL,
                offered_qos_profiles TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                topic_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                data BLOB NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO topics (id, name, type, serialization_format, "
            "offered_qos_profiles) VALUES (1, ?, ?, 'cdr', '')",
            (topic, "sensor_msgs/msg/PointCloud2"),
        )
        next_topic_id = 2
        for name, type_name in extra_topics or ():
            conn.execute(
                "INSERT INTO topics (id, name, type, serialization_format, "
                "offered_qos_profiles) VALUES (?, ?, ?, 'cdr', '')",
                (next_topic_id, name, type_name),
            )
            next_topic_id += 1

        for i, (points, stamp) in enumerate(frames):
            payload = serialise_pointcloud2(points, stamp)
            conn.execute(
                "INSERT INTO messages (id, topic_id, timestamp, data) VALUES (?, 1, ?, ?)",
                (i + 1, int(round((stamp + receive_offset_s) * 1e9)), sqlite3.Binary(payload)),
            )
        conn.commit()
    finally:
        conn.close()

    if write_metadata_yaml:
        # rosbag2 writes this; the reader does not need it, which is the point of
        # including it — an unused file must not break anything.
        (db3_path.parent / "metadata.yaml").write_text(
            "rosbag2_bagfile_information:\n"
            "  version: 5\n"
            "  storage_identifier: sqlite3\n"
            f"  relative_file_paths:\n    - {db3_path.name}\n"
            f"  message_count: {len(frames)}\n",
            encoding="utf-8",
        )
    return db3_path
