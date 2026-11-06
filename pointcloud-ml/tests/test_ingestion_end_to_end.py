"""The whole ingestion path, on a synthetic multi-session recording set.

This is the test that says the toolchain is ready for the first real bag. It
runs, in one go:

    3 synthetic sessions (one of them a real CDR .db3 bag)
      -> open_recording
      -> ground removal + clustering            -> cluster crops + index
      -> label once per object, propagate       -> label stores
      -> split BY SESSION                       -> train/val/test manifests
      -> torch Dataset with augmentation
      -> batched forward pass through the deployed PointNet

Nothing here measures accuracy, and nothing here is a result: the labels are
assigned by construction to synthetic objects. What it establishes is that the
shapes, the contracts and the file formats line up end to end, so the only thing
missing on the day a real recording lands is the recording.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn", reason="needs a DBSCAN backend")

import torch  # noqa: E402
import yaml  # noqa: E402

from pcml.data.config import load_preprocess_config  # noqa: E402
from pcml.data.crops import extract_many, find_crop_indices, load_crop_index  # noqa: E402
from pcml.data.dataset import ClusterCropDataset  # noqa: E402
from pcml.data.io import open_recording  # noqa: E402
from pcml.data.label_mapping import load_label_map  # noqa: E402
from pcml.data.labels import LabelStore  # noqa: E402
from pcml.data.splits import (  # noqa: E402
    SPLIT_NAMES,
    class_distribution_report,
    format_class_distribution_report,
    write_manifests,
)
from pcml.models.model_factory import build_model  # noqa: E402
from synthetic_recordings import (  # noqa: E402
    synthetic_session_frames,
    write_meta_sidecar,
    write_npy_frame_dir,
    write_npz_session,
    write_pointcloud2_bag,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = REPO_ROOT / "configs" / "preprocess.yaml"
REAL_MAP = REPO_ROOT / "data" / "label_map.yaml"
DEPLOY_MODEL = REPO_ROOT / "configs" / "models" / "pointnet_deploy.yaml"


def build_recording_set(root: Path) -> list[Path]:
    """Three sessions, deliberately in three different formats.

    One is a genuine rosbag2 ``.db3`` with CDR ``PointCloud2`` messages, because
    that is what ``ros2 bag record /livox/lidar`` actually produces and it is the
    format the first real recording will arrive in.
    """
    raw = root / "raw"
    paths: list[Path] = []

    # Session 1: a rosbag2 directory â€” the real target format.
    frames = synthetic_session_frames(n_frames=3, extent_m=8.0, density_per_m2=25.0, seed=11)
    bag_dir = raw / "session_01_hover_12m"
    write_pointcloud2_bag(bag_dir / "session_01_hover_12m_0.db3", frames)
    write_meta_sidecar(bag_dir, "session_01_hover_12m", altitude_m=12.0)
    paths.append(bag_dir)

    # Session 2: a packed .npz.
    frames = synthetic_session_frames(n_frames=3, extent_m=8.0, density_per_m2=25.0, seed=31)
    npz = write_npz_session(raw / "session_02_hover_15m.npz", frames)
    write_meta_sidecar(npz, "session_02_hover_15m", altitude_m=15.0)
    paths.append(npz)

    # Session 3: a directory of per-frame .npy files.
    frames = synthetic_session_frames(n_frames=3, extent_m=8.0, density_per_m2=25.0, seed=57)
    frame_dir = write_npy_frame_dir(raw / "session_03_hover_18m", frames)
    write_meta_sidecar(frame_dir, "session_03_hover_18m", altitude_m=18.0)
    paths.append(frame_dir)

    return paths


def label_session(index, label_map, labels_dir: Path) -> LabelStore:
    """Annotate once per object in the first frame, then propagate (Â§5.2).

    Classes are assigned by descending height, which on the simulated field maps
    the 6 m column to ``pole``, the tree to ``vegetation`` and the 1.75 m column
    to ``person``. Assigned by construction â€” this is a plumbing test, not a
    measurement.
    """
    store = LabelStore.load_or_create(labels_dir, index.session_id)
    frame0 = [c for c in index.crops if c.frame_index == min(index.frames())]
    by_height = sorted(frame0, key=lambda c: -c.extent_m[2])

    assignments = [
        (label_map.id_for("pole"), {"subtype": "floodlight"}),
        (label_map.id_for("vegetation"), {"extended": False}),
        (label_map.id_for("person"), {"merged_instances": 1}),
    ]
    for crop, (label_id, attributes) in zip(by_height, assignments):
        store.annotate(crop, label_id, label_map, attributes=attributes)
    # Anything else in the first frame is a ground fragment: `unknown`.
    for crop in by_height[len(assignments) :]:
        store.annotate(crop, label_map.reserved_class_id, label_map)

    store.propagate(index.crops)
    store.save()
    return store


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the whole path once; the tests below assert on its products."""
    root = tmp_path_factory.mktemp("e2e")
    config = load_preprocess_config(PREPROCESS)
    label_map = load_label_map(REAL_MAP)

    recordings = build_recording_set(root)
    indices = extract_many(
        [open_recording(p) for p in recordings], root / "processed", config=config
    )
    stores = {ix.session_id: label_session(ix, label_map, root / "labels") for ix in indices}
    written = write_manifests(
        indices,
        stores,
        label_map,
        out_dir=root / "manifests",
        fractions=(1 / 3, 1 / 3, 1 / 3),
        seed=0,
    )
    return {
        "root": root,
        "config": config,
        "label_map": label_map,
        "recordings": recordings,
        "indices": indices,
        "stores": stores,
        "manifests": written,
    }


# ======================================================================
class TestTheWholePath:
    def test_three_sessions_in_three_formats_were_read(self, pipeline):
        indices = pipeline["indices"]
        assert len(indices) == 3
        assert {ix.session_id for ix in indices} == {
            "session_01_hover_12m",
            "session_02_hover_15m",
            "session_03_hover_18m",
        }
        for ix in indices:
            assert ix.stats["n_frames_processed"] == 3
            assert ix.stats["n_frames_skipped_no_ground_plane"] == 0
            assert len(ix.crops) >= 9  # >= 3 objects x 3 frames

    def test_the_bag_session_carried_sensor_stamps(self, pipeline):
        """The .db3 session must come through with real header stamps, not a
        frame counter."""
        bag_index = next(
            ix for ix in pipeline["indices"] if ix.session_id == "session_01_hover_12m"
        )
        assert bag_index.stamp_source == "sensor"
        stamps = sorted({c.stamp for c in bag_index.crops})
        assert len(stamps) == 3
        assert stamps[0] > 1_000_000_000  # a real epoch stamp, not 0.0/0.1/0.2
        assert stamps[1] - stamps[0] == pytest.approx(0.1, abs=1e-3)

    def test_session_metadata_survived_to_the_crop_index(self, pipeline):
        """Â§4.2: altitude and mount angle cannot be reconstructed later."""
        for ix in pipeline["indices"]:
            assert ix.recording_meta is not None
            assert ix.recording_meta["mount_angle_deg"] == 35.0
            assert ix.recording_meta["motion"] == "hover"
        altitudes = {
            ix.session_id: ix.recording_meta["altitude_m"] for ix in pipeline["indices"]
        }
        assert altitudes["session_01_hover_12m"] == 12.0
        assert altitudes["session_03_hover_18m"] == 18.0

    def test_propagation_did_most_of_the_labelling(self, pipeline):
        """The Â§5.2 payoff, measured on the synthetic set."""
        total_annotated = sum(s.summary()["n_annotated"] for s in pipeline["stores"].values())
        total_propagated = sum(
            s.summary()["n_propagated"] for s in pipeline["stores"].values()
        )
        assert total_propagated > total_annotated
        # 3 frames per session means ~2/3 of labels are propagated.
        assert total_propagated / (total_annotated + total_propagated) > 0.5

    def test_attributes_travelled_with_the_propagated_labels(self, pipeline):
        store = pipeline["stores"]["session_01_hover_12m"]
        with_subtype = [
            l for l in store.labels.values() if l.attributes.get("subtype") == "floodlight"
        ]
        assert len(with_subtype) == 3  # one per frame
        assert sum(1 for l in with_subtype if l.is_annotated) == 1

    def test_crop_indices_are_discoverable_and_reloadable(self, pipeline):
        found = find_crop_indices(pipeline["root"] / "processed")
        assert len(found) == 3
        for path in found:
            reloaded = load_crop_index(path)
            assert reloaded.crops
            points = reloaded.points_for(reloaded.crops[0])
            assert points.shape[1] == 3
            assert points.dtype == np.float32


class TestManifestsAndLeakage:
    def test_each_session_went_to_exactly_one_split(self, pipeline):
        docs = {
            name: json.loads(pipeline["manifests"][name].read_text(encoding="utf-8"))
            for name in SPLIT_NAMES
        }
        sessions = [s for d in docs.values() for s in d["sessions"]]
        assert sorted(sessions) == sorted(
            ix.session_id for ix in pipeline["indices"]
        ), "a session was lost or duplicated"
        assert len(sessions) == len(set(sessions))
        for name in SPLIT_NAMES:
            assert len(docs[name]["sessions"]) == 1

    def test_the_test_split_is_human_labels_only(self, pipeline):
        doc = json.loads(pipeline["manifests"]["test"].read_text(encoding="utf-8"))
        assert doc["crops"]
        assert all(r["provenance"] == "annotated" for r in doc["crops"])

    def test_unknown_crops_reached_no_manifest(self, pipeline):
        for name in SPLIT_NAMES:
            doc = json.loads(pipeline["manifests"][name].read_text(encoding="utf-8"))
            assert all(r["label_id"] >= 0 for r in doc["crops"])
            assert all(r["label_name"] != "unknown" for r in doc["crops"])

    def test_the_distribution_report_renders(self, pipeline):
        report = class_distribution_report(
            {name: pipeline["manifests"][name] for name in SPLIT_NAMES},
            pipeline["label_map"],
        )
        text = format_class_distribution_report(report)
        assert "person" in text and "pole" in text and "TOTAL" in text
        # The synthetic set has 3 object types, so 3 of 6 classes are empty and
        # every class is far below the 10-instance floor. Both must be warned
        # about: a quiet report here would be the Â§4.4 mistake.
        assert any("no examples at all" in w for w in report["warnings"])
        assert any("distinct physical objects" in w for w in report["warnings"])


class TestThroughTheModel:
    def test_a_training_batch_reaches_the_deployed_pointnet(self, pipeline):
        """The last link: manifest -> Dataset -> one batched forward pass.

        One forward pass for the whole batch, which is the batching property
        ``docs/task_definition.md`` Â§9.2 calls mandatory rather than an
        optimisation.
        """
        from torch.utils.data import DataLoader

        label_map = pipeline["label_map"]
        ds = ClusterCropDataset.from_manifest(
            pipeline["manifests"]["train"],
            label_map=label_map,
            config=pipeline["config"],
        )
        assert len(ds) > 0
        assert ds.augment is True

        # build_model takes the `model` section, and K comes from the label map
        # rather than the model config, so the two can never disagree.
        model_config = yaml.safe_load(DEPLOY_MODEL.read_text(encoding="utf-8"))["model"]
        model = build_model(model_config, num_classes=label_map.num_classes)
        model.eval()

        batch_size = min(4, len(ds))
        points, aux, labels = next(
            iter(DataLoader(ds, batch_size=batch_size, shuffle=False))
        )
        assert points.shape == (batch_size, 3, pipeline["config"].sampling.num_points)

        with torch.no_grad():
            logits = model(points) if model.aux_dim == 0 else model(points, aux)

        assert logits.shape == (batch_size, label_map.num_classes)
        assert torch.isfinite(logits).all()
        assert labels.min().item() >= 0
        assert labels.max().item() < label_map.num_classes

    def test_one_training_step_reduces_the_loss_on_this_batch(self, pipeline):
        """Not a result â€” a wiring check.

        Overfitting one batch proves gradients flow from the loss through the
        head and backbone to the input. It says nothing whatever about accuracy
        on real data, and the labels here were assigned by construction.
        """
        from torch.utils.data import DataLoader

        label_map = pipeline["label_map"]
        ds = ClusterCropDataset.from_manifest(
            pipeline["manifests"]["train"],
            label_map=label_map,
            config=pipeline["config"],
            augment=False,  # a fixed batch, so the loss curve means something
        )
        # build_model takes the `model` section, and K comes from the label map
        # rather than the model config, so the two can never disagree.
        model_config = yaml.safe_load(DEPLOY_MODEL.read_text(encoding="utf-8"))["model"]
        model = build_model(model_config, num_classes=label_map.num_classes)
        model.train()

        batch_size = min(8, len(ds))
        points, aux, labels = next(
            iter(DataLoader(ds, batch_size=batch_size, shuffle=False))
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        def step() -> float:
            optimiser.zero_grad()
            logits = model(points) if model.aux_dim == 0 else model(points, aux)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            return float(loss.item())

        first = step()
        for _ in range(29):
            last = step()

        assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"

    def test_val_and_test_splits_load_without_augmentation(self, pipeline):
        for name in ("val", "test"):
            ds = ClusterCropDataset.from_manifest(
                pipeline["manifests"][name],
                label_map=pipeline["label_map"],
                config=pipeline["config"],
            )
            assert ds.augment is False
            if len(ds):
                points, aux, _label = ds[0]
                assert points.shape == (3, pipeline["config"].sampling.num_points)
                assert aux.shape == (6,)


class TestReproducibleFromRawPlusConfigPlusSeed:
    def test_re_extraction_reproduces_the_crops_exactly(self, pipeline):
        """The claim in crops.py's docstring, checked on the whole set."""
        again = extract_many(
            [open_recording(p) for p in pipeline["recordings"]],
            pipeline["root"] / "processed_again",
            config=pipeline["config"],
        )
        by_session = {ix.session_id: ix for ix in pipeline["indices"]}
        for ix in again:
            original = by_session[ix.session_id]
            assert [c.crop_id for c in ix.crops] == [c.crop_id for c in original.crops]
            assert ix.preprocess_sha256 == original.preprocess_sha256
            assert ix.seed == original.seed
            for a, b in zip(ix.crops, original.crops):
                assert a == b
                assert np.array_equal(ix.points_for(a), original.points_for(b))

    def test_the_index_names_the_exact_bytes_it_came_from(self, pipeline):
        for ix in pipeline["indices"]:
            assert ix.preprocess_sha256 == pipeline["config"].sha256
            assert ix.seed == pipeline["config"].seed
            # Directory recordings (a bag, a frame dir) have no single file to
            # hash; a single-file recording does, and it is recorded.
            if Path(ix.source_file).is_file():
                assert ix.raw_sha256 and len(ix.raw_sha256) == 64
