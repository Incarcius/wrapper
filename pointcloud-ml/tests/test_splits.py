"""Splits by session. ``TestNoLeakage`` is the test that holds the line.

``docs/task_definition.md`` §7: splitting by random frame would put frame 211 of
a hover in train and frame 212 in val — the same objects, 100 ms apart. The
validation score would measure memorisation and would look excellent. So the
invariant asserted here is that **no session id ever appears in two splits**,
checked over many random session counts and seeds rather than one happy case.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pcml.data.crops import CropIndex, CropRecord
from pcml.data.label_mapping import load_label_map
from pcml.data.labels import LabelStore
from pcml.data.splits import (
    SPLIT_NAMES,
    SplitError,
    assign_sessions_to_splits,
    class_distribution_report,
    format_class_distribution_report,
    load_manifest_sessions,
    write_manifests,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MAP = REPO_ROOT / "data" / "label_map.yaml"


@pytest.fixture
def label_map():
    return load_label_map(REAL_MAP)


# ======================================================================
class TestNoLeakage:
    """The invariant. Everything else in this module is secondary to it."""

    @pytest.mark.parametrize("n_sessions", [3, 4, 5, 8, 13, 40])
    @pytest.mark.parametrize("seed", [0, 1, 7, 99])
    def test_no_session_appears_in_two_splits(self, n_sessions, seed):
        sessions = [f"session_{i:02d}" for i in range(n_sessions)]
        a = assign_sessions_to_splits(sessions, seed=seed)

        all_assigned = list(a.train) + list(a.val) + list(a.test)
        assert len(all_assigned) == len(set(all_assigned)), "a session is in two splits"
        assert set(all_assigned) == set(sessions), "a session was lost or invented"

    @pytest.mark.parametrize("n_sessions", [3, 6, 11])
    def test_every_split_is_non_empty(self, n_sessions):
        """An empty val or test split makes every metric from it vacuous."""
        sessions = [f"s{i}" for i in range(n_sessions)]
        a = assign_sessions_to_splits(sessions, seed=3)
        for name in SPLIT_NAMES:
            assert getattr(a, name), f"{name} is empty with {n_sessions} sessions"

    def test_the_dataclass_itself_rejects_an_overlap(self):
        """Belt and braces: a hand-edited manifest must not reach training."""
        from pcml.data.splits import SplitAssignment

        with pytest.raises(SplitError, match="is in both"):
            SplitAssignment(
                train=("s1", "s2"),
                val=("s2",),
                test=("s3",),
                seed=0,
                fractions=(0.7, 0.15, 0.15),
                pinned={},
            )

    def test_manifests_on_disk_share_no_session(self, tmp_path, label_map):
        written = _build_manifests(tmp_path, label_map, n_sessions=6)
        session_sets = {
            name: load_manifest_sessions(written[name]) for name in SPLIT_NAMES
        }
        assert not (session_sets["train"] & session_sets["val"])
        assert not (session_sets["train"] & session_sets["test"])
        assert not (session_sets["val"] & session_sets["test"])

    def test_no_crop_id_appears_in_two_manifests(self, tmp_path, label_map):
        """The consequence of the session rule, checked at crop level too."""
        written = _build_manifests(tmp_path, label_map, n_sessions=6)
        seen: dict[str, str] = {}
        for name in SPLIT_NAMES:
            doc = json.loads(written[name].read_text(encoding="utf-8"))
            for row in doc["crops"]:
                assert row["crop_id"] not in seen, (
                    f"{row['crop_id']} is in both {seen.get(row['crop_id'])} and {name}"
                )
                seen[row["crop_id"]] = name


class TestAssignment:
    def test_is_reproducible_from_sessions_fractions_and_seed(self):
        sessions = [f"s{i}" for i in range(10)]
        a = assign_sessions_to_splits(sessions, seed=5)
        b = assign_sessions_to_splits(sessions, seed=5)
        assert a.train == b.train and a.val == b.val and a.test == b.test

    def test_input_order_does_not_change_the_result(self):
        """Sorted before shuffling, so filesystem or dict order cannot leak in."""
        sessions = [f"s{i}" for i in range(10)]
        a = assign_sessions_to_splits(sessions, seed=5)
        b = assign_sessions_to_splits(list(reversed(sessions)), seed=5)
        assert a.train == b.train and a.val == b.val and a.test == b.test

    def test_different_seeds_give_different_assignments(self):
        sessions = [f"s{i}" for i in range(12)]
        a = assign_sessions_to_splits(sessions, seed=0)
        b = assign_sessions_to_splits(sessions, seed=1)
        assert (a.train, a.val, a.test) != (b.train, b.val, b.test)

    def test_roughly_honours_the_fractions(self):
        sessions = [f"s{i}" for i in range(100)]
        a = assign_sessions_to_splits(sessions, fractions=(0.7, 0.15, 0.15), seed=0)
        assert 60 <= len(a.train) <= 78
        assert 10 <= len(a.val) <= 22
        assert 10 <= len(a.test) <= 22

    def test_fractions_are_normalised(self):
        sessions = [f"s{i}" for i in range(10)]
        a = assign_sessions_to_splits(sessions, fractions=(7, 1.5, 1.5), seed=0)
        assert sum(a.fractions) == pytest.approx(1.0)

    def test_pinned_sessions_stay_where_they_are_put(self):
        """How a held-out location or the range-calibration recording is kept out
        of training (§4.1, Q4)."""
        sessions = [f"s{i}" for i in range(10)]
        a = assign_sessions_to_splits(
            sessions, seed=0, pinned={"s3": "test", "s7": "test", "s0": "val"}
        )
        assert "s3" in a.test and "s7" in a.test
        assert "s0" in a.val
        assert "s3" not in a.train and "s7" not in a.train

    def test_a_zero_fraction_split_is_allowed_when_asked_for_explicitly(self):
        a = assign_sessions_to_splits(["s1", "s2"], fractions=(0.8, 0.2, 0.0), seed=0)
        assert a.test == ()
        assert a.train and a.val

    def test_too_few_sessions_fails_loudly(self):
        with pytest.raises(SplitError, match="cannot fill"):
            assign_sessions_to_splits(["only_one", "and_two"], seed=0)

    def test_no_sessions_at_all(self):
        with pytest.raises(SplitError, match="no sessions"):
            assign_sessions_to_splits([])

    def test_bad_pinned_split_name(self):
        with pytest.raises(SplitError, match="valid:"):
            assign_sessions_to_splits(["a", "b", "c"], pinned={"a": "holdout"})

    def test_pinned_session_must_exist(self):
        with pytest.raises(SplitError, match="not among the available"):
            assign_sessions_to_splits(["a", "b", "c"], pinned={"zz": "test"})

    def test_negative_fraction(self):
        with pytest.raises(SplitError, match="non-negative"):
            assign_sessions_to_splits(["a", "b", "c"], fractions=(0.8, -0.1, 0.3))


# ======================================================================
def _fake_index(
    tmp_path: Path,
    session_id: str,
    n_frames: int = 4,
    n_objects: int = 3,
    preprocess_sha256: str = "abc123",
) -> CropIndex:
    """A crop index with no points file — enough for manifest generation.

    Manifest writing reads only the index, never the points, which is what makes
    it cheap to run over a whole dataset.
    """
    crops = []
    for frame in range(n_frames):
        for obj in range(n_objects):
            crops.append(
                CropRecord(
                    crop_id=f"{session_id}/{frame:06d}/{obj:02d}",
                    session_id=session_id,
                    frame_index=frame,
                    stamp=frame * 0.1,
                    n_points=200 + obj,
                    centroid_m=(float(obj) * 4.0, 0.0, 1.0),
                    extent_m=(0.4, 0.4, 1.8),
                    ground_z_m=0.0,
                    offset=0,
                    length=200 + obj,
                )
            )
    index_path = tmp_path / "processed" / f"{session_id}_crops.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("{}", encoding="utf-8")
    return CropIndex(
        session_id=session_id,
        index_path=index_path,
        points_path=index_path.with_suffix(".npz"),
        crops=tuple(crops),
        source_file=f"raw/{session_id}.npz",
        raw_sha256="f" * 64,
        preprocess_sha256=preprocess_sha256,
        seed=0,
        stamp_source="sensor",
    )


def _build_manifests(
    tmp_path: Path,
    label_map,
    n_sessions: int = 4,
    n_frames: int = 4,
    annotate_every_frame: bool = False,
    **kwargs,
):
    """Build indices + label stores for n_sessions and write the manifests.

    Each session has three objects, labelled once each and propagated — i.e.
    exactly the §5.2 workflow.
    """
    indices = []
    stores = {}
    for i in range(n_sessions):
        session_id = f"session_{i:02d}"
        index = _fake_index(tmp_path, session_id, n_frames=n_frames)
        store = LabelStore.load_or_create(tmp_path / "labels", session_id)
        by_frame: dict[int, list[CropRecord]] = {}
        for c in index.crops:
            by_frame.setdefault(c.frame_index, []).append(c)

        # One human decision per object, in frame 0; the rest propagate.
        for obj_i, c in enumerate(by_frame[0]):
            store.annotate(c, obj_i % label_map.num_classes, label_map)
        if annotate_every_frame:
            for frame, crops in by_frame.items():
                if frame == 0:
                    continue
                for obj_i, c in enumerate(crops):
                    store.annotate(
                        c, obj_i % label_map.num_classes, label_map, new_object=False
                    )
        store.propagate(index.crops)
        store.save()
        indices.append(index)
        stores[session_id] = store

    return write_manifests(
        indices,
        stores,
        label_map,
        out_dir=tmp_path / "manifests",
        seed=0,
        **kwargs,
    )


class TestManifests:
    def test_writes_the_three_manifests_plus_splits_json(self, tmp_path, label_map):
        written = _build_manifests(tmp_path, label_map, n_sessions=4)
        for name in SPLIT_NAMES:
            assert written[name].name == f"{name}.json"
            assert written[name].exists()
        assert written["splits"].name == "splits.json"

    def test_manifests_record_the_vocabulary_and_config_they_assume(
        self, tmp_path, label_map
    ):
        written = _build_manifests(tmp_path, label_map, n_sessions=4)
        doc = json.loads(written["train"].read_text(encoding="utf-8"))
        assert doc["label_map_sha256"] == label_map.sha256
        assert doc["num_classes"] == 6
        assert doc["class_names"] == list(label_map.names)
        assert doc["preprocess_sha256"] == "abc123"
        assert doc["split_by"] == "session"

    def test_test_split_is_annotated_only_by_default(self, tmp_path, label_map):
        """§5.3: the final number must not be partly a measure of propagation."""
        written = _build_manifests(tmp_path, label_map, n_sessions=4, n_frames=5)
        test_doc = json.loads(written["test"].read_text(encoding="utf-8"))
        assert test_doc["annotated_only"] is True
        assert test_doc["crops"], "test split is empty"
        assert all(r["provenance"] == "annotated" for r in test_doc["crops"])

        # Train is allowed propagated labels, and in a hover most of it is.
        train_doc = json.loads(written["train"].read_text(encoding="utf-8"))
        assert train_doc["provenance_counts"].get("propagated", 0) > 0

    def test_propagated_test_labels_can_be_allowed_deliberately(
        self, tmp_path, label_map
    ):
        written = _build_manifests(
            tmp_path, label_map, n_sessions=4, n_frames=5, test_annotated_only=False
        )
        doc = json.loads(written["test"].read_text(encoding="utf-8"))
        assert doc["annotated_only"] is False
        assert any(r["provenance"] == "propagated" for r in doc["crops"])

    def test_reject_labelled_crops_are_excluded_but_findable(self, tmp_path, label_map):
        """`unknown` is an output, not a training target — but the crops are kept
        so the threshold can be calibrated against them."""
        index = _fake_index(tmp_path, "session_00")
        other = _fake_index(tmp_path, "session_01")
        third = _fake_index(tmp_path, "session_02")
        stores = {}
        for ix in (index, other, third):
            store = LabelStore.load_or_create(tmp_path / "labels", ix.session_id)
            frame0 = [c for c in ix.crops if c.frame_index == 0]
            store.annotate(frame0[0], label_map.id_for("person"), label_map)
            store.annotate(frame0[1], label_map.reserved_class_id, label_map)
            store.propagate(ix.crops)
            stores[ix.session_id] = store

        written = write_manifests(
            [index, other, third],
            stores,
            label_map,
            out_dir=tmp_path / "manifests",
            seed=0,
        )
        for name in SPLIT_NAMES:
            doc = json.loads(written[name].read_text(encoding="utf-8"))
            assert all(r["label_id"] >= 0 for r in doc["crops"])

        splits = json.loads(written["splits"].read_text(encoding="utf-8"))
        assert splits["dropped"]["reject_class"] > 0
        assert splits["reject_labelled_crops"], "the unknown crops must stay findable"

    def test_ambiguous_crops_are_excluded_by_default(self, tmp_path, label_map):
        index = _fake_index(tmp_path, "session_00", n_frames=2, n_objects=2)
        a, b = [c for c in index.crops if c.frame_index == 0]
        store = LabelStore.load_or_create(tmp_path / "labels", "session_00")
        # Two objects with radii wide enough to both match the frame-1 crops.
        store.annotate(a, label_map.id_for("person"), label_map, radius_m=20.0)
        store.annotate(b, label_map.id_for("pole"), label_map, radius_m=20.0)
        store.propagate(index.crops, radius_m=20.0)
        assert store.summary()["n_ambiguous"] > 0

        others = [_fake_index(tmp_path, f"session_{i:02d}") for i in (1, 2)]
        stores = {"session_00": store}
        for ix in others:
            s = LabelStore.load_or_create(tmp_path / "labels", ix.session_id)
            s.annotate(ix.crops[0], label_map.id_for("pole"), label_map)
            s.propagate(ix.crops)
            stores[ix.session_id] = s

        written = write_manifests(
            [index, *others], stores, label_map, out_dir=tmp_path / "m", seed=0
        )
        splits = json.loads(written["splits"].read_text(encoding="utf-8"))
        assert splits["dropped"]["ambiguous"] > 0

    def test_unlabelled_crops_are_counted_not_silently_dropped(self, tmp_path, label_map):
        written = _build_manifests(tmp_path, label_map, n_sessions=4)
        splits = json.loads(written["splits"].read_text(encoding="utf-8"))
        assert "unlabelled" in splits["dropped"]

    def test_a_session_with_no_label_store_is_reported(self, tmp_path, label_map):
        indices = [_fake_index(tmp_path, f"session_{i:02d}") for i in range(4)]
        stores = {}
        for ix in indices[:3]:
            s = LabelStore.load_or_create(tmp_path / "labels", ix.session_id)
            s.annotate(ix.crops[0], label_map.id_for("pole"), label_map)
            s.propagate(ix.crops)
            stores[ix.session_id] = s

        written = write_manifests(
            indices, stores, label_map, out_dir=tmp_path / "m", seed=0
        )
        splits = json.loads(written["splits"].read_text(encoding="utf-8"))
        assert "session_03" in splits["sessions_without_label_store"]
        assert splits["dropped"]["no_label_store"] > 0

    def test_index_paths_are_relative_so_a_clone_still_works(self, tmp_path, label_map):
        written = _build_manifests(tmp_path, label_map, n_sessions=4)
        doc = json.loads(written["train"].read_text(encoding="utf-8"))
        for row in doc["crops"]:
            assert not Path(row["index_file"]).is_absolute(), row["index_file"]

    def test_mixed_preprocess_versions_are_refused(self, tmp_path, label_map):
        """Crops from two different DBSCAN radii must not land in one dataset."""
        a = _fake_index(tmp_path, "session_00", preprocess_sha256="aaa")
        b = _fake_index(tmp_path, "session_01", preprocess_sha256="bbb")
        c = _fake_index(tmp_path, "session_02", preprocess_sha256="aaa")
        stores = {}
        for ix in (a, b, c):
            s = LabelStore.load_or_create(tmp_path / "labels", ix.session_id)
            s.annotate(ix.crops[0], label_map.id_for("pole"), label_map)
            s.propagate(ix.crops)
            stores[ix.session_id] = s

        with pytest.raises(SplitError, match="different preprocess.yaml"):
            write_manifests([a, b, c], stores, label_map, out_dir=tmp_path / "m", seed=0)

    def test_duplicate_session_indices_are_refused(self, tmp_path, label_map):
        a = _fake_index(tmp_path, "same")
        b = _fake_index(tmp_path, "same")
        with pytest.raises(SplitError, match="share a session_id"):
            write_manifests([a, b], {}, label_map, out_dir=tmp_path / "m", seed=0)


# ======================================================================
class TestClassDistributionReport:
    def test_counts_crops_and_distinct_objects_separately(self, tmp_path, label_map):
        """§4.4's counting mistake, made impossible to commit by accident:
        4,800 crops from 8 objects is 8 objects."""
        written = _build_manifests(tmp_path, label_map, n_sessions=5, n_frames=10)
        report = class_distribution_report(
            {name: written[name] for name in SPLIT_NAMES}, label_map
        )
        train = report["splits"]["train"]
        assert train["n_crops"] > 0
        assert sum(train["distinct_objects_per_class"].values()) < train["n_crops"]
        assert train["n_sessions"] == len(load_manifest_sessions(written["train"]))

    def test_warns_about_classes_with_no_examples(self, tmp_path, label_map):
        """The fake data labels only 3 of the 6 classes."""
        written = _build_manifests(tmp_path, label_map, n_sessions=4)
        report = class_distribution_report(
            {name: written[name] for name in SPLIT_NAMES}, label_map
        )
        warnings = " ".join(report["warnings"])
        assert "no examples at all" in warnings
        assert "structure" in warnings or "vehicle" in warnings

    def test_warns_below_the_ten_instance_floor(self, tmp_path, label_map):
        written = _build_manifests(tmp_path, label_map, n_sessions=4)
        report = class_distribution_report(
            {name: written[name] for name in SPLIT_NAMES}, label_map
        )
        assert any("distinct physical objects" in w for w in report["warnings"])

    def test_suggests_class_weights_from_the_training_split_only(
        self, tmp_path, label_map
    ):
        written = _build_manifests(tmp_path, label_map, n_sessions=5)
        report = class_distribution_report(
            {name: written[name] for name in SPLIT_NAMES}, label_map
        )
        weights = report["suggested_class_weights_from_train"]
        assert set(weights) == set(label_map.names)
        # Mean-normalised over the classes that are present, so the loss scale
        # is unchanged.
        present = [w for w in weights.values() if w > 0]
        assert sum(present) / len(present) == pytest.approx(1.0, abs=0.05)
        assert "not a decision" in report["suggested_class_weights_note"]

    def test_renders_as_a_text_table(self, tmp_path, label_map):
        written = _build_manifests(tmp_path, label_map, n_sessions=4)
        report = class_distribution_report(
            {name: written[name] for name in SPLIT_NAMES}, label_map
        )
        text = format_class_distribution_report(report)
        assert "split BY SESSION" in text
        assert "person" in text
        assert "TOTAL" in text
        assert "WARNINGS:" in text
        # Every line prints; nothing is a bare repr of a dict.
        assert "{" not in text.replace("{}", "")
