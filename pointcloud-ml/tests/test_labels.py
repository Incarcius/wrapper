"""Label store, §5.2 propagation, §5.3 provenance, and the text rendering.

``TestPropagation`` is the class that backs the claim that labelling is
tractable: one human decision per static object, propagated to every frame. The
arithmetic is asserted directly — N frames of the same object cost one
annotation and yield N labels.

``TestProvenanceIsMandatory`` backs §5.3: there is no code path that stores a
label without saying whether a human or the heuristic produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pcml.data.crops import CropRecord
from pcml.data.label_mapping import load_label_map
from pcml.data.labels import (
    DEFAULT_MATCH_RADIUS_M,
    PROVENANCE_ANNOTATED,
    PROVENANCE_PROPAGATED,
    CropLabel,
    LabelStore,
    LabelStoreError,
    ascii_projection,
    describe_crop,
    validate_attributes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MAP = REPO_ROOT / "data" / "label_map.yaml"


@pytest.fixture
def label_map():
    return load_label_map(REAL_MAP)


def crop(
    crop_id: str = "s1/000000/00",
    centroid=(0.0, 0.0, 1.0),
    extent=(0.4, 0.4, 1.8),
    frame_index: int = 0,
    n_points: int = 300,
) -> CropRecord:
    return CropRecord(
        crop_id=crop_id,
        session_id=crop_id.split("/")[0],
        frame_index=frame_index,
        stamp=float(frame_index) * 0.1,
        n_points=n_points,
        centroid_m=tuple(float(v) for v in centroid),
        extent_m=tuple(float(v) for v in extent),
        ground_z_m=0.0,
        offset=0,
        length=n_points,
    )


def hover_crops(n_frames: int, centroid=(0.0, 0.0, 1.0), wobble_m: float = 0.05):
    """The same static object seen in N frames, with a small centroid wobble.

    The wobble is the real effect propagation has to tolerate: the Mid-70's
    non-repetitive rosette samples different points of the same object each
    frame, so the centroid moves slightly. A matcher requiring exact equality
    would propagate nothing.
    """
    rng = np.random.default_rng(0)
    out = []
    for i in range(n_frames):
        offset = rng.uniform(-wobble_m, wobble_m, 3)
        out.append(
            crop(
                crop_id=f"s1/{i:06d}/00",
                centroid=tuple(np.asarray(centroid) + offset),
                frame_index=i,
            )
        )
    return out


# ======================================================================
class TestAnnotate:
    def test_records_a_human_decision_with_its_object(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        c = crop()
        label, obj = store.annotate(c, label_map.id_for("person"), label_map)

        assert label.provenance == PROVENANCE_ANNOTATED
        assert label.label_name == "person"
        assert label.label_id == 0
        assert obj is not None
        assert obj.position_m == c.centroid_m
        assert store.objects[obj.object_id] is obj

    def test_name_comes_from_the_map_not_a_literal(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        for entry in label_map.classes:
            label, _ = store.annotate(
                crop(crop_id=f"s1/00000{entry.class_id}/00"), entry.class_id, label_map
            )
            assert label.label_name == entry.name

    def test_the_reject_id_is_storable(self, tmp_path, label_map):
        """Negatives are recorded, not discarded: they calibrate the threshold."""
        store = LabelStore.load_or_create(tmp_path, "s1")
        label, _ = store.annotate(crop(), label_map.reserved_class_id, label_map)
        assert label.label_id == -1
        assert label.label_name == "unknown"
        assert not label_map.is_trained_class(label.label_id)

    def test_an_id_outside_the_map_is_refused(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        with pytest.raises(LabelStoreError, match="neither a trained class"):
            store.annotate(crop(), 99, label_map)
        with pytest.raises(LabelStoreError, match="neither a trained class"):
            store.annotate(crop(), -2, label_map)

    def test_moving_object_registers_no_position(self, tmp_path, label_map):
        """§5.2 only holds for static objects; a walking person must not become
        a position that later frames propagate from."""
        store = LabelStore.load_or_create(tmp_path, "s1")
        label, obj = store.annotate(
            crop(), label_map.id_for("person"), label_map, new_object=False
        )
        assert obj is None
        assert store.objects == {}
        assert label.provenance == PROVENANCE_ANNOTATED
        # And nothing propagates from it.
        counts = store.propagate(hover_crops(5))
        assert counts["propagated"] == 0

    def test_skip_is_remembered(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        c = crop()
        store.skip(c)
        assert c.crop_id in store.skipped
        assert store.unlabelled([c]) == []


class TestProvenanceIsMandatory:
    """§5.3: a label with no provenance cannot be excluded from the test split."""

    def test_no_label_without_a_provenance(self):
        with pytest.raises(LabelStoreError, match="provenance must be one of"):
            CropLabel(crop_id="c", label_id=0, label_name="person", provenance="")
        with pytest.raises(LabelStoreError, match="provenance must be one of"):
            CropLabel(
                crop_id="c", label_id=0, label_name="person", provenance="probably_fine"
            )

    def test_annotated_and_propagated_are_distinguishable(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        crops = hover_crops(4)
        store.annotate(crops[0], label_map.id_for("pole"), label_map)
        store.propagate(crops)

        labels = [store.labels[c.crop_id] for c in crops]
        assert labels[0].is_annotated
        assert all(not l.is_annotated for l in labels[1:])
        assert sum(1 for l in labels if l.provenance == PROVENANCE_PROPAGATED) == 3


class TestPropagation:
    def test_one_decision_labels_every_frame(self, tmp_path, label_map):
        """The §5.2 arithmetic, asserted: 1 annotation -> N labels."""
        store = LabelStore.load_or_create(tmp_path, "s1")
        crops = hover_crops(60)
        store.annotate(crops[0], label_map.id_for("vegetation"), label_map)
        counts = store.propagate(crops)

        assert counts["propagated"] == 59
        assert counts["already_labelled"] == 1  # the annotated one
        assert len(store.labels) == 60
        assert store.summary()["n_annotated"] == 1
        assert store.summary()["n_propagated"] == 59
        # One human decision, one distinct physical object.
        assert store.summary()["distinct_objects"] == 1

    def test_distant_clusters_do_not_match(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        near = crop(crop_id="s1/000000/00", centroid=(0.0, 0.0, 1.0))
        far = crop(crop_id="s1/000000/01", centroid=(5.0, 0.0, 1.0))
        store.annotate(near, label_map.id_for("person"), label_map)
        counts = store.propagate([near, far])
        assert counts["propagated"] == 0
        assert counts["unmatched"] == 1
        assert far.crop_id not in store.labels

    def test_radius_is_configurable(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        a = crop(crop_id="s1/000000/00", centroid=(0.0, 0.0, 1.0))
        b = crop(crop_id="s1/000001/00", centroid=(1.2, 0.0, 1.0))
        store.annotate(a, label_map.id_for("pole"), label_map)

        assert store.propagate([b], radius_m=0.5)["propagated"] == 0
        assert store.propagate([b], radius_m=2.0)["propagated"] == 1

    def test_default_radius_is_the_dbscan_eps(self):
        """Not an arbitrary number: two objects closer than eps_m would have
        merged into one cluster anyway."""
        assert DEFAULT_MATCH_RADIUS_M == 0.5

    def test_tolerates_the_centroid_wobble_of_a_real_hover(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        crops = hover_crops(20, wobble_m=0.2)
        store.annotate(crops[0], label_map.id_for("pole"), label_map)
        counts = store.propagate(crops)
        assert counts["propagated"] == 19

    def test_nearest_object_wins_and_ambiguity_is_flagged(self, tmp_path, label_map):
        """Two labelled objects inside the radius: nearest wins, and the crop is
        marked ambiguous so it can be counted or excluded later."""
        store = LabelStore.load_or_create(tmp_path, "s1")
        a = crop(crop_id="s1/000000/00", centroid=(0.0, 0.0, 1.0))
        b = crop(crop_id="s1/000000/01", centroid=(0.4, 0.0, 1.0))
        store.annotate(a, label_map.id_for("person"), label_map, radius_m=1.0)
        store.annotate(b, label_map.id_for("pole"), label_map, radius_m=1.0)

        probe = crop(crop_id="s1/000001/00", centroid=(0.05, 0.0, 1.0))
        counts = store.propagate([probe], radius_m=1.0)
        label = store.labels[probe.crop_id]

        assert counts["ambiguous"] == 1
        assert label.ambiguous is True
        assert label.label_name == "person"  # the nearer object
        assert label.runner_up_distance_m is not None
        assert label.match_distance_m < label.runner_up_distance_m

    def test_annotation_is_never_overwritten_by_propagation(self, tmp_path, label_map):
        """A human decision outranks the heuristic, always."""
        store = LabelStore.load_or_create(tmp_path, "s1")
        a = crop(crop_id="s1/000000/00", centroid=(0.0, 0.0, 1.0))
        b = crop(crop_id="s1/000001/00", centroid=(0.05, 0.0, 1.0))
        store.annotate(a, label_map.id_for("person"), label_map)
        # The annotator disagrees with what propagation would say for b.
        store.annotate(b, label_map.id_for("vehicle"), label_map, new_object=False)

        store.propagate([a, b], overwrite_propagated=True)
        assert store.labels[b.crop_id].label_name == "vehicle"
        assert store.labels[b.crop_id].is_annotated

    def test_repropagation_after_moving_an_object(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        a = crop(crop_id="s1/000000/00", centroid=(0.0, 0.0, 1.0))
        far = crop(crop_id="s1/000001/00", centroid=(3.0, 0.0, 1.0))
        _label, obj = store.annotate(a, label_map.id_for("pole"), label_map)
        assert store.propagate([far])["unmatched"] == 1

        store.move_object(obj.object_id, (3.0, 0.0, 1.0))
        counts = store.propagate([far], overwrite_propagated=True)
        assert counts["propagated"] == 1
        assert store.labels[far.crop_id].label_name == "pole"

    def test_moving_an_unknown_object_is_an_error(self, tmp_path):
        store = LabelStore.load_or_create(tmp_path, "s1")
        with pytest.raises(LabelStoreError, match="no object 5"):
            store.move_object(5, (0, 0, 0))


class TestAttributes:
    """The annotation policies in label_map.yaml ask for more than a class id."""

    def test_person_merge_count_is_stored_and_propagated(self, tmp_path, label_map):
        """The `person` policy: label a merged cluster `person` and record how
        many people, so evaluation can report merged and single cases apart."""
        store = LabelStore.load_or_create(tmp_path, "s1")
        crops = hover_crops(5)
        store.annotate(
            crops[0],
            label_map.id_for("person"),
            label_map,
            attributes={"merged_instances": 3},
        )
        store.propagate(crops)

        assert store.labels[crops[0].crop_id].attributes["merged_instances"] == 3
        # Attributes are properties of the object, so they travel with the label.
        for c in crops[1:]:
            assert store.labels[c.crop_id].attributes["merged_instances"] == 3

    @pytest.mark.parametrize(
        "class_name,attributes",
        [
            ("person", {"merged_instances": 2}),
            ("vegetation", {"extended": True}),
            ("pole", {"subtype": "goal"}),
            ("net_fence", {"edge_on": True}),
        ],
    )
    def test_every_policy_attribute_round_trips(
        self, tmp_path, label_map, class_name, attributes
    ):
        store = LabelStore.load_or_create(tmp_path, "s1")
        store.annotate(
            crop(), label_map.id_for(class_name), label_map, attributes=attributes
        )
        store.save()

        reloaded = LabelStore.load(store.path)
        assert reloaded.labels[crop().crop_id].attributes == attributes

    def test_unknown_keys_are_kept_with_a_warning(self):
        """The schema stays open: a policy written later needs no code change."""
        attrs, warnings = validate_attributes({"weather_note": "wet"}, label_name="person")
        assert attrs == {"weather_note": "wet"}
        assert any("not one of the keys" in w for w in warnings)

    def test_a_key_from_another_class_warns(self):
        attrs, warnings = validate_attributes({"edge_on": True}, label_name="person")
        assert attrs == {"edge_on": True}
        assert any("net_fence" in w for w in warnings)

    def test_wrong_value_type_is_refused(self):
        with pytest.raises(LabelStoreError, match="expects int"):
            validate_attributes({"merged_instances": "three"}, label_name="person")
        with pytest.raises(LabelStoreError, match="expects bool"):
            validate_attributes({"edge_on": 3}, label_name="net_fence")

    def test_boolean_is_not_accepted_as_a_count(self):
        """True == 1 in Python; a boolean merge count is a bug, not a count."""
        with pytest.raises(LabelStoreError, match="boolean"):
            validate_attributes({"merged_instances": True}, label_name="person")

    def test_integral_float_is_accepted_as_a_count(self):
        """YAML and JSON both happily produce 2.0 where 2 was meant."""
        attrs, _ = validate_attributes({"merged_instances": 2.0}, label_name="person")
        assert attrs["merged_instances"] == 2
        assert isinstance(attrs["merged_instances"], int)

    def test_non_json_value_is_refused(self):
        with pytest.raises(LabelStoreError, match="must be bool, int, float, str"):
            validate_attributes({"subtype": ["goal"]}, label_name="pole")

    def test_attributes_are_counted_in_the_summary(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        crops = hover_crops(4)
        store.annotate(
            crops[0],
            label_map.id_for("net_fence"),
            label_map,
            attributes={"edge_on": True},
        )
        store.propagate(crops)
        assert store.summary()["attribute_counts"]["edge_on"] == 4


class TestResumability:
    """The tool is restarted constantly; losing work is the failure to avoid."""

    def test_round_trips_through_disk(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        crops = hover_crops(6)
        store.annotate(crops[0], label_map.id_for("pole"), label_map)
        store.propagate(crops)
        store.skip(crop(crop_id="s1/000099/00"))
        store.save()

        reloaded = LabelStore.load_or_create(tmp_path, "s1")
        assert reloaded.labels.keys() == store.labels.keys()
        assert reloaded.objects.keys() == store.objects.keys()
        assert reloaded.skipped == store.skipped
        assert reloaded.summary() == store.summary()

    def test_resuming_presents_only_what_is_left(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        crops = hover_crops(5) + [crop(crop_id="s1/000000/01", centroid=(9.0, 0.0, 1.0))]
        store.annotate(crops[0], label_map.id_for("pole"), label_map)
        store.propagate(crops)
        store.save()

        reloaded = LabelStore.load_or_create(tmp_path, "s1")
        remaining = reloaded.unlabelled(crops)
        assert [c.crop_id for c in remaining] == ["s1/000000/01"]

    def test_the_write_is_atomic(self, tmp_path, label_map):
        """A crash mid-write must not truncate an afternoon of annotation."""
        store = LabelStore.load_or_create(tmp_path, "s1")
        store.annotate(crop(), label_map.id_for("pole"), label_map)
        store.save()
        assert store.path.exists()
        assert not store.path.with_name(store.path.name + ".tmp").exists()
        # And the file is complete, parseable JSON.
        doc = json.loads(store.path.read_text(encoding="utf-8"))
        assert doc["session_id"] == "s1"
        assert doc["labels"]

    def test_a_changed_label_map_is_refused_not_reinterpreted(self, tmp_path, label_map):
        """Class ids may mean different things now; existing annotation must not
        be silently re-read against a different vocabulary."""
        store = LabelStore.load_or_create(
            tmp_path, "s1", label_map_sha256=label_map.sha256
        )
        store.annotate(crop(), label_map.id_for("pole"), label_map)
        store.save()

        with pytest.raises(LabelStoreError, match="class ids may mean different"):
            LabelStore.load_or_create(tmp_path, "s1", label_map_sha256="deadbeef" * 8)

    def test_store_version_mismatch_is_refused(self, tmp_path, label_map):
        store = LabelStore.load_or_create(tmp_path, "s1")
        store.annotate(crop(), label_map.id_for("pole"), label_map)
        store.save()
        doc = json.loads(store.path.read_text(encoding="utf-8"))
        doc["version"] = 77
        store.path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(LabelStoreError, match="holds human work"):
            LabelStore.load(store.path)


class TestTextRendering:
    """It has to be usable over ssh from a field laptop, with no display."""

    def test_projection_shape_and_charset(self):
        points = np.random.default_rng(0).normal(0, 1, (500, 3))
        rows = ascii_projection(points, width=20, height=8)
        assert len(rows) == 8
        assert all(len(r) == 20 for r in rows)
        assert set("".join(rows)) <= set(" .:-=+*#%@")

    def test_empty_cluster_renders_blank_not_crashes(self):
        rows = ascii_projection(np.zeros((0, 3)), width=10, height=4)
        assert rows == [" " * 10] * 4

    def test_non_finite_points_are_dropped(self):
        points = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [1.0, 1.0, 1.0]])
        rows = ascii_projection(points, width=10, height=4)
        assert any(ch != " " for ch in "".join(rows))

    def test_aspect_ratio_is_preserved(self):
        """A 6 m pole must not be stretched to fill the box — the shape is the
        thing the annotator is reading."""
        g = np.random.default_rng(1)
        pole = np.column_stack(
            [
                g.uniform(-0.1, 0.1, 400),
                g.uniform(-0.1, 0.1, 400),
                g.uniform(0.0, 6.0, 400),
            ]
        )
        rows = ascii_projection(pole, width=31, height=15, axes=(0, 2))
        occupied_cols = {i for r in rows for i, ch in enumerate(r) if ch != " "}
        # Thin in x, tall in z: the ink occupies a narrow column band.
        assert len(occupied_cols) <= 6, f"pole spread over {len(occupied_cols)} columns"

    def test_tiny_grid_is_refused(self):
        with pytest.raises(ValueError, match="at least 2x2"):
            ascii_projection(np.zeros((3, 3)), width=1, height=1)

    def test_describe_crop_reports_metres_and_both_views(self):
        c = crop(extent=(0.4, 0.4, 1.8), centroid=(3.0, 4.0, 1.0))
        points = np.random.default_rng(0).normal(0, 0.2, (200, 3)) + np.array([3, 4, 1])
        lines = describe_crop(c, points, width=20, height=8)
        text = "\n".join(lines)

        assert "s1/000000/00" in text
        assert "range 5.10 m" in text  # sqrt(9 + 16 + 1)
        assert "extent    x=0.40  y=0.40  z=1.80 m" in text
        assert "top view  x-y" in text
        assert "elevation x-z" in text
        assert "height above fitted ground" in text

    def test_describe_crop_without_points_is_geometry_only(self):
        lines = describe_crop(crop())
        assert not any("top view" in l for l in lines)
        assert any("centroid" in l for l in lines)
