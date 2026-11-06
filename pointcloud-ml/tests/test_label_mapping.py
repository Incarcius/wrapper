"""Label-map validation: every rule corresponds to a way a checkpoint goes wrong."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pcml.data.label_mapping import LabelMapError, load_label_map

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MAP = REPO_ROOT / "data" / "label_map.yaml"
TEMPLATE_MAP = REPO_ROOT / "data" / "label_map.template.yaml"

#: The vocabulary the owner confirmed on 15 Oct 2026.
CONFIRMED_CLASSES = (
    "person",
    "vegetation",
    "pole",
    "structure",
    "net_fence",
    "vehicle",
)


def write_map(tmp_path: Path, doc: dict, name: str = "label_map.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def valid_doc(n_classes: int = 3) -> dict:
    return {
        "version": 1,
        "reject": {"reserved_class_id": -1, "reserved_class_name": "unknown"},
        "classes": [
            {
                "class_id": i,
                "name": f"class_{i}",
                "definition": f"what class {i} physically is",
                "annotation_policy": f"what to do with an ambiguous class {i}",
            }
            for i in range(n_classes)
        ],
        "class_weights": None,
    }


class TestValidMaps:
    def test_k_is_derived_from_the_file(self, tmp_path):
        for k in (2, 3, 6, 11):
            lm = load_label_map(write_map(tmp_path, valid_doc(k), f"m{k}.yaml"))
            assert lm.num_classes == k
            assert lm.names == tuple(f"class_{i}" for i in range(k))

    def test_round_trips_ids_and_names(self, tmp_path):
        lm = load_label_map(write_map(tmp_path, valid_doc(4)))
        for i in range(4):
            assert lm.id_for(lm.name_for(i)) == i

    def test_reject_outcome_resolves_but_is_not_a_trained_class(self, tmp_path):
        lm = load_label_map(write_map(tmp_path, valid_doc(3)))
        assert lm.name_for(-1) == "unknown"
        assert lm.id_for("unknown") == -1
        assert not lm.is_trained_class(-1)
        assert lm.is_trained_class(0)
        assert not lm.is_trained_class(3)

    def test_sha256_is_recorded_and_changes_with_content(self, tmp_path):
        a = load_label_map(write_map(tmp_path, valid_doc(3), "a.yaml"))
        b = load_label_map(write_map(tmp_path, valid_doc(4), "b.yaml"))
        assert a.sha256 and len(a.sha256) == 64
        assert a.sha256 != b.sha256

    def test_definitions_and_policies_are_carried_through(self, tmp_path):
        doc = valid_doc(1)
        doc["classes"] = [
            {
                "class_id": 0,
                "name": "person",
                "definition": "a human",
                "annotation_policy": "label merges as person",
                "expected_extent_m": [1.2, 2.1],
                "notes": "n",
            },
            {
                "class_id": 1,
                "name": "pole",
                "definition": "thin vertical",
                "annotation_policy": "a goal is a pole",
            },
        ]
        lm = load_label_map(write_map(tmp_path, doc))
        assert lm.classes[0].annotation_policy.startswith("label merges")
        assert lm.classes[0].expected_extent_m == (1.2, 2.1)
        assert lm.classes[1].expected_extent_m is None

    def test_class_weights_as_list_and_as_mapping(self, tmp_path):
        doc = valid_doc(3)
        doc["class_weights"] = [1.0, 2.0, 0.5]
        assert load_label_map(write_map(tmp_path, doc, "l.yaml")).class_weights == (
            1.0,
            2.0,
            0.5,
        )
        doc["class_weights"] = {"class_2": 0.5, "class_0": 1.0, "class_1": 2.0}
        # Mapping form is reordered into id order, which is what the loss needs.
        assert load_label_map(write_map(tmp_path, doc, "m.yaml")).class_weights == (
            1.0,
            2.0,
            0.5,
        )


class TestRejectedMaps:
    def test_gap_in_ids(self, tmp_path):
        doc = valid_doc(3)
        doc["classes"][2]["class_id"] = 5
        with pytest.raises(LabelMapError, match="contiguous from 0"):
            load_label_map(write_map(tmp_path, doc))

    def test_ids_not_starting_at_zero(self, tmp_path):
        doc = valid_doc(3)
        for i, entry in enumerate(doc["classes"]):
            entry["class_id"] = i + 1
        with pytest.raises(LabelMapError, match="contiguous from 0"):
            load_label_map(write_map(tmp_path, doc))

    def test_duplicate_id(self, tmp_path):
        doc = valid_doc(3)
        doc["classes"][2]["class_id"] = 1
        with pytest.raises(LabelMapError, match="duplicate class_id"):
            load_label_map(write_map(tmp_path, doc))

    def test_duplicate_name(self, tmp_path):
        doc = valid_doc(3)
        doc["classes"][2]["name"] = "class_0"
        with pytest.raises(LabelMapError, match="duplicate class name"):
            load_label_map(write_map(tmp_path, doc))

    def test_reserved_id_as_a_trained_class(self, tmp_path):
        """-1 is the reject *outcome*, never a class. Â§3/Â§5."""
        doc = valid_doc(2)
        doc["classes"].append({"class_id": -1, "name": "junk"})
        with pytest.raises(LabelMapError, match="reserved class_id"):
            load_label_map(write_map(tmp_path, doc))

    def test_reserved_name_as_a_trained_class(self, tmp_path):
        doc = valid_doc(2)
        doc["classes"].append({"class_id": 2, "name": "unknown"})
        with pytest.raises(LabelMapError, match="reserved for"):
            load_label_map(write_map(tmp_path, doc))

    def test_empty_class_list_is_refused(self, tmp_path):
        doc = valid_doc(0)
        doc["classes"] = []
        with pytest.raises(LabelMapError, match="K would be 0"):
            load_label_map(write_map(tmp_path, doc))

    def test_missing_definition_is_refused(self, tmp_path):
        """A class two annotators would label differently is not loadable."""
        doc = valid_doc(2)
        del doc["classes"][1]["definition"]
        with pytest.raises(LabelMapError, match="no definition"):
            load_label_map(write_map(tmp_path, doc))

    def test_missing_annotation_policy_is_refused(self, tmp_path):
        doc = valid_doc(2)
        del doc["classes"][1]["annotation_policy"]
        with pytest.raises(LabelMapError, match="no annotation_policy"):
            load_label_map(write_map(tmp_path, doc))

    def test_blank_definition_is_refused(self, tmp_path):
        """Whitespace is not a definition."""
        doc = valid_doc(2)
        doc["classes"][1]["definition"] = "   \n  "
        with pytest.raises(LabelMapError, match="no definition"):
            load_label_map(write_map(tmp_path, doc))

    def test_missing_name_or_id(self, tmp_path):
        doc = valid_doc(2)
        del doc["classes"][1]["name"]
        with pytest.raises(LabelMapError, match="missing `name`"):
            load_label_map(write_map(tmp_path, doc))

        doc = valid_doc(2)
        del doc["classes"][1]["class_id"]
        with pytest.raises(LabelMapError, match="missing `class_id`"):
            load_label_map(write_map(tmp_path, doc))

    def test_non_integer_id(self, tmp_path):
        doc = valid_doc(2)
        doc["classes"][1]["class_id"] = "1"
        with pytest.raises(LabelMapError, match="must be an integer"):
            load_label_map(write_map(tmp_path, doc))

    def test_non_negative_reserved_id(self, tmp_path):
        doc = valid_doc(2)
        doc["reject"]["reserved_class_id"] = 7
        with pytest.raises(LabelMapError, match="must be negative"):
            load_label_map(write_map(tmp_path, doc))

    def test_threshold_out_of_range(self, tmp_path):
        doc = valid_doc(2)
        doc["reject"]["confidence_threshold"] = 1.5
        with pytest.raises(LabelMapError, match=r"\[0, 1\]"):
            load_label_map(write_map(tmp_path, doc))

    def test_wrong_number_of_class_weights(self, tmp_path):
        doc = valid_doc(3)
        doc["class_weights"] = [1.0, 2.0]
        with pytest.raises(LabelMapError, match="2 entries but there are 3"):
            load_label_map(write_map(tmp_path, doc))

    def test_class_weights_mapping_missing_a_class(self, tmp_path):
        doc = valid_doc(3)
        doc["class_weights"] = {"class_0": 1.0, "class_1": 1.0}
        with pytest.raises(LabelMapError, match="missing a weight"):
            load_label_map(write_map(tmp_path, doc))

    def test_unknown_class_id_resolution_raises(self, tmp_path):
        lm = load_label_map(write_map(tmp_path, valid_doc(2)))
        with pytest.raises(LabelMapError, match="not in this label map"):
            lm.name_for(9)
        with pytest.raises(LabelMapError, match="not in this label map"):
            lm.id_for("nope")

    def test_missing_file_points_at_the_template_and_the_example(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            load_label_map(tmp_path / "absent.yaml")
        message = str(exc.value)
        assert "data/label_map.yaml" in message

    def test_empty_file(self, tmp_path):
        p = tmp_path / "e.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(LabelMapError, match="is empty"):
            load_label_map(p)

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("classes: [\n  - {", encoding="utf-8")
        with pytest.raises(LabelMapError, match="not valid YAML"):
            load_label_map(p)


class TestTheConfirmedLabelMap:
    """data/label_map.yaml is the vocabulary of record. It must satisfy every
    rule above, and the facts the rest of the codebase depends on must hold."""

    def test_k_is_six_with_the_confirmed_class_names(self):
        lm = load_label_map(REAL_MAP)
        assert lm.num_classes == 6
        assert lm.names == CONFIRMED_CLASSES

    def test_ids_are_the_index_positions(self):
        """Anything that indexes a softmax by class_id depends on this."""
        lm = load_label_map(REAL_MAP)
        for i, entry in enumerate(lm.classes):
            assert entry.class_id == i
            assert lm.name_for(i) == entry.name
            assert lm.id_for(entry.name) == i

    def test_reject_is_an_output_not_a_trained_class(self):
        lm = load_label_map(REAL_MAP)
        assert lm.reserved_class_id == -1
        assert lm.reserved_class_name == "unknown"
        assert lm.reject_trained_as_class is False
        assert "unknown" not in lm.names
        # Storable by an annotator (negatives calibrate the threshold), but not
        # a training target. Those are deliberately different questions.
        assert lm.is_storable_label(-1)
        assert not lm.is_trained_class(-1)

    def test_threshold_is_not_guessed(self):
        """PROVISIONAL 0.0 = rejection disabled until the curve is measured."""
        assert load_label_map(REAL_MAP).confidence_threshold == 0.0

    def test_class_weights_are_still_unset(self):
        """They come from the measured training distribution, not a guess."""
        assert load_label_map(REAL_MAP).class_weights is None

    def test_it_is_not_flagged_as_an_example(self):
        """The example flag suppresses reported numbers; a confirmed map must
        not carry it, or real results would be silently caveated away."""
        assert load_label_map(REAL_MAP).is_example is False

    def test_every_class_carries_a_definition_policy_and_extent(self):
        lm = load_label_map(REAL_MAP)
        for entry in lm.classes:
            assert entry.definition and entry.definition.strip()
            assert entry.annotation_policy and entry.annotation_policy.strip()
            assert entry.expected_extent_m is not None
            lo, hi = entry.expected_extent_m
            assert lo is not None and hi is not None and 0 < lo <= hi

    def test_sha256_is_recorded(self):
        """Manifests and checkpoints name the vocabulary by this hash."""
        lm = load_label_map(REAL_MAP)
        assert lm.sha256 and len(lm.sha256) == 64

    def test_template_refuses_to_load(self):
        """The template is an empty structural skeleton, not a vocabulary."""
        with pytest.raises(LabelMapError, match="K would be 0"):
            load_label_map(TEMPLATE_MAP)


class TestExtentWarning:
    """expected_extent_m as a mislabel check, used by scripts/label_clusters.py."""

    def test_no_warning_inside_the_expected_range(self):
        lm = load_label_map(REAL_MAP)
        assert lm.extent_warning(lm.id_for("person"), 1.75) is None

    def test_warns_above_the_maximum(self):
        """A 9 m person is a merged cluster or a mis-keyed label."""
        lm = load_label_map(REAL_MAP)
        warning = lm.extent_warning(lm.id_for("person"), 9.0)
        assert warning is not None
        assert "ABOVE" in warning and "person" in warning

    def test_warns_below_the_minimum(self):
        lm = load_label_map(REAL_MAP)
        warning = lm.extent_warning(lm.id_for("structure"), 0.4)
        assert warning is not None
        assert "BELOW" in warning and "structure" in warning

    def test_no_warning_for_the_reject_outcome(self):
        """`unknown` has no expected size — it is whatever was not a class."""
        lm = load_label_map(REAL_MAP)
        assert lm.extent_warning(-1, 500.0) is None

    def test_no_warning_when_a_class_declares_no_extent(self, tmp_path):
        doc = valid_doc(2)
        lm = load_label_map(write_map(tmp_path, doc))
        assert lm.extent_warning(0, 1e6) is None
