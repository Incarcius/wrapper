"""The command-line tools, driven as the README documents them.

Scripts are the part of a toolchain that rots quietly: the library stays tested
while an argument name drifts and nobody notices until the day the real data
arrives. So the three commands the README gives are executed here, in order,
against synthetic recordings — ``extract_crops``, ``label_clusters``,
``make_splits`` — with stdin piped, which is also what proves the labelling tool
genuinely works headless.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn", reason="needs a DBSCAN backend")

from pcml.data.crops import load_crop_index  # noqa: E402
from pcml.data.label_mapping import load_label_map  # noqa: E402
from pcml.data.labels import LabelStore  # noqa: E402
from synthetic_recordings import (  # noqa: E402
    synthetic_session_frames,
    write_meta_sidecar,
    write_npz_session,
    write_pointcloud2_bag,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
REAL_MAP = REPO_ROOT / "data" / "label_map.yaml"


def load_script(name: str):
    """Import a script by path. They are tools, not an installed package."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recordings(tmp_path):
    """Three sessions: a real .db3 bag and two .npz, each with §4.2 metadata."""
    raw = tmp_path / "raw"
    frames = synthetic_session_frames(n_frames=2, extent_m=8.0, density_per_m2=25.0, seed=11)
    bag_dir = raw / "session_01_hover_12m"
    write_pointcloud2_bag(bag_dir / "session_01_hover_12m_0.db3", frames)
    write_meta_sidecar(bag_dir, "session_01_hover_12m")

    for i, seed in enumerate((31, 57), start=2):
        frames = synthetic_session_frames(
            n_frames=2, extent_m=8.0, density_per_m2=25.0, seed=seed
        )
        p = write_npz_session(raw / f"session_{i:02d}_hover_15m.npz", frames)
        write_meta_sidecar(p, f"session_{i:02d}_hover_15m")
    return raw


# ======================================================================
class TestExtractCrops:
    def test_extracts_every_session_in_a_directory(self, tmp_path, recordings, capsys):
        extract = load_script("extract_crops")
        out = tmp_path / "processed"
        rc = extract.main([str(recordings), "--glob", "--out", str(out)])
        captured = capsys.readouterr().out

        assert rc == 0
        indices = sorted(out.glob("*_crops.json"))
        assert len(indices) == 3
        for path in indices:
            index = load_crop_index(path)
            assert index.crops
            assert index.preprocess_sha256
        # The config identity and the clustering parameters are echoed, so a run
        # recorded in a terminal log says what it did.
        assert "preprocess.yaml sha256" in captured
        assert "eps_m" in captured
        # And it points at the next command.
        assert "label_clusters.py" in captured

    def test_summary_reports_the_distributions_for_the_caps(
        self, tmp_path, recordings, capsys
    ):
        extract = load_script("extract_crops")
        summary_path = tmp_path / "reports" / "crops.json"
        rc = extract.main(
            [
                str(recordings),
                "--glob",
                "--out",
                str(tmp_path / "processed"),
                "--summary",
                "--summary-json",
                str(summary_path),
            ]
        )
        assert rc == 0
        assert "clusters_per_frame" in capsys.readouterr().out
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["n_crops"] > 0
        assert "p95" in summary["points_per_cluster"]

    def test_a_session_id_collision_is_refused(self, tmp_path, capsys):
        """It would leak near-duplicate frames across splits."""
        extract = load_script("extract_crops")
        frames = synthetic_session_frames(n_frames=1, extent_m=6.0, density_per_m2=20.0)
        a = write_npz_session(tmp_path / "a" / "same.npz", frames)
        b = write_npz_session(tmp_path / "b" / "same.npz", frames)
        rc = extract.main([str(a), str(b), "--out", str(tmp_path / "out")])
        captured = capsys.readouterr().out
        assert rc == 0  # one succeeded
        assert "already used by" in captured
        assert len(list((tmp_path / "out").glob("*_crops.json"))) == 1

    def test_missing_metadata_is_warned_about(self, tmp_path, capsys):
        extract = load_script("extract_crops")
        frames = synthetic_session_frames(n_frames=1, extent_m=6.0, density_per_m2=20.0)
        p = write_npz_session(tmp_path / "bare_session.npz", frames)
        extract.main([str(p), "--out", str(tmp_path / "out")])
        assert "no §4.2 metadata sidecar" in capsys.readouterr().out

    def test_glob_ignores_files_that_are_not_recordings(
        self, tmp_path, recordings, capsys
    ):
        """A collection directory also holds sidecars, notes and stray exports."""
        (recordings / "notes.txt").write_text("mount angle 35 deg", encoding="utf-8")
        (recordings / "scan.lvx").write_bytes(b"")
        rc = load_script("extract_crops").main(
            [str(recordings), "--glob", "--out", str(tmp_path / "out")]
        )
        capsys.readouterr()
        assert rc == 0
        assert len(list((tmp_path / "out").glob("*_crops.json"))) == 3

    def test_an_unreadable_recording_named_explicitly_is_skipped_not_fatal(
        self, tmp_path, recordings, capsys
    ):
        """Naming an unsupported format must report it and carry on with the rest."""
        junk = recordings / "scan.lvx"
        junk.write_bytes(b"")
        good = recordings / "session_02_hover_15m.npz"
        rc = load_script("extract_crops").main(
            [str(junk), str(good), "--out", str(tmp_path / "out")]
        )
        captured = capsys.readouterr().out
        assert rc == 0
        assert "SKIP" in captured
        assert "not implemented" in captured
        assert len(list((tmp_path / "out").glob("*_crops.json"))) == 1


# ======================================================================
class TestLabelClusters:
    """Driven with piped stdin, which is exactly the headless path."""

    def _crops(self, tmp_path, recordings):
        load_script("extract_crops").main(
            [str(recordings), "--glob", "--out", str(tmp_path / "processed")]
        )
        return sorted((tmp_path / "processed").glob("*_crops.json"))[0]

    def test_labels_once_per_object_and_propagates(
        self, tmp_path, recordings, monkeypatch, capsys
    ):
        """The §5.2 workflow through the CLI: a few keypresses, many labels."""
        crops_path = self._crops(tmp_path, recordings)
        index = load_crop_index(crops_path)
        label_map = load_label_map(REAL_MAP)

        n_objects = len(index.frames()[min(index.frames())])
        # One key per object in the first frame, then 'q'. Each `2` (pole) is
        # followed by a blank line for the `subtype` prompt the pole policy asks.
        keys = "".join("2\n\n" for _ in range(n_objects)) + "q\n"
        monkeypatch.setattr(sys, "stdin", io.StringIO(keys))

        labeller = load_script("label_clusters")
        rc = labeller.main([str(crops_path), "--label-map", str(REAL_MAP)])
        captured = capsys.readouterr().out
        assert rc == 0

        store = LabelStore.load_or_create(crops_path.parent, index.session_id)
        summary = store.summary()
        assert summary["n_annotated"] == n_objects
        # Every later frame was labelled without a human touching it.
        assert summary["n_propagated"] > 0
        assert summary["n_labels"] == summary["n_annotated"] + summary["n_propagated"]
        assert all(l.label_name == "pole" for l in store.labels.values())

        # The text UI really printed geometry and a projection.
        assert "UNMATCHED CLUSTER" in captured
        assert "top view  x-y" in captured
        assert "elevation x-z" in captured
        assert "propagated to" in captured

    def test_is_resumable(self, tmp_path, recordings, monkeypatch, capsys):
        crops_path = self._crops(tmp_path, recordings)
        index = load_crop_index(crops_path)
        labeller = load_script("label_clusters")

        # First run: label one object, then quit.
        monkeypatch.setattr(sys, "stdin", io.StringIO("1\nq\n"))
        labeller.main([str(crops_path), "--label-map", str(REAL_MAP), "--limit", "1"])
        capsys.readouterr()
        first = LabelStore.load_or_create(crops_path.parent, index.session_id).summary()
        assert first["n_annotated"] == 1

        # Second run: it must not re-ask about what is already decided.
        monkeypatch.setattr(sys, "stdin", io.StringIO("2\n\nq\n"))
        labeller.main([str(crops_path), "--label-map", str(REAL_MAP)])
        capsys.readouterr()
        second = LabelStore.load_or_create(crops_path.parent, index.session_id).summary()
        assert second["n_annotated"] == 2
        assert second["n_labels"] > first["n_labels"]

    def test_unknown_is_selectable(self, tmp_path, recordings, monkeypatch, capsys):
        """`reject.trained_as_class` is false, so negatives still get recorded."""
        crops_path = self._crops(tmp_path, recordings)
        index = load_crop_index(crops_path)
        monkeypatch.setattr(sys, "stdin", io.StringIO("u\nq\n"))
        load_script("label_clusters").main(
            [str(crops_path), "--label-map", str(REAL_MAP)]
        )
        capsys.readouterr()
        store = LabelStore.load_or_create(crops_path.parent, index.session_id)
        assert any(l.label_id == -1 for l in store.labels.values())
        assert any(l.label_name == "unknown" for l in store.labels.values())

    def test_skip_is_remembered_across_runs(self, tmp_path, recordings, monkeypatch, capsys):
        crops_path = self._crops(tmp_path, recordings)
        index = load_crop_index(crops_path)
        labeller = load_script("label_clusters")

        monkeypatch.setattr(sys, "stdin", io.StringIO("s\nq\n"))
        labeller.main([str(crops_path), "--label-map", str(REAL_MAP)])
        capsys.readouterr()
        store = LabelStore.load_or_create(crops_path.parent, index.session_id)
        assert len(store.skipped) == 1
        skipped_id = next(iter(store.skipped))

        monkeypatch.setattr(sys, "stdin", io.StringIO("q\n"))
        labeller.main([str(crops_path), "--label-map", str(REAL_MAP)])
        captured = capsys.readouterr().out
        assert skipped_id not in captured.split("UNMATCHED CLUSTER")[-1]

    def test_unrecognised_key_reprompts(self, tmp_path, recordings, monkeypatch, capsys):
        """A mis-keyed class must not silently become class 0."""
        crops_path = self._crops(tmp_path, recordings)
        monkeypatch.setattr(sys, "stdin", io.StringIO("z\n9\nq\n"))
        load_script("label_clusters").main(
            [str(crops_path), "--label-map", str(REAL_MAP)]
        )
        captured = capsys.readouterr().out
        assert "unrecognised key" in captured

    def test_eof_saves_rather_than_crashing(self, tmp_path, recordings, monkeypatch, capsys):
        crops_path = self._crops(tmp_path, recordings)
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # immediate EOF
        rc = load_script("label_clusters").main(
            [str(crops_path), "--label-map", str(REAL_MAP)]
        )
        assert rc == 0
        assert "end of input" in capsys.readouterr().out

    def test_report_mode_changes_nothing(self, tmp_path, recordings, monkeypatch, capsys):
        crops_path = self._crops(tmp_path, recordings)
        index = load_crop_index(crops_path)
        monkeypatch.setattr(sys, "stdin", io.StringIO("2\n\nq\n"))
        labeller = load_script("label_clusters")
        labeller.main([str(crops_path), "--label-map", str(REAL_MAP)])
        capsys.readouterr()

        store_path = LabelStore.path_for(crops_path.parent, index.session_id)
        before = store_path.read_bytes()
        rc = labeller.main([str(crops_path), "--label-map", str(REAL_MAP), "--report"])
        captured = capsys.readouterr().out
        assert rc == 0
        assert store_path.read_bytes() == before
        assert "distinct physical objects" in captured
        assert "annotated" in captured and "propagated" in captured

    def test_dry_run_writes_nothing(self, tmp_path, recordings, monkeypatch, capsys):
        crops_path = self._crops(tmp_path, recordings)
        index = load_crop_index(crops_path)
        monkeypatch.setattr(sys, "stdin", io.StringIO("2\n\nq\n"))
        load_script("label_clusters").main(
            [str(crops_path), "--label-map", str(REAL_MAP), "--dry-run"]
        )
        capsys.readouterr()
        assert not LabelStore.path_for(crops_path.parent, index.session_id).exists()

    def test_writes_a_metadata_template(self, tmp_path, capsys):
        """So the §4.2 file exists before the flight, not after it."""
        out = tmp_path / "meta" / "session_09.meta.json"
        rc = load_script("label_clusters").main(["--write-meta-template", str(out)])
        assert rc == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["session_id"] == "session_09"
        for key in ("altitude_m", "mount_angle_deg", "motion", "field"):
            assert key in doc and doc[key] is None
        assert "AT CAPTURE TIME" in capsys.readouterr().out


# ======================================================================
class TestMakeSplits:
    def _labelled(self, tmp_path, recordings, monkeypatch, capsys):
        load_script("extract_crops").main(
            [str(recordings), "--glob", "--out", str(tmp_path / "processed")]
        )
        labeller = load_script("label_clusters")
        label_keys = ("0\n\n", "1\n\n", "2\n\n")
        for i, crops_path in enumerate(sorted((tmp_path / "processed").glob("*_crops.json"))):
            index = load_crop_index(crops_path)
            n_objects = len(index.frames()[min(index.frames())])
            keys = "".join(label_keys[(i + j) % 3] for j in range(n_objects)) + "q\n"
            monkeypatch.setattr(sys, "stdin", io.StringIO(keys))
            labeller.main([str(crops_path), "--label-map", str(REAL_MAP)])
        capsys.readouterr()
        return tmp_path / "processed"

    def test_writes_manifests_split_by_session(
        self, tmp_path, recordings, monkeypatch, capsys
    ):
        processed = self._labelled(tmp_path, recordings, monkeypatch, capsys)
        rc = load_script("make_splits").main(
            [
                "--crops",
                str(processed),
                "--out",
                str(tmp_path / "manifests"),
                "--label-map",
                str(REAL_MAP),
                "--fractions",
                "0.34,0.33,0.33",
            ]
        )
        captured = capsys.readouterr().out
        assert rc == 0
        assert "BY SESSION" in captured

        sessions_seen: set[str] = set()
        for name in ("train", "val", "test"):
            doc = json.loads(
                (tmp_path / "manifests" / f"{name}.json").read_text(encoding="utf-8")
            )
            assert doc["split_by"] == "session"
            for session in doc["sessions"]:
                assert session not in sessions_seen, f"{session} is in two splits"
                sessions_seen.add(session)
        assert len(sessions_seen) == 3
        assert (tmp_path / "manifests" / "splits.json").exists()

    def test_pinning_holds_a_session_out(self, tmp_path, recordings, monkeypatch, capsys):
        processed = self._labelled(tmp_path, recordings, monkeypatch, capsys)
        rc = load_script("make_splits").main(
            [
                "--crops",
                str(processed),
                "--out",
                str(tmp_path / "manifests"),
                "--label-map",
                str(REAL_MAP),
                "--pin",
                "session_01_hover_12m:test",
            ]
        )
        capsys.readouterr()
        assert rc == 0
        test_doc = json.loads(
            (tmp_path / "manifests" / "test.json").read_text(encoding="utf-8")
        )
        assert "session_01_hover_12m" in test_doc["sessions"]

    def test_dry_run_writes_nothing(self, tmp_path, recordings, monkeypatch, capsys):
        processed = self._labelled(tmp_path, recordings, monkeypatch, capsys)
        rc = load_script("make_splits").main(
            [
                "--crops",
                str(processed),
                "--out",
                str(tmp_path / "manifests"),
                "--label-map",
                str(REAL_MAP),
                "--dry-run",
            ]
        )
        captured = capsys.readouterr().out
        assert rc == 0
        assert "DRY RUN" in captured
        assert not (tmp_path / "manifests").exists()

    def test_writes_the_distribution_report(
        self, tmp_path, recordings, monkeypatch, capsys
    ):
        processed = self._labelled(tmp_path, recordings, monkeypatch, capsys)
        report = tmp_path / "reports" / "splits.md"
        load_script("make_splits").main(
            [
                "--crops",
                str(processed),
                "--out",
                str(tmp_path / "manifests"),
                "--label-map",
                str(REAL_MAP),
                "--fractions",
                "0.34,0.33,0.33",
                "--report",
                str(report),
            ]
        )
        capsys.readouterr()
        text = report.read_text(encoding="utf-8")
        assert "split BY SESSION" in text
        assert "TOTAL" in text

    def test_no_crops_is_a_clear_error(self, tmp_path):
        with pytest.raises(SystemExit, match="extract_crops"):
            load_script("make_splits").main(
                ["--crops", str(tmp_path), "--label-map", str(REAL_MAP)]
            )

    def test_bad_pin_is_rejected(self, tmp_path, recordings, monkeypatch, capsys):
        processed = self._labelled(tmp_path, recordings, monkeypatch, capsys)
        with pytest.raises(SystemExit, match="split must be one of"):
            load_script("make_splits").main(
                [
                    "--crops",
                    str(processed),
                    "--label-map",
                    str(REAL_MAP),
                    "--pin",
                    "session_01_hover_12m:holdout",
                ]
            )
