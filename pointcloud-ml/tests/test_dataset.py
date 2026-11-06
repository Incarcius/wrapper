"""The torch ``Dataset``: shapes, the transform order, and seeded augmentation.

``TestAugmentationIsInMetres`` is the class that matters for correctness. The
augmentation parameters in ``preprocess.yaml`` are metric — ``jitter_sigma_m:
0.01`` is a centimetre of Mid-70 ranging noise — so they must be applied to the
raw metric cluster, before normalisation. Applied afterwards they would mean "1%
of this object's own size", which scales with the object and is a different,
wrong augmentation. The tests assert the metric interpretation directly.

``TestReproducibility`` backs the claim that an item is identical on every run
with the same seed, whatever the batch size, shuffle order or worker count.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="the Dataset is the only torch-dependent module")
pytest.importorskip("sklearn", reason="needs a DBSCAN backend")

import torch  # noqa: E402

from pcml.data.clustering import AUX_FEATURE_NAMES  # noqa: E402
from pcml.data.config import AugmentationConfig, load_preprocess_config  # noqa: E402
from pcml.data.crops import extract_session_crops  # noqa: E402
from pcml.data.dataset import (  # noqa: E402
    ClusterCropDataset,
    DatasetError,
    DatasetItem,
    augment_cluster_points,
    rotation_matrix,
)
from pcml.data.label_mapping import load_label_map  # noqa: E402
from pcml.data.labels import LabelStore  # noqa: E402
from pcml.data.splits import write_manifests  # noqa: E402
from synthetic_recordings import (  # noqa: E402
    synthetic_session_frames,
    write_npz_session,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = REPO_ROOT / "configs" / "preprocess.yaml"
REAL_MAP = REPO_ROOT / "data" / "label_map.yaml"


@pytest.fixture
def label_map():
    return load_label_map(REAL_MAP)


@pytest.fixture
def config():
    return load_preprocess_config(PREPROCESS)


@pytest.fixture
def crops(tmp_path, config):
    """One session of real extracted crops, from the simulated field."""
    frames = synthetic_session_frames(n_frames=3, extent_m=8.0, density_per_m2=25.0)
    raw = write_npz_session(tmp_path / "session_01.npz", frames)
    return extract_session_crops(raw, tmp_path / "processed", config=config)


@pytest.fixture
def labelled(tmp_path, crops, label_map):
    """The crops, labelled once per object and propagated (the §5.2 workflow)."""
    store = LabelStore.load_or_create(tmp_path / "labels", crops.session_id)
    frame0 = [c for c in crops.crops if c.frame_index == 0]
    for i, c in enumerate(sorted(frame0, key=lambda c: -c.extent_m[2])):
        store.annotate(c, i % label_map.num_classes, label_map)
    store.propagate(crops.crops)
    store.save()
    return store


def items_from(store, crops):
    return [
        DatasetItem(
            crop_id=c.crop_id,
            session_id=c.session_id,
            label_id=store.labels[c.crop_id].label_id,
            provenance=store.labels[c.crop_id].provenance,
        )
        for c in crops.crops
        if c.crop_id in store.labels and store.labels[c.crop_id].label_id >= 0
    ]


# ======================================================================
class TestShapesAndContract:
    def test_returns_points_aux_and_label(self, crops, labelled, label_map, config):
        ds = ClusterCropDataset(
            items_from(labelled, crops), {crops.session_id: crops}, label_map, config
        )
        assert len(ds) > 0
        points, aux, label = ds[0]

        assert isinstance(points, torch.Tensor)
        assert points.shape == (3, config.sampling.num_points)
        assert points.dtype == torch.float32
        assert aux.shape == (len(AUX_FEATURE_NAMES),)
        assert aux.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label < label_map.num_classes

    def test_points_are_normalised_but_aux_is_metric(self, crops, labelled, label_map, config):
        """The split that makes the aux features worth having: the network sees a
        unit sphere, the head sees real metres (task_definition.md §4)."""
        ds = ClusterCropDataset(
            items_from(labelled, crops), {crops.session_id: crops}, label_map, config
        )
        points, aux, _ = ds[0]

        radius = points.norm(dim=0).max().item()
        assert radius == pytest.approx(1.0, abs=1e-4)
        assert points.mean(dim=1).abs().max().item() < 1e-5

        extents = aux[:3].numpy()
        assert extents.max() > 0.1, "aux extents must be in metres, not normalised"

    def test_aux_order_is_the_model_contract(self, crops, labelled, label_map, config):
        assert AUX_FEATURE_NAMES == (
            "extent_x_m",
            "extent_y_m",
            "extent_z_m",
            "log_n_points",
            "range_to_centroid_m",
            "height_above_ground_m",
        )
        ds = ClusterCropDataset(
            items_from(labelled, crops), {crops.session_id: crops}, label_map, config
        )
        assert ds.aux_feature_names == AUX_FEATURE_NAMES

    def test_aux_is_produced_even_when_use_aux_is_false(
        self, crops, labelled, label_map, config
    ):
        """So the aux ablation is a model choice, not a dataset regeneration."""
        assert config.use_aux is False
        ds = ClusterCropDataset(
            items_from(labelled, crops), {crops.session_id: crops}, label_map, config
        )
        _points, aux, _label = ds[0]
        assert aux.shape == (6,)
        assert np.isfinite(aux.numpy()).all()

    def test_works_with_a_torch_dataloader(self, crops, labelled, label_map, config):
        from torch.utils.data import DataLoader

        ds = ClusterCropDataset(
            items_from(labelled, crops), {crops.session_id: crops}, label_map, config
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        points, aux, labels = next(iter(loader))
        assert points.shape == (4, 3, config.sampling.num_points)
        assert aux.shape == (4, 6)
        assert labels.shape == (4,)

    def test_reports_class_and_provenance_counts(self, crops, labelled, label_map, config):
        ds = ClusterCropDataset(
            items_from(labelled, crops), {crops.session_id: crops}, label_map, config
        )
        counts = ds.class_counts()
        assert set(counts) == set(range(label_map.num_classes))
        assert sum(counts.values()) == len(ds)

        provenance = ds.provenance_counts()
        assert provenance.get("annotated", 0) >= 1
        assert provenance.get("propagated", 0) >= 1
        assert ds.describe()["n_items"] == len(ds)


class TestRejectedConstruction:
    def test_the_reject_id_is_not_a_training_target(self, crops, label_map, config):
        items = [
            DatasetItem(crops.crops[0].crop_id, crops.session_id, label_id=-1),
        ]
        with pytest.raises(DatasetError, match="is an output, not a training target"):
            ClusterCropDataset(items, {crops.session_id: crops}, label_map, config)

    def test_an_out_of_range_label_is_refused(self, crops, label_map, config):
        items = [DatasetItem(crops.crops[0].crop_id, crops.session_id, label_id=42)]
        with pytest.raises(DatasetError, match="not a trained class"):
            ClusterCropDataset(items, {crops.session_id: crops}, label_map, config)

    def test_a_missing_crop_index_is_refused(self, crops, label_map, config):
        items = [DatasetItem(crops.crops[0].crop_id, "nowhere", label_id=0)]
        with pytest.raises(DatasetError, match="no crop index loaded"):
            ClusterCropDataset(items, {crops.session_id: crops}, label_map, config)

    def test_a_crop_the_store_does_not_have_is_refused(self, crops, label_map, config):
        """Manifest and crop store out of sync — most likely a re-extraction."""
        items = [DatasetItem("session_01/999999/00", crops.session_id, label_id=0)]
        with pytest.raises(DatasetError, match="Re-generate the manifest"):
            ClusterCropDataset(items, {crops.session_id: crops}, label_map, config)


# ======================================================================
class TestAugmentationIsInMetres:
    """The parameters are metric, so they are applied to the metric cluster."""

    def test_jitter_sigma_is_metres_not_a_fraction(self):
        """0.01 must move points by ~1 cm regardless of how big the object is."""
        aug = AugmentationConfig(
            yaw_degrees=0.0,
            pitch_roll_degrees=0.0,
            scale_range=(1.0, 1.0),
            point_dropout_max=0.0,
            jitter_sigma_m=0.01,
            jitter_clip_m=0.03,
        )
        rng = np.random.default_rng(0)
        for size in (0.5, 20.0):
            points = rng.uniform(-size, size, (4000, 3))
            out = augment_cluster_points(points, aug, np.random.default_rng(1))
            displacement = np.linalg.norm(out - points, axis=1)
            # Same absolute displacement for a 0.5 m object and a 20 m one.
            assert displacement.mean() == pytest.approx(0.016, abs=0.004)

    def test_jitter_is_clipped(self):
        aug = AugmentationConfig(
            yaw_degrees=0.0,
            pitch_roll_degrees=0.0,
            scale_range=(1.0, 1.0),
            point_dropout_max=0.0,
            jitter_sigma_m=0.05,
            jitter_clip_m=0.02,
        )
        points = np.zeros((5000, 3))
        out = augment_cluster_points(points, aug, np.random.default_rng(2))
        assert np.abs(out).max() <= 0.02 + 1e-6

    def test_rotation_is_about_the_cluster_centroid(self):
        """Not the sensor origin: that would move the object elsewhere in the
        scene and fabricate a different range_to_centroid_m."""
        aug = AugmentationConfig(
            yaw_degrees=180.0,
            pitch_roll_degrees=0.0,
            scale_range=(1.0, 1.0),
            point_dropout_max=0.0,
            jitter_sigma_m=0.0,
        )
        g = np.random.default_rng(3)
        points = g.normal(0, 0.3, (500, 3)) + np.array([20.0, 5.0, 2.0])
        before = points.mean(axis=0)
        out = augment_cluster_points(points, aug, np.random.default_rng(4))
        assert np.allclose(out.mean(axis=0), before, atol=1e-4)
        assert np.linalg.norm(out.mean(axis=0)) == pytest.approx(
            np.linalg.norm(before), abs=1e-4
        )

    def test_yaw_rotation_actually_rotates(self):
        aug = AugmentationConfig(
            yaw_degrees=180.0,
            pitch_roll_degrees=0.0,
            scale_range=(1.0, 1.0),
            point_dropout_max=0.0,
            jitter_sigma_m=0.0,
        )
        # An L-shape, so a yaw is detectable rather than symmetric away.
        points = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        )
        out = augment_cluster_points(points, aug, np.random.default_rng(5))
        assert not np.allclose(out, points, atol=1e-3)
        # Rigid: pairwise distances are preserved.
        def pdist(a):
            return np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)

        assert np.allclose(pdist(out), pdist(points), atol=1e-6)

    def test_scale_is_limited_and_about_the_centroid(self):
        aug = AugmentationConfig(
            yaw_degrees=0.0,
            pitch_roll_degrees=0.0,
            scale_range=(0.9, 1.1),
            point_dropout_max=0.0,
            jitter_sigma_m=0.0,
        )
        g = np.random.default_rng(6)
        points = g.normal(0, 1.0, (1000, 3)) + np.array([3.0, 0.0, 1.0])
        before_extent = points.max(axis=0) - points.min(axis=0)
        out = augment_cluster_points(points, aug, np.random.default_rng(7))
        after_extent = out.max(axis=0) - out.min(axis=0)
        ratio = after_extent / before_extent
        assert np.allclose(ratio, ratio[0], atol=1e-6)  # isotropic
        assert 0.9 - 1e-6 <= ratio[0] <= 1.1 + 1e-6
        assert np.allclose(out.mean(axis=0), points.mean(axis=0), atol=1e-4)

    def test_dropout_removes_points_but_keeps_a_floor(self):
        aug = AugmentationConfig(
            yaw_degrees=0.0,
            pitch_roll_degrees=0.0,
            scale_range=(1.0, 1.0),
            point_dropout_max=0.25,
            jitter_sigma_m=0.0,
        )
        points = np.random.default_rng(8).normal(0, 1, (400, 3))
        sizes = [
            augment_cluster_points(points, aug, np.random.default_rng(s)).shape[0]
            for s in range(20)
        ]
        assert min(sizes) >= int(400 * 0.75) - 1
        assert max(sizes) <= 400
        assert len(set(sizes)) > 1, "dropout fraction should vary between draws"

    def test_dropout_never_guts_a_tiny_cluster(self):
        aug = AugmentationConfig(
            yaw_degrees=0.0,
            pitch_roll_degrees=0.0,
            scale_range=(1.0, 1.0),
            point_dropout_max=0.99,
            jitter_sigma_m=0.0,
        )
        points = np.random.default_rng(9).normal(0, 1, (12, 3))
        out = augment_cluster_points(points, aug, np.random.default_rng(10))
        assert out.shape[0] == 12  # below min_points: untouched

    def test_identity_config_changes_nothing(self):
        aug = AugmentationConfig(
            yaw_degrees=0.0,
            pitch_roll_degrees=0.0,
            translation_m=0.0,
            scale_range=(1.0, 1.0),
            point_dropout_max=0.0,
            jitter_sigma_m=0.0,
            occlusion_dropout_max=0.0,
        )
        assert aug.is_identity
        points = np.random.default_rng(11).normal(0, 1, (50, 3))
        out = augment_cluster_points(points, aug, np.random.default_rng(12))
        assert np.allclose(out, points)

    def test_empty_cluster(self):
        out = augment_cluster_points(
            np.zeros((0, 3)), AugmentationConfig(), np.random.default_rng(0)
        )
        assert out.shape == (0, 3)

    def test_rotation_matrix_is_orthonormal(self):
        r = rotation_matrix(0.7, 0.1, -0.2)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)

    def test_the_configured_augmentation_is_the_one_from_the_file(self, config):
        """The real preprocess.yaml values, so a change there is visible here."""
        aug = config.augmentation
        assert aug.yaw_degrees == 180.0  # clusters arrive at arbitrary heading
        assert aug.pitch_roll_degrees == 10.0  # the drone is roughly level
        assert aug.scale_range == (0.9, 1.1)  # object size is meaningful
        assert aug.point_dropout_max == 0.25
        assert aug.jitter_sigma_m == 0.01  # ~Mid-70 ranging noise
        assert not aug.is_identity


class TestAugmentationThroughTheDataset:
    def test_augmentation_changes_the_output(self, crops, labelled, label_map, config):
        items = items_from(labelled, crops)
        plain = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=False
        )
        augmented = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=True
        )
        assert not torch.allclose(plain[0][0], augmented[0][0], atol=1e-3)
        # The label is untouched by augmentation.
        assert plain[0][2] == augmented[0][2]

    def test_augmented_points_are_still_normalised(
        self, crops, labelled, label_map, config
    ):
        """Augmentation goes before normalisation, so the contract still holds."""
        ds = ClusterCropDataset(
            items_from(labelled, crops),
            {crops.session_id: crops},
            label_map,
            config,
            augment=True,
        )
        for i in range(min(5, len(ds))):
            points, _aux, _label = ds[i]
            assert points.shape == (3, config.sampling.num_points)
            assert points.norm(dim=0).max().item() == pytest.approx(1.0, abs=1e-4)
            assert torch.isfinite(points).all()

    def test_dropout_reaches_the_aux_point_count(self, crops, labelled, label_map, config):
        """log_n_points must reflect the augmented cluster, not the stored one —
        otherwise the head is told a point count the network never saw."""
        items = items_from(labelled, crops)
        plain = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=False
        )
        augmented = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=True
        )
        plain_counts = np.array([plain[i][1][3].item() for i in range(len(plain))])
        aug_counts = np.array([augmented[i][1][3].item() for i in range(len(augmented))])
        assert (aug_counts <= plain_counts + 1e-6).all()
        assert (aug_counts < plain_counts - 1e-6).any()


# ======================================================================
class TestReproducibility:
    def test_the_same_item_is_identical_across_instances(
        self, crops, labelled, label_map, config
    ):
        items = items_from(labelled, crops)
        a = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=True, seed=5
        )
        b = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=True, seed=5
        )
        for i in range(len(items)):
            assert torch.equal(a[i][0], b[i][0])
            assert torch.equal(a[i][1], b[i][1])

    def test_an_item_does_not_depend_on_access_order(
        self, crops, labelled, label_map, config
    ):
        """Per-item RNG, not one stream advanced by iteration — so batch size,
        shuffling and worker count cannot change what item 2 is."""
        items = items_from(labelled, crops)
        ds = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=True, seed=5
        )
        straight = ds[2][0].clone()
        for i in (0, 1, 3, 2, 1, 0):
            _ = ds[i]
        assert torch.equal(ds[2][0], straight)

    def test_a_different_seed_augments_differently(
        self, crops, labelled, label_map, config
    ):
        items = items_from(labelled, crops)
        a = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=True, seed=1
        )
        b = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=True, seed=2
        )
        assert not torch.allclose(a[0][0], b[0][0], atol=1e-4)

    def test_set_epoch_changes_the_draw(self, crops, labelled, label_map, config):
        items = items_from(labelled, crops)
        ds = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=True, seed=0
        )
        first = ds[0][0].clone()
        ds.set_epoch(1)
        assert not torch.allclose(ds[0][0], first, atol=1e-4)
        ds.set_epoch(0)
        assert torch.equal(ds[0][0], first)

    def test_an_eval_set_does_not_drift_between_epochs(
        self, crops, labelled, label_map, config
    ):
        """Epoch is pinned to 0 without augmentation, so the FPS start index —
        and therefore the validation set — cannot change."""
        items = items_from(labelled, crops)
        ds = ClusterCropDataset(
            items, {crops.session_id: crops}, label_map, config, augment=False, seed=0
        )
        first = ds[0][0].clone()
        for epoch in (1, 2, 37):
            ds.set_epoch(epoch)
            assert torch.equal(ds[0][0], first)


class TestMultipleViews:
    """Two independently augmented views of one crop, for a consistency arm.

    The reason this has to be the dataset's job and not the trainer's: the
    dataset returns unit-sphere points, and while yaw commutes with centring and
    scaling, the jitter and point dropout are specified in **metres** and applied
    to the metric cluster before normalisation. A second view produced by
    re-rotating the dataset's output would carry the same dropout pattern and the
    same jitter, so it would be a strictly weaker perturbation than it appears.
    """

    def _two_view(self, crops, labelled, label_map, config, **kwargs):
        return ClusterCropDataset(
            items_from(labelled, crops),
            {crops.session_id: crops},
            label_map,
            config,
            augment=True,
            n_views=2,
            **kwargs,
        )

    def test_shapes_gain_a_view_axis(self, crops, labelled, label_map, config):
        ds = self._two_view(crops, labelled, label_map, config)
        points, aux, label = ds[0]
        assert points.shape == (2, 3, config.sampling.num_points)
        assert aux.shape == (2, 6)
        assert isinstance(label, int)

    def test_one_view_keeps_the_original_contract(self, crops, labelled, label_map, config):
        """n_views=1 is the default and must return exactly what it always did."""
        ds = ClusterCropDataset(
            items_from(labelled, crops), {crops.session_id: crops}, label_map, config
        )
        assert ds.n_views == 1
        points, aux, _label = ds[0]
        assert points.shape == (3, config.sampling.num_points)
        assert aux.shape == (6,)

    def test_the_two_views_differ(self, crops, labelled, label_map, config):
        """The bug this fixes: before keying on `view`, these were identical."""
        ds = self._two_view(crops, labelled, label_map, config)
        points, _aux, _label = ds[0]
        assert not torch.allclose(points[0], points[1], atol=1e-3)

    def test_the_views_differ_in_more_than_rotation(
        self, crops, labelled, label_map, config
    ):
        """Specifically: the dropout draw differs, so the point counts differ.

        A trainer-side second view could not produce this — it only sees the
        already-resampled, already-normalised cloud. log_n_points is aux[3].
        """
        ds = self._two_view(crops, labelled, label_map, config)
        differing = 0
        for i in range(len(ds)):
            _points, aux, _label = ds[i]
            if abs(aux[0][3].item() - aux[1][3].item()) > 1e-6:
                differing += 1
        assert differing > 0, "no item had a different dropout draw between views"

    def test_the_label_is_shared(self, crops, labelled, label_map, config):
        """The views are the same object; that is what consistency asserts."""
        ds = self._two_view(crops, labelled, label_map, config)
        plain = ClusterCropDataset(
            items_from(labelled, crops), {crops.session_id: crops}, label_map, config
        )
        for i in range(len(ds)):
            assert ds[i][2] == plain[i][2]

    def test_views_are_reproducible(self, crops, labelled, label_map, config):
        a = self._two_view(crops, labelled, label_map, config, seed=4)
        b = self._two_view(crops, labelled, label_map, config, seed=4)
        for i in range(len(a)):
            assert torch.equal(a[i][0], b[i][0])

    def test_get_view_is_stable_and_matches_the_stack(
        self, crops, labelled, label_map, config
    ):
        ds = self._two_view(crops, labelled, label_map, config)
        stacked, _aux, _label = ds[0]
        for v in range(2):
            points, _a, _l = ds.get_view(0, v)
            assert torch.equal(points, stacked[v])
            # And calling it again gives the same cloud, not a fresh draw.
            assert torch.equal(ds.get_view(0, v)[0], points)

    def test_more_than_two_views(self, crops, labelled, label_map, config):
        ds = self._two_view(crops, labelled, label_map, config)
        ds4 = ClusterCropDataset(
            items_from(labelled, crops),
            {crops.session_id: crops},
            label_map,
            config,
            augment=True,
            n_views=4,
        )
        points, aux, _label = ds4[0]
        assert points.shape == (4, 3, config.sampling.num_points)
        assert aux.shape == (4, 6)
        # All four are mutually distinct.
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(points[i], points[j], atol=1e-3)
        # And the first two agree with the 2-view dataset: the view key does not
        # depend on how many views were requested.
        assert torch.equal(points[:2], ds[0][0])

    def test_epoch_still_changes_the_draw(self, crops, labelled, label_map, config):
        ds = self._two_view(crops, labelled, label_map, config)
        first = ds[0][0].clone()
        ds.set_epoch(1)
        assert not torch.allclose(ds[0][0], first, atol=1e-4)

    def test_collates_to_batch_by_view(self, crops, labelled, label_map, config):
        """(B, V, 3, N) — what a consistency loss wants, from default collate."""
        from torch.utils.data import DataLoader

        ds = self._two_view(crops, labelled, label_map, config)
        batch_size = min(3, len(ds))
        points, aux, labels = next(
            iter(DataLoader(ds, batch_size=batch_size, shuffle=False))
        )
        assert points.shape == (batch_size, 2, 3, config.sampling.num_points)
        assert aux.shape == (batch_size, 2, 6)
        assert labels.shape == (batch_size,)

    def test_both_views_pass_through_the_model(self, crops, labelled, label_map, config):
        """Flattening (B, V, 3, N) to (B*V, 3, N) is one forward pass."""
        import yaml
        from torch.utils.data import DataLoader

        from pcml.models.model_factory import build_model

        ds = self._two_view(crops, labelled, label_map, config)
        batch_size = min(3, len(ds))
        points, _aux, _labels = next(
            iter(DataLoader(ds, batch_size=batch_size, shuffle=False))
        )
        b, v = points.shape[:2]
        model_config = yaml.safe_load(
            (REPO_ROOT / "configs" / "models" / "pointnet_deploy.yaml").read_text(
                encoding="utf-8"
            )
        )["model"]
        model = build_model(model_config, num_classes=label_map.num_classes)
        model.eval()
        with torch.no_grad():
            logits = model(points.reshape(b * v, *points.shape[2:]))
        assert logits.shape == (b * v, label_map.num_classes)
        # The two views of one crop give different logits — which is what a
        # consistency loss has something to work on.
        per_view = logits.reshape(b, v, -1)
        assert not torch.allclose(per_view[:, 0], per_view[:, 1], atol=1e-4)


class TestMultipleViewsRefusals:
    def test_views_without_augmentation_are_refused(
        self, crops, labelled, label_map, config
    ):
        """They would differ only by the FPS draw — a trivially satisfied loss."""
        with pytest.raises(DatasetError, match="differ only by the FPS draw"):
            ClusterCropDataset(
                items_from(labelled, crops),
                {crops.session_id: crops},
                label_map,
                config,
                augment=False,
                n_views=2,
            )

    def test_views_with_identity_augmentation_are_refused(
        self, crops, labelled, label_map, config
    ):
        off = replace(
            config,
            augmentation=AugmentationConfig(
                yaw_degrees=0.0,
                pitch_roll_degrees=0.0,
                translation_m=0.0,
                scale_range=(1.0, 1.0),
                point_dropout_max=0.0,
                jitter_sigma_m=0.0,
                occlusion_dropout_max=0.0,
            ),
        )
        with pytest.raises(DatasetError, match="views would be identical"):
            ClusterCropDataset(
                items_from(labelled, crops),
                {crops.session_id: crops},
                label_map,
                off,
                augment=True,
                n_views=2,
            )

    def test_zero_views_is_refused(self, crops, labelled, label_map, config):
        with pytest.raises(DatasetError, match="n_views must be >= 1"):
            ClusterCropDataset(
                items_from(labelled, crops),
                {crops.session_id: crops},
                label_map,
                config,
                n_views=0,
            )


class TestFromManifest:
    def test_builds_from_written_manifests(self, tmp_path, crops, labelled, label_map, config):
        written = write_manifests(
            [crops],
            {crops.session_id: labelled},
            label_map,
            out_dir=tmp_path / "manifests",
            fractions=(1.0, 0.0, 0.0),
            seed=0,
        )
        ds = ClusterCropDataset.from_manifest(
            written["train"], label_map=label_map, config=config
        )
        assert len(ds) > 0
        assert ds.session_ids() == (crops.session_id,)
        # train => augmentation on by default; getting that backwards silently
        # would be worse than a slightly magic default.
        assert ds.augment is True
        points, aux, label = ds[0]
        assert points.shape == (3, config.sampling.num_points)
        assert aux.shape == (6,)
        assert 0 <= label < label_map.num_classes

    def test_n_views_passes_through(self, tmp_path, crops, labelled, label_map, config):
        written = write_manifests(
            [crops],
            {crops.session_id: labelled},
            label_map,
            out_dir=tmp_path / "manifests",
            fractions=(1.0, 0.0, 0.0),
            seed=0,
        )
        ds = ClusterCropDataset.from_manifest(
            written["train"], label_map=label_map, config=config, n_views=2
        )
        assert ds.n_views == 2
        points, aux, _label = ds[0]
        assert points.shape == (2, 3, config.sampling.num_points)
        assert aux.shape == (2, 6)

    def test_views_on_an_eval_manifest_are_refused(
        self, tmp_path, crops, labelled, label_map, config
    ):
        """val/test do not augment, so a consistency objective there measures
        nothing — the constraint is intended, not incidental."""
        written = write_manifests(
            [crops],
            {crops.session_id: labelled},
            label_map,
            out_dir=tmp_path / "manifests",
            fractions=(0.0, 1.0, 0.0),
            seed=0,
        )
        with pytest.raises(DatasetError, match="differ only by the FPS draw"):
            ClusterCropDataset.from_manifest(
                written["val"], label_map=label_map, config=config, n_views=2
            )

    def test_val_and_test_manifests_do_not_augment(
        self, tmp_path, crops, labelled, label_map, config
    ):
        written = write_manifests(
            [crops],
            {crops.session_id: labelled},
            label_map,
            out_dir=tmp_path / "manifests",
            fractions=(0.0, 1.0, 0.0),
            seed=0,
        )
        ds = ClusterCropDataset.from_manifest(
            written["val"], label_map=label_map, config=config
        )
        assert ds.augment is False

    def test_a_changed_label_map_is_refused(
        self, tmp_path, crops, labelled, label_map, config
    ):
        written = write_manifests(
            [crops],
            {crops.session_id: labelled},
            label_map,
            out_dir=tmp_path / "manifests",
            fractions=(1.0, 0.0, 0.0),
            seed=0,
        )
        stale = replace(label_map, sha256="0" * 64)
        with pytest.raises(DatasetError, match="class ids may mean different things"):
            ClusterCropDataset.from_manifest(
                written["train"], label_map=stale, config=config
            )

    def test_a_changed_preprocess_config_is_refused(
        self, tmp_path, crops, labelled, label_map, config
    ):
        """The crops it names were produced with different DBSCAN parameters."""
        written = write_manifests(
            [crops],
            {crops.session_id: labelled},
            label_map,
            out_dir=tmp_path / "manifests",
            fractions=(1.0, 0.0, 0.0),
            seed=0,
        )
        stale = replace(config, sha256="1" * 64)
        with pytest.raises(DatasetError, match="different clustering parameters"):
            ClusterCropDataset.from_manifest(
                written["train"], label_map=label_map, config=stale
            )
