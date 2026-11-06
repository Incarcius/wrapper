"""Model and adapter contract tests. No training, no data — shapes and invariants only."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pcml.data.clustering import AUX_FEATURE_NAMES  # noqa: E402
from pcml.deploy.adapter import to_model_input  # noqa: E402
from pcml.models import PointNetClassifier, build_model, count_parameters  # noqa: E402

K = 5  # placeholder class count; the real K comes from label_map.yaml


def model(**kwargs) -> PointNetClassifier:
    kwargs.setdefault("num_classes", K)
    m = PointNetClassifier(**kwargs)
    return m.eval()


class TestForwardShapes:
    @pytest.mark.parametrize("m_clusters", [1, 3, 16])
    def test_one_pass_classifies_a_whole_frame(self, m_clusters):
        """B is the cluster count: M objects, one forward pass."""
        net = model()
        x = torch.randn(m_clusters, 3, 1024)
        with torch.inference_mode():
            logits = net(x)
        assert logits.shape == (m_clusters, K)

    @pytest.mark.parametrize("n_points", [256, 1024, 2048])
    def test_point_count_is_not_baked_in(self, n_points):
        """A shared MLP + max-pool is point-count agnostic; only export fixes N."""
        net = model()
        with torch.inference_mode():
            assert net(torch.randn(2, 3, n_points)).shape == (2, K)

    def test_embedding_width_follows_config(self):
        net = model(widths=(32, 32, 64, 128))
        with torch.inference_mode():
            assert net.encode(torch.randn(2, 3, 512)).shape == (2, 128)


class TestAuxFeatures:
    def test_aux_is_accepted_when_configured(self):
        net = model(aux_dim=len(AUX_FEATURE_NAMES))
        x = torch.randn(4, 3, 1024)
        aux = torch.randn(4, len(AUX_FEATURE_NAMES))
        with torch.inference_mode():
            assert net(x, aux).shape == (4, K)

    def test_missing_aux_fails_loudly(self):
        net = model(aux_dim=6)
        with pytest.raises(ValueError, match="aux=None"):
            net(torch.randn(2, 3, 128))

    def test_unexpected_aux_fails_loudly(self):
        net = model(aux_dim=0)
        with pytest.raises(ValueError, match="aux_dim=0"):
            net(torch.randn(2, 3, 128), torch.randn(2, 6))

    def test_wrong_aux_width_fails_loudly(self):
        net = model(aux_dim=6)
        with pytest.raises(ValueError, match="expected aux shape"):
            net(torch.randn(2, 3, 128), torch.randn(2, 4))


class TestInvariants:
    def test_permutation_invariance(self):
        """Max-pool aggregation must make the output independent of point order."""
        net = model()
        x = torch.randn(2, 3, 512)
        perm = torch.randperm(512)
        with torch.inference_mode():
            a = net(x)
            b = net(x[:, :, perm])
        assert torch.allclose(a, b, atol=1e-5)

    def test_duplicate_tolerance(self):
        """Padding a cluster with duplicates must not change the prediction.

        This is why max-pool matters for this task: undersized clusters are
        padded by resampling with replacement, so duplicates are the norm.
        """
        net = model()
        base = torch.randn(1, 3, 256)
        padded = torch.cat([base, base[:, :, :128]], dim=2)  # 384 pts, 128 duplicated
        with torch.inference_mode():
            a = net(base)
            b = net(padded)
        assert torch.allclose(a, b, atol=1e-5)

    def test_not_rotation_invariant(self):
        """Documents a real limitation rather than assuming it away.

        Measured on the **embedding**, not the logits: at random initialisation
        the head compresses the two logit vectors to within ~1e-3 of each other,
        which would make a logit-level assertion a test of the init scheme rather
        than of the architecture.

        If this ever starts passing trivially, the model gained rotation
        invariance and the yaw-augmentation decision in task_definition.md §9.5
        should be revisited.
        """
        net = model()
        x = torch.randn(1, 3, 512)
        theta = torch.tensor(1.0)
        c, s = torch.cos(theta), torch.sin(theta)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        with torch.inference_mode():
            a = net.encode(x)
            b = net.encode(torch.matmul(rot, x))  # (3,3) @ (B,3,N) -> (B,3,N)
        relative_change = ((a - b).norm() / a.norm()).item()
        assert relative_change > 1e-3, (
            f"embedding changed by only {relative_change:.2e} under a 57-degree "
            "yaw rotation, which would imply rotation invariance"
        )


class TestErrorHandling:
    def test_rejects_bnc_layout(self):
        net = model()
        with pytest.raises(ValueError, match="channels at dim 1"):
            net(torch.randn(2, 1024, 3))

    def test_rejects_wrong_rank(self):
        net = model()
        with pytest.raises(ValueError, match=r"expected \(B, C, N\)"):
            net(torch.randn(3, 1024))

    def test_rejects_degenerate_class_count(self):
        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            PointNetClassifier(num_classes=1)


class TestTNetAblation:
    def test_input_tnet_runs_and_costs_parameters(self):
        plain = model()
        with_tnet = model(use_input_tnet=True)
        with torch.inference_mode():
            assert with_tnet(torch.randn(2, 3, 512)).shape == (2, K)
        assert count_parameters(with_tnet) > count_parameters(plain)

    def test_tnet_initialises_to_identity(self):
        """A random init would scramble input geometry for the first steps."""
        net = model(use_input_tnet=True)
        t = net.input_tnet(torch.randn(4, 3, 256))
        assert torch.allclose(t, torch.eye(3).expand(4, 3, 3), atol=1e-5)

    def test_feature_tnet_requires_width_64(self):
        with pytest.raises(ValueError, match="widths\\[0\\] == 64"):
            PointNetClassifier(num_classes=K, widths=(32, 64, 128), use_feature_tnet=True)


class TestCapacity:
    def test_deployment_default_is_small(self):
        """Budget check, not a benchmark: the deploy model stays under ~1 M params."""
        net = model()
        n = count_parameters(net)
        assert 100_000 < n < 1_000_000, f"unexpected parameter count {n}"

    def test_student_is_much_smaller_than_teacher(self):
        teacher = model(widths=(64, 64, 128, 256, 1024), head_widths=(512, 256))
        student = model(widths=(32, 32, 64, 128), head_widths=(128, 64))
        assert count_parameters(student) < count_parameters(teacher) / 5


class TestFactory:
    def test_builds_from_config(self):
        net = build_model({"name": "pointnet", "widths": [32, 32, 64]}, num_classes=K)
        with torch.inference_mode():
            assert net.eval()(torch.randn(2, 3, 256)).shape == (2, K)

    def test_unknown_architecture_names_the_alternatives(self):
        with pytest.raises(ValueError, match="available:"):
            build_model({"name": "transformer3000"}, num_classes=K)

    def test_missing_name(self):
        with pytest.raises(KeyError, match="'name'"):
            build_model({"widths": [32]}, num_classes=K)

    def test_unknown_kwarg_is_not_swallowed(self):
        """A silently ignored config key makes an experiment untraceable."""
        with pytest.raises(TypeError):
            build_model({"name": "pointnet", "nonexistent_knob": 7}, num_classes=K)


class TestAdapter:
    def test_bcn_layout_and_contiguity(self):
        """The wrapper's inline version produced a non-contiguous view (I5)."""
        pts = np.random.rand(1024, 3).astype(np.float32)
        t = to_model_input(pts, layout="bcn")
        assert t.shape == (1, 3, 1024)
        assert t.is_contiguous()

    def test_bnc_layout(self):
        t = to_model_input(np.random.rand(1024, 3).astype(np.float32), layout="bnc")
        assert t.shape == (1, 1024, 3)
        assert t.is_contiguous()

    def test_batched_input_is_preserved(self):
        t = to_model_input(np.random.rand(7, 512, 3).astype(np.float32), layout="bcn")
        assert t.shape == (7, 3, 512)

    def test_values_are_not_scrambled_by_the_transpose(self):
        pts = np.arange(12, dtype=np.float32).reshape(4, 3)
        t = to_model_input(pts, layout="bcn")
        assert t[0, :, 0].tolist() == [0.0, 1.0, 2.0]
        assert t[0, 0, :].tolist() == [0.0, 3.0, 6.0, 9.0]

    def test_accepts_torch_input(self):
        t = to_model_input(torch.rand(256, 3), layout="bcn")
        assert t.shape == (1, 3, 256)

    def test_rejects_unknown_layout(self):
        with pytest.raises(ValueError, match="unknown layout"):
            to_model_input(np.random.rand(10, 3).astype(np.float32), layout="nchw")

    def test_rejects_wrong_trailing_dim(self):
        with pytest.raises(ValueError, match="3 coordinates"):
            to_model_input(np.random.rand(3, 1024).astype(np.float32))

    def test_end_to_end_adapter_into_model(self):
        """The real integration contract: numpy cluster batch -> logits."""
        net = model()
        batch = np.random.rand(4, 1024, 3).astype(np.float32)
        x = to_model_input(batch, layout="bcn")
        with torch.inference_mode():
            assert net(x).shape == (4, K)


class TestPredict:
    def test_returns_ids_and_confidences(self):
        net = model()
        cls, conf = net.predict(torch.randn(6, 3, 512))
        assert cls.shape == (6,)
        assert conf.shape == (6,)
        assert bool(((conf >= 0) & (conf <= 1)).all())
        assert bool(((cls >= 0) & (cls < K)).all())

    def test_eval_mode_is_deterministic(self):
        """Dropout must be inactive, or per-frame predictions would flicker."""
        net = model(dropout=0.5)
        x = torch.randn(3, 3, 512)
        with torch.inference_mode():
            assert torch.allclose(net(x), net(x))
