"""PointNet classification backbone, sized for per-cluster inference on an Orin Nano.

Why this architecture is the deployment candidate (see ``docs/task_definition.md``
§9 for the full reasoning):

* **Zero neighbourhood searches.** The per-frame cost is ``M`` clusters x one
  shared-MLP pass. DGCNN rebuilds a kNN graph every layer (~5 NxN distance
  matrices at N=1024); at M=8 clusters that is the entire 100 ms frame budget.
* **Every op is TensorRT-native.** Conv1d, BatchNorm1d, ReLU, ReduceMax, Gemm,
  Dropout. No ball query, no FPS-in-the-graph, no gather/scatter, no custom
  plugins. PointNet++ needs custom CUDA extensions rewritten in pure torch
  before it will export at all.
* **Max-pool is duplicate-invariant**, which matters because padding a small
  cluster to 1024 points produces duplicates by construction.

The classes here are width-parameterised so that "teacher" and "student" are
config, not separate code. ``widths`` is the only knob that separates a 3.5 M
parameter model from a 0.15 M one.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PointNetClassifier", "TNet", "count_parameters"]


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters, for the comparison matrix."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _shared_mlp(in_ch: int, widths: Sequence[int]) -> nn.Sequential:
    """Point-wise MLP as Conv1d over the point axis: the PointNet primitive.

    Conv1d with kernel_size=1 applies the same linear map to every point
    independently, which is what makes the backbone permutation-equivariant
    (and the subsequent max-pool permutation-invariant).
    """
    layers: list[nn.Module] = []
    prev = in_ch
    for w in widths:
        layers.append(nn.Conv1d(prev, w, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm1d(w))
        layers.append(nn.ReLU(inplace=True))
        prev = w
    return nn.Sequential(*layers)


class TNet(nn.Module):
    """Spatial transformer that regresses a ``k x k`` alignment matrix.

    Off by default. It is included for the ablation only, because the honest
    summary is that T-Net does **not** give rotation invariance — it gives a
    learned, input-dependent affine warp that partially compensates for
    orientation seen during training. For this task the drone can supply a far
    better prior (yaw from the flight controller, or PCA on the cluster), and
    T-Net costs a second backbone plus a kxk matmul per forward pass.
    """

    def __init__(self, k: int = 3, widths: Sequence[int] = (64, 128, 512)) -> None:
        super().__init__()
        self.k = k
        self.mlp = _shared_mlp(k, widths)
        self.fc = nn.Sequential(
            nn.Linear(widths[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, k * k),
        )
        # Initialise to predict the identity: a random init here would scramble
        # the input geometry for the first few hundred steps.
        nn.init.zeros_(self.fc[-1].weight)
        with torch.no_grad():
            self.fc[-1].bias.copy_(torch.eye(k).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, k, N) -> (B, k, k)"""
        feat = self.mlp(x).max(dim=2).values
        return self.fc(feat).view(-1, self.k, self.k)


class PointNetClassifier(nn.Module):
    """Per-cluster point-cloud classifier.

    Args:
        num_classes: read from ``label_map.yaml``. Never hard-coded by a caller.
        in_channels: point feature channels. 3 (xyz) for this project; the
            wrapper discards intensity at ingestion.
        widths: shared-MLP widths. The last entry is the global embedding size.
            Deployment default ``(64, 64, 64, 128, 512)``; the reference PointNet
            uses 1024 at the end, which doubles the head cost for a gain that has
            to be demonstrated on this data, not assumed.
        head_widths: fully-connected head widths.
        dropout: applied in the head only.
        aux_dim: size of the auxiliary scale-feature vector concatenated at the
            head (0 disables it). See ``AUX_FEATURE_NAMES`` in
            ``pcml.data.clustering``; 6 when enabled.
        use_input_tnet: enable the 3x3 input T-Net (ablation).
        use_feature_tnet: enable the 64x64 feature T-Net (ablation). Expensive:
            a 64x64 regression head plus a (B,64,64)x(B,64,N) matmul.

    Shapes:
        forward: ``(B, in_channels, N)`` -> ``(B, num_classes)``.
        ``B`` is the number of clusters in the frame, not a training batch:
        one forward pass classifies the whole frame.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        widths: Sequence[int] = (64, 64, 64, 128, 512),
        head_widths: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        aux_dim: int = 0,
        use_input_tnet: bool = False,
        use_feature_tnet: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if len(widths) < 2:
            raise ValueError("widths needs at least two entries")
        if use_feature_tnet and widths[0] != 64:
            raise ValueError(
                "use_feature_tnet assumes widths[0] == 64 (the reference PointNet "
                f"layout); got widths[0]={widths[0]}"
            )

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.aux_dim = aux_dim
        self.embed_dim = int(widths[-1])

        self.input_tnet = TNet(in_channels) if use_input_tnet else None
        # Split the MLP so the feature T-Net can be inserted after the first block.
        self.mlp_early = _shared_mlp(in_channels, widths[:1])
        self.feature_tnet = TNet(int(widths[0])) if use_feature_tnet else None
        self.mlp_late = _shared_mlp(int(widths[0]), widths[1:])

        head: list[nn.Module] = []
        prev = self.embed_dim + aux_dim
        for w in head_widths:
            head.append(nn.Linear(prev, w))
            head.append(nn.BatchNorm1d(w))
            head.append(nn.ReLU(inplace=True))
            if dropout > 0:
                head.append(nn.Dropout(p=dropout))
            prev = w
        head.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*head)

    # ------------------------------------------------------------------
    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """Global shape embedding. ``(B, C, N)`` -> ``(B, embed_dim)``.

        Exposed separately because feature distillation matches *this* tensor
        between teacher and student; the logits are matched by the KD loss.
        """
        if points.dim() != 3:
            raise ValueError(f"expected (B, C, N), got shape {tuple(points.shape)}")
        if points.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels at dim 1, got {points.shape[1]}. "
                "A (B, N, C) tensor was probably passed; use the adapter to convert."
            )

        x = points
        if self.input_tnet is not None:
            t = self.input_tnet(x)
            x = torch.bmm(t, x)

        x = self.mlp_early(x)
        if self.feature_tnet is not None:
            t = self.feature_tnet(x)
            x = torch.bmm(t, x)

        x = self.mlp_late(x)
        # Symmetric aggregation: the step that makes the output invariant to
        # point order, and tolerant of the duplicate points that padding creates.
        return x.max(dim=2).values

    def forward(
        self, points: torch.Tensor, aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``(B, C, N)`` [+ ``(B, aux_dim)``] -> ``(B, num_classes)`` logits."""
        feat = self.encode(points)

        if self.aux_dim > 0:
            if aux is None:
                raise ValueError(
                    f"model was built with aux_dim={self.aux_dim} but forward() got "
                    "aux=None. Pass the auxiliary features, or rebuild with aux_dim=0."
                )
            if aux.shape != (feat.shape[0], self.aux_dim):
                raise ValueError(
                    f"expected aux shape {(feat.shape[0], self.aux_dim)}, "
                    f"got {tuple(aux.shape)}"
                )
            feat = torch.cat([feat, aux], dim=1)
        elif aux is not None:
            raise ValueError("model was built with aux_dim=0 but aux was provided")

        return self.head(feat)

    @torch.no_grad()
    def predict(
        self, points: torch.Tensor, aux: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference convenience: returns ``(class_ids, confidences)``.

        Confidence is the max softmax probability. It is **not** evidence of
        correctness; calibration and a reject threshold are applied downstream
        (``docs/task_definition.md`` §7).
        """
        logits = self.forward(points, aux)
        probs = F.softmax(logits, dim=1)
        conf, cls = probs.max(dim=1)
        return cls, conf
