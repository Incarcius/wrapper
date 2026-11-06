"""The wrapper <-> model boundary.

This is the *entire* coupling between ``livox_wrapper`` and whatever network is
chosen. Keeping it to one function is what allows PointNet, PointNet++ and DGCNN
to be A/B tested without touching the pipeline — and it is the fix for the
wrapper's finding **I5**, where the layout was hardcoded inline in the main loop
and produced a non-contiguous, CPU-resident tensor.

Two defects in the wrapper's four-line version:

1. ``np.expand_dims(processed.T, 0)`` is a **view with permuted strides**, not a
   contiguous tensor. Many CUDA kernels either require contiguity or force a
   hidden copy on every frame.
2. ``(B, 3, N)`` is an assumption, not a standard: it is right for the PyTorch
   reference PointNet/PointNet++/DGCNN implementations, and wrong for Open3D-ML,
   torch-points3d and most torch_geometric models, which expect ``(B, N, 3)``.
   That belongs in config, not in Python.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import torch

__all__ = ["Layout", "to_model_input", "LAYOUTS"]

Layout = Literal["bcn", "bnc"]

#: ``bcn`` = (B, C, N), for Conv1d-based reference implementations (PointNet,
#: PointNet++, DGCNN). ``bnc`` = (B, N, C), for Open3D-ML / torch-points3d / PyG.
LAYOUTS: tuple[str, ...] = ("bcn", "bnc")


def to_model_input(
    points: Union[np.ndarray, torch.Tensor],
    layout: Layout = "bcn",
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Batched, contiguous, on-device tensor in the layout the model wants.

    Args:
        points: ``(N, 3)`` for a single cluster, or ``(B, N, 3)`` for a frame's
            worth of clusters. A ``(N, 3)`` input gains a leading batch axis of 1.
        layout: ``"bcn"`` -> ``(B, 3, N)``; ``"bnc"`` -> ``(B, N, 3)``.
        device: target device. ``non_blocking=True`` is used, which is only
            actually asynchronous from pinned host memory; it is harmless
            otherwise.
        dtype: output dtype. float16 here is **not** a substitute for a real
            half-precision engine — it only casts the input.

    Returns:
        A contiguous tensor on ``device``.

    Raises:
        ValueError: on an unknown layout, a wrong rank, or a trailing dimension
            that is not 3. Failing loudly here is the point: a silently
            transposed cloud produces a model that appears to work and is
            quietly wrong.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")

    if isinstance(points, np.ndarray):
        # ascontiguousarray first: torch.as_tensor on a strided view would carry
        # the stride permutation into the tensor.
        tensor = torch.as_tensor(np.ascontiguousarray(points))
    elif isinstance(points, torch.Tensor):
        tensor = points
    else:
        raise ValueError(f"expected numpy array or torch tensor, got {type(points)!r}")

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 3:
        raise ValueError(
            f"expected (N, 3) or (B, N, 3), got shape {tuple(tensor.shape)}"
        )

    if tensor.shape[-1] != 3:
        raise ValueError(
            f"expected 3 coordinates in the last dimension, got shape "
            f"{tuple(tensor.shape)}. Input may already be in (B, C, N) layout."
        )

    tensor = tensor.to(dtype=dtype)
    if layout == "bcn":
        tensor = tensor.transpose(1, 2)

    return tensor.contiguous().to(device, non_blocking=True)
