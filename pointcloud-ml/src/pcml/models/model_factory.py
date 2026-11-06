"""Config -> model. One place where architecture names resolve to classes.

Architectures are added to :data:`REGISTRY` as they are implemented. PointNet is
first because it is the deployment candidate; PointNet++ (SSG) and DGCNN arrive
with the Stage 4 comparison study, where they are trained under the identical
protocol so the comparison is fair.

A model is always built from a dict, never from positional arguments scattered
through a training script, so that the exact architecture is recoverable from the
experiment manifest.
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from .pointnet import PointNetClassifier

__all__ = ["build_model", "REGISTRY", "available_architectures"]


def _build_pointnet(num_classes: int, **kwargs: Any) -> nn.Module:
    return PointNetClassifier(num_classes=num_classes, **kwargs)


#: name -> builder. Builders take ``num_classes`` plus architecture kwargs.
REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "pointnet": _build_pointnet,
}


def available_architectures() -> list[str]:
    return sorted(REGISTRY)


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    """Instantiate the architecture named by ``config["name"]``.

    Args:
        config: the ``model`` section of an experiment config. Must contain
            ``name``; all other keys are passed to the architecture verbatim.
        num_classes: taken from the label map, not from the model config, so a
            checkpoint can never disagree with its label map about ``K``.

    Raises:
        KeyError: if ``name`` is missing.
        ValueError: if the architecture is not registered.
        TypeError: if an unexpected architecture kwarg is given (propagated from
            the constructor — deliberately not swallowed, because a silently
            ignored config key is how an experiment ends up untraceable).
    """
    if "name" not in config:
        raise KeyError("model config needs a 'name' key")

    name = str(config["name"]).lower()
    if name not in REGISTRY:
        raise ValueError(
            f"unknown architecture {name!r}; available: {available_architectures()}. "
            "PointNet++ and DGCNN land with the Stage 4 comparison study."
        )

    kwargs = {k: v for k, v in config.items() if k != "name"}
    return REGISTRY[name](num_classes=num_classes, **kwargs)
