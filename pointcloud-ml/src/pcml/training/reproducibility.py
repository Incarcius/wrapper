"""Independent, named RNG streams — the thing that makes the arm comparison fair.

The distillation benchmark compares arms that consume *different numbers of
random draws*. The EMA mean-teacher arm pulls two augmented views per item where
the from-scratch baseline pulls one. With a single global RNG that difference
shifts every subsequent draw, so the data order diverges too, and "arm 1 vs arm
0" silently becomes "different loss **and** different batch order". The
comparison is then confounded in a way no amount of seed-averaging repairs,
because the confound is correlated with the arm.

The fix is one independent stream per concern, each derived from the run seed by
**name**:

    streams = RngStreams(seed=0)
    streams.numpy("order")      # batch order
    streams.numpy("augment")    # augmentation draws
    streams.torch("init")       # weight initialisation

Drawing from ``augment`` cannot perturb ``order``. ``tests/test_reproducibility.py``
asserts exactly that, and asserts the batch-index sequence is byte-identical
across arms at equal seed.

Names are hashed with ``zlib.crc32``, **not** Python's ``hash()``: string hashing
is salted per process by default, so ``hash("order")`` differs between runs and
would make every "reproducible" run irreproducible across processes.
"""

from __future__ import annotations

import os
import random
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = ["RngStreams", "stream_seed", "seed_everything", "describe_rng_env"]


def stream_seed(seed: int, name: str) -> int:
    """A stable 64-bit seed for ``name`` under ``seed``.

    Stable across processes, Python versions and platforms, which
    ``hash(name)`` is not (PYTHONHASHSEED randomises str hashing).
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"stream name must be a non-empty string, got {name!r}")
    tag = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    # Mix with the 64-bit golden ratio so neighbouring run seeds do not give
    # correlated streams. Done in Python ints, not numpy uint64: the multiply
    # overflows 64 bits and numpy would warn on every call (and wrap), whereas
    # Python ints are arbitrary precision and the modulus is explicit.
    mixed = (int(seed) * 0x9E3779B97F4A7C15 + tag) % (2**64)
    return mixed % (2**63)


@dataclass
class RngStreams:
    """Named generators derived from one run seed.

    Each name gets its own generator, created on first use and cached, so a
    given ``(seed, name)`` always produces the same sequence regardless of which
    other streams exist or how much they have been drawn from.
    """

    seed: int
    _numpy: dict[str, np.random.Generator] = field(default_factory=dict, repr=False)
    _torch: dict[str, Any] = field(default_factory=dict, repr=False)

    def numpy(self, name: str) -> np.random.Generator:
        """The NumPy generator for ``name``."""
        if name not in self._numpy:
            self._numpy[name] = np.random.default_rng(stream_seed(self.seed, name))
        return self._numpy[name]

    def torch(self, name: str):
        """The torch generator for ``name``. Imported lazily so this module
        stays usable without torch (the deployment side has no torch)."""
        import torch

        if name not in self._torch:
            gen = torch.Generator()
            gen.manual_seed(stream_seed(self.seed, name))
            self._torch[name] = gen
        return self._torch[name]

    def names(self) -> tuple[str, ...]:
        """Streams actually used so far — recorded in the run manifest."""
        return tuple(sorted(set(self._numpy) | set(self._torch)))

    def fork(self, name: str) -> "RngStreams":
        """A child RngStreams, for a sub-experiment that must not disturb this one."""
        return RngStreams(seed=stream_seed(self.seed, name))


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed the global RNGs as a backstop, and return what was set.

    Named streams are preferred for anything that affects results — this exists
    because third-party code reaches for the global RNGs and we would rather
    those be pinned than free-running. It is a safety net, not the mechanism.

    Args:
        deterministic: also request deterministic algorithms from torch. This
            can be *slower* and raises on ops with no deterministic
            implementation; preferred anyway, because a benchmark whose numbers
            move between identical runs cannot settle a 1-point difference.

    Returns:
        A dict for the run manifest, recording what was actually applied — so a
        report can say "deterministic algorithms: requested and enabled" rather
        than assuming it.
    """
    applied: dict[str, Any] = {"seed": int(seed), "deterministic_requested": deterministic}

    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import torch
    except ImportError:  # pragma: no cover - torch-free environments
        applied["torch"] = "absent"
        return applied

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    applied["torch"] = torch.__version__
    applied["cuda_available"] = bool(torch.cuda.is_available())

    if deterministic:
        # cuBLAS needs this set before the first matmul to be deterministic.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            applied["deterministic_algorithms"] = "enabled (warn_only)"
        except Exception as exc:  # pragma: no cover - version dependent
            applied["deterministic_algorithms"] = f"unavailable: {type(exc).__name__}"
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            applied["cudnn_deterministic"] = True

    return applied


def describe_rng_env(seed: int, streams: Optional[RngStreams] = None) -> dict[str, Any]:
    """RNG provenance for the run manifest.

    A published result needs the reader to be able to reconstruct the exact
    random state, which means recording the derived stream seeds and not just
    the run seed.
    """
    out: dict[str, Any] = {"run_seed": int(seed)}
    if streams is not None:
        out["streams"] = {name: stream_seed(seed, name) for name in streams.names()}
    return out
