"""Make ``pcml`` importable when it has not been pip-installed.

The wrapper depends on ``pcml.data`` for clustering and sampling and on
``pcml.deploy`` for the classifier (see ``pointcloud-ml/docs/wrapper_audit.md``
§5, "Ownership boundary"). In a deployment that package is installed —
``pip install -e ../pointcloud-ml`` on a dev box, or the wheel built for the
Jetson. For a checkout where the two repositories sit side by side and nothing
has been installed yet, this adds the sibling ``src`` directory to ``sys.path``
so the test suite runs out of the box.

Deliberately test-only: library code does not edit ``sys.path``, because an
import that works under pytest and fails under systemd is worse than one that
fails in both.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SIBLING_SRC = Path(__file__).resolve().parents[2] / "pointcloud-ml" / "src"

try:  # pragma: no cover - environment-dependent
    import pcml  # noqa: F401
except ImportError:  # pragma: no cover - environment-dependent
    if not _SIBLING_SRC.is_dir():
        raise ImportError(
            "The wrapper needs the 'pcml' package (pointcloud-ml). Either "
            "`pip install -e ../pointcloud-ml`, or check out the two repositories "
            f"side by side so that {_SIBLING_SRC} exists."
        ) from None
    sys.path.insert(0, str(_SIBLING_SRC))
