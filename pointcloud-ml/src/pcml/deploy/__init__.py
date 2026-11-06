"""Deployment-side code. The only subpackage that ships to the Jetson.

Keep this import-light: no datasets, no training utilities, no matplotlib.

``classifier`` is deliberately not re-exported here: it imports torch, and the
wrapper's preprocessing path must stay importable without it. Import it
explicitly as ``from pcml.deploy.classifier import ClusterClassifier``.
"""

from .adapter import LAYOUTS, to_model_input

__all__ = ["LAYOUTS", "to_model_input"]
