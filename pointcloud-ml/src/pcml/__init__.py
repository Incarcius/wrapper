"""pcml — point-cloud ML for the Livox Mid-70 / Jetson Orin Nano perception stack.

Layout:
    pcml.data        dataset, clustering, sampling, augmentation, label mapping
    pcml.models      architectures and the factory that builds them from config
    pcml.deploy      the only subpackage that ships to the Jetson

Nothing in ``pcml.deploy`` may import from training-only modules: the deployed
artifact must not depend on datasets, teacher checkpoints or training code.
"""

__version__ = "0.1.0"
