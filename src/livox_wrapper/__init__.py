"""
livox_wrapper
-------------
Real-time preprocessing for Livox Mid-70 point clouds: ingests frames over
ROS 2, crops, de-noises, normalises and resamples them to a fixed-size array
ready for a point-cloud classifier.

    from livox_wrapper import PointCloudPipeline, LivoxInterface
"""
from .livox_interface import LivoxInterface
from .pipeline import PointCloudPipeline, StageTimings

__all__ = ["LivoxInterface", "PointCloudPipeline", "StageTimings"]
__version__ = "0.1.0"
