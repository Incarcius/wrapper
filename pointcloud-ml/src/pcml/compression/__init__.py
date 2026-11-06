"""Compression and export: the path from a trained checkpoint to a deployable engine.

Stage order on the drone is ONNX first, TensorRT second. This subpackage owns the
ONNX half, which is the half that can be verified without the device:

* :mod:`pcml.compression.export` — ONNX export with a dynamic batch axis, plus the
  PyTorch-vs-onnxruntime parity check and the graph op-type inventory that
  substantiate the "every op is TensorRT-native" claim the architecture choice
  rests on (``docs/task_definition.md`` §9).

Quantisation and structured pruning (Phase Six) land here too, and deliberately
after export parity is proven: a compression pass is only meaningful once there
is a verified reference graph to compare it against.

Nothing here is imported by :mod:`pcml.deploy`. Export is a build-time activity;
the drone loads an engine, it does not build one.
"""

from .export import (
    EXPECTED_OPS,
    ExportResult,
    ParityReport,
    check_parity,
    export_onnx,
    graph_io_shapes,
    onnx_op_types,
    onnxruntime_latency,
    parity_inputs,
    unexpected_op_types,
)

__all__ = [
    "EXPECTED_OPS",
    "ExportResult",
    "ParityReport",
    "check_parity",
    "export_onnx",
    "graph_io_shapes",
    "onnx_op_types",
    "onnxruntime_latency",
    "parity_inputs",
    "unexpected_op_types",
]
