"""Precision names: the dtype an artifact stores and computes in; graph inputs and outputs stay fp32 everywhere.

ONNX graphs take ``fp32`` or ``fp16`` (ONNX Runtime has no CPU bf16 kernels to check parity with); TensorRT
engines take ``fp32`` / ``fp16`` / ``bf16`` and pick their kernels from the fp32 graph.
"""

import math

import onnx
import torch

ONNX_PRECISIONS = ("fp32", "fp16")
ENGINE_PRECISIONS = ("fp32", "fp16", "bf16")
# (rtol, atol) against the fp32 PyTorch reference: fp16 keeps ~3 significant digits, bf16 ~2.
TOLERANCES = {"fp32": (2e-3, 2e-4), "fp16": (1e-2, 2e-3), "bf16": (2e-2, 1e-2)}
_ONNX_FLOATS = {"FLOAT": "fp32", "FLOAT16": "fp16", "BFLOAT16": "bf16"}
_TORCH_FLOATS = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def check_precision(precision, allowed):
    if precision not in allowed:
        raise ValueError(f"precision={precision!r}; choose one of {list(allowed)}")
    return precision


def tolerance(precision, rtol=None, atol=None):
    """``(rtol, atol)`` for a precision, either one overridden; an unlisted label (``mixed``) is held to fp32."""
    default_rtol, default_atol = TOLERANCES.get(precision, TOLERANCES["fp32"])
    return (default_rtol if rtol is None else float(rtol), default_atol if atol is None else float(atol))


def convert_onnx(model_onnx, precision):
    """Cast a traced fp32 graph's weights to ``precision``; graph inputs and outputs stay fp32."""
    check_precision(precision, ONNX_PRECISIONS)
    if precision == "fp16":
        from onnxruntime.transformers import float16

        model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
    return model_onnx


def onnx_precision(model_onnx):
    """The precision holding most stored weight elements, and ``{onnx dtype: elements}`` over the initializers
    (an fp16 conversion keeps the initializers of a few blocked ops in fp32)."""
    elements = {}
    for tensor in model_onnx.graph.initializer:
        name = onnx.TensorProto.DataType.Name(tensor.data_type)
        elements[name] = elements.get(name, 0) + math.prod(tensor.dims)
    floats = {_ONNX_FLOATS[name]: count for name, count in elements.items() if name in _ONNX_FLOATS}
    return (max(floats, key=floats.get) if floats else "none"), elements


def model_precision(model):
    """The precision of a PyTorch module's floating-point parameters."""
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    return _TORCH_FLOATS[dtypes.pop()] if len(dtypes) == 1 else "mixed"
