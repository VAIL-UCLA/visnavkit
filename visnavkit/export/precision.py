"""Precision names: the dtype a graph stores and its engine computes in; graph inputs and outputs stay fp32.

``fp32`` | ``fp16`` | ``bf16``. A TensorRT engine is strongly typed: it computes at its graph's precision.
ONNX Runtime has no bf16 kernels, so a bf16 graph is checked through its engine.
"""

import math

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper

PRECISIONS = ("fp32", "fp16", "bf16")
# (rtol, atol) against the fp32 PyTorch reference: fp16 keeps ~3 significant digits, bf16 ~2; tf32 is an fp32
# engine whose matmuls / convolutions round to 10 mantissa bits (TensorRT's default), so it is held to fp16.
TOLERANCES = {"fp32": (2e-3, 2e-4), "tf32": (1e-2, 2e-3), "fp16": (1e-2, 2e-3), "bf16": (2e-2, 1e-2)}
_ONNX_FLOATS = {"FLOAT": "fp32", "FLOAT16": "fp16", "BFLOAT16": "bf16"}
_TORCH_FLOATS = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def check_precision(precision):
    if precision not in PRECISIONS:
        raise ValueError(f"precision={precision!r}; choose one of {list(PRECISIONS)}")
    return precision


def tolerance(precision, rtol=None, atol=None):
    """``(rtol, atol)`` for a precision, either one overridden; an unlisted label (``mixed``) is held to fp32."""
    default_rtol, default_atol = TOLERANCES.get(precision, TOLERANCES["fp32"])
    return (default_rtol if rtol is None else float(rtol), default_atol if atol is None else float(atol))


def bf16_bits(values):
    """fp32 values as bf16 bit patterns (uint16): the upper half of the fp32 word, rounded to nearest even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _fp16_to_bf16(graph, source):
    """Retype every fp16 tensor of ``graph`` as bf16, re-encoded from its fp32 ``source`` (fp16 clamps range)."""

    def retype(tensor):
        if tensor.data_type != TensorProto.FLOAT16:
            return
        values = source.get(tensor.name)
        if values is None or values.shape != tuple(tensor.dims):
            values = numpy_helper.to_array(tensor)
        name, dims = tensor.name, list(tensor.dims)
        tensor.Clear()
        tensor.name, tensor.data_type, tensor.raw_data = name, TensorProto.BFLOAT16, bf16_bits(values).tobytes()
        tensor.dims.extend(dims)

    for tensor in graph.initializer:
        retype(tensor)
    for value in (*graph.value_info, *graph.input, *graph.output):
        if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = TensorProto.BFLOAT16
    for node in graph.node:
        for attribute in node.attribute:
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
                attribute.i = TensorProto.BFLOAT16
            elif attribute.type == onnx.AttributeProto.TENSOR:
                retype(attribute.t)
            elif attribute.type == onnx.AttributeProto.GRAPH:
                _fp16_to_bf16(attribute.g, source)


def convert_onnx(model_onnx, precision):
    """Cast a traced fp32 graph's weights to ``precision``; graph inputs and outputs stay fp32. bf16 takes the
    fp16 conversion's graph (the same casts and blocked ops) with its tensors re-encoded from fp32."""
    check_precision(precision)
    if precision == "fp32":
        return model_onnx
    from onnxruntime.transformers import float16

    fp32 = [t for t in model_onnx.graph.initializer if t.data_type == TensorProto.FLOAT and precision == "bf16"]
    source = {tensor.name: numpy_helper.to_array(tensor) for tensor in fp32}
    model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
    if precision == "bf16":
        _fp16_to_bf16(model_onnx.graph, source)
    return model_onnx


def onnx_precision(model_onnx):
    """The precision holding most stored weight elements, and ``{onnx dtype: elements}`` over the initializers
    (an fp16 / bf16 conversion keeps the initializers of a few blocked ops in fp32)."""
    elements = {}
    for tensor in model_onnx.graph.initializer:
        name = TensorProto.DataType.Name(tensor.data_type)
        elements[name] = elements.get(name, 0) + math.prod(tensor.dims)
    floats = {_ONNX_FLOATS[name]: count for name, count in elements.items() if name in _ONNX_FLOATS}
    return (max(floats, key=floats.get) if floats else "none"), elements


def model_precision(model):
    """The precision of a PyTorch module's floating-point parameters."""
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    return _TORCH_FLOATS[dtypes.pop()] if len(dtypes) == 1 else "mixed"
