"""Export artifacts: precision names, ONNX weight casting, the ``.pth`` / metadata sidecars, output parity.

A precision names the dtype an artifact stores and computes in; graph inputs and outputs stay fp32 everywhere.
ONNX graphs take ``fp32`` or ``fp16`` (ONNX Runtime has no CPU bf16 kernels to check parity with); TensorRT
engines take ``fp32`` / ``fp16`` / ``bf16`` and pick their kernels from the fp32 graph.
"""

import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import torch
from omegaconf import DictConfig, OmegaConf

from visnavkit.benchmark.export import load_native_model, sha256_file

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


def compare_outputs(names, reference, observed, rtol, atol):
    """Per output: the max abs / rel error of ``observed`` against ``reference`` and whether every element meets
    ``|observed - reference| <= atol + rtol * |reference|`` (NumPy's ``assert_allclose`` rule)."""
    report = {}
    for name, expected, actual in zip(names, reference, observed):
        expected, actual = _as_float64(expected), _as_float64(actual)
        if expected.shape != actual.shape:
            raise ValueError(f"{name}: shape {actual.shape} differs from the reference {expected.shape}")
        diff = np.abs(actual - expected)
        report[name] = {
            "max_abs": float(diff.max()) if diff.size else 0.0,
            "max_rel": float((diff / np.maximum(np.abs(expected), atol)).max()) if diff.size else 0.0,
            "pass": bool(np.all(diff <= atol + rtol * np.abs(expected))),
            "finite": bool(np.isfinite(actual).all()),
        }
    return report


def _as_float64(value):
    value = value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return value.astype(np.float64)


@contextmanager
def mha_fastpath_disabled():
    """Trace and compare with the same attention kernels: the MHA fast path is not traceable."""
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def save_pth(path, cfg, model):
    """Lightning-free weights: ``{"cfg": <resolved config>, "state_dict": <policy weights>}`` for ``load_model``."""
    torch.save({"cfg": OmegaConf.to_container(cfg, resolve=True), "state_dict": model.state_dict()}, path)
    return Path(path)


def load_model(path, cfg=None):
    """The eval-mode policy from a Lightning ``.ckpt`` or an exporter ``.pth``; a bare state dict needs ``cfg``."""
    stored = torch.load(path, map_location="cpu", weights_only=False)
    saved = stored.get("hyper_parameters", {}).get("cfg", stored.get("cfg")) if isinstance(stored, dict) else None
    if saved is not None:
        cfg = saved if isinstance(saved, DictConfig) else OmegaConf.create(saved)
    if cfg is None:
        raise ValueError(f"{path} stores no config; pass the checkpoint it was exported from as well")
    return load_native_model(cfg, path), cfg


def describe(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def size_mb(path):
    return f"{Path(path).stat().st_size / 2**20:.1f}MB"


def write_metadata(path, meta):
    Path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return Path(path)


def read_metadata(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def print_export_summary(meta):
    """The block to paste in a report: source weights, artifacts, io, parity."""
    print("=" * 40 + " EXPORT " + "=" * 40)
    source = f"{meta['weights']}"
    if meta.get("checkpoint"):
        source += f" {meta['checkpoint']} (sha256 {meta['checkpoint_sha256'][:12]})"
    print(f"model   : {meta['model']} {meta['parameters_total'] / 1e6:.1f}M params, {source}")
    print(
        f"onnx    : {meta['onnx']} ({size_mb(meta['onnx'])}) {meta['stored_precision']} weights, fp32 io,"
        f" opset {meta['opset']}, sha256 {meta['onnx_sha256'][:12]}"
    )
    pth = meta["pth"]
    print(f"pth     : {pth} ({size_mb(pth)}) fp32 weights + config, sha256 {meta['pth_sha256'][:12]}")
    dtypes = meta["input_dtypes"]
    print("inputs  : " + ", ".join(f"{n}{tuple(s)} {dtypes[n]}" for n, s in meta["input_shapes"].items()))
    print("outputs : " + ", ".join(f"{name}{tuple(shape)}" for name, shape in meta["output_shapes"].items()))
    tol = meta["parity_tolerance"]
    print(
        "parity  : "
        + ", ".join(f"{name} {error:.3g}" for name, error in meta["parity_max_abs_error"].items())
        + f" max abs error vs PyTorch (rtol {tol['rtol']}, atol {tol['atol']})"
    )
