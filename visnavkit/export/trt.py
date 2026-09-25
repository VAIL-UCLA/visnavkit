"""TensorRT 10: build an engine from an ONNX graph and run it through plain ``libcudart`` buffers.

``tensorrt`` is not a dependency: ``uv pip install --python .venv tensorrt-cu12`` (``tensorrt-cu13`` on a CUDA 13
driver) adds it; the runner loads the CUDA runtime library through ctypes, so neither building nor running an
engine needs a CUDA-enabled PyTorch. An engine is bound to the GPU, TensorRT version and precision it
was built with.
"""

import ctypes
import ctypes.util
import site
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx

from visnavkit.export.artifacts import describe, write_metadata
from visnavkit.export.precision import ENGINE_PRECISIONS, check_precision, onnx_precision
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)
_STATE = {}  # the TensorRT logger and runtime outlive every engine they produce


def _tensorrt():
    try:
        import tensorrt as trt
    except ImportError as error:
        raise ImportError(
            "TensorRT is not installed: uv pip install --python .venv tensorrt-cu12 (or -cu13)"
        ) from error
    if int(trt.__version__.split(".")[0]) < 10:
        raise ImportError(f"TensorRT {trt.__version__} found; the TensorRT 10 API is required")
    return trt


def _logger(trt, verbose=False):
    if "logger" not in _STATE:
        _STATE["logger"] = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    return _STATE["logger"]


def build_engine(onnx_path, output, *, precision=None, tf32=False, workspace_gb=4.0, verbose=False):
    """Parse ``onnx_path``, build the engine and write it plus ``<output>.metadata.json``.

    ``precision`` is the engine's compute dtype, by default the graph's stored one. TensorRT 11 builds strongly
    typed engines, so it must equal the graph's: export the ONNX at that precision. TensorRT 10 can still cast an
    fp32 graph to fp16 / bf16 with a builder flag. io keeps the graph's dtypes. An fp32 engine is exact fp32
    unless ``tf32`` lets its matmuls / convolutions run on tensor cores at 10 mantissa bits (TensorRT's own
    default; same speed on small models, a visible feature drift). Graphs keep their fixed export shapes.
    """
    trt = _tensorrt()
    onnx_path, output = Path(onnx_path), Path(output)
    stored, _ = onnx_precision(onnx.load(str(onnx_path), load_external_data=False))
    precision = check_precision(precision or stored, ENGINE_PRECISIONS)
    flags = []
    if precision != stored:
        if not hasattr(trt.BuilderFlag, "FP16"):
            raise ValueError(
                f"TensorRT {trt.__version__} builds strongly typed engines: {onnx_path} stores {stored} weights,"
                f" so export the ONNX with precision={precision} instead of casting here"
            )
        flags = [trt.BuilderFlag.FP16 if precision == "fp16" else trt.BuilderFlag.BF16]
    output.parent.mkdir(parents=True, exist_ok=True)
    trt_logger = _logger(trt, verbose)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)  # explicit batch, the only mode in TensorRT 10
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise ValueError(f"TensorRT could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 2**30))
    for flag in flags:
        config.set_flag(flag)
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    tf32 = tf32 and precision == "fp32"
    for tensor in (network.get_input(i) for i in range(network.num_inputs)):
        if -1 in tensor.shape:
            raise ValueError(f"{tensor.name} has a dynamic axis {list(tensor.shape)}; export at fixed shapes")
    source = describe(onnx_path)
    logger.info(f"Building a {precision} engine from {onnx_path} ({source['bytes'] / 2**20:.1f}MB) ...")
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed; rerun with verbose=true for the builder log")
    seconds = time.perf_counter() - start
    with output.open("wb") as stream:
        stream.write(serialized)
    engine = load_engine(output)
    meta = {
        "onnx": source["path"],
        "onnx_sha256": source["sha256"],
        "engine_sha256": describe(output)["sha256"],
        "engine_bytes": output.stat().st_size,
        "precision": precision,
        "graph_precision": stored,
        "builder_flags": [flag.name for flag in flags],
        "strongly_typed": not flags,
        "tf32": tf32,
        "tensorrt": trt.__version__,
        "gpu": gpu_name(),
        "workspace_bytes": int(workspace_gb * 2**30),
        "io": engine_io(engine),
        "num_layers": engine.num_layers,
        "build_seconds": round(seconds, 1),
    }
    write_metadata(output.with_suffix(".metadata.json"), meta)
    logger.info(f"Saved {output} ({output.stat().st_size / 2**20:.1f}MB) and metadata in {seconds:.0f}s")
    return meta


def load_engine(path):
    trt = _tensorrt()
    if "runtime" not in _STATE:
        _STATE["runtime"] = trt.Runtime(_logger(trt))
    engine = _STATE["runtime"].deserialize_cuda_engine(Path(path).read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not load {path}; an engine only runs on the GPU and TensorRT version that built it")
    return engine


def engine_io(engine):
    """``{name: {mode, dtype, shape}}`` over the engine's io tensors; -1 marks a dynamic axis."""
    trt = _tensorrt()
    io = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        io[name] = {
            "mode": "input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output",
            "dtype": engine.get_tensor_dtype(name).name.lower(),
            "shape": list(engine.get_tensor_shape(name)),
        }
    return io


_NUMPY_DTYPES = {"FLOAT": np.float32, "HALF": np.float16, "INT32": np.int32, "INT64": np.int64, "BOOL": np.bool_}
_NUMPY_DTYPES |= {"INT8": np.int8, "UINT8": np.uint8}


def gpu_name():
    try:
        query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        return subprocess.run(query, capture_output=True, text=True, timeout=5, check=True).stdout.split("\n")[0]
    except (OSError, subprocess.SubprocessError):
        return None


def _cudart():
    """The CUDA runtime library: the pip ``nvidia-cuda-runtime`` wheels (TensorRT's own comes first), then the
    system one; the first that initializes on this driver wins."""
    if "cudart" in _STATE:
        return _STATE["cudart"]
    roots = [Path(p) for p in (*site.getsitepackages(), site.getusersitepackages(), *sys.path) if p]
    candidates = sorted({c for root in roots if root.is_dir() for c in root.glob("nvidia/cu*/lib/libcudart.so.*")})
    for root in (r for r in roots if r.is_dir()):
        candidates += sorted(root.glob("nvidia/cuda_runtime/lib/libcudart.so.*"))
    if found := ctypes.util.find_library("cudart"):
        candidates.append(Path(found))
    candidates += sorted(Path("/usr/local/cuda/lib64").glob("libcudart.so.*"))
    errors = []
    for candidate in sorted(candidates, key=lambda c: (not c.name.endswith(".12"), str(c))):
        try:
            lib = ctypes.CDLL(str(candidate))
            lib.cudaFree.argtypes = [ctypes.c_void_p]
            if lib.cudaFree(None) == 0:
                lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
                lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                lib.cudaGetErrorString.restype = ctypes.c_char_p
                _STATE["cudart"] = lib
                return lib
            errors.append(f"{candidate}: {lib.cudaGetErrorString(lib.cudaGetLastError()).decode()}")
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("No usable libcudart for this driver:\n" + "\n".join(errors))


class _DeviceBuffer:
    def __init__(self, cudart, nbytes):
        self.cudart, self.nbytes, self.ptr = cudart, nbytes, ctypes.c_void_p()
        _check(cudart, cudart.cudaMalloc(ctypes.byref(self.ptr), max(nbytes, 1)))

    def upload(self, array):
        _check(self.cudart, self.cudart.cudaMemcpy(self.ptr, array.ctypes.data, array.nbytes, 1))  # host -> device

    def download(self, array):
        _check(self.cudart, self.cudart.cudaMemcpy(array.ctypes.data, self.ptr, array.nbytes, 2))  # device -> host

    def __del__(self):
        if getattr(self, "ptr", None) is not None and self.ptr.value:
            self.cudart.cudaFree(self.ptr)


def _check(cudart, status):
    if status != 0:
        raise RuntimeError(f"CUDA runtime error {status}: {cudart.cudaGetErrorString(status).decode()}")


class EngineRunner:
    """Run an engine on NumPy feeds through ``libcudart`` device buffers (no PyTorch CUDA needed)."""

    def __init__(self, engine):
        self.trt, self.engine, self.cudart = _tensorrt(), engine, _cudart()
        self.context = engine.create_execution_context()
        self.io = engine_io(engine)
        self.buffers = {}

    @property
    def input_names(self):
        return [name for name, spec in self.io.items() if spec["mode"] == "input"]

    @property
    def output_names(self):
        return [name for name, spec in self.io.items() if spec["mode"] == "output"]

    def _dtype(self, name):
        kind = self.engine.get_tensor_dtype(name).name
        if kind not in _NUMPY_DTYPES:
            raise ValueError(f"{name} is {kind}: no NumPy dtype for it (exported graphs keep fp32 io)")
        return _NUMPY_DTYPES[kind]

    def _buffer(self, name, nbytes):
        if name not in self.buffers or self.buffers[name].nbytes < nbytes:
            self.buffers[name] = _DeviceBuffer(self.cudart, nbytes)
        return self.buffers[name]

    def __call__(self, feeds):
        """``{output name: array}``."""
        missing = [name for name in self.input_names if name not in feeds]
        if missing:
            raise ValueError(f"Missing engine inputs {missing}; the engine takes {self.input_names}")
        for name in self.input_names:
            value = np.ascontiguousarray(feeds[name], dtype=self._dtype(name))
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise ValueError(f"{name}{value.shape} is outside the engine's profile {self.io[name]['shape']}")
            buffer = self._buffer(name, value.nbytes)
            buffer.upload(value)
            self.context.set_tensor_address(name, buffer.ptr.value)
        outputs = {}
        for name in self.output_names:
            outputs[name] = np.empty(tuple(self.context.get_tensor_shape(name)), dtype=self._dtype(name))
            self.context.set_tensor_address(name, self._buffer(name, outputs[name].nbytes).ptr.value)
        if not self.context.execute_async_v3(0):  # the default stream
            raise RuntimeError("TensorRT execution failed")
        _check(self.cudart, self.cudart.cudaDeviceSynchronize())
        for name, array in outputs.items():
            self.buffers[name].download(array)
        return outputs
