"""TensorRT 10: build an engine from an ONNX graph and run it through PyTorch CUDA buffers.

``tensorrt`` is not a dependency: ``uv pip install tensorrt`` (CUDA 12 wheels; ``tensorrt-cu13`` on a CUDA 13
stack) adds it. An engine is bound to the GPU, TensorRT version and precision it was built with.
"""

import time
from pathlib import Path

import numpy as np
import torch

from visnavkit.export.artifacts import describe, write_metadata
from visnavkit.export.precision import ENGINE_PRECISIONS, check_precision
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)
_STATE = {}  # the TensorRT logger and runtime outlive every engine they produce


def _tensorrt():
    try:
        import tensorrt as trt
    except ImportError as error:
        raise ImportError("TensorRT is not installed: uv pip install tensorrt (or tensorrt-cu13)") from error
    if int(trt.__version__.split(".")[0]) < 10:
        raise ImportError(f"TensorRT {trt.__version__} found; the TensorRT 10 API is required")
    return trt


def _logger(trt, verbose=False):
    if "logger" not in _STATE:
        _STATE["logger"] = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    return _STATE["logger"]


def _torch_dtype(trt, dtype):
    kinds = trt.DataType
    table = {
        kinds.FLOAT: torch.float32,
        kinds.HALF: torch.float16,
        kinds.BF16: torch.bfloat16,
        kinds.INT32: torch.int32,
        kinds.INT64: torch.int64,
        kinds.BOOL: torch.bool,
        kinds.INT8: torch.int8,
        kinds.UINT8: torch.uint8,
    }
    if dtype not in table:
        raise ValueError(f"No PyTorch buffer dtype for TensorRT {dtype}")
    return table[dtype]


def build_engine(onnx_path, output, *, precision="fp16", workspace_gb=4.0, batch=(1, 1, 1), verbose=False):
    """Parse ``onnx_path``, build it at ``precision`` and write the engine plus ``<output>.metadata.json``.

    fp32 keeps TensorRT's TF32 default; fp16 / bf16 let the builder choose those kernels while the io keeps the
    graph's dtypes. ``batch`` = (min, opt, max) sizes a dynamic leading input axis; other dynamic axes are refused.
    """
    trt = _tensorrt()
    check_precision(precision, ENGINE_PRECISIONS)
    onnx_path, output = Path(onnx_path), Path(output)
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
    flags = {"fp32": [], "fp16": [trt.BuilderFlag.FP16], "bf16": [trt.BuilderFlag.BF16]}[precision]
    for flag in flags:
        config.set_flag(flag)
    profile, dynamic = builder.create_optimization_profile(), []
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = list(tensor.shape)
        if -1 not in shape:
            continue
        if -1 in shape[1:]:
            raise ValueError(f"{tensor.name} has a dynamic axis beyond the batch axis: {shape}")
        profile.set_shape(tensor.name, *[[size, *shape[1:]] for size in batch])
        dynamic.append(tensor.name)
    if dynamic:
        config.add_optimization_profile(profile)
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
        "builder_flags": [flag.name for flag in flags],
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "workspace_bytes": int(workspace_gb * 2**30),
        "batch_profile": dict(zip(("min", "opt", "max"), batch)) if dynamic else None,
        "dynamic_inputs": dynamic,
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


class EngineRunner:
    """Run an engine on NumPy feeds; the buffers are PyTorch CUDA tensors, so no other CUDA binding is needed."""

    def __init__(self, engine):
        if not torch.cuda.is_available():
            raise RuntimeError("Running an engine needs CUDA-enabled PyTorch for the device buffers")
        self.trt, self.engine = _tensorrt(), engine
        self.context = engine.create_execution_context()
        self.io = engine_io(engine)
        self.buffers = {}

    @property
    def input_names(self):
        return [name for name, spec in self.io.items() if spec["mode"] == "input"]

    @property
    def output_names(self):
        return [name for name, spec in self.io.items() if spec["mode"] == "output"]

    def __call__(self, feeds):
        """``{output name: array}``; float outputs come back as float32."""
        missing = [name for name in self.input_names if name not in feeds]
        if missing:
            raise ValueError(f"Missing engine inputs {missing}; the engine takes {self.input_names}")
        for name in self.input_names:
            dtype = _torch_dtype(self.trt, self.engine.get_tensor_dtype(name))
            value = torch.as_tensor(np.ascontiguousarray(feeds[name])).to("cuda", dtype)
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise ValueError(f"{name}{tuple(value.shape)} is outside the engine's profile {self.io[name]['shape']}")
            self.buffers[name] = value
            self.context.set_tensor_address(name, value.data_ptr())
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _torch_dtype(self.trt, self.engine.get_tensor_dtype(name))
            if name not in self.buffers or self.buffers[name].shape != shape:
                self.buffers[name] = torch.empty(shape, dtype=dtype, device="cuda")
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        torch.cuda.synchronize()
        return {
            name: (value.float() if value.is_floating_point() else value).cpu().numpy()
            for name, value in ((name, self.buffers[name]) for name in self.output_names)
        }
