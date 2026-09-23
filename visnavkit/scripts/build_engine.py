"""Build a TensorRT engine from an exported ONNX graph and write ``<engine>.metadata.json`` beside it.

    uv run visnavkit-build-engine onnx=outputs/policy.onnx precision=fp16       # -> outputs/policy.fp16.engine
    uv run visnavkit-build-engine onnx=flowpilot_dst.onnx precision=bf16 workspace_gb=8

Needs ``tensorrt`` (``uv pip install tensorrt``, or ``tensorrt-cu13`` on a CUDA 13 stack) and the GPU the engine
will run on. Build from the fp32 export: the builder keeps the graph's fp32 io and picks fp16 / bf16 kernels itself.
``batch=[min,opt,max]`` sizes the dynamic batch axis of ``visnavkit-export`` graphs. Compare the engine with the
checkpoint and the ONNX graph through ``visnavkit-check-export``.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from visnavkit.export.trt import build_engine


def print_engine_summary(meta):
    print("=" * 40 + " TENSORRT ENGINE " + "=" * 40)
    print(f"source : {meta['onnx']} (sha256 {meta['onnx_sha256'][:12]})")
    print(
        f"engine : {meta['engine_bytes'] / 2**20:.1f}MB, {meta['precision']} (flags {meta['builder_flags'] or 'none'}),"
        f" TensorRT {meta['tensorrt']}, {meta['gpu'] or 'GPU unknown to torch'}, {meta['num_layers']} layers,"
        f" built in {meta['build_seconds']}s"
    )
    for name, spec in meta["io"].items():
        print(f"{spec['mode']:6} : {name}{tuple(spec['shape'])} {spec['dtype']}")
    if meta["batch_profile"]:
        print(f"profile: batch {meta['batch_profile']} on {meta['dynamic_inputs']}")


@hydra.main(version_base=None, config_path="../configs", config_name="build_engine")
def main(cfg: DictConfig):
    output = cfg.output or Path(cfg.onnx).with_suffix(f".{cfg.precision}.engine")
    meta = build_engine(
        cfg.onnx,
        output,
        precision=cfg.precision,
        workspace_gb=cfg.workspace_gb,
        batch=tuple(int(size) for size in cfg.batch),
        verbose=cfg.verbose,
    )
    print_engine_summary(meta)
    print(f"Saved {output} and {Path(output).with_suffix('.metadata.json')}")
    return meta


if __name__ == "__main__":
    main()
