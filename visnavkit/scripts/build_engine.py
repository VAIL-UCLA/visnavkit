"""Build a TensorRT engine from an exported ONNX graph and write ``<engine>.metadata.json`` beside it.

    uv run visnavkit-build-engine onnx=outputs/policy.onnx                 # -> outputs/policy.<graph precision>.engine
    uv run visnavkit-build-engine onnx=policy.onnx workspace_gb=8 tf32=true

Needs ``tensorrt`` (``uv pip install tensorrt-cu12`` or ``tensorrt-cu13``, matching the driver's CUDA) and the GPU
the engine will run on. The engine computes at the graph's stored precision (TensorRT 11 is strongly typed): export
with ``precision=fp16`` for an fp16 engine. Compare the engine with the checkpoint and the graph through
``visnavkit-check-export``.
"""

from pathlib import Path

import hydra
import onnx
from omegaconf import DictConfig

from visnavkit.export.precision import onnx_precision
from visnavkit.export.trt import build_engine


def print_engine_summary(meta):
    print("=" * 40 + " TENSORRT ENGINE " + "=" * 40)
    print(f"source : {meta['onnx']} (sha256 {meta['onnx_sha256'][:12]})")
    print(
        f"engine : {meta['engine_bytes'] / 2**20:.1f}MB, {meta['precision']}"
        f" ({'strongly typed' if meta['strongly_typed'] else 'flags ' + str(meta['builder_flags'])}"
        f"{', tf32' if meta['tf32'] else ''}),"
        f" TensorRT {meta['tensorrt']}, {meta['gpu'] or 'GPU unknown to torch'}, {meta['num_layers']} layers,"
        f" built in {meta['build_seconds']}s"
    )
    for name, spec in meta["io"].items():
        print(f"{spec['mode']:6} : {name}{tuple(spec['shape'])} {spec['dtype']}")


@hydra.main(version_base=None, config_path="../configs", config_name="build_engine")
def main(cfg: DictConfig):
    precision = cfg.precision or onnx_precision(onnx.load(str(cfg.onnx), load_external_data=False))[0]
    output = cfg.output or Path(cfg.onnx).with_suffix(f".{precision}.engine")
    meta = build_engine(
        cfg.onnx,
        output,
        precision=precision,
        tf32=cfg.tf32,
        workspace_gb=cfg.workspace_gb,
        verbose=cfg.verbose,
    )
    print_engine_summary(meta)
    print(f"Saved {output} and {Path(output).with_suffix('.metadata.json')}")
    return meta


if __name__ == "__main__":
    main()
