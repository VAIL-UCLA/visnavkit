"""Build a TensorRT engine from an exported ONNX graph and write ``<engine>.metadata.json`` beside it.

    uv run visnavkit-build-engine onnx=outputs/policy.onnx                 # -> outputs/policy.<precision>.engine
    uv run visnavkit-build-engine onnx=policy.onnx workspace_gb=8 tf32=true

Needs ``tensorrt`` (``uv pip install tensorrt-cu12`` or ``tensorrt-cu13``, matching the driver's CUDA) and the GPU
the engine will run on. The engine computes at the graph's stored precision: export with ``precision=fp16`` or
``bf16`` for such an engine. Compare it with the checkpoint and the graph through ``visnavkit-check-export``.
"""

import hydra
from omegaconf import DictConfig

from visnavkit.export.trt import build_engine


def print_engine_summary(meta):
    print("=" * 40 + " TENSORRT ENGINE " + "=" * 40)
    print(f"source : {meta['onnx']} (sha256 {meta['onnx_sha256'][:12]})")
    print(
        f"engine : {meta['engine']} ({meta['engine_bytes'] / 2**20:.1f}MB) {meta['precision']}"
        f"{' tf32' if meta['tf32'] else ''}, TensorRT {meta['tensorrt']}, {meta['gpu'] or 'GPU unknown'},"
        f" {meta['num_layers']} layers, built in {meta['build_seconds']}s, sha256 {meta['engine_sha256'][:12]}"
    )
    for name, spec in meta["io"].items():
        print(f"{spec['mode']:6} : {name}{tuple(spec['shape'])} {spec['dtype']}")


@hydra.main(version_base=None, config_path="../configs", config_name="build_engine")
def main(cfg: DictConfig):
    meta = build_engine(cfg.onnx, cfg.output, tf32=cfg.tf32, workspace_gb=cfg.workspace_gb, verbose=cfg.verbose)
    print_engine_summary(meta)
    return meta


if __name__ == "__main__":
    main()
