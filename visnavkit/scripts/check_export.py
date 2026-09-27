"""Compare a model's checkpoint, ``.pth``, ONNX graph and TensorRT engine on the same inputs (``export/check.py``).

    uv run visnavkit-check-export checkpoint=last.ckpt onnx=policy.onnx engine=policy.fp16.engine
    uv run visnavkit-check-export pth=policy.pth onnx=policy.onnx                # Lightning-free reference

Exit status 1 when an output leaves its tolerance or a sidecar hash does not match.
"""

import sys

import hydra
from omegaconf import DictConfig

from visnavkit.export.check import check_export


@hydra.main(version_base=None, config_path="../configs", config_name="check_export")
def main(cfg: DictConfig):
    report = check_export(
        checkpoint=cfg.checkpoint,
        pth=cfg.pth,
        onnx_path=cfg.onnx,
        engine=cfg.engine,
        inputs=cfg.inputs,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        device=cfg.device,
        provider=cfg.provider,
        atol=cfg.atol,
        rtol=cfg.rtol,
        iterations=cfg.iterations,
    )
    if not report["ok"]:
        sys.exit(1)
    return report


if __name__ == "__main__":
    main()
