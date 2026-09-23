"""Export FlowPilot-DST to its window ONNX graph and verify parity (``export/dst.py``).

    uv run visnavkit-export-dst checkpoint=/path/last.ckpt output=flowpilot_dst.onnx precision=fp32

Same options and sidecars as ``visnavkit-export``; ``docs/flowpilot_dst_onnx.md`` documents the io contract.
"""

import hydra
from omegaconf import DictConfig

from visnavkit.export.dst import export_dst


@hydra.main(version_base=None, config_path="../configs", config_name="export_dst")
def main(cfg: DictConfig):
    return export_dst(
        cfg,
        cfg.output,
        checkpoint=cfg.checkpoint,
        batch_size=cfg.batch_size,
        top_k=cfg.top_k,
        opset=cfg.onnx_opset_version,
        precision=cfg.precision,
    )


if __name__ == "__main__":
    main()
