"""Export a policy to its deployment ONNX graph and verify parity (``export/policy.py``).

    uv run visnavkit-export checkpoint=/path/last.ckpt output=policy.onnx precision=fp16
    uv run visnavkit-export checkpoint=null model=gnm  # untrained pipeline check

``precision`` (``fp32`` | ``fp16``) is the stored weight dtype; io stays fp32. Beside the graph: ``<output>.pth``
(the fp32 weights and config, Lightning-free), ``.metadata.json`` and ``.inputs.npz`` (the traced inputs), read
by ``visnavkit-check-export`` and ``visnavkit-build-engine``.
"""

import hydra
from omegaconf import DictConfig

from visnavkit.export.policy import export_policy


@hydra.main(version_base=None, config_path="../configs", config_name="export")
def main(cfg: DictConfig):
    return export_policy(cfg, cfg.output)


if __name__ == "__main__":
    main()
