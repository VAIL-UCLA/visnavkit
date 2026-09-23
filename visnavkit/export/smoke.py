"""The whole deployment path on randomly initialized weights: a temporary checkpoint, its ONNX export, the
TensorRT engine (when ``tensorrt`` is importable) and the check that compares all of them, with latencies."""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from visnavkit.export.artifacts import instantiate_model
from visnavkit.export.check import check_export
from visnavkit.export.dst import export_dst
from visnavkit.export.policy import export_policy
from visnavkit.models.flowpilot_dst import FlowPilotDST
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def save_random_checkpoint(cfg, path):
    """A checkpoint shaped like training's (config + ``model.``-prefixed weights), untrained."""
    model = instantiate_model(cfg)
    state = {f"model.{name}": value for name, value in model.state_dict().items()}
    torch.save({"hyper_parameters": {"cfg": OmegaConf.to_container(cfg, resolve=True)}, "state_dict": state}, path)
    return model


def export_smoke(
    cfg,
    output_dir,
    *,
    precision="fp32",
    engine_precision=None,
    engine=True,
    batch_size=1,
    iterations=10,
    workspace_gb=2.0,
    provider="CPUExecutionProvider",
    device="cpu",
):
    """Returns the check's report; parity is reported, not enforced (random weights amplify float noise)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.exp_name
    checkpoint = output_dir / f"{name}.random.ckpt"
    model = save_random_checkpoint(cfg, checkpoint)
    logger.info(f"Random-weight checkpoint: {checkpoint} ({type(model).__name__})")
    onnx_path = output_dir / f"{name}.{precision}.onnx"
    if isinstance(model, FlowPilotDST):
        meta = export_dst(
            cfg, onnx_path, checkpoint=str(checkpoint), batch_size=batch_size, precision=precision, strict=False
        )
    else:
        meta = export_policy(
            cfg,
            onnx_path,
            checkpoint=str(checkpoint),
            batch_size=batch_size,
            precision=precision,
            strict=False,
            device=device,
        )
    engine_path = None
    if engine:
        try:
            from visnavkit.export.trt import build_engine
            from visnavkit.scripts.build_engine import print_engine_summary

            engine_path = output_dir / f"{name}.{engine_precision or precision}.engine"
            batch = (batch_size,) * 3
            meta_engine = build_engine(
                onnx_path, engine_path, precision=engine_precision, workspace_gb=workspace_gb, batch=batch
            )
            print_engine_summary(meta_engine)
        except ImportError as error:
            logger.warning(f"Skipping the TensorRT engine: {error}")
            engine_path = None
    report = check_export(
        checkpoint=str(checkpoint),
        pth=meta["pth"],
        onnx_path=str(onnx_path),
        engine=str(engine_path) if engine_path else None,
        iterations=iterations,
        provider=provider,
        device=device,
    )
    print(f"Artifacts in {output_dir}")
    return report
