"""Run the deployment path end to end on random weights: checkpoint -> ONNX -> TensorRT engine -> check.

    uv run visnavkit-export-smoke model=gnm                              # fp32 ONNX, fp16 engine, CPU check
    uv run visnavkit-export-smoke model=gnm --precision fp16 --engine-precision bf16 --device cuda
    uv run visnavkit-export-smoke experiment=flowpilot_dst_clips1k --no-engine

Positional arguments are Hydra overrides on the export config. Parity is reported, not enforced: random weights
amplify float noise (fp16 features can leave their tolerance; trained weights are the real test). The engine step
is skipped when ``tensorrt`` is not installed.
"""

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_module

from visnavkit.export.precision import ENGINE_PRECISIONS, ONNX_PRECISIONS
from visnavkit.export.smoke import export_smoke
from visnavkit.models.lit_model import disable_pretrained_downloads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=gnm model/action_decoder=mhp")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/export_smoke"))
    parser.add_argument("--precision", choices=ONNX_PRECISIONS, default="fp32", help="ONNX weight dtype")
    parser.add_argument("--engine-precision", choices=ENGINE_PRECISIONS, default="fp16")
    parser.add_argument("--no-engine", action="store_true", help="Skip the TensorRT build")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10, help="Latency samples per backend; 0 skips timing")
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider")
    parser.add_argument("--device", default="cpu", help="PyTorch reference device")
    parser.add_argument("--threads", type=int, default=torch.get_num_threads(), help="CPU threads")
    args = parser.parse_intermixed_args(argv)
    torch.set_num_threads(args.threads)
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="export", overrides=args.overrides)
    disable_pretrained_downloads(cfg.model)
    report = export_smoke(
        cfg,
        args.output_dir,
        precision=args.precision,
        engine_precision=args.engine_precision,
        engine=not args.no_engine,
        batch_size=args.batch_size,
        iterations=args.iterations,
        workspace_gb=args.workspace_gb,
        provider=args.provider,
        device=args.device,
    )
    return 0 if report["artifacts"] else 1  # the pipeline ran; RESULT on random weights is informative only


if __name__ == "__main__":
    raise SystemExit(main())
