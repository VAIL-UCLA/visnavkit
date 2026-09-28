"""Export a policy to a deployment ONNX graph (single frame + feature buffer) and verify parity.

    uv run visnavkit-export checkpoint=/path/last.ckpt output=policy.onnx
    uv run visnavkit-export checkpoint=null model=gnm  # untrained pipeline check

Graph inputs are presence-driven: ``vision``, ``feature_buffer``, then one ``goal`` input per goal
encoder, one input per key its modality encoders read (``ego``, ``intrinsics``/``extrinsics``,
...), and ``noise`` for generative decoders.
Outputs: ``plan``, ``feat_out``, plus ``speed`` when the recipe enables the auxiliary speed head.
"""

import copy
from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.utils.fusion import fuse_linear_bn_eval

from visnavkit.models.action.outputs import parse_plan_output as parse_tensor_plan_output
from visnavkit.models.checkpoint import read, saved_config
from visnavkit.models.lit_model import LitModel
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def enforce_output_order(model_onnx: onnx.ModelProto, output_names: list[str]) -> onnx.ModelProto:
    output_map = {out.name: out for out in model_onnx.graph.output}
    if not all(name in output_map for name in output_names):
        return model_onnx
    del model_onnx.graph.output[:]
    for name in output_names:
        model_onnx.graph.output.append(output_map[name])
    return model_onnx


def fuse_linear_bn_pairs(module: nn.Module) -> None:
    """Fold Linear -> BatchNorm1d pairs."""
    children = list(module.named_children())
    for (name, child), (next_name, next_child) in zip(children, children[1:]):
        if isinstance(child, nn.Linear) and isinstance(next_child, nn.BatchNorm1d):
            setattr(module, name, fuse_linear_bn_eval(child, next_child))
            setattr(module, next_name, nn.Identity())
    for _, child in module.named_children():
        fuse_linear_bn_pairs(child)


def reparameterize_model(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    fuse_linear_bn_pairs(model)
    return model


def parse_plan_output(output, M, num_pts, pose_width):
    """Single-sample NumPy helper for deployment consumers (xy trajectories only)."""
    output = np.asarray(output).reshape(1, M * (num_pts * 2 * pose_width + 1))
    parsed = parse_tensor_plan_output(torch.from_numpy(output), num_modes=M, num_pts=num_pts, pose_size=pose_width)
    return dict(
        pred_logits=output.reshape(M, -1)[:, -1],
        pred_confs=parsed["confs"][0].numpy(),
        pred_plans=parsed["plans"][0, :, :, :2].numpy(),
        best_plan=parsed["best_plan"][0, :, :2].numpy(),
    )


class _ExportPolicy(nn.Module):
    """Positional ``predict`` wrapper so absent goal/noise inputs never appear in the graph."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, *inputs):
        kwargs = dict(zip(self.policy.export_input_names(), inputs))
        goal_names = self.policy.goal_input_names()
        goal = [kwargs[name] for name in goal_names] if len(goal_names) > 1 else kwargs.get("goal")
        modalities = {name: kwargs[name] for name in self.policy.modality_input_names if name in kwargs}
        return self.policy.predict(
            kwargs["vision"], kwargs["feature_buffer"], goal=goal, noise=kwargs.get("noise"), **modalities
        )


def check_parity(model_path, feeds, reference, output_names, *, half=False, strict=True):
    """Run ONNX Runtime on nonzero inputs and compare with the PyTorch outputs."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    graph_inputs = {node.name for node in sess.get_inputs()}
    output = sess.run(output_names, {name: value for name, value in feeds.items() if name in graph_inputs})
    errors = {}
    for name, expected, observed in zip(output_names, reference, output):
        expected = expected.detach().cpu().numpy()
        errors[name] = (
            float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64)))) if expected.size else 0.0
        )
        try:
            np.testing.assert_allclose(
                observed, expected, rtol=1e-2 if half else 2e-3, atol=2e-3 if half else 2e-4, err_msg=name
            )
        except AssertionError:
            if strict:
                raise
            logger.warning(
                f"Parity tolerance exceeded for {name} (max abs error {errors[name]:.3g}); untrained weights, not enforced"
            )
    logger.info("Nonzero-input PyTorch/ONNX numerical parity: " + ", ".join(f"{k}={v:.3g}" for k, v in errors.items()))
    return output, errors


def prepare_export_config(cfg):
    """Restore checkpoint architecture/preprocessing while retaining export options."""
    cfg = copy.deepcopy(cfg)
    if cfg.checkpoint:
        saved = saved_config(read(cfg.checkpoint))
        if saved is not None:
            options = {key: cfg[key] for key in ("checkpoint", "output", "onnx_opset_version", "half", "export_heads")}
            cfg = OmegaConf.merge(OmegaConf.to_container(saved, resolve=True), options)
    return cfg


def export_policy(cfg, output, *, half=None, checkpoint=..., batch_size=1, opset=None, export_heads=None):
    """Export the deployment graph, slim it, optionally convert to fp16, and verify parity.

    Returns the per-output maximum absolute parity errors. Parity is enforced for checkpoints
    and reported for untrained pipeline checks.
    """
    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        if checkpoint is not ...:
            cfg.checkpoint = checkpoint
        cfg.setdefault("checkpoint", None)
        cfg.setdefault("output", str(output))
        cfg.setdefault("onnx_opset_version", 14)
        cfg.setdefault("half", True)
        cfg.setdefault("export_heads", [])
    cfg = prepare_export_config(cfg)
    half = cfg.get("half", True) if half is None else half
    opset = cfg.get("onnx_opset_version", 14) if opset is None else opset
    export_heads = list(cfg.get("export_heads", []) if export_heads is None else export_heads)
    if cfg.model.temporal_encoder.reduction == "none":
        # Training predicts per frame; deployment wants the newest frame's decision only.
        cfg.model.temporal_encoder.reduction = "last"

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if cfg.checkpoint is None:  # untrained weights: pipeline check
        lmodel = LitModel(cfg, initialize_pretrained=False)
    else:
        lmodel = LitModel.load_from_checkpoint(cfg.checkpoint, cfg=cfg)
    lmodel.eval()
    infer_model = reparameterize_model(lmodel.model)
    infer_model.export_heads = [name for name in export_heads if name in infer_model.vision_encoder.heads]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_model = infer_model.to(device)

    img_w = int(cfg.common.crop_wh[0] // cfg.common.downscale_factor)
    img_h = int(cfg.common.crop_wh[1] // cfg.common.downscale_factor)
    infer_model.vision_encoder = infer_model.vision_encoder.prepare_for_export((img_h, img_w))
    torch.manual_seed(0)
    inputs = infer_model.example_inputs(batch_size, (img_h, img_w), device)
    input_names = infer_model.export_input_names()
    output_names = infer_model.export_output_names()
    logger.info("Export inputs: " + ", ".join(f"{name}{tuple(t.shape)}" for name, t in zip(input_names, inputs)))

    wrapper = _ExportPolicy(infer_model).eval()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            reference = wrapper(*inputs)
        if len(reference) != len(output_names):
            raise ValueError(f"Expected {len(output_names)} outputs ({output_names}), got {len(reference)}")
        dynamic_axes = {name: {0: "batch_size"} for name in [*input_names, *output_names]}
        logger.info("Exporting from torch model...")
        torch.onnx.export(
            wrapper,
            inputs,
            str(output),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            verbose=False,
            dynamo=False,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)

    model_onnx = onnx.load(str(output))
    try:
        import onnxslim

        logger.info("Slimming...")
        model_onnx = onnxslim.slim(model_onnx)
    except ImportError:
        logger.warning("onnxslim not installed (uv sync --extra export); skipping graph slimming")
    if half:
        from onnxruntime.transformers import float16

        logger.info("Converting to fp16...")
        model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
    model_onnx = enforce_output_order(model_onnx, output_names)  # index-based runtimes rely on the order
    onnx.save(model_onnx, str(output))
    logger.info(f"Saved {output}")

    feeds = {name: tensor.cpu().numpy() for name, tensor in zip(input_names, inputs)}
    outputs, errors = check_parity(output, feeds, reference, output_names, half=half, strict=cfg.checkpoint is not None)
    decoder = infer_model.action_decoder
    parsed = parse_plan_output(
        outputs[0][:1], M=decoder.num_modes, num_pts=decoder.num_pts, pose_width=decoder.pose_size
    )
    print("=" * 40 + " SANITY CHECK " + "=" * 40)
    if "speed" in output_names:
        speed = outputs[output_names.index("speed")]
        print(f"speed: {float(np.asarray(speed).reshape(-1)[0]):.4f}")
    print(f"logits: {np.round(parsed['pred_logits'], 3)}")
    print(f"best_plan p0: {np.round(parsed['best_plan'][0], 2)}")
    print(f"best_plan pN: {np.round(parsed['best_plan'][-1], 2)}")
    return errors


@hydra.main(version_base=None, config_path="../configs", config_name="export")
def main(cfg: DictConfig):
    return export_policy(cfg, cfg.output)


if __name__ == "__main__":
    main()
