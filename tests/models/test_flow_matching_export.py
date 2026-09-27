"""FlowMatchingPolicy through the generic export: the deterministic graph and the one with a ``noise`` input."""

import pytest
import torch
from hydra import compose, initialize_config_module

from visnavkit.export.check import check_export
from visnavkit.export.graph import export_onnx
from visnavkit.models.lit_model import disable_pretrained_downloads

SMALL = [
    "model=flow_matching_policy",
    "model.frame_encoder.pretrained=false",
    "model.frame_encoder.backbone_name=fastvit_t8",
    "model.dim=32",
    "model.temporal.num_layers=1",
    "model.temporal.num_heads=2",
    "model.action_decoder.num_layers=1",
    "model.action_decoder.num_heads=2",
    "model.action_decoder.sample_steps=2",
    "plan_len_points=8",
    "plan_len_seconds=0.4",
    "common.seq_length=4",
    "common.uniform_t_anchors=true",
    "common.crop_wh=[64,32]",
    "common.downscale_factor=1",
]


@pytest.mark.parametrize(
    "graph, inputs, plans",
    [
        ({}, ["vision", "route_patch", "ego", "action_bounds"], 1),
        ({"noise": "randn", "num_samples": 3}, ["vision", "route_patch", "ego", "action_bounds", "noise"], 3),
    ],
)
def test_window_export_takes_no_goal_and_an_optional_noise_input(tmp_path, graph, inputs, plans):
    torch.set_num_threads(1)
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="train", overrides=SMALL)
    disable_pretrained_downloads(cfg.model)
    path = tmp_path / "flow_matching_policy.onnx"
    meta = export_onnx(cfg, path, precision="fp32", device="cpu", graph=graph)
    assert meta["input_names"] == inputs and meta["model"] == "FlowMatchingPolicy"
    assert meta["output_shapes"] == {"modes": [1, plans, 8, 5], "probs": [1, plans], "speed": [1, 1]}
    report = check_export(pth=str(path.with_suffix(".pth")), onnx_path=str(path), iterations=0)
    assert report["ok"] and report["artifacts"]["onnx"]["outputs"]["modes"]["pass"], report
