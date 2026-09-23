"""Deployment export: presence-driven inputs, ONNX Runtime parity, precision, sidecars, checkpoint round trip."""

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from hydra import compose, initialize_config_module

from visnavkit.benchmark.export import sha256_file
from visnavkit.scripts.check_export import check_export
from visnavkit.scripts.export import export_policy, parse_plan_output
from visnavkit.utils.artifacts import compare_outputs, convert_onnx, onnx_precision, tolerance

SMALL = [
    "common.seq_length=2",
    "common.crop_wh=[32,32]",
    "common.downscale_factor=1",
    "model.feat_size=8",
    "plan_len_points=4",
    "model.vision_encoder.pretrained=false",
    "model.vision_encoder.img_embed_size=16",
    "model.vision_encoder.neck_cfg.n_res_blocks=0",
    "model.temporal_encoder.num_heads=2",
    "model.temporal_encoder.ff_dim=16",
]


def _cfg(*overrides):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        return compose(config_name="export", overrides=[*SMALL, *overrides])


def _inputs(path):
    return [node.name for node in ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_inputs()]


@pytest.mark.parametrize(
    "overrides, inputs",
    [
        (["model/vision_encoder=resnet18", "model.action_decoder.hidden=16"], ["vision", "feature_buffer"]),
        (
            [
                "model/vision_encoder=resnet18",
                "model/goal_encoder=point",
                "model/action_decoder=flow_dit",
                "model.action_decoder.sample_steps=2",
                "model.action_decoder.denoiser.hidden=16",
                "model.action_decoder.denoiser.depth=1",
                "model.action_decoder.denoiser.num_heads=2",
                "model.vision_encoder.token_mode=fused",
                "model.vision_encoder.patch_grid=[2,2]",
            ],
            ["vision", "feature_buffer", "goal", "noise"],
        ),
        (
            [
                "model/vision_encoder=resnet18",
                "model/goal_encoder=image",
                "model/action_decoder=diffusion_unet",
                "model.action_decoder.sample_steps=2",
                "model.action_decoder.denoiser.down_dims=[8,16]",
                "model.action_decoder.denoiser.n_groups=4",
                "model.goal_encoder.pretrained=false",
            ],
            ["vision", "feature_buffer", "goal", "noise"],
        ),
    ],
)
def test_untrained_export_has_presence_driven_inputs_and_parity(tmp_path, overrides, inputs):
    torch.set_num_threads(1)
    path = tmp_path / "policy.onnx"
    errors = export_policy(_cfg(*overrides), path, precision="fp32")
    assert _inputs(path) == inputs
    assert set(errors) == {"plan", "feat_out"}
    # These are absolute errors on untrained outputs; export_policy itself applies the relative
    # check (rtol 2e-3), and an untrained denoiser amplifies float noise over its sampling loop.
    assert all(error < 5e-3 for error in errors.values()), errors


def test_checkpoint_round_trip_enforces_parity(tmp_path):
    lightning = pytest.importorskip("lightning")
    from visnavkit.models.lit_model import LitModel

    torch.set_num_threads(1)
    cfg = _cfg(
        "model/vision_encoder=resnet18",
        "model/action_decoder=regression",
        "model.action_decoder.hidden=16",
        "model/goal_encoder=point",
    )
    model = LitModel(cfg)
    checkpoint = tmp_path / "last.ckpt"
    trainer = lightning.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu", devices=1, max_steps=0)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(checkpoint)
    restored = LitModel.load_from_checkpoint(checkpoint, cfg=cfg)
    assert not any(p.requires_grad is None for p in restored.parameters())
    path = tmp_path / "trained.onnx"
    errors = export_policy(cfg, path, precision="fp32", checkpoint=str(checkpoint))
    assert errors["plan"] < 2e-4
    meta = json.loads(path.with_suffix(".metadata.json").read_text())
    assert meta["weights"] == "checkpoint" and meta["checkpoint_sha256"] == sha256_file(checkpoint)
    # The checkpoint, the .pth and the graph agree on the traced inputs; the .pth runs the same weights bit for bit.
    pth = str(path.with_suffix(".pth"))
    report = check_export(checkpoint=str(checkpoint), pth=pth, onnx_path=str(path), iterations=0)
    assert report["ok"] and set(report["artifacts"]) == {"pth", "onnx"}, report
    assert report["artifacts"]["pth"]["outputs"]["plan"]["max_abs"] < 1e-6
    assert [item["check"] for item in report["provenance"]] == [
        "onnx metadata: onnx_sha256",
        "onnx metadata: checkpoint_sha256",
        "onnx metadata: pth_sha256",
    ]
    # A replaced graph no longer matches its sidecar.
    path.with_suffix(".metadata.json").write_text(json.dumps({**meta, "onnx_sha256": "0" * 64}))
    assert not check_export(checkpoint=str(checkpoint), onnx_path=str(path), iterations=0)["ok"]


def test_numpy_plan_parser_keeps_xy_layout():
    means = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    logits = np.array([-2, 2], dtype=np.float32)
    flat = np.concatenate([means.reshape(2, -1), -means.reshape(2, -1), logits[:, None]], axis=1).reshape(1, -1)
    parsed = parse_plan_output(flat, M=2, num_pts=3, pose_width=3)
    assert set(parsed) == {"pred_logits", "pred_confs", "pred_plans", "best_plan"}
    np.testing.assert_array_equal(parsed["pred_logits"], logits)
    np.testing.assert_array_equal(parsed["pred_plans"], means[:, :, :2])
    np.testing.assert_array_equal(parsed["best_plan"], means[1, :, :2])


REGRESSION = ["model/vision_encoder=resnet18", "model/action_decoder=regression", "model.action_decoder.hidden=16"]


def test_untrained_export_writes_sidecars_the_check_replays(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "policy.onnx"
    export_policy(_cfg(*REGRESSION), path, precision="fp32")
    meta = json.loads(path.with_suffix(".metadata.json").read_text())
    assert (meta["precision"], meta["stored_precision"], meta["weights"]) == ("fp32", "fp32", "untrained")
    assert meta["onnx_sha256"] == sha256_file(path) and meta["pth_sha256"] == sha256_file(path.with_suffix(".pth"))
    assert set(np.load(path.with_suffix(".inputs.npz"))) == set(meta["input_names"]) == {"vision", "feature_buffer"}
    report = check_export(pth=str(path.with_suffix(".pth")), onnx_path=str(path), iterations=1)
    assert report["ok"] and set(report["artifacts"]) == {"onnx"}, report
    assert report["inputs"]["source"] == str(path.with_suffix(".inputs.npz"))
    assert report["artifacts"]["onnx"]["latency_ms"] > 0 and report["artifacts"]["onnx"]["endpoint_delta_m"] < 1e-3


def test_fp16_export_keeps_fp32_io_and_stays_within_its_tolerance(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "policy.onnx"
    export_policy(_cfg(*REGRESSION), path, precision="fp16")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert {node.type for node in (*session.get_inputs(), *session.get_outputs())} == {"tensor(float)"}
    assert onnx_precision(onnx.load(str(path)))[0] == "fp16"
    report = check_export(pth=str(path.with_suffix(".pth")), onnx_path=str(path), iterations=0)
    assert report["ok"] and report["artifacts"]["onnx"]["precision"] == "fp16", report
    assert report["artifacts"]["onnx"]["tolerance"] == {"rtol": 1e-2, "atol": 2e-3}


def test_bf16_is_an_engine_precision_not_an_onnx_one():
    empty = onnx.helper.make_model(onnx.helper.make_graph([], "empty", [], []))
    with pytest.raises(ValueError, match="precision='bf16'"):
        convert_onnx(empty, "bf16")
    assert tolerance("bf16") == (2e-2, 1e-2) and tolerance("fp16", atol=1e-3) == (1e-2, 1e-3)


def test_compare_outputs_applies_the_allclose_rule():
    report = compare_outputs(["a"], [np.array([1.0, 100.0])], [np.array([1.0001, 100.5])], rtol=1e-2, atol=1e-3)
    assert report["a"]["pass"] and report["a"]["max_abs"] == pytest.approx(0.5)
    assert not compare_outputs(["a"], [np.array([1.0])], [np.array([1.1])], rtol=1e-2, atol=1e-3)["a"]["pass"]
    assert not compare_outputs(["a"], [np.array([1.0])], [np.array([np.nan])], rtol=1e-2, atol=1e-3)["a"]["finite"]
    with pytest.raises(ValueError, match="shape"):
        compare_outputs(["a"], [np.zeros(2)], [np.zeros(3)], rtol=1e-2, atol=1e-3)
