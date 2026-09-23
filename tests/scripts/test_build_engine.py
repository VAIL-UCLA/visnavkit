"""TensorRT engine build and check against the .pth and ONNX references; skipped without tensorrt and a GPU."""

import pytest
import torch
from hydra import compose, initialize_config_module

pytest.importorskip("tensorrt")

SMALL = [
    "common.seq_length=2",
    "common.crop_wh=[32,32]",
    "common.downscale_factor=1",
    "model.feat_size=8",
    "plan_len_points=4",
    "model/vision_encoder=resnet18",
    "model.vision_encoder.pretrained=false",
    "model.vision_encoder.img_embed_size=16",
    "model.vision_encoder.neck_cfg.n_res_blocks=0",
    "model.temporal_encoder.num_heads=2",
    "model.temporal_encoder.ff_dim=16",
    "model/action_decoder=regression",
    "model.action_decoder.hidden=16",
]


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_engine_matches_the_pytorch_reference(tmp_path, precision):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from visnavkit.scripts.build_engine import print_engine_summary
    from visnavkit.scripts.check_export import check_export
    from visnavkit.scripts.export import export_policy
    from visnavkit.utils.trt import build_engine

    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="export", overrides=SMALL)
    path = tmp_path / "policy.onnx"
    export_policy(cfg, path, precision="fp32")
    engine = tmp_path / f"policy.{precision}.engine"
    meta = build_engine(path, engine, precision=precision, workspace_gb=1)
    print_engine_summary(meta)
    assert meta["precision"] == precision and meta["dynamic_inputs"] == ["vision", "feature_buffer"]
    assert {spec["dtype"] for spec in meta["io"].values()} == {"float"}
    report = check_export(pth=str(path.with_suffix(".pth")), onnx_path=str(path), engine=str(engine), iterations=1)
    assert report["ok"] and report["artifacts"]["engine"]["precision"] == precision, report
