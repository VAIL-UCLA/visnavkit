"""The random-weight smoke: temporary checkpoint -> ONNX -> check, on the CPU without TensorRT."""

from pathlib import Path

from visnavkit.scripts.export_smoke import main

SMALL = [
    "exp_name=smoke",
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


def test_smoke_writes_every_artifact_and_passes_the_check(tmp_path, capsys):
    status = main([*SMALL, "--output-dir", str(tmp_path), "--no-engine", "--iterations", "1", "--threads", "1"])
    assert status == 0
    names = {path.name for path in Path(tmp_path).iterdir()}
    assert {"smoke.random.ckpt", "smoke.fp32.onnx", "smoke.fp32.pth", "smoke.fp32.metadata.json"} <= names
    out = capsys.readouterr().out
    assert "EXPORT CHECK" in out and "RESULT: PASS" in out and "smoke.fp32.pth" in out
