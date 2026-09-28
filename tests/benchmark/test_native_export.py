import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from visnavkit.benchmark import export as benchmark_export
from visnavkit.benchmark.export import architecture_config

SMALL = [
    "common.seq_length=2",
    "common.crop_wh=[32,32]",
    "common.downscale_factor=1",
    "model.feat_size=8",
    "plan_len_points=4",
    "model/vision_encoder=resnet18",
    "model.vision_encoder.img_embed_size=16",
    "model.vision_encoder.neck_cfg.n_res_blocks=0",
    "model.temporal_encoder.num_heads=2",
    "model.temporal_encoder.ff_dim=16",
]


def _cfg(*overrides):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        return compose(config_name="benchmark", overrides=[*SMALL, *overrides])


def test_architecture_config_ignores_initialization_only_fields():
    requested, saved = _cfg(), _cfg()
    saved.model.vision_encoder.pretrained = True
    saved.model.vision_encoder.weights = "initial.pt"
    assert architecture_config(requested.model) == architecture_config(saved.model)
    assert architecture_config(requested.model) != architecture_config(_cfg("model/action_decoder=regression").model)


def test_native_export_rejects_mismatched_checkpoint_config(tmp_path):
    requested = _cfg()
    saved = _cfg("model/action_decoder=regression")
    checkpoint = tmp_path / "other.ckpt"
    torch.save({"hyper_parameters": {"cfg": benchmark_export.OmegaConf.to_container(saved, resolve=True)}}, checkpoint)
    with pytest.raises(ValueError, match="Checkpoint model config differs"):
        benchmark_export.export_native(requested, tmp_path / "rejected.onnx", checkpoint=checkpoint)
    assert not (tmp_path / "rejected.onnx").exists()


def test_complete_checkpoint_does_not_load_component_initialization_weights(tmp_path, monkeypatch):
    cfg = _cfg("model/goal_encoder=image")
    cfg.model.vision_encoder.weights = "old-vision.pt"
    cfg.model.vision_encoder.pretrained = True
    cfg.model.goal_encoder.pretrained = True
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, checkpoint)

    def fake_instantiate(model_cfg):
        assert model_cfg.vision_encoder.pretrained is False
        assert model_cfg.vision_encoder.weights is None
        assert model_cfg.goal_encoder.pretrained is False
        return torch.nn.Identity()

    from visnavkit.models import checkpoint as checkpoints

    monkeypatch.setattr(checkpoints, "instantiate", fake_instantiate)
    model = checkpoints.load_model(cfg, checkpoint)
    assert not model.training
    assert cfg.model.vision_encoder.weights == "old-vision.pt"


def test_native_export_labels_null_goal_conditioning(tmp_path):
    torch.set_num_threads(1)
    meta = benchmark_export.export_native(_cfg("model/goal_encoder=point"), tmp_path / "point.onnx", model_id="point")
    assert meta["conditioning"] == "null_goal_token"
    model = instantiate(_cfg().model).eval()
    assert benchmark_export.SequencePolicy(model).conditioning == "goal_free"


def test_native_export_of_generative_recipe_is_deterministic(tmp_path):
    torch.set_num_threads(1)
    cfg = _cfg(
        "model/action_decoder=flow_mlp",
        "model.action_decoder.sample_steps=2",
        "model.action_decoder.denoiser.hidden=16",
    )
    meta = benchmark_export.export_native(cfg, tmp_path / "flow.onnx", model_id="flow")
    assert meta["input_shapes"]["initial_noise"] == [1, 5, 4, 3]
    assert meta["denoising_steps"] == 2 and meta["action_space"] == "waypoint"
    assert all(value["max_abs_error"] < 2e-4 for value in meta["parity"].values())
