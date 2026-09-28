"""Checkpoints load across versions: missing initialization files, and configs saved before a key existed."""

import lightning
import numpy as np
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from visnavkit.models.checkpoint import disable_pretrained_downloads, load_checkpoint
from visnavkit.models.lit_model import LitModel

SMALL = [
    "experiment=flow_matching_policy_clips1k",
    "model.frame_encoder.backbone_name=fastvit_t8",
    "model.frame_encoder.pretrained=false",
    "model.dim=32",
    "model.temporal.num_layers=1",
    "model.temporal.num_heads=2",
    "model.action_decoder.num_layers=1",
    "model.action_decoder.num_heads=2",
    "model.action_decoder.sample_steps=2",
    "plan_len_points=8",
    "plan_len_seconds=0.4",
    "common.seq_length=4",
    "common.crop_wh=[64,32]",
]


def _deploy(model):
    torch.manual_seed(0)
    vision, goal, mods = model.example_batch(1, 4, (32, 64))
    with torch.no_grad():
        return model.deploy(vision, goal, mods["route_patch"], mods["ego"], mods["action_bounds"])[0]


def test_a_missing_anchors_file_is_skipped_and_an_existing_one_kept(tmp_path):
    kept = tmp_path / "anchors.npy"
    np.save(kept, np.zeros((4, 8, 2), dtype=np.float32))
    cfg = OmegaConf.create(
        {
            "pretrained": True,
            "head": {"anchors_path": "/elsewhere/kmeans64.npy", "route": {"weights": "/elsewhere/vae"}},
        }
    )
    cfg.other = {"anchors_path": str(kept)}
    assert disable_pretrained_downloads(cfg) == {
        "pretrained": False,
        "head": {"anchors_path": None, "route": {"weights": None}},
        "other": {"anchors_path": str(kept)},
    }


def test_a_checkpoint_from_before_the_token_kv_and_the_scheduler_loads_and_decodes_the_same(tmp_path):
    """Saved before ``single_kv`` and ``scheduler`` existed: the config lacks both keys; the weights tell."""
    torch.set_num_threads(1)
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(
            config_name="train", overrides=[*SMALL, "+model.single_kv=true", "~model.action_decoder.scheduler"]
        )
    disable_pretrained_downloads(cfg.model)
    old = LitModel(cfg, initialize_pretrained=False)
    assert "model.kv.weight" in old.state_dict() and old.model.action_decoder.scheduler is None
    saved = OmegaConf.to_container(cfg, resolve=True)
    del saved["model"]["single_kv"]
    checkpoint = tmp_path / "old.ckpt"
    version = {"pytorch-lightning_version": lightning.__version__}  # as a training checkpoint has
    torch.save({"hyper_parameters": {"cfg": saved}, "state_dict": old.state_dict(), **version}, checkpoint)

    model, _ = load_checkpoint(checkpoint)
    reference = _deploy(old.model.eval())
    torch.testing.assert_close(_deploy(model), reference, rtol=0, atol=0)
    resumed = LitModel.load_from_checkpoint(checkpoint, map_location="cpu")
    torch.testing.assert_close(_deploy(resumed.model.eval()), reference, rtol=0, atol=0)


def test_step_flow_without_a_scheduler_integrates_from_noise_at_t0():
    from visnavkit.models.action.step_flow import StepFlowHead

    torch.manual_seed(0)
    head = StepFlowHead(16, num_pts=4, num_layers=1, num_heads=2, sample_steps=3).eval()
    kv, bounds = torch.randn(2, 1, 16), torch.tensor([[-0.1, -0.1, -0.1, -0.1, -1.0], [0.3, 0.1, 0.1, 3.0, 1.0]])
    bounds, ego_vw = bounds.expand(2, 2, 5), torch.zeros(2, 2)
    with torch.no_grad():
        x, ego = torch.zeros(2, 4, 5), head.ego(head.ego_cond(ego_vw, bounds))
        for i in range(3):  # the original: velocity x0 - eps, Euler steps of 1/3 from t = 0 (noise) up
            x = x + head.denoise(x, torch.full((2,), i / 3), head.kv_norm(kv), ego) / 3
        torch.testing.assert_close(head.sample(kv, bounds, ego_vw)[:, 0], head.metric(x, bounds))
