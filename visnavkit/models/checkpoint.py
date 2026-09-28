"""Load a model from a checkpoint: a Lightning ``.ckpt`` (``hyper_parameters.cfg`` + ``model.``-prefixed weights) or a
bare state dict.

A checkpoint is complete: it rebuilds from the config it was trained with, and its weights replace every
initialization, so backbone downloads and initialization files (``weights``, a missing ``anchors_path``) are skipped.
A config key added after the checkpoint was saved takes its constructor default, which keeps the old architecture;
where the default cannot tell, the model class's ``legacy_config(model_cfg, weights)`` sets the key from the weights.
"""

import copy
from pathlib import Path

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf


def disable_pretrained_downloads(model_cfg: DictConfig) -> DictConfig:
    """Skip backbone downloads and initialization files, however nested. An anchors file is dropped only when it is
    missing (it stayed on the training machine): the checkpoint carries the anchors."""
    for key, value in model_cfg.items():
        if key == "pretrained":
            model_cfg[key] = False
        elif key == "weights" or (key == "anchors_path" and value and not Path(value).exists()):
            model_cfg[key] = None
        elif isinstance(value, DictConfig):
            disable_pretrained_downloads(value)
        elif isinstance(value, ListConfig):
            for item in value:
                if isinstance(item, DictConfig):
                    disable_pretrained_downloads(item)
    return model_cfg


def read(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def saved_config(stored):
    """The config a loaded Lightning checkpoint was trained with, or None (a bare state dict)."""
    saved = stored.get("hyper_parameters", {}).get("cfg") if isinstance(stored, dict) else None
    return saved if saved is None or isinstance(saved, DictConfig) else OmegaConf.create(saved)


def model_weights(stored):
    """The model's state dict from a loaded checkpoint (``LitModel`` keys lose their ``model.`` prefix)."""
    weights = stored.get("state_dict", stored)
    if any(key.startswith("model.") for key in weights):
        weights = {key.removeprefix("model."): value for key, value in weights.items() if key.startswith("model.")}
    return weights


def legacy_model_config(model_cfg, weights):
    """``model_cfg`` with the keys a checkpoint predates set from its weights, by the model class's ``legacy_config``
    hook when it has one (a copy; unchanged otherwise)."""
    model_cfg = copy.deepcopy(model_cfg)
    hook = getattr(get_class(model_cfg._target_), "legacy_config", None)
    return hook(model_cfg, weights) if hook else model_cfg


def checkpoint_model_config(model_cfg, weights):
    """``model_cfg`` ready to rebuild a checkpoint: legacy keys set, no downloads or initialization files."""
    return disable_pretrained_downloads(legacy_model_config(model_cfg, weights))


def load_model(cfg, checkpoint=None):
    """The eval-mode CPU model of ``cfg.model``; ``checkpoint`` (a path or a loaded checkpoint) is loaded strictly."""
    if checkpoint is None:
        return instantiate(cfg.model).cpu().eval()
    weights = model_weights(checkpoint if isinstance(checkpoint, dict) else read(checkpoint))
    model = instantiate(checkpoint_model_config(cfg.model, weights))
    model.load_state_dict(weights, strict=True)
    return model.cpu().eval()


def load_checkpoint(path, cfg=None):
    """``(model, cfg)`` from a checkpoint: its own saved config, else ``cfg`` (needed for a bare state dict)."""
    stored = read(path)
    cfg = saved_config(stored) or cfg
    if cfg is None:
        raise ValueError(f"{path} stores no config; pass the config it was trained with")
    return load_model(cfg, stored), cfg
