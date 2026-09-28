"""Train a navigation policy with configurable logging and checkpoint selection."""

import subprocess
import sys

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf, open_dict

from visnavkit.models.checkpoint import legacy_model_config, model_weights, read
from visnavkit.models.lit_model import LitModel
from visnavkit.utils.display import display_callbacks

torch.set_float32_matmul_precision("medium")


def check_git_status(strict=False):
    """Record source provenance; require a clean checkout only when requested."""
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if strict and (status.returncode or revision.returncode):
        raise RuntimeError("strict_git=true requires running from a Git checkout.")
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    if strict and dirty:
        raise RuntimeError("strict_git=true requires committed or stashed changes before training.")
    return {"revision": revision.stdout.strip() if revision.returncode == 0 else None, "dirty": dirty}


def create_trainer(cfg):
    """Construct the logger and callbacks from the public Hydra configuration."""
    experiment_logger = instantiate(cfg.logger) if cfg.get("logger") else False
    if experiment_logger:
        experiment_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    logging = cfg.trainer.logging
    callbacks = [
        ModelCheckpoint(
            monitor=logging.monitor,
            save_top_k=logging.save_top_k_ckpts,
            save_last=True,
            mode=logging.mode,
            filename="epoch_{epoch:02d}-step_{step}",
            auto_insert_metric_name=False,
        )
    ]
    if cfg.get("ema"):
        callbacks.append(instantiate(cfg.ema))
    if experiment_logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    shown, kwargs = display_callbacks(cfg.trainer.get("display"), cfg.trainer.kwargs)
    return L.Trainer(**cfg.trainer.kwargs, **kwargs, logger=experiment_logger, callbacks=callbacks + shown)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    provenance = check_git_status(strict=cfg.get("strict_git", False))
    with open_dict(cfg):
        cfg.provenance = provenance
    if not cfg.get("dataset"):
        sys.exit("ERROR: no dataset config yet. Add visnavkit/configs/dataset/<name>.yaml and pass dataset=<name>.")
    L.seed_everything(cfg.seed, workers=True)
    datamodule = instantiate(cfg.dataset)
    trainer = create_trainer(cfg)
    if resume := cfg.trainer.resume.ckpt_path:  # a checkpoint older than a config key: set it from the weights
        cfg.model = legacy_model_config(cfg.model, model_weights(read(resume)))
    model = LitModel(cfg, initialize_pretrained=not bool(resume))
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.trainer.resume.ckpt_path)


if __name__ == "__main__":
    main()
