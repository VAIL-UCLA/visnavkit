"""Console display of a training run: the progress bar and the model parameter summary, by style name.

``trainer.display.progress_bar``:
- ``tqdm``: Lightning's bar, the reference's (``Epoch 0: 1%| 100/11516 [01:32<2:55:49, 1.08it/s, loss=.., train/rps=..]``).
- ``rich``: Lightning's rich bar (the ``rich`` extra).
- ``lines``: one plain line every ``refresh_every_n_steps`` steps, for ``nohup`` / ``tee`` logs without carriage returns.
- ``none``: no bar.

``trainer.display.model_summary``:
- ``lightning``: Lightning's depth-1 table (name, type, params, mode), the reference's.
- ``deep``: the same table ``summary_depth`` levels deep.
- ``rich``: the rich table ``summary_depth`` levels deep (the ``rich`` extra).
- ``stages``: one row per policy stage: total / trainable / frozen parameters (``36,614,802 (36.61M)``), share of
  the model, fp32 size and a clickable ``path:line`` of its source.
- ``none``: nothing.

``trainer.display.print_config``: true prints the resolved config (the run's hyperparameters) at fit start.
``trainer.display.batch_summary``: N > 0 prints the first train batch (every key's shape, dtype, range) and its first N
windows (source clip and frame when the dataset returns ``index``, current-slot ego / goal, plan endpoint).
"""

import inspect
import os
import sys
import time
from datetime import timedelta

import torch
from lightning.pytorch.callbacks import Callback, ModelSummary, TQDMProgressBar
from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

PROGRESS_BARS = ("tqdm", "rich", "lines", "none")
MODEL_SUMMARIES = ("lightning", "deep", "rich", "stages", "none")


def _format_metrics(metrics):
    return " | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items())


def count(n):
    """``36614802`` -> ``36,614,802 (36.61M)``; ``B`` from a billion."""
    return f"{n:,} ({n / 1e9:.2f}B)" if n >= 1e9 else f"{n:,} ({n / 1e6:.2f}M)"


class LineProgressBar(ProgressBar):
    """``[train] epoch 0 step 100/11516 (0.9%) | 1.08 it/s | eta 2:55:49 | loss 2.889 | train/rps 337``."""

    def __init__(self, every_n_steps=50):
        super().__init__()
        self.every_n_steps, self._enabled, self._t0 = max(1, int(every_n_steps)), True, None

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        done = batch_idx + 1
        if self._t0 is None:  # the rate counts from the epoch's first finished batch (dataloader warm-up excluded)
            self._t0, self._b0 = time.monotonic(), done
        if not self._enabled or (done % self.every_n_steps and done != self.total_train_batches):
            return
        rate = (done - self._b0) / max(time.monotonic() - self._t0, 1e-9)
        total = self.total_train_batches
        progress, eta = f"{done}", ""
        if isinstance(total, int) and total:
            progress = f"{done}/{total} ({100 * done / total:.1f}%)"
            eta = f" | eta {timedelta(seconds=round((total - done) / max(rate, 1e-9)))}"
        metrics = self.get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        line = (
            f"[train] epoch {trainer.current_epoch} step {progress} | {rate:.2f} it/s{eta} | {_format_metrics(metrics)}"
        )
        print(line, file=sys.stdout, flush=True)

    def on_validation_end(self, trainer, pl_module):
        if self._enabled and not trainer.sanity_checking:
            metrics = {k: v for k, v in self.get_metrics(trainer, pl_module).items() if k.startswith("val/")}
            print(f"[val] epoch {trainer.current_epoch} | {_format_metrics(metrics)}", file=sys.stdout, flush=True)


def source(owner, name, module):
    """Clickable ``path:line``: a visnavkit class's definition, else the line in ``owner.__init__`` that builds
    ``name`` (a torch ``Sequential`` / ``Linear`` / ``Parameter`` says more there than in torch's source)."""
    target, line = type(module), None
    if not target.__module__.startswith("visnavkit"):
        target = type(owner)
        lines, start = inspect.getsourcelines(target.__init__)
        line = next((start + i for i, text in enumerate(lines) if f"self.{name}" in text), start)
    return f"{os.path.relpath(inspect.getsourcefile(target))}:{line or inspect.getsourcelines(target)[1]}"


class StageSummary(Callback):
    """One row per top-level stage of the policy (``pl_module.model``): parameters total / trainable / frozen."""

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        model = getattr(pl_module, "model", pl_module)
        rows = []
        for name, module in model.named_children():
            params = list(module.parameters())
            total = sum(p.numel() for p in params)
            trainable = sum(p.numel() for p in params if p.requires_grad)
            rows.append((name, type(module).__name__, total, trainable, total - trainable, source(model, name, module)))
        loose = [p for n, p in model.named_parameters(recurse=False)]  # parameters on the policy itself
        if loose:
            total = sum(p.numel() for p in loose)
            trainable = sum(p.numel() for p in loose if p.requires_grad)
            rows.append(("(own)", type(model).__name__, total, trainable, 0, source(model, "__init__", model)))
        grand = sum(r[2] for r in rows) or 1
        header = f"{'stage':<20} {'type':<22} {'params':>22} {'trainable':>22} {'frozen':>22} {'share':>7} {'fp32':>9}  source"
        lines = [header, "-" * len(header)]
        for name, kind, total, trainable, frozen, src in rows:
            lines.append(
                f"{name:<20} {kind:<22} {count(total):>22} {count(trainable):>22} {count(frozen):>22} "
                f"{100 * total / grand:>6.1f}% {total * 4 / 2**20:>7.1f}MB  {src}"
            )
        trainable = sum(r[3] for r in rows)
        lines += ["-" * len(header), f"{'total':<43} {count(grand):>22} {count(trainable):>22} "
                  f"{count(grand - trainable):>22} {100.0:>6.1f}% {grand * 4 / 2**20:>7.1f}MB"]  # fmt: skip
        print("\n".join(lines), flush=True)


class ConfigPrint(Callback):
    """The resolved run config (``pl_module.cfg``) at fit start."""

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        print(OmegaConf.to_yaml(pl_module.cfg, resolve=True), flush=True)


def _stats(value):
    if not hasattr(value, "shape"):
        return f"{type(value).__name__} x {len(value)}"
    text = f"{str(tuple(value.shape)):<24} {str(value.dtype).removeprefix('torch.'):<8}"
    if value.numel() and (value.is_floating_point() or value.dtype == torch.bool):
        v = value.float()
        text += f" min {v.min().item():9.3f}  max {v.max().item():9.3f}  mean {v.mean().item():9.3f}"
    return text


class BatchSummary(Callback):
    """The first train batch: every key, then ``num_samples`` windows (source window, current-slot values)."""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step or batch_idx:
            return
        width = max(len(k) for k in batch)
        lines = ["first train batch:"] + [f"  {k:<{width}}  {_stats(v)}" for k, v in batch.items()]
        windows = getattr(getattr(trainer.train_dataloader, "dataset", None), "windows", None)
        for i in range(min(self.num_samples, len(batch["vision"]))):
            line = []
            if "index" in batch and windows is not None:
                video_fp, current = windows[int(batch["index"][i])]
                line.append(f"{video_fp} frame {current}")
            for key in ("embodiment_id", "frame_mask", "route_mask"):
                if key in batch:
                    value = batch[key][i]
                    line.append(f"{key} {int(value.sum()) if value.dim() else int(value)}")
            for key in ("ego", "goal"):
                if key in batch:
                    line.append(f"{key}[-1] {[round(x, 3) for x in batch[key][i, -1].tolist()]}")
            if "future_poses" in batch:
                end = (
                    batch["future_poses"][i, -1, -1]
                    if batch["future_poses"].dim() == 4
                    else batch["future_poses"][i, -1]
                )
                line.append(f"plan end {[round(x, 2) for x in end.tolist()]}")
            lines.append(f"  window {i}: " + " | ".join(line))
        print("\n".join(lines), flush=True)


def display_callbacks(display, trainer_kwargs=None):
    """``trainer.display`` -> (callbacks, Trainer kwargs) for the chosen progress bar and model summary styles;
    ``enable_progress_bar`` / ``enable_model_summary`` false in ``trainer_kwargs`` win (style ``none``)."""
    display, trainer_kwargs = display or {}, trainer_kwargs or {}
    bar = display.get("progress_bar", "tqdm") if trainer_kwargs.get("enable_progress_bar", True) else "none"
    summary = display.get("model_summary", "lightning") if trainer_kwargs.get("enable_model_summary", True) else "none"
    every = display.get("refresh_every_n_steps", 1)
    depth = display.get("summary_depth", 2)
    if bar not in PROGRESS_BARS:
        raise ValueError(f"trainer.display.progress_bar must be one of {PROGRESS_BARS}, got {bar!r}")
    if summary not in MODEL_SUMMARIES:
        raise ValueError(f"trainer.display.model_summary must be one of {MODEL_SUMMARIES}, got {summary!r}")
    callbacks, kwargs = [], {}
    if bar == "tqdm":
        callbacks.append(TQDMProgressBar(refresh_rate=every))
    elif bar == "rich":
        from lightning.pytorch.callbacks import RichProgressBar

        callbacks.append(RichProgressBar(refresh_rate=every))
    elif bar == "lines":
        callbacks.append(LineProgressBar(every))
    elif "enable_progress_bar" not in trainer_kwargs:
        kwargs["enable_progress_bar"] = False
    if summary == "lightning":
        callbacks.append(ModelSummary(max_depth=1))
    elif summary == "deep":
        callbacks.append(ModelSummary(max_depth=depth))
    elif summary == "rich":
        from lightning.pytorch.callbacks import RichModelSummary

        callbacks.append(RichModelSummary(max_depth=depth))
    elif summary == "stages":
        callbacks.append(StageSummary())
    if display.get("print_config"):
        callbacks.append(ConfigPrint())
    if display.get("batch_summary"):
        callbacks.append(BatchSummary(int(display["batch_summary"])))
    if summary in ("stages", "none") and "enable_model_summary" not in trainer_kwargs:
        kwargs["enable_model_summary"] = False
    return callbacks, kwargs
