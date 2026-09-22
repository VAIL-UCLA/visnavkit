"""Per-corpus action bounds and k-means anchors: the normalised action space FlowPilot-DST decodes in.

A ``dataset=pose`` window's action is its current frame's future ``[x, y, yaw, v, w]``, read as
per-step ``[dx, dy, dyaw, v, w]``. Every corpus — a clip's first directory under ``data_root`` —
gets its own bounds (p0.5 / p99.5 per channel; dy, dyaw and w symmetric, so a horizontal flip stays
a mirror), and ``(a - lo) / (hi - lo)`` puts every embodiment on one ~[0, 1] scale. The anchors are
the k-means centres of the normalised dx, dy over all corpora (unclipped), every window
joined by its mirror so the vocabulary is left / right balanced.
"""

import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset

from visnavkit.data.pose_dataset import corpus_of
from visnavkit.scripts.dataset.anchors import kmeans
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["action_steps", "actions_dir", "cache_actions", "fit_action_anchors", "fit_bounds", "load_actions"]

CHANNELS = ("dx", "dy", "dyaw", "v", "w")
MIRRORED = (1, 2, 4)  # the channels a horizontal flip negates


def actions_dir(output_dir, split: str) -> Path:
    return Path(output_dir) / f"actions_{split}"


class _CurrentActions(Dataset):
    """Each window's current-frame action ``(T, 5)``: all the statistics read."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]["future_poses"][-1]


def cache_actions(cfg, split: str = "train", output_dir="outputs/dataset") -> Path:
    """Write ``<corpus>.npy`` (N, T, 5) [x, y, yaw, v, w] for every window of a split; no frame is decoded."""
    key = f"{split}_loader"
    if cfg.get("dataset") is None or key not in cfg.dataset or not cfg.dataset._target_.endswith("PoseDataModule"):
        raise ValueError(f"The action commands read dataset=pose windows; the config has no such dataset.{key}")
    dataset = instantiate(
        cfg.dataset[key],
        _target_="visnavkit.data.pose_dataset.PoseWindowDataset",
        _convert_="all",
        pose_size=5,
        frames=False,
        route_hw=None,
        goal_type="none",
        ego_features=[],
        embodiment_ids=None,
        action_bounds=None,
        p_hflip=0.0,
    )
    corpora = np.asarray([corpus_of(video_fp, dataset.data_root) or "" for video_fp, _ in dataset.windows])
    if not len(corpora) or "" in corpora:
        raise ValueError(f"Split {split!r} needs windows of clips inside corpus directories under {dataset.data_root}")

    loader = DataLoader(_CurrentActions(dataset), batch_size=1024, num_workers=cfg.dataset.num_workers)
    chunks = []
    for step, batch in enumerate(loader):
        chunks.append(batch)
        if step % 200 == 0:
            logger.info(f"Cached {sum(map(len, chunks))} / {len(dataset)} windows")
    actions = torch.cat(chunks).numpy()

    directory = actions_dir(output_dir, split)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.npy"):  # a corpus that left the split must leave the statistics too
        stale.unlink()
    for corpus in np.unique(corpora):
        np.save(directory / f"{corpus}.npy", actions[corpora == corpus])
    logger.info(f"Cached {len(actions)} windows of {len(np.unique(corpora))} corpora to {directory}")
    return directory


def load_actions(output_dir, split: str = "train") -> dict:
    paths = sorted(actions_dir(output_dir, split).glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No cached actions in {actions_dir(output_dir, split)}; run command=action_cache")
    return {path.stem: np.load(path, mmap_mode="r") for path in paths}


def action_steps(actions) -> np.ndarray:
    """``(N, T, 5)`` [x, y, yaw, v, w] -> per-step [dx, dy, dyaw, v, w], differences from the origin on."""
    delta = np.diff(actions[..., :3], axis=1, prepend=0)
    delta[..., 2] = (delta[..., 2] + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate([delta, actions[..., 3:]], axis=-1).astype(np.float32)


def fit_bounds(steps, percentiles=(0.5, 99.5)) -> np.ndarray:
    """Per-step actions ``(N, T, 5)`` -> ``(2, 5)`` [lo, hi]; mirrored channels symmetric, no empty range."""
    lo, hi = np.percentile(steps.reshape(-1, 5), percentiles, axis=0)
    for channel in MIRRORED:
        hi[channel] = max(abs(lo[channel]), abs(hi[channel]))
        lo[channel] = -hi[channel]
    flat = hi - lo < 1e-6  # a constant channel (a corpus that never turns) still needs a range to divide by
    return np.stack([np.where(flat, lo - 5e-7, lo), np.where(flat, hi + 5e-7, hi)]).astype(np.float32)


def fit_action_anchors(
    output_dir="outputs/dataset", split: str = "train", num_anchors: int = 64, per_corpus: int = 20000, seed: int = 42
) -> Path:
    """Cached actions -> ``action_bounds.json`` {corpus: [[lo x 5], [hi x 5]]} (what the pose dataset's
    ``action_bounds`` reads), ``kmeans<K>.npy`` (K, T, 2) normalised dx, dy (``model.head.anchors_path``)
    and one ``kmeans<K>_<corpus>.png`` per corpus.

    The bounds see every window; the k-means sees at most ``per_corpus`` of each corpus, so a large
    corpus does not own the vocabulary.
    """
    output_dir, rng = Path(output_dir), np.random.default_rng(seed)
    bounds, samples = {}, {}
    for corpus, actions in load_actions(output_dir, split).items():
        steps = action_steps(np.asarray(actions))
        bounds[corpus] = fit_bounds(steps)
        samples[corpus] = steps[np.sort(rng.choice(len(steps), min(per_corpus, len(steps)), replace=False))]
        ranges = "  ".join(f"{c} [{lo:+.4f}, {hi:+.4f}]" for c, lo, hi in zip(CHANNELS, *bounds[corpus]))
        logger.info(f"{corpus}: {len(steps)} windows, {len(samples[corpus])} for the k-means  {ranges}")
    (output_dir / "action_bounds.json").write_text(json.dumps({c: b.tolist() for c, b in bounds.items()}, indent=1))

    unit = {c: (samples[c][..., :2] - bounds[c][0, :2]) / (bounds[c][1, :2] - bounds[c][0, :2]) for c in samples}
    points = torch.from_numpy(np.concatenate(list(unit.values())))
    points = torch.cat([points, points * torch.tensor([1.0, -1.0]) + torch.tensor([0.0, 1.0])])  # + the mirrors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    centres, assignment = kmeans(points.flatten(1).to(device), num_anchors, iters=300, seed=seed)
    anchors = centres.reshape(num_anchors, -1, 2).cpu().numpy().astype(np.float32)
    path = output_dir / f"kmeans{num_anchors}.npy"
    np.save(path, anchors)
    logger.info(f"Fitted {num_anchors} anchors over {len(points)} windows (half mirrored) to {path}")

    start = 0
    for corpus, steps in samples.items():  # the originals come first, corpus by corpus
        labels = assignment[start : start + len(steps)].cpu().numpy()
        start += len(steps)
        figure_path = output_dir / f"kmeans{num_anchors}_{corpus}.png"
        ade, fde = plot_anchors(corpus, steps, bounds[corpus], anchors, labels, figure_path)
        logger.info(f"{corpus}: minADE {ade:.3f} m, minFDE {fde:.3f} m to the assigned anchor -> {figure_path}")
    return path


def plot_anchors(corpus, steps, bounds, anchors, labels, path):
    """The corpus' view of the vocabulary: the anchors in its metres, coloured by their share of its windows.

    Top: endpoint density with the anchor endpoints | paths (grey: 300 windows) | dx, dy per step.
    Bottom: every per-step channel with its bounds. Returns the mean and final distance of the
    windows to their assigned anchor.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the optional extra
        raise ImportError("Plotting needs matplotlib: uv sync --extra plot") from error

    metric = anchors * (bounds[1, :2] - bounds[0, :2]) + bounds[0, :2]  # (K, T, 2) dx, dy in this corpus' metres
    centre_paths, paths = metric.cumsum(1), steps[..., :2].cumsum(1)
    share = np.bincount(labels, minlength=len(anchors)) / len(steps)
    distance = np.linalg.norm(paths - centre_paths[labels], axis=-1)  # (N, T)
    ade, fde = float(distance.mean()), float(distance[:, -1].mean())

    figure = plt.figure(figsize=(20, 10), layout="constrained")
    top, bottom = figure.add_gridspec(2, 1, height_ratios=[2, 1])
    top = top.subgridspec(2, 4, width_ratios=[5, 5, 0.15, 4])
    ends, bev, bar = (figure.add_subplot(top[:, column]) for column in range(3))
    per_step = [figure.add_subplot(top[row, 3]) for row in range(2)]
    cmap, top = plt.get_cmap("viridis"), max(share.max(), 1e-6)
    reach = max(1.0, float(np.abs(centre_paths).max()) * 1.1)
    inside = (np.abs(paths[:, -1, 1]) < reach) & (paths[:, -1, 0] > -reach / 4) & (paths[:, -1, 0] < reach)
    ends.hexbin(-paths[inside, -1, 1], paths[inside, -1, 0], gridsize=60, bins="log", mincnt=1, cmap="Greys")
    ends.scatter(-centre_paths[:, -1, 1], centre_paths[:, -1, 0], c=share, cmap=cmap, vmin=0, vmax=top, s=26, ec="w")
    ends.set_title(f"endpoints of {len(steps)} windows + anchors")
    for index in np.random.default_rng(0).choice(len(paths), min(300, len(paths)), replace=False):
        bev.plot(-paths[index, :, 1], paths[index, :, 0], color="0.75", lw=0.5, alpha=0.4)
    for k in np.argsort(share):  # the most used anchors on top
        bev.plot(-centre_paths[k, :, 1], centre_paths[k, :, 0], color=cmap(share[k] / top), lw=1.6, alpha=0.9)
        for axis, channel in zip(per_step, (0, 1)):
            axis.plot(np.arange(1, metric.shape[1] + 1), metric[k, :, channel], color=cmap(share[k] / top), lw=1.0)
    bev.set_title("anchor paths (cumulative dx, dy)")
    for axis in (ends, bev):
        axis.plot(0, 0, "r^", ms=7)
        axis.set_xlim(-reach, reach), axis.set_ylim(-reach / 4, reach)
        axis.set_aspect("equal", adjustable="box"), axis.grid(alpha=0.3)
        axis.set_xlabel("left [m]"), axis.set_ylabel("forward [m]")
    for axis, channel in zip(per_step, (0, 1)):
        axis.set_ylabel(f"{CHANNELS[channel]} [m / step]"), axis.grid(alpha=0.3)
    per_step[1].set_xlabel("step")
    colours = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100 * top))
    figure.colorbar(colours, cax=bar, label=f"share of {corpus} windows [%]")
    bottom = bottom.subgridspec(1, len(CHANNELS))
    for channel, name in enumerate(CHANNELS):
        axis = figure.add_subplot(bottom[channel])
        axis.hist(steps[..., channel].reshape(-1), bins=100, color="#4477aa")
        for bound in bounds[:, channel]:
            axis.axvline(bound, color="#aa3377", ls="--", lw=1)
        axis.set_yscale("log"), axis.grid(alpha=0.3)
        axis.set_title(f"{name} per step, bounds [{bounds[0, channel]:+.3f}, {bounds[1, channel]:+.3f}]", fontsize=9)
    used = int((share > 0).sum())
    figure.suptitle(f"{corpus}: K={len(anchors)}, used {used}, minADE {ade:.2f} m, minFDE {fde:.2f} m (n={len(steps)})")
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return ade, fde
