"""Windows on a fixed slot grid: the past second of ego state (and frames) plus the future trajectory."""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as TF

from visnavkit.data.file_list import parse_file_list_frame_ranges, resolve_path
from visnavkit.data.pose_targets import (
    load_pose_arrays,
    point_goal_from_local,
    sample_goal_frame,
    target_safe_frame_ranges,
)
from visnavkit.data.torch_datamodule import TorchDataModule
from visnavkit.utils.common import anchor_times
from visnavkit.utils.orientation import yaw_from_quat

GOAL_TYPES = ("none", "point", "gps")
EGO_FEATURES = {"past_xy": 2, "yaw": 1, "speed": 1, "yaw_rate": 1}  # name -> channels
POSE_SIZES = {2: "x, y", 3: "x, y, v", 5: "x, y, yaw, v, w"}
CAMERA = "camera.json"  # {"camera": [fx, fy, cx, cy, k1..k4, cam_height_m, cam_type], "width", "height", "principal_point_delta"}
ROUTE_LABELS = "route_labels"  # .npy (memory-mapped) or .npz (compressed, key `labels`): (N, h, w) uint8 class ids


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _local(yaw, dx, dy):
    """World offsets -> the frame of a pose with heading ``yaw`` (x forward, y left); broadcasts."""
    c, s = np.cos(yaw), np.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def corpus_of(video_fp, data_root):
    """A clip's corpus: the first directory of its path under ``data_root``, None outside of one."""
    parts = Path(os.path.relpath(Path(video_fp).parent, data_root)).parts
    return parts[0] if parts and parts[0] != ".." else None


def load_route_labels(clip):
    """A clip's route class ids ``(N, h, w)``: ``route_labels.npy`` or the compressed ``route_labels.npz``; None without."""
    if (clip / f"{ROUTE_LABELS}.npy").exists():
        return np.load(clip / f"{ROUTE_LABELS}.npy", mmap_mode="r")
    if (clip / f"{ROUTE_LABELS}.npz").exists():
        with np.load(clip / f"{ROUTE_LABELS}.npz") as archive:
            return archive["labels"]
    return None


def load_camera(clip, frame_wh):
    """The clip's camera sidecar at ``frame_wh``: the camera vector ``(10,)`` and the principal-point offset ``(2,)`` px."""
    meta = json.loads((clip / CAMERA).read_text())
    sx, sy = frame_wh[0] / meta["width"], frame_wh[1] / meta["height"]
    camera = np.asarray(meta["camera"], dtype=np.float32)
    camera[[0, 2]] *= sx
    camera[[1, 3]] *= sy
    delta = np.asarray(meta.get("principal_point_delta", (0.0, 0.0)), dtype=np.float32) * [sx, sy]
    return camera, delta


def _shift(x, k, axis):
    """``out[j] = x[j + k]`` along ``axis``, zeros past the edge."""
    out = torch.zeros_like(x)
    n = x.shape[axis]
    if abs(k) < n:
        src = x.narrow(axis, max(k, 0), n - abs(k))
        out.narrow(axis, max(-k, 0), n - abs(k)).copy_(src)
    return out


def translate(frames, dx, dy):
    """``(N, C, H, W)`` uint8 -> ``out(x, y) = in(x + dx, y + dy)``, bilinear, zeros outside (the reference's warp)."""
    x = frames.float()
    for axis, d in ((-1, float(dx)), (-2, float(dy))):
        i = math.floor(d)
        a = d - i
        x = (1 - a) * _shift(x, i, axis) + a * _shift(x, i + 1, axis) if a else _shift(x, i, axis)
    return x.round_().clamp_(0, 255).to(torch.uint8)


def slot_frames(times, slot_times, hz):
    """Per slot the nearest source frame and whether it lies within half a slot of the slot time."""
    j = np.clip(np.searchsorted(times, slot_times), 1, len(times) - 1)
    j = np.where(np.abs(times[j - 1] - slot_times) <= np.abs(times[j] - slot_times), j - 1, j)
    return j, np.abs(times[j] - slot_times) <= 0.5 / hz + 1e-6


class PoseWindowDataset(Dataset):
    """Mp4WindowDataset's manifests, targets and goals on a fixed slot grid, frames optional.

    One sample per current frame, every ``stride_s`` seconds inside a manifest row's ``[start, end)``.
    The observed window is ``seq_len`` slots ``1 / hz`` s apart ending at the current frame (20 slots at
    20 Hz = the past second); the ego state is linearly interpolated from the pose sidecars, so every
    source frame rate yields the same window; the heading is the quaternion's yaw. A window needs its
    past inside the row and ``plan_len_seconds`` of future inside the clip. Keys (S = seq_len, T = anchors):

    - ``ego`` (S, E): the ``ego_features`` columns in the current frame's ego frame — ``past_xy`` (x, y),
      ``yaw`` (heading relative to the current one), ``speed``, ``yaw_rate``.
    - ``future_poses`` (S, T, pose_size) per slot in that slot's frame: ``x, y`` | ``x, y, v`` |
      ``x, y, yaw, v, w``; ``target_times_s`` (T,), ``frame_times_s`` (S,), ``frame_speeds`` (S, 1) as
      Mp4WindowDataset gives them; ``past_poses`` (S, 3) [x, y, yaw] of every slot in the current frame.
    - ``goal``: ``point`` (S, 3) distance (m) / cos / sin or ``gps`` (S, 2) to a frame ``goal_horizon_s``
      after the current one, in each slot's ego frame. Image goals need frames and are not offered.
    - ``frames``: ``vision`` (S, 3, h, w) uint8, each slot the source frame nearest to its time when
      that lies within half a slot (all slots at 20 fps, 1 in 4 at 5 fps), zeros otherwise, and
      ``frame_mask`` (S,) True where the slot holds a frame; ``frame_wh`` resizes the decoded frames.
    - ``route_hw``: ``route_patch`` (S, h, w) float class ids from the clip's ``route_labels.npy``
      (N, h, w) uint8 sidecar (or ``route_labels.npz``, key ``labels``: the compressed form a large corpus
      needs) at the slot's frame, and ``route_mask`` (S,) True where it is real
      (zeros + False without the sidecar or the frame).
    - ``embodiment_ids`` {corpus: id}: ``embodiment_id`` of the clip's corpus, the first directory of
      the clip path under ``data_root``; ``action_bounds`` JSON {corpus: [[lo x 5], [hi x 5]]} gives
      ``action_bounds`` (2, 5), the per-step [dx, dy, dyaw, v, w] range of that embodiment.

    ``camera``: ``camera`` (10,) [fx, fy, cx, cy, k1, k2, k3, k4, cam_height_m, cam_type] at the frame size
    from the clip's ``camera.json`` (cam_type 0 pinhole, 1 Kannala-Brandt fisheye, 2 unknown).
    ``principal_point_calibration``: a clip whose sidecar carries a ``principal_point_delta`` (clips1k's
    per-clip SLAM offset) has its frames shifted by it, so the principal point lands at the nominal cx, cy
    (the reference's principal-point calibration: deterministic, train and val alike); without it the
    offset moves ``camera``'s cx, cy instead.
    ``p_hflip`` mirrors frames, route patches, y, yaw, yaw rate and the goal; clips whose path contains
    a ``no_flip`` substring are never mirrored (driving corpora keep their side of the road).
    ``shuffle`` belongs to the loader.
    """

    def __init__(
        self,
        file_list,
        data_root,
        seq_len=20,
        hz=20,
        stride_s=1.0,
        plan_len_seconds=4,
        plan_len_points=80,
        offset_t_anchors=False,
        uniform_t_anchors=True,
        pose_size=3,
        goal_type="none",
        goal_horizon_s=(3.0, 15.0),
        ego_features=("past_xy", "yaw", "speed", "yaw_rate"),
        frames=False,
        frame_wh=None,
        route_hw=None,
        embodiment_ids=None,
        action_bounds=None,
        camera=False,
        principal_point_calibration=False,
        p_hflip=0.0,
        no_flip=(),
        shuffle=True,
    ):
        self.data_root = Path(data_root)
        self.file_list = resolve_path(file_list, self.data_root)
        if not isinstance(seq_len, int) or seq_len < 1 or hz <= 0 or stride_s <= 0:
            raise ValueError("seq_len must be a positive integer, hz and stride_s positive")
        if plan_len_points < 2 or not np.isfinite(plan_len_seconds) or plan_len_seconds <= 0:
            raise ValueError("plan_len_points must be >= 2 and plan_len_seconds must be positive and finite")
        if pose_size not in POSE_SIZES:
            raise ValueError(f"pose_size must be one of {tuple(POSE_SIZES)}, got {pose_size!r}")
        if goal_type not in GOAL_TYPES:
            raise ValueError(f"goal_type must be one of {GOAL_TYPES}, got {goal_type!r}")
        self.ego_features = tuple(ego_features or ())
        for name in self.ego_features:
            if name not in EGO_FEATURES:
                raise ValueError(f"ego_features must be from {tuple(EGO_FEATURES)}, got {name!r}")
        self.seq_len, self.hz, self.stride_s = seq_len, float(hz), float(stride_s)
        self.t_anchors = anchor_times(plan_len_seconds, plan_len_points, offset_t_anchors, uniform_t_anchors)
        self.num_pts = plan_len_points
        self.pose_size = pose_size
        self.goal_type = goal_type
        self.goal_horizon_s = tuple(float(v) for v in goal_horizon_s)
        self.frames = bool(frames)
        self.frame_wh = tuple(int(v) for v in frame_wh) if frame_wh else None
        self.route_hw = tuple(int(v) for v in route_hw) if route_hw else None
        self.embodiment_ids = dict(embodiment_ids) if embodiment_ids else None
        self.bounds = None
        if action_bounds is not None:
            self.bounds = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in json.loads(Path(resolve_path(action_bounds, self.data_root)).read_text()).items()
            }
            if any(v.shape != (2, 5) for v in self.bounds.values()):
                raise ValueError(f"{action_bounds} must map each corpus to [[lo x 5], [hi x 5]]")
        self.camera, self.calibrate = bool(camera), bool(principal_point_calibration)
        if (self.camera or self.calibrate) and self.frame_wh is None:
            raise ValueError("camera / principal_point_calibration need frame_wh (the camera is scaled to it)")
        self.p_hflip = p_hflip
        self.no_flip = tuple(no_flip or ())
        if self.frames:
            from torchcodec.decoders import VideoDecoder  # only the frame path needs a decoder

            self._decoder = VideoDecoder

        past_s = (seq_len - 1) / self.hz
        rows = parse_file_list_frame_ranges(self.file_list, data_root=self.data_root)
        self.windows: list[tuple[str, int]] = []  # (video path, current frame)
        self.corpora: dict[str, str] = {}  # video path -> corpus, for embodiment ids and action bounds
        for video_fp, _, start, end in target_safe_frame_ranges(rows, float(self.t_anchors[-1])):
            *_, times = load_pose_arrays(Path(video_fp).parent)
            first = start + int(np.searchsorted(times[start:] - times[start], past_s - 1e-6))
            stride = max(1, round(self.stride_s * (len(times) - 1) / (times[-1] - times[0])))
            self.windows += [(video_fp, current) for current in range(first, end, stride)]
            if self.embodiment_ids is not None or self.bounds is not None:
                self.corpora[video_fp] = self._corpus(video_fp)

    def _corpus(self, video_fp):
        corpus = corpus_of(video_fp, self.data_root)
        for table, name in ((self.embodiment_ids, "embodiment_ids"), (self.bounds, "action_bounds")):
            if table is not None and corpus not in table:
                raise ValueError(
                    f"{video_fp}: corpus {corpus!r} (its first directory under data_root) is not in {name}"
                )
        return corpus

    def __len__(self):
        return len(self.windows)

    def _decode(self, video_fp, frame_idxs, mask):
        """The masked slots' frames -> ``vision`` (S, 3, h, w) uint8, zeros where the slot holds none."""
        unique = sorted(set(frame_idxs[mask].tolist()))
        decoded = self._decoder(video_fp, device="cpu", dimension_order="NCHW").get_frames_at(indices=unique).data
        if self.frame_wh is not None:
            decoded = TF.resize(decoded, [self.frame_wh[1], self.frame_wh[0]], antialias=True)
        vision = torch.zeros(self.seq_len, *decoded.shape[1:], dtype=torch.uint8)
        position = {frame: i for i, frame in enumerate(unique)}
        vision[np.flatnonzero(mask)] = decoded[[position[frame] for frame in frame_idxs[mask]]]
        return vision

    def __getitem__(self, idx):
        video_fp, current = self.windows[idx]
        clip = Path(video_fp).parent
        positions, orientations, speeds, times = load_pose_arrays(clip)
        x, y = positions[:, 0], positions[:, 1]
        yaw = np.unwrap(yaw_from_quat(np.asarray(orientations)))
        yaw_rate = np.gradient(yaw, times)

        slot_times = times[current] + (np.arange(self.seq_len) - (self.seq_len - 1)) / self.hz
        xs, ys, yaws, vs, ws = (np.interp(slot_times, times, column) for column in (x, y, yaw, speeds, yaw_rate))
        past_xy = np.column_stack(_local(yaw[current], xs - x[current], ys - y[current]))
        heading = _wrap(yaws - yaw[current])

        anchors = slot_times[:, None] + self.t_anchors[None, :]  # (S, T)
        fx, fy, fyaw, fv, fw = (
            np.interp(anchors.ravel(), times, column).reshape(anchors.shape) for column in (x, y, yaw, speeds, yaw_rate)
        )
        lx, ly = _local(yaws[:, None], fx - xs[:, None], fy - ys[:, None])
        columns = {2: [lx, ly], 3: [lx, ly, fv], 5: [lx, ly, _wrap(fyaw - yaws[:, None]), fv, fw]}[self.pose_size]
        future = np.stack(columns, axis=-1).astype(np.float32)  # (S, T, pose_size)

        goal = None
        if self.goal_type != "none":
            goal_idx = sample_goal_frame(times, current, self.goal_horizon_s, f"{video_fp}:{current}")
            goal = np.column_stack(_local(yaws, x[goal_idx] - xs, y[goal_idx] - ys)).astype(np.float32)

        vision = route = None
        if self.frames or self.route_hw is not None:
            frame_idxs, mask = slot_frames(times, slot_times, self.hz)
        camera = delta = None
        if self.camera or self.calibrate:
            wh = self.frame_wh
            camera, delta = load_camera(clip, wh)
        if self.frames:
            vision = self._decode(video_fp, frame_idxs, mask)
            if self.calibrate and delta.any():
                vision[mask] = translate(vision[mask], *delta)
        if camera is not None and not (self.frames and self.calibrate):
            camera[2:4] += delta  # the image keeps its offset: the principal point sits there
        if self.route_hw is not None:
            route, route_mask = np.zeros((self.seq_len, *self.route_hw), np.float32), np.zeros(self.seq_len, bool)
            if (labels := load_route_labels(clip)) is not None:
                if labels.shape[1:] != self.route_hw:
                    raise ValueError(f"{clip / ROUTE_LABELS} must be (N, {self.route_hw[0]}, {self.route_hw[1]})")
                route[mask], route_mask = labels[frame_idxs[mask]], mask.copy()

        flip = self.p_hflip > 0 and not any(s in video_fp for s in self.no_flip) and bool(torch.rand(1) < self.p_hflip)
        if flip:
            past_xy[:, 1] *= -1.0
            heading *= -1.0
            ws = -ws
            future[..., 1] *= -1.0
            if self.pose_size == 5:
                future[..., [2, 4]] *= -1.0
            if goal is not None:
                goal[:, 1] *= -1.0
            if vision is not None:
                vision = torch.flip(vision, dims=[-1])
            if route is not None:
                route = np.ascontiguousarray(route[..., ::-1])
            if camera is not None:
                camera[2] = wh[0] - camera[2]  # mirroring the image mirrors cx
        if goal is not None and self.goal_type == "point":
            goal = point_goal_from_local(goal)

        ego = {"past_xy": past_xy, "yaw": heading[:, None], "speed": vs[:, None], "yaw_rate": ws[:, None]}
        sample = dict(
            index=torch.tensor(idx),  # -> self.windows[idx], the source (video, current frame)
            frame_times_s=torch.tensor(slot_times, dtype=torch.float64),
            future_poses=torch.from_numpy(future),
            target_times_s=torch.tensor(self.t_anchors, dtype=torch.float32),
            frame_speeds=torch.from_numpy(vs.astype(np.float32)).reshape(-1, 1),
            past_poses=torch.from_numpy(np.column_stack([past_xy, heading]).astype(np.float32)),
        )
        if camera is not None:
            sample["camera"] = torch.from_numpy(camera)
        if self.ego_features:
            sample["ego"] = torch.from_numpy(np.concatenate([ego[n] for n in self.ego_features], -1).astype(np.float32))
        if goal is not None:
            sample["goal"] = torch.from_numpy(goal)
        if vision is not None:
            sample["vision"], sample["frame_mask"] = vision, torch.from_numpy(mask)
        if route is not None:
            sample["route_patch"], sample["route_mask"] = torch.from_numpy(route), torch.from_numpy(route_mask)
        if self.embodiment_ids is not None:
            sample["embodiment_id"] = torch.tensor(self.embodiment_ids[self.corpora[video_fp]], dtype=torch.long)
        if self.bounds is not None:
            sample["action_bounds"] = torch.from_numpy(self.bounds[self.corpora[video_fp]])
        return sample


class PoseDataModule(TorchDataModule):
    dataset_cls = PoseWindowDataset
