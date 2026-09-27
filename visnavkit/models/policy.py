"""NavigationPolicy: vision + any number of modalities + goals -> context -> action decoder."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from visnavkit.models.action.outputs import parse_plan_output
from visnavkit.models.outputs import PolicyOutput, VisionOutput
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def _detach(v):
    return v.detach() if isinstance(v, torch.Tensor) else v


def _matches(name: str, prefixes: list[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _add_frame_axis(value: torch.Tensor | None) -> torch.Tensor | None:
    """Deployment passes the newest frame's modality input without a frame axis."""
    return value if value is None else value[:, None]


def _as_list(value) -> list:
    """A single module/tensor or any sequence of them (Hydra hands lists back as ListConfig)."""
    if value is None:
        return []
    if isinstance(value, nn.Module) or torch.is_tensor(value):
        return [value]
    return list(value)


class _ExportPolicy(nn.Module):
    """Positional ``predict`` wrapper so absent goal / noise inputs never appear in the graph."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, *inputs):
        kwargs = dict(zip(self.policy.export_input_names(), inputs))
        goal_names = self.policy.goal_input_names()
        goal = [kwargs[name] for name in goal_names] if len(goal_names) > 1 else kwargs.get("goal")
        modalities = {name: kwargs[name] for name in self.policy.modality_input_names if name in kwargs}
        return self.policy.predict(
            kwargs["vision"], kwargs["feature_buffer"], goal=goal, noise=kwargs.get("noise"), **modalities
        )


class NavigationPolicy(nn.Module):
    """Compose the encoders and the decoder; every stage exchanges ``feat_size``-wide tokens.

    The policy takes the three raw inputs and owns the wiring between them:

    - ``vision`` ``(B, F, 3, H, W)`` RGB frames in [0, 1] -> per-frame vision tokens.
    - ``modality_encoders`` is an open ``{name: encoder}`` mapping. Each encoder declares the
      batch keys it reads (``ego``, ``intrinsics``/``extrinsics``, ``depth``, a spatial raster,
      ...) and returns per-frame tokens that are concatenated with the vision tokens of the same
      frame, so the temporal encoder mixes them across time and they travel in the deployment
      feature buffer. Adding a modality needs no change here.
    - ``goal`` one goal per goal encoder -> goal tokens for the action decoder. ``goal_encoder``
      may be a list, in which case ``goal`` is the matching list and the tokens are concatenated.

    Inference: ``act(vision, goal=None, noise=None, **modality_inputs)`` -> trajectories
    ``(B, M, T, P)`` and scores ``(B, M)`` for the newest frame of the window.
    Training: ``forward(...)`` with the same inputs, one decision per frame under
    ``reduction=none``, packed in the flat layout the losses read.
    Deployment: ``predict(frame, feature_buffer, goal=None, noise=None, **modality_inputs)``
    encodes one frame and reuses past frame tokens from the buffer ``(B, history, K * D)``; the
    newest frame's modality inputs are passed without a frame axis.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        temporal_encoder: nn.Module,
        action_decoder: nn.Module,
        goal_encoder: nn.Module | Sequence[nn.Module] | None = None,
        modality_encoders: Mapping[str, nn.Module] | None = None,
        feat_size: int = 256,
        loss_cfg: DictConfig | None = None,
        export_cfg: DictConfig | None = None,
        frozen_modules: list[str] | None = None,
        trainable_modules: list[str] | None = None,
    ):
        super().__init__()
        goal_encoders = _as_list(goal_encoder)
        modalities = {name: encoder for name, encoder in dict(modality_encoders or {}).items() if encoder is not None}
        modules = [("vision_encoder", vision_encoder), ("action_decoder", action_decoder)]
        modules += [(f"goal_encoders.{i}", encoder) for i, encoder in enumerate(goal_encoders)]
        modules += [(f"modality_encoders.{name}", encoder) for name, encoder in modalities.items()]
        for name, module in modules:
            if getattr(module, "feat_size", feat_size) != feat_size:
                raise ValueError(f"{name}.feat_size must equal model.feat_size={feat_size}")
        if temporal_encoder.embed_dim != feat_size:
            raise ValueError(f"temporal_encoder.embed_dim must equal model.feat_size={feat_size}")
        self.vision_encoder = vision_encoder
        self.temporal_encoder = temporal_encoder
        self.goal_encoders = nn.ModuleList(goal_encoders)
        self.modality_encoders = nn.ModuleDict(modalities)
        duplicates = [name for name in self.modality_input_names if self.modality_input_names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Modality encoders must read distinct batch keys; {sorted(set(duplicates))} repeat")
        self.action_decoder = action_decoder
        self.feat_size = feat_size
        self.loss_cfg = loss_cfg

        step, seq_len = export_cfg.seq_step, export_cfg.seq_len
        if step < 1 or seq_len < 1:
            raise ValueError("export_cfg.seq_step and seq_len must be positive")
        self.register_buffer(
            "feature_idxs", torch.arange(-(seq_len - 1) * step, 0, step, dtype=torch.long), persistent=False
        )
        self.export_heads: list[str] = []
        self._trainable_modules = list(trainable_modules or [])
        self._configure_trainable_modules(list(frozen_modules or []))
        if self._trainable_modules:
            self.train(True)

    # ---- freezing --------------------------------------------------------------------------
    def _configure_trainable_modules(self, frozen_modules: list[str]) -> None:
        if self._trainable_modules:
            logger.warning(f"Only {self._trainable_modules} train; frozen_modules={frozen_modules} is ignored")
            for name, p in self.named_parameters():
                p.requires_grad = _matches(name, self._trainable_modules)
            return
        for frozen in frozen_modules:
            module = self.get_submodule(frozen) if frozen in dict(self.named_modules()) else None
            if module is None:
                logger.warning(f"Module {frozen} not found. It will not be frozen.")
                continue
            logger.info(f"Freezing {frozen}")
            for p in module.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._trainable_modules:
            for module in self.children():
                module.eval()
            for name, module in self.named_modules():
                if _matches(name, self._trainable_modules):
                    module.train(True)
        return self

    # ---- token budget ----------------------------------------------------------------------
    @property
    def vision_tokens(self) -> int:
        return self.vision_encoder.num_tokens

    @property
    def modality_tokens(self) -> int:
        return sum(encoder.num_tokens for encoder in self.modality_encoders.values())

    @property
    def modality_input_names(self) -> list[str]:
        """Batch keys read by the modality encoders, in the order the export graph takes them."""
        return [name for encoder in self.modality_encoders.values() for name in encoder.input_names]

    @property
    def goal_tokens(self) -> int:
        return sum(encoder.num_tokens for encoder in self.goal_encoders)

    @property
    def num_tokens(self) -> int:
        """Tokens per frame entering the temporal encoder."""
        return self.vision_tokens + self.modality_tokens

    @property
    def token_dim(self) -> int:
        """Width of one frame in the deployment feature buffer."""
        return self.num_tokens * self.feat_size

    # ---- stages ----------------------------------------------------------------------------
    def encode_frames(self, vision: torch.Tensor) -> VisionOutput:
        b, f = vision.shape[:2]
        return self.vision_encoder(vision.reshape(b * f, *vision.shape[2:]))

    def encode_modalities(self, inputs: Mapping, batch: int, frames: int, image_hw) -> dict[str, torch.Tensor]:
        """``{batch key: value}`` -> ``{modality: (B, F, G, D)}``, skipping zero-token slots."""
        unknown = set(inputs) - set(self.modality_input_names)
        if unknown:
            raise ValueError(f"Unknown policy inputs {sorted(unknown)}; expected {self.modality_input_names}")
        tokens = {}
        for name, encoder in self.modality_encoders.items():
            if encoder.num_tokens == 0:
                continue
            values = tuple(inputs.get(key) for key in encoder.input_names)
            encoded = encoder(*values, batch_size=batch, frames=frames, image_hw=image_hw)
            if encoded.shape[:2] != (batch, frames):
                raise ValueError(f"{name} must cover ({batch}, {frames}) frames, got {tuple(encoded.shape[:2])}")
            tokens[name] = encoded
        return tokens

    def _per_frame_tokens(self, vision_tokens, modality_tokens: Mapping, dim: int):
        extra = list(modality_tokens.values())
        return torch.cat([vision_tokens, *extra], dim=dim) if extra else vision_tokens

    def _encode_goals(self, goals: list, batch: int, observation=None) -> torch.Tensor | None:
        """Per-encoder goals (already shaped for the decision batch) -> concatenated tokens."""
        if self.goal_tokens == 0:
            return None
        tokens = [
            encoder(goal, batch_size=batch, observation=observation)
            for encoder, goal in zip(self.goal_encoders, goals)
            if encoder.num_tokens
        ]
        return torch.cat(tokens, dim=1)

    def null_goal_tokens(self, batch_size: int) -> torch.Tensor | None:
        """Goal tokens for goal-free inference: each encoder's learned null token."""
        return self._encode_goals([None] * len(self.goal_encoders), batch_size)

    def _match_goals(self, goal) -> list:
        goals = [None] * len(self.goal_encoders) if goal is None else _as_list(goal)
        if len(goals) != len(self.goal_encoders):
            raise ValueError(f"{len(self.goal_encoders)} goal encoders expect {len(self.goal_encoders)} goals")
        return goals

    def _window_goals(self, goal, batch: int, frames: int, decisions: int) -> list:
        """Reshape each goal from its dataset layout to one entry per decision."""
        prepared = []
        for encoder, value in zip(self.goal_encoders, self._match_goals(goal)):
            if value is None or encoder.num_tokens == 0:
                prepared.append(None)
            elif encoder.per_frame:
                if value.shape[:2] != (batch, frames):
                    raise ValueError(
                        f"{type(encoder).__name__} expects per-frame goals (B, F, ...), got {tuple(value.shape)}"
                    )
                prepared.append(value[:, -1] if decisions == 1 else value.flatten(0, 1))
            else:
                if value.shape[0] != batch:
                    raise ValueError(
                        f"{type(encoder).__name__} expects one goal per window (B, ...), got {tuple(value.shape)}"
                    )
                prepared.append(value.repeat_interleave(decisions, dim=0) if decisions > 1 else value)
        return prepared

    def encode_window(self, vision: torch.Tensor, inputs: Mapping) -> tuple[VisionOutput, torch.Tensor, dict]:
        """``(B, F, 3, H, W)`` plus modality inputs -> vision output, context ``(B, F', K, D)``, modality tokens."""
        b, f = vision.shape[:2]
        vision_out = self.encode_frames(vision)
        modality_tokens = self.encode_modalities(inputs, b, f, vision.shape[-2:])
        tokens = self._per_frame_tokens(
            vision_out.tokens.reshape(b, f, self.vision_tokens, self.feat_size), modality_tokens, dim=2
        )
        return vision_out, self.temporal_encoder(tokens), modality_tokens

    def act(self, vision: torch.Tensor, goal=None, noise: torch.Tensor | None = None, **inputs):
        """One decision for the newest frame: trajectories ``(B, M, T, P)`` in pose space, scores ``(B, M)``."""
        b, f = vision.shape[:2]
        _, context, _ = self.encode_window(vision, inputs)
        goal_tokens = self._encode_goals(self._window_goals(goal, b, f, 1), b, vision[:, -1])
        parsed = self.action_decoder.parse_output(self.action_decoder(context[:, -1], goal_tokens, noise).plans)
        return parsed["plans"], parsed["confs"]

    def forward(self, vision: torch.Tensor, goal=None, noise: torch.Tensor | None = None, **inputs) -> PolicyOutput:
        b, f = vision.shape[:2]
        vision_out, context, modality_tokens = self.encode_window(vision, inputs)
        decisions = context.shape[1]
        observation = vision.flatten(0, 1) if decisions > 1 else vision[:, -1]
        goal_tokens = self._encode_goals(self._window_goals(goal, b, f, decisions), b * decisions, observation)
        plan = self.action_decoder(context.reshape(b * decisions, self.num_tokens, self.feat_size), goal_tokens, noise)
        return PolicyOutput(
            vision=vision_out, plan=plan, goal_tokens=goal_tokens, modality_tokens=modality_tokens or None
        )

    def predict(self, frame: torch.Tensor, feature_buffer: torch.Tensor, goal=None, noise=None, **inputs):
        """One decision from the newest frame plus buffered past-frame tokens (export graph).

        The newest frame's modality inputs are given without a frame axis, e.g. ``ego (B, E)``,
        ``intrinsics (B, 3, 3)``.
        """
        b = frame.shape[0]
        vision = self.vision_encoder(frame, export_heads=self.export_heads)
        newest = {name: _add_frame_axis(value) for name, value in inputs.items()}
        modality_tokens = self.encode_modalities(newest, b, 1, frame.shape[-2:])
        tokens = self._per_frame_tokens(
            vision.tokens, {name: value[:, 0] for name, value in modality_tokens.items()}, dim=1
        )
        current = tokens.reshape(b, 1, self.token_dim)
        window = torch.cat([feature_buffer[:, self.feature_idxs], current], dim=1)
        context = self.temporal_encoder(window.reshape(b, -1, self.num_tokens, self.feat_size))[:, -1]
        goal_tokens = self._encode_goals(self._match_goals(goal), b, frame)
        plan = self.action_decoder(context, goal_tokens, noise).plans
        heads = tuple(vision.heads[name] for name in self.vision_encoder.get_head_output_names(self.export_heads))
        speed = (vision.speed,) if self.vision_encoder.has_speed_head else ()
        return (plan, current.flatten(1), *speed, *heads)

    # ---- export contract -------------------------------------------------------------------
    def goal_input_names(self) -> list[str]:
        names = [f"goal_{i}" for i, encoder in enumerate(self.goal_encoders) if encoder.num_tokens]
        return ["goal"] if len(names) == 1 else names

    def export_input_names(self) -> list[str]:
        names = ["vision", "feature_buffer", *self.goal_input_names(), *self.modality_input_names]
        if self.action_decoder.uses_noise:
            names.append("noise")
        return names

    def export_output_names(self) -> list[str]:
        names = ["plan", "feat_out"]
        if self.vision_encoder.has_speed_head:
            names.append("speed")
        return [*names, *self.vision_encoder.get_head_output_names(self.export_heads)]

    def example_inputs(self, batch_size: int, image_hw: tuple[int, int], device=None) -> tuple:
        frame = torch.rand(batch_size, 3, *image_hw, device=device)
        history = int(-self.feature_idxs[0]) if len(self.feature_idxs) else 0
        buffer = torch.randn(batch_size, history, self.token_dim, device=device) * 0.1
        inputs = [frame, buffer]
        inputs += [
            encoder.example_input(batch_size, device, image_hw=tuple(image_hw))
            for encoder in self.goal_encoders
            if encoder.num_tokens
        ]
        for encoder in self.modality_encoders.values():
            inputs += [value.squeeze(1) for value in encoder.example_inputs(batch_size, 1, tuple(image_hw), device)]
        if self.action_decoder.uses_noise:
            inputs.append(self.action_decoder.example_noise(batch_size, device))
        return tuple(inputs)

    def export_graph(self, cfg, batch_size=1, export_heads=(), **_):
        """``(wrapper, inputs, input_names, output_names)`` of the deployment graph, on an export copy: the
        newest frame's decision from one frame plus the feature buffer, at the recipe's export resolution."""
        if self.temporal_encoder.reduction == "none":
            self.temporal_encoder.reduction = "last"  # training predicts per frame; deployment wants the newest
        self.export_heads = [name for name in export_heads if name in self.vision_encoder.heads]
        w, h = (int(v // cfg.common.downscale_factor) for v in cfg.common.crop_wh)
        self.vision_encoder = self.vision_encoder.prepare_for_export((h, w))
        inputs = self.example_inputs(batch_size, (h, w), next(self.parameters()).device)
        return _ExportPolicy(self).eval(), inputs, self.export_input_names(), self.export_output_names()

    def decision(self, outputs):
        """The newest decision of NumPy ``outputs`` (by name): a label, its endpoint (x, y) in metres and the
        SANITY CHECK lines."""
        decoder = self.action_decoder
        plan = torch.as_tensor(np.asarray(outputs["plan"])[:1])
        parsed = parse_plan_output(
            plan, num_modes=decoder.num_modes, num_pts=decoder.num_pts, pose_size=decoder.pose_size
        )
        logits, best_plan = plan.reshape(decoder.num_modes, -1)[:, -1].numpy(), parsed["best_plan"][0, :, :2].numpy()
        best = int(np.argmax(logits))
        lines = [f"speed: {float(np.asarray(outputs['speed']).reshape(-1)[0]):.4f}"] if "speed" in outputs else []
        lines += [
            f"logits: {np.round(logits, 3)}",
            f"best_plan p0: {np.round(best_plan[0], 2)}",
            f"best_plan pN: {np.round(best_plan[-1], 2)}",
        ]
        return f"mode {best} logit={logits[best]:.3f}", best_plan[-1].astype(np.float64), lines

    def example_batch(self, batch_size: int, frames: int, image_hw: tuple[int, int], device=None):
        """Synthetic ``(vision, goal, {modality key: value})`` inputs for shape checks."""
        vision = torch.rand(batch_size, frames, 3, *image_hw, device=device)
        goals = []
        for encoder in self.goal_encoders:
            if encoder.num_tokens == 0:
                goals.append(None)
                continue
            count = batch_size * frames if encoder.per_frame else batch_size
            goal = encoder.example_input(count, device, image_hw=tuple(image_hw))
            goals.append(goal.reshape(batch_size, frames, -1) if encoder.per_frame else goal)
        if len(self.goal_encoders) > 1:
            goal = goals
        else:
            goal = goals[0] if goals else None
        modalities = {}
        for encoder in self.modality_encoders.values():
            values = encoder.example_inputs(batch_size, frames, tuple(image_hw), device)
            modalities.update(dict(zip(encoder.input_names, values)))
        return vision, goal, modalities

    # ---- losses ---------------------------------------------------------------------------
    def get_losses(self, preds: PolicyOutput, targets):
        vision_loss_dict = self.vision_encoder.get_losses(preds.vision, targets.get("vision", {}))
        action_loss_dict, action_loss_debug = self.action_decoder.get_losses(preds.plan, targets["action"])
        total_loss = self.loss_cfg.action_weight * action_loss_dict["total"]
        if vision_loss_dict:
            total_loss = total_loss + self.loss_cfg.vision_weight * vision_loss_dict["total"]
        loss_dict = dict(loss=total_loss)
        loss_dict.update({f"vision_{k}": _detach(v) for k, v in vision_loss_dict.items()})
        loss_dict.update({f"action_{k}": _detach(v) for k, v in action_loss_dict.items()})
        return loss_dict, dict(action_loss_debug=action_loss_debug)
