---
name: downstream-impl
description: Implement downstream features in visnavkit (new components, datasets, experiments, export consumers). Use when adding or modifying components in this repo so contracts and verification steps are honored.
---

# Downstream implementation in visnavkit

## Ground rules
- Smallest viable diff; reuse the stage bases (`BaseVisionEncoder`, `BaseTemporalEncoder`, `BaseGoalEncoder`, `BaseActionDecoder`) over new abstractions.
- `uv run` for everything. Training records Git provenance; `strict_git=true` requires a clean tree (untracked files count).
- Configs live under `visnavkit/configs/` only (shipped as `visnavkit.configs` package-data).

## Contracts (do not break)

### Batch (both dataloaders emit exactly this)
| key | shape | dtype |
|---|---|---|
| `vision` | (B, S, 3, h, w) — RGB frames | uint8 |
| `future_poses` | (B, S, plan_len_points, 3) — x, y, v | float32 |
| `frame_speeds` | (B, S, 1) | float32 |
| `frame_times_s` | (B, S) | float64 |
| `target_times_s` | (B, T) relative seconds | float32 |
| `goal` (optional) | point (B, S, 3) · gps (B, S, 2) · image (B, 3, h, w) uint8 · route_image (B, C, h, w) · instruction (B, E); a list when the recipe has several goal encoders | — |
| `ego` (optional) | (B, S, E) — `common.ego_features` | float32 |
| `intrinsics` / `extrinsics` (optional) | (B, S, 3, 3) / (B, S, 4, 4) — `common.use_camera` | float32 |
| any other modality key | (B, S, ...) — whatever its encoder's `input_names` declare | — |

`dataset=torch` (torchcodec, CPU) supports every goal type, goal lists, `common.ego_features` and `common.use_camera`; `dataset=dali` (GPU) supports `none`/`point` goals and speed-only ego. The dataset reads `goal_type` from `${model.goal_encoder.goal_type}` (set it explicitly when the recipe uses a list of goal encoders). Pose and goal targets come from `visnavkit/data/pose_targets.py` — reuse, never reimplement.

### Layout
`models/vision` · `models/temporal` · `models/goal` · `models/action` (+ `denoisers/`, `schedulers/`) · `models/policy.py` (NavigationPolicy) · `models/lit_model.py` · `data/` · `evaluation/` · `export/` (ONNX, TensorRT, artifact check) · `benchmark/` · `scripts/` (thin entry points) · `configs/model/<group>/` mirrors the packages.

### Policy
- Training: `policy(vision, goal=None, noise=None, **modality_inputs)` with vision (B, S, 3, h, w) float in [0,1] and one keyword per modality batch key (ego (B, S, E), intrinsics (B, S, 3, 3), extrinsics (B, S, 4, 4), ...) → `PolicyOutput(vision=VisionOutput(tokens (B*S, Kv, D), speed), plan=PlanOutput(plans (decisions, M*(2*T*P+1))), goal_tokens, modality_tokens)`. Every non-vision input is optional and falls back to its encoder's null token.
- Deployment: `policy.predict(frame, feature_buffer, goal=None, noise=None, **modality_inputs)` — newest-frame modality inputs carry no frame axis — → `(plan, feat_out, [speed], *heads)`; `export_input_names()` / `export_output_names()` define the ONNX contract and are presence-driven. A new input modality is a `BaseModalityEncoder` subclass declaring `input_names`; `policy.py` needs no change.
- Losses: `policy.get_losses(out, targets)` combines `vision_encoder.get_losses` and `action_decoder.get_losses` with `loss_cfg` weights.

### Configs
- Experiments: self-contained `# @package _global_` files in `configs/experiment/`; never edit recipe files for an experiment.
- Component groups are nested: `model/vision_encoder=...`, `model/temporal_encoder=...`, `model/goal_encoder=...`, `model/action_decoder=...` (+ `model/action_decoder/denoiser=...`, `.../scheduler=...`). Interpolate widths from `${model.feat_size}`.

## Verify before claiming done (in order)
```bash
uv run visnavkit-sanity-check --onnx <overrides>          # shapes, loss/backward, buffer parity, ONNX parity
uv run pytest tests/ -q
uv run ruff check visnavkit tests
uv run python -m visnavkit.scripts.benchmark_dataloader --batches 20 common.data_root=<root>  # if loaders touched
uv run visnavkit-export checkpoint=<ckpt> output=/tmp/test.onnx                             # if model/export touched
```
Report real shapes/losses from these runs. Then commit, push, and give the exact train command:
`uv run visnavkit-train experiment=<name> dataset=<dali|torch>`.
