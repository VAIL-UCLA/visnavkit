# Architecture

A policy takes vision, any number of extra input *modalities* and any number of goals, connected
by tokens of width `feat_size`. Hydra instantiates each stage from its own config group;
`NavigationPolicy` owns the wiring, `LitModel` the optimization and metrics. The policy knows
vision by name and everything else through the modality and goal contracts, so the stage set is
open. The decomposition follows [diffusers](https://github.com/huggingface/diffusers) (typed
outputs, denoiser and scheduler as separate objects) and
[LeRobot](https://github.com/huggingface/lerobot) (one training forward, one deployment predict).

```text
visnavkit/models/
├── policy.py            NavigationPolicy: forward (training) / predict (deployment)
├── outputs.py           VisionOutput, PlanOutput, PolicyOutput
├── lit_model.py         Lightning module: targets, losses, metrics, optimizer
├── normalization.py     Normalizer shared by supervision targets and input signals
├── vision/              one RGB frame -> tokens: timm_cnn.py (features_only), timm_vit.py, speed_head.py
├── modality/            any non-image input -> per-frame tokens: vector.py, camera.py, none.py
├── temporal/            tokens across frames -> context: causal.py, bidirectional.py, identity.py
├── goal/                goal specification(s) -> goal tokens: point, gps, image, route, instruction, none
└── action/              context (+ goal) tokens -> trajectories
    ├── spaces.py        ActionSpace: waypoint | velocity
    ├── anchors.py       AnchorSet: arc fan or NPZ vocabulary
    ├── regression.py / mhp.py / anchor.py
    ├── generative.py    GenerativeDecoder = denoiser x scheduler (x anchors)
    ├── denoisers/       mlp.py, dit.py, unet.py
    └── schedulers/      ddim.py, flow.py
```

## Tensor contracts

| Stage | Input | Output |
| --- | --- | --- |
| Vision encoder | `(N, 3, H, W)` RGB in [0, 1] | `VisionOutput(tokens (N, Kv, D), speed (N, 1) or None)` |
| Modality encoder | its `input_names` batch keys, each `(B, F, ...)`, or `None` | `(B, F, Km, D)`; `Km = 0` disables the slot |
| Temporal encoder | `(B, F, K, D)`, `K = Kv + sum(Km)` | `(B, F', K, D)`; `F' = F` for `reduction=none`, else 1 |
| Goal encoder(s) | one goal batch each, or `None` | `(N, G, D)` concatenated; `G = 0` for `none` |
| Action decoder | context `(N, K, D)`, goal `(N, G, D)`, optional noise | `PlanOutput(plans (N, M * (2 * T * P + 1)), ...)` |

- **Tokens per frame** `K`: `token_mode=global` (1), `patch` (`gh * gw` pooled patches, learned
  positions) or `fused` (1 + grid), plus whatever the modalities add. The temporal encoder shares
  its frame position across the `K` tokens; `K * D` is one deployment feature-buffer slot, so a
  new modality widens the buffer and the export graph by itself.
- **Modalities** are an open `{name: encoder}` mapping (`model/modality_encoder`, or
  `+model.modality_encoders.<name>=...`). An encoder's `input_names` are at once the dataset
  keys, the `forward`/`predict` keywords and the ONNX input names. Shipped: `VectorEncoder`
  (any per-frame vector; `common.ego_features` fills the `ego` key) and `PinholeCameraEncoder`
  (intrinsics normalized by the running resolution, so weights transfer across crops).
- **Optional inputs**: every non-vision encoder has a learned null token, and `p_drop`
  substitutes it for a fraction of training samples, so one set of weights works with and
  without the input (goal-free exploration, NoMaD style).
- **Goal batches**: `point` is per frame, `(B, F, 3)` as (distance, cos, sin) in that frame's
  ego frame; `gps` the same goal as a raw `(B, F, 2)` metre offset; image `(B, 3, h, w)`,
  route image `(B, C, h, w)` and instruction `(B, E)` describe the whole window. A **list** of
  goal encoders takes a list of goals in the same order; the dataset's `goal_type` follows via
  the `${goal_types:...}` resolver. A `route_image` encoder takes `weights=<ckpt>` from
  `visnavkit-train-route`, whose AE/VAE encoder is the same CNN.
- **Auxiliary heads**: per-frame speed regression (`vision_encoder.speed_head=true`) is
  recipe-specific; without it there is no `speed` output and no `vision_*` loss.
- **Conditioning**: the decoder concatenates context and goal tokens with a type embedding and
  pools them with one learned attention query (or the mean). DiT denoisers also cross-attend to
  the raw tokens.
- **Flat layout**: every decoder packs `[mu, log_scale, confidence_logit]` per mode in pose
  space; `parse_plan_output` yields trajectories, scales, confidences and the best plan.

## Cross-cutting knobs

- `model.vision_encoder.token_mode=global|patch|fused`, `patch_grid=[4,4]`.
- `model.action_decoder.action_space.kind=waypoint|velocity`: `velocity` derives unicycle
  (speed, yaw rate) per anchor segment and integrates back, so losses act on commands while
  metrics and export see poses.
- `model.vision_encoder.speed_head=true`.
- A `normalizer` (`none|meanstd|minmax|scale`) on the supervision targets and on each input
  encoder, one per signal: a `gps` goal in metres, an ego vector mixing m/s with rad/s and the
  action targets share no statistics. Statistics live in buffers, so checkpoint, ONNX graph and
  deployment cannot drift apart. Generative decoders expect roughly unit-scale targets, so fit
  `meanstd` or `minmax` once a corpus exists (`visnavkit-dataset command=stats`).
- Generative decoders: `scheduler.time_sampling=uniform|logit_normal|beta`, `scheduler.shift`
  for flow matching; `scheduler.beta_schedule` and `clip_sample` for DDIM.

## Generative decoders

`GenerativeDecoder(denoiser, scheduler, anchors=None)`:

- **Schedulers** run on continuous `t in [0, 1]` (1 = noise). `DDIMScheduler` trains with DDPM
  noise prediction and samples deterministic DDIM; its `clip_sample` is in *normalized* action
  units, so with `normalizer.mode=none` it must exceed the longest plan in metres (the decoder
  warns). `FlowMatchingScheduler` uses linear interpolation, a velocity target and Euler steps;
  `beta` time sampling is openpi pi0's `Beta(1.5, 1)`, `shift` the diffusers SD3/Flux time shift.
- **Denoisers** share `forward(x_t, t, cond, tokens)`: `MLPDenoiser`, `DiTDenoiser` (adaLN-Zero
  self-attention over steps, cross-attention to tokens), `UNet1DDenoiser` (diffusion-policy FiLM
  U-Net).
- **Anchors** make it per-anchor residual generation: one trajectory per anchor, confidences
  from an anchor classifier (`num_modes = K`).
- Training skips sampling; inference takes explicit `noise (N, M, T, A)`, so the same noise gives
  the same plan in PyTorch and ONNX.

## Weight averaging

`ema=default` adds `EMACallback`: the diffusers `EMAModel` schedule
`1 - (1 + step / inv_gamma) ** -power` capped at `decay`, swapped in for validation and written
into the checkpoint's `state_dict` (online weights ride along under `ema_online_state_dict` for
resumes). Denoising decoders benefit most; off by default.

## Inference, deployment, benchmark

`NavigationPolicy.act(vision, goal=None, noise=None, **modality_inputs)` is the window-in,
actions-out contract: trajectories `(B, M, T, P)` in pose space and scores `(B, M)` for the
newest frame. `forward` runs the same encoders and returns the flat packed layout, one decision
per frame under `reduction=none`; both share `encode_window`.

`NavigationPolicy.predict(frame, feature_buffer, goal=None, noise=None, **modality_inputs)`
encodes one frame and reuses buffered past tokens `(B, history, K * D)`; the newest frame's
modality inputs carry no frame axis. A model owns its deployment graph: `export_graph(cfg,
batch_size)` returns the traced wrapper, example inputs and io names, `decision(outputs)` reads the
newest decision back. `NavigationPolicy` traces `predict` with presence-driven inputs — `vision`,
`feature_buffer`, one `goal` per goal encoder, one per modality key, `noise` — after flipping
`reduction=none` to `last` and precomputing ViT position embeddings for the export resolution;
`FlowPilotDST` traces `deploy`, the window in and the current frame's top-k plans out
([contract](flowpilot_dst_onnx.md)). `export/graph.py` folds the model (FastViT branches,
Linear-BatchNorm, RMSNorm), traces at fixed shapes with the MHA fast path disabled, casts the
weights to `precision` (`fp32` | `fp16`, io stays fp32), verifies ONNX Runtime parity and writes
`.pth` (weights + config), `.metadata.json` and `.inputs.npz` beside the graph. `export/trt.py`
builds a TensorRT engine at the graph's precision (TensorRT 11 is strongly typed); `export/check.py`
replays the traced inputs through checkpoint, `.pth`, ONNX Runtime and engine at each precision's
tolerance. `benchmark/export.py` traces the full window and outputs `trajectories`, `scores` (plus
`speed`) for latency and open-loop measurements.

## Add a component

1. Subclass the stage base (`BaseVisionEncoder._encode`, `BaseTemporalEncoder`,
   `BaseGoalEncoder.encode`, `BaseModalityEncoder.encode`, `BaseActionDecoder.decode/loss`, or a
   denoiser with the shared signature). A new input is a `BaseModalityEncoder` with its own
   `input_names`; `policy.py` does not change.
2. Add a yaml under `configs/model/<group>/` with an explicit `_target_` and
   `feat_size: ${model.feat_size}`. Modality files carry `# @package model.modality_encoders`
   and write one named slot, so options compose.
3. `uv run visnavkit-sanity-check model/<group>=<name> --onnx`, then add the entry to the group
   test in `tests/models/test_model_configs.py`.
