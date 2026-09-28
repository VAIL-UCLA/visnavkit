# Model catalog

Research identity is recorded separately from this repository's architecture adaptations.
**Downloadable ONNX artifacts are not yet validated open-loop policy adapters.**

## Recipes

Paper-named recipes adapt each architecture to this repo's data contract (single RGB frames,
fixed-horizon x/y/v targets, goals from the episode's own future). They are not reproductions and
load no upstream checkpoint; `model=base` is the skeleton they inherit.

| Recipe | Vision | Temporal | Goal | Decoder |
| --- | --- | --- | --- | --- |
| `gnm` | MobileNetV2 | single frame | image (stacked with observation) | regression |
| `vint` | EfficientNet-B0 | causal x4, 512-d, 4 heads | image (stacked) | regression |
| `nomad` | EfficientNet-B0 | causal x4, 256-d | image, 50% goal dropout | diffusion U-Net, 10 steps, 8 candidates |
| `citywalker` | DINOv2 ViT-B (frozen) | causal x16, 768-d | point + past odometry | regression |
| `mbra` | EfficientNet-B0 | causal x4, 1024-d, 4 heads | gps | regression |
| `navdp` | DINOv2 ViT-S | causal x2, 384-d | point | diffusion DiT (384, x16), 10 steps, 16 candidates |
| `s2e` | DINOv3 ViT-S | causal x6, 768-d | point, 55% goal dropout | anchor, 64 k-means anchors |
| `socialnav` | SigLIP ViT-B/16 (frozen VLM tower) | causal x1, 1536-d | point + past positions | flow DiT (1536, x12), 5 steps |
| `internvla_n1` | DINOv2 ViT-S | causal x1 | instruction (System 2 latent, 4 tokens) | flow DiT (384, x12), 10 steps |
| `mimic` | DINOv3 ViT-S | causal x4, 512-d | point + camera token | anchor, 64 anchors |
| `flowpilot` | FastViT-MA36 + speed head | causal x4, 1280-d | gps, 90% goal dropout | anchored flow DiT (1280, x4), 64 anchors, 4 steps, Beta(1.5, 1) times |
| `flowpilot_dst` | FastViT-T12 on [frame_t, frame_t-1] + speed head, frozen route VAE; `dataset=pose` 20 Hz slots | causal x2 over the slots, 512-d | point, 50% goal dropout, embodiment token | anchored flow DiT (512, x4, cross-attn only), 64 anchors, 4 steps, modes from noise 0 and one draw |

## FlowPilot family

`flowpilot` is the paper recipe in the table above, a `NavigationPolicy`. The other variants are their own model classes on
`dataset=pose` windows of 20 Hz slots (frames, route patches, ego [v, w], action bounds) and decide for the current
slot through `deploy`.

| Variant | Model | Config | Frame encoder | Head |
| --- | --- | --- | --- | --- |
| FlowPilot | `NavigationPolicy` | `model=flowpilot` | FastViT-MA36 on frame pairs | anchored flow DiT |
| FlowPilot-DST | `FlowPilotDST` | `experiment=flowpilot_dst_clips1k`, `flowpilot_dst_tiny`, `flowpilot_dst_overfit` | FastViT-T12 on frame pairs + speed head | anchored flow DiT over per-frame kv tokens |
| FlowPilot-DUNE-DST | `FlowPilotDST` | `experiment=flowpilot_dune_dst_clips1k` | frozen DUNE ViT-B/14 + adapter | as FlowPilot-DST, at dim 1024 |
| FlowMatchingPolicy | `FlowMatchingPolicy` | `experiment=flow_matching_policy_clips1k` | `model/frame_encoder=fastvit_sa12` or `dune` | per-step flow (`StepFlowHead`) on a flow or DDIM scheduler |
| FlowMatchingPolicy, S2E | `FlowMatchingPolicy` | `experiment=flow_matching_policy_s2e_clips1k` | as above | anchors in, trajectories out (`S2EHead`) |
| FlowMatchingPolicy, FlowBridge | `FlowMatchingPolicy` | `experiment=flow_matching_policy_bridge_clips1k` | as above | anchor-to-trajectory flow (`FlowBridgeHead`) |

Each recipe's yaml under [`configs/model/`](../visnavkit/configs/model/) and
[`configs/experiment/`](../visnavkit/configs/experiment/) describes its layers and the files it needs (route VAE,
k-means anchors, action bounds). The window graph's inputs and outputs: [FlowPilot-DST ONNX](flowpilot_dst_onnx.md).

## Pretrained weights

One row per deployable variant: the paper authors' **official** weights, a VisNavKit
**reproduced-ckpt**, and its `visnavkit-export` **reproduced-onnx**, hosted in the
[model zoo](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints).

| Weights | Config | Model | Checkpoints (official) | Checkpoints (reproduced-ckpt) | Checkpoints (reproduced-onnx) |
| --- | --- | --- | --- | --- | --- |
| `gnm-point` | `gnm` + point goal | MobileNetV2 -> regression, 3.5M | — | — | — |
| `flowpilot-edge` | `flowpilot` | FastViT-MA36 -> anchored flow DiT, 300M | — | — | — |
| `flowpilot-dst-small` | `flowpilot_dst_clips1k` | FastViT-T12 pairs -> anchored flow DiT (256, x2), 21.4M ([ONNX IO](flowpilot_dst_onnx.md)) | — | [ckpt](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-small/flowpilot_dst_fastvit_t12.ckpt) | [onnx](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-small/flowpilot_dst_fastvit_t12.onnx) |
| `flowpilot-dst-dune` | `flowpilot_dune_dst_clips1k` | frozen DUNE ViT-B/14 -> anchored flow DiT (1024, x2), 208.2M ([ONNX IO](flowpilot_dst_onnx.md)) | — | [ckpt](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-dune/flowpilot_dst_dune_vitb14.ckpt) | [onnx](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-dune/flowpilot_dst_dune_vitb14.onnx) |

```bash
uv run visnavkit-train dataset=torch model=gnm model/goal_encoder=point \
  ~model.goal_encoder.backbone_name ~model.goal_encoder.stack_observation  # gnm-point
uv run visnavkit-train dataset=torch model=flowpilot                       # flowpilot-edge
uv run visnavkit-train experiment=flowpilot_dst_clips1k                   # flowpilot-dst-small
uv run visnavkit-train experiment=flowpilot_dune_dst_clips1k \
  model.frame_encoder.weights=<DUNE ViT-B/14 ckpt>                        # flowpilot-dst-dune
```

Where each paper publishes its own weights: [Weights](#weights). The published ONNX graphs the benchmark
downloads: [Published exports](#published-exports).

## Loading a checkpoint

A checkpoint rebuilds from the config it was trained with, so no recipe needs composing:

```python
from visnavkit.models.checkpoint import load_checkpoint

model, cfg = load_checkpoint("flowpilot_dst_fastvit_t12.ckpt")  # eval mode, weights loaded strictly
```

Older checkpoints keep loading. Initialization files that stayed on the training machine (a route VAE, k-means
anchors) are skipped, since the checkpoint carries those weights. A config key added after a checkpoint was saved takes
its constructor default, which keeps the old architecture, or is set from the checkpoint's weights where the default
cannot tell. `LitModel.load_from_checkpoint`, `visnavkit-export`, `visnavkit-export-dst`, `profile_dst` and
resuming training (`trainer.resume.ckpt_path`) load the same way.

## Published exports

The [UCLA-VAIL model zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public) holds
the published ONNX exports and wrappers. The catalog pins revision
`9c1f523ef8dddbccb6dee1d3d588112f754bc8f2` (verified 2026-09-07); sizes and SHA-256 digests are
in [`catalog.py`](../visnavkit/benchmark/catalog.py). The downloader takes only `.onnx` and
`.onnx.data`, verifies length and digest, and installs atomically; no remote Python is imported.

```python
from visnavkit.benchmark.catalog import download_model, get_model, list_models

for model in list_models():
    print(model.id, model.status, model.download_bytes)
paths = download_model("gnm", "artifacts/models")  # explicit network download
```

`register_model(ModelSpec(...))` adds an entry without validating inference.

| ID | Source | Published export | Local recipe |
| --- | --- | --- | --- |
| `gnm` | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | image-goal graph; six RGB frames, 64×85 | image-goal adaptation, no temporal-distance head |
| `vint` | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | image-goal graph; six RGB frames, 64×85 | image-goal adaptation, no temporal-distance head |
| `nomad` | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | three component graphs; four RGB frames, 96×96 | 1D U-Net denoiser with goal dropout; sampler and normalization differ |
| `citywalker` | [ai4ce/CityWalker](https://github.com/ai4ce/CityWalker) | five RGB frames, 350×630, past coordinates, point goal | frozen DINOv2-B, 16-layer stack; past odometry as `past_xy`; causal, not bidirectional |
| `s2e` | [VAIL-UCLA/S2E](https://github.com/VAIL-UCLA/S2E) | BC Web100 variant; eleven RGB frames, 256×256, point goal | ICLR 2026 camera-ready: DINOv3, 6 layers, 64 k-means anchors; no RL stage |
| `mimic` | [VAIL-UCLA/MIMIC](https://github.com/VAIL-UCLA/MIMIC) | goal-free variant; sixteen RGB frames, 288×512; batch one | DINOv3-S, goal and camera tokens, 64-anchor decoder |
| `navdp` | [InternRobotics/NavDP](https://github.com/InternRobotics/NavDP) | none verified; checkpoints by author form | point-goal diffusion DiT; RGB only, no privileged critic |
| `mbra` | [MBRA / LogoNav](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) | six RGB frames, 96×96, point-goal pose | LogoNav: EfficientNet-B0, local GPS goal, regression; heading omitted |

Resolution is `(height, width)`. A paper name in Hydra establishes no checkpoint compatibility;
published exports stay `artifacts_available` until reference parity is verified, and NavDP is
`unavailable` (downloading raises `ModelUnavailableError`).

## Wrapper audit

What each published bundle needs before its numbers mean anything:

- **GNM, ViNT**: the zoo wrappers feed random goal images; use a real goal image and label
  random-goal ablations. Cadence and metric spacing must match the checkpoint.
  [GNM](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/gnm/gnm.py),
  [ViNT](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/vint.py)
- **NoMaD**: the denoiser graph is one component; full-policy timing includes the encoder, the
  ten DDPM steps and decoding, with a fixed candidate count and seed.
  [metadata](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/NoMaD_GL_Official/model_info.yaml)
- **CityWalker**: the wrapper ignores `goal_xy`, randomizes the last coordinate row and can
  mutate the past-trajectory array; an adapter needs explicit odometry, goal, coordinate
  conversion and the original normalization.
  [wrapper](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/CityWalker_PG_Official/inference.py),
  [config](https://github.com/ai4ce/CityWalker/blob/main/config/citywalk_2000hr.yaml)
- **S2E**: BC weights, not the RL policy. Wrapper comments disagree with its code, which scales
  output XY by 0.25 and encodes (clipped distance / 200, cos, sin) — audit findings, not
  validated conversions.
  [wrapper](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/S2E/inference.py)
- **MIMIC**: the goal-free graph differs from the paper's DINOv3 + goal/camera model; the wrapper
  has a stale docstring and imports `urbansim`. Output times are
  `[1,2,4,6,7,8,10,12,14,15,17,19,21,23,25]/5` s — keep them and declare truncation; scaling is
  unvalidated.
  [metadata](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/MIMIC/model_info.yaml),
  [wrapper](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/MIMIC/inference.py)
- **NavDP**: RGB-D, diffusion sampling and critic selection must be preserved; upstream defaults
  are eight 224×224 frames plus depth. [agent](https://github.com/InternRobotics/NavDP/blob/master/baselines/navdp/policy_agent.py)

`output_names` are source-reported hints; only MIMIC has source-verified output timestamps.
`output_contract_verified=False` means the metre conversion is pending.

Keep RGB goal-free, image-goal, point-goal-with-odometry and RGB-D policies in separate cohorts,
report full-policy latency apart from component timing, and artifact bytes apart from parameter
count. The zoo is Apache-2.0 over the original terms: visualnav-transformer MIT, CityWalker
Apache-2.0, NavDP CC-BY-NC-SA-4.0
([zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public#license),
[NavDP](https://github.com/InternRobotics/NavDP#-license)).

## Weights

VisNavKit ships no trained policies. What loads:

| Weights | How |
| --- | --- |
| timm backbone (ImageNet, DINOv2/v3, CLIP, SigLIP, ...) | `model.vision_encoder.pretrained=true`, the default |
| a vision encoder you trained | `model.vision_encoder.weights=/path/encoder.pt` |
| a full VisNavKit checkpoint | `pretrained.ckpt_path=/path/last.ckpt`; `pretrained.strict=false` takes the stages that match |
| published ONNX exports | `visnavkit-benchmark command=download`, catalogued above |

Upstream checkpoints are **not** loadable into the recipes — the adapted architectures do not
line up tensor for tensor. Treat them as baselines and follow each project's licence.

| Recipe | Released weights | Notes |
| --- | --- | --- |
| `gnm`, `vint`, `nomad` | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | one checkpoint per model |
| `citywalker` | [ai4ce/CityWalker](https://github.com/ai4ce/CityWalker) | — |
| `mbra` | [Learning-to-Drive-Anywhere-with-MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) | LogoNav image-goal and GPS-goal |
| `navdp` | [InternRobotics/NavDP](https://github.com/InternRobotics/NavDP) | by author form |
| `s2e` | [VAIL-UCLA/S2E](https://github.com/VAIL-UCLA/S2E) | BC weights; the RL stage is unreleased |
| `socialnav` | [AMAP-EAI/SocialNav](https://github.com/AMAP-EAI/SocialNav) | Qwen2-VL and Qwen2.5-VL Brains; dataset unreleased |
| `internvla_n1` | [InternRobotics/InternVLA-N1](https://huggingface.co/InternRobotics/InternVLA-N1) | `-System2`, `-DualVLN`, `-Preview`, `-wo-dagger` |
| `mimic` | [VAIL-UCLA/MIMIC](https://github.com/VAIL-UCLA/MIMIC), [zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public) | inference and augmentation code, goal-free ONNX; training code and checkpoints planned |
| `flowpilot` | [VAIL-UCLA/FlowPilot](https://github.com/VAIL-UCLA/FlowPilot) | repository is still a placeholder |
