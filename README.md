<h1 align="center">
  <img src="docs/assets/logo.png" alt="" width="64" align="absmiddle"> VisNavKit
</h1>

**A composable toolkit for visual navigation policies: train, export to ONNX, benchmark.**

Vision in, trajectory out. Every stage is a Hydra group exchanging tokens of one width, so any
encoder works with any goal, extra input and decoder. Ego state and camera calibration ship as
inputs; anything else (depth, LiDAR, a spatial raster) is one encoder subclass plus a yaml.

```text
context   vision   (B, F, 3, H, W)                            ─┐
          ego      (B, F, E)                                  ─┤
          camera   K (B, F, 3, 3), RT (B, F, 4, 4)            ─┼──▶  policy  ──▶  actions (B, M, T, D)
          <yours>  (B, F, ...)                                ─┤
goal      point · gps · image · route · instruction · <yours> ─┘
```

B batch, F frames, M candidates, T steps, D pose; every input but vision is optional. That function is
`policy.act(vision, goal, **inputs)`; `forward` is its training view (one decision per frame, packed
for the losses) and `predict` its streaming view (the newest frame plus a feature buffer).

[Architecture](docs/architecture.md) · [Data](docs/data.md) · [Models & weights](docs/models.md) · [Benchmark](docs/benchmark.md) · [Roadmap](docs/roadmap.md)

**Contents:** [Install](#install) · [Quick start](#quick-start) · [Models](#models) ([compose](#compose-a-policy),
[paper recipes](#paper-recipes), [FlowPilot family](#flowpilot-family), [pretrained weights](#pretrained-weights),
[loading a checkpoint](#loading-a-checkpoint)) · [Data](#data) · [Train](#train) · [Deploy and benchmark](#deploy-and-benchmark)
· [Development](#development)

## Install

```bash
uv sync                    # CPU ONNX Runtime included; Linux resolves CUDA 13 torch wheels
uv sync --extra export     # + onnxslim for deployment graphs
uv sync --extra dali       # + NVIDIA DALI GPU video decoding (Linux)
export VISNAVKIT_DATA_ROOT=/data/nav_clips   # what common.data_root reads
```

Python 3.10+; the TorchCodec loader needs FFmpeg on the system (`apt install ffmpeg`).

## Quick start

No data, no downloads. Composes the pipeline, prints its shapes, runs a training forward and
backward, and checks that `act`, `forward` and the feature-buffer `predict` agree on the newest
decision; `--onnx` also exports and verifies ONNX Runtime parity:

```bash
uv run visnavkit-sanity-check model=s2e model/action_decoder=anchor_flow_dit --onnx
```

## Models

### Compose a policy

One entry per group, or a recipe plus overrides:

| Group | Choices |
| --- | --- |
| `model/vision_encoder` | `cnn` or `vit` with any timm backbone; 22 presets from `fastvit_t8` to `dinov3_b` |
| `model/temporal_encoder` | `identity`, `causal`, `causal_4layer`, `bidirectional` |
| `model/goal_encoder` | `none`, `point`, `gps`, `image`, `route_image`, `instruction`; a list combines them |
| `model/modality_encoder` | `none`, `ego`, `camera`, `ego_camera`; `+model.modality_encoders.<name>=...` adds yours |
| `model/action_decoder` | `regression`, `mhp`, `anchor`, `{diffusion,flow}_{mlp,dit,unet}`, `anchor_{diffusion,flow}_dit` |

```bash
uv run visnavkit-train dataset=torch model=vint \
  model/vision_encoder=dinov3_s model/goal_encoder=gps model/action_decoder=flow_dit
```

Token modes, action spaces, per-signal normalizers and the denoiser × scheduler split behind the
generative decoders: [architecture](docs/architecture.md).

### Paper recipes

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

### FlowPilot family

`flowpilot` is the paper recipe above, a `NavigationPolicy`. The other variants are their own model classes on
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

Each recipe's yaml under [`configs/model/`](visnavkit/configs/model/) and
[`configs/experiment/`](visnavkit/configs/experiment/) describes its layers and the files it needs (route VAE,
k-means anchors, action bounds). The window graph's inputs and outputs: [FlowPilot-DST ONNX](docs/flowpilot_dst_onnx.md).

### Pretrained weights

One row per deployable variant: the paper authors' **official** weights, a VisNavKit
**reproduced-ckpt**, and its `visnavkit-export` **reproduced-onnx**, hosted in the
[model zoo](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints).

| Weights | Config | Model | Checkpoints (official) | Checkpoints (reproduced-ckpt) | Checkpoints (reproduced-onnx) |
| --- | --- | --- | --- | --- | --- |
| `gnm-point` | `gnm` + point goal | MobileNetV2 -> regression, 3.5M | — | — | — |
| `flowpilot-edge` | `flowpilot` | FastViT-MA36 -> anchored flow DiT, 300M | — | — | — |
| `flowpilot-dst-small` | `flowpilot_dst_clips1k` | FastViT-T12 pairs -> anchored flow DiT (256, x2), 21.4M ([ONNX IO](docs/flowpilot_dst_onnx.md)) | — | [ckpt](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-small/flowpilot_dst_fastvit_t12.ckpt) | [onnx](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-small/flowpilot_dst_fastvit_t12.onnx) |
| `flowpilot-dst-dune` | `flowpilot_dune_dst_clips1k` | frozen DUNE ViT-B/14 -> anchored flow DiT (1024, x2), 208.2M ([ONNX IO](docs/flowpilot_dst_onnx.md)) | — | [ckpt](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-dune/flowpilot_dst_dune_vitb14.ckpt) | [onnx](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-dune/flowpilot_dst_dune_vitb14.onnx) |

```bash
uv run visnavkit-train dataset=torch model=gnm model/goal_encoder=point \
  ~model.goal_encoder.backbone_name ~model.goal_encoder.stack_observation  # gnm-point
uv run visnavkit-train dataset=torch model=flowpilot                       # flowpilot-edge
uv run visnavkit-train experiment=flowpilot_dst_clips1k                   # flowpilot-dst-small
uv run visnavkit-train experiment=flowpilot_dune_dst_clips1k \
  model.frame_encoder.weights=<DUNE ViT-B/14 ckpt>                        # flowpilot-dst-dune
```

Where each paper publishes its own weights, and the published ONNX graphs the benchmark downloads:
[model catalog](docs/models.md).

### Loading a checkpoint

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

## Data

A clip is a directory with `video.mp4` and four NumPy sidecars (times, positions, orientations,
speeds); a manifest lists `video_path label start end`. Targets are interpolated at fixed relative
times, so any frame rate works.

```bash
uv run visnavkit-dataset command=preprocess                         # validate clips, write manifests
uv run visnavkit-dataset command=stats   dataset=torch name=city    # normalizer NPZ + plots
uv run visnavkit-dataset command=anchors dataset=torch num_anchors=64
```

Sidecar layout, the opt-in ego and calibration inputs, and the public corpora this format targets:
[data guide](docs/data.md).

## Train

```bash
uv run visnavkit-train dataset=torch model=mimic ema=default
uv run visnavkit-train-route route=vae                      # route-patch AE/VAE from route_images.npy
```

A route checkpoint seeds the route goal encoder: `model/goal_encoder=route_image
model.goal_encoder.weights=<ckpt>`. Training records Git provenance (`strict_git=true` requires
a clean tree) and keeps overrides in
[`configs/experiment/`](visnavkit/configs/experiment/).

## Deploy and benchmark

```bash
uv run visnavkit-export checkpoint=logs/baseline/.../last.ckpt output=outputs/policy.onnx
uv run visnavkit-export-dst checkpoint=<ckpt> output=flowpilot_dst.onnx [streaming=true]
uv run python -m visnavkit.scripts.profile_dst checkpoint=<ckpt>
uv run visnavkit-benchmark command=export model=gnm output_dir=outputs/benchmark/gnm
```

- `visnavkit-export`: the `NavigationPolicy` deployment graph, which streams one frame through a feature buffer.
- `visnavkit-export-dst`: the FlowPilot-DST / FlowMatchingPolicy window graph. `streaming=true` takes one frame per
  call plus a buffer of the past slots (FlowPilot-DST); `noise=randn` adds a noise input (FlowMatchingPolicy).
- `profile_dst`: FlowPilot-DST sanity checks and per-stage latency.
- `visnavkit-benchmark`: the full window, for latency and open-loop metrics.

More in [architecture](docs/architecture.md#inference-deployment-benchmark) and [benchmark](docs/benchmark.md).

## Development

```bash
uv run pytest tests/ -q && uv run ruff check visnavkit tests
```

Use it from another project with `uv add --editable /path/to/visnavkit`; the configs ship in the
package as `visnavkit.configs`. Coding-agent contract: [AGENTS.md](AGENTS.md).

## Credits

VisNavKit is built on the following amazing open-source projects:

- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) Training loop, checkpointing and callbacks.
- [Hydra](https://github.com/facebookresearch/hydra) + [OmegaConf](https://github.com/omry/omegaconf) Composable configuration for every stage.
- [timm](https://github.com/huggingface/pytorch-image-models) Every vision backbone, pretrained and feature-ready.
- [TorchCodec](https://github.com/pytorch/torchcodec) and [NVIDIA DALI](https://github.com/NVIDIA/DALI) CPU and GPU video decoding.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [OnnxSlim](https://github.com/inisis/OnnxSlim) Deployment graphs, parity checks and benchmarks.
- [uv](https://github.com/astral-sh/uv) + [Ruff](https://github.com/astral-sh/ruff) Environments, linting and formatting.

The following repositories greatly inspire VisNavKit:

- [diffusers](https://github.com/huggingface/diffusers) — typed outputs, denoiser and scheduler as separate objects, the EMA schedule.
- [openpi](https://github.com/Physical-Intelligence/openpi) — flow-matching conventions and its Beta time sampling.
- [LeRobot](https://github.com/huggingface/lerobot) — one training forward, one deployment predict.
- [diffusion_policy](https://github.com/real-stanford/diffusion_policy) — the conditional 1D U-Net denoiser.
- [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) — the GNM / ViNT / NoMaD recipes.
- [openpilot](https://github.com/commaai/openpilot) — quadratically spaced trajectory anchors.

Thanks to the maintainers of these projects for their contribution to the community!

## Research references

<details>
<summary>Navigation policies, world models, generative building blocks, simulators and benchmarks</summary>

Navigation policies, oldest first; every one but ViKiNG has a recipe above.

- **ViKiNG**: Vision-Based Kilometer-Scale Navigation with Geographic Hints — [arXiv:2202.11271](https://arxiv.org/abs/2202.11271)
- **GNM**: A General Navigation Model to Drive Any Robot — [arXiv:2210.03370](https://arxiv.org/abs/2210.03370), [code](https://github.com/robodhruv/drive-any-robot)
- **ViNT**: A Foundation Model for Visual Navigation — [arXiv:2306.14846](https://arxiv.org/abs/2306.14846), [code](https://github.com/robodhruv/visualnav-transformer)
- **NoMaD**: Goal Masked Diffusion Policies for Navigation and Exploration — [arXiv:2310.07896](https://arxiv.org/abs/2310.07896), [code](https://github.com/robodhruv/visualnav-transformer)
- **CityWalker**: Learning Embodied Urban Navigation from Web-Scale Videos — [arXiv:2411.17820](https://arxiv.org/abs/2411.17820), [code](https://github.com/ai4ce/CityWalker)
- **MBRA**: Learning to Drive Anywhere with Model-Based Reannotation — [arXiv:2505.05592](https://arxiv.org/abs/2505.05592), [code](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA)
- **NavDP**: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance — [arXiv:2505.08712](https://arxiv.org/abs/2505.08712), [code](https://github.com/InternRobotics/NavDP)
- **S2E**: From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning — [arXiv:2507.22028](https://arxiv.org/abs/2507.22028), [code](https://github.com/VAIL-UCLA/S2E)
- **SocialNav**: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation — [arXiv:2511.21135](https://arxiv.org/abs/2511.21135), [code](https://github.com/AMAP-EAI/SocialNav)
- **InternVLA-N1**: Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-Language Navigation — [arXiv:2512.08186](https://arxiv.org/abs/2512.08186), [code](https://github.com/InternRobotics/InternNav)
- **MIMIC**: Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion — [arXiv:2603.22527](https://arxiv.org/abs/2603.22527), [code](https://github.com/VAIL-UCLA/MIMIC)
- **FlowPilot**: From Imitation to Alignment: Human-Preference Flow Policies for Long-Horizon Sidewalk Navigation — [arXiv:2606.12603](https://arxiv.org/abs/2606.12603), [code](https://github.com/VAIL-UCLA/FlowPilot), [project](https://vail.cs.ucla.edu/FlowPilot)

World models, generative building blocks, simulators and benchmarks:

- **NWM**: Navigation World Models — [arXiv:2412.03572](https://arxiv.org/abs/2412.03572), [code](https://github.com/facebookresearch/nwm)
- **Diffusion Policy** — [arXiv:2303.04137](https://arxiv.org/abs/2303.04137), [code](https://github.com/real-stanford/diffusion_policy); **DiT** — [arXiv:2212.09748](https://arxiv.org/abs/2212.09748), [code](https://github.com/facebookresearch/DiT); **Flow matching** — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **MetaUrban** — [arXiv:2407.08725](https://arxiv.org/abs/2407.08725), [code](https://github.com/metadriverse/metaurban); **SidewalkBench** — [arXiv:2606.16953](https://arxiv.org/abs/2606.16953)

</details>

## Citation

If VisNavKit helps your work, please consider citing it:

```bibtex
@Misc{visnavkit2026,
  author       = {Honglin He and Bolei Zhou},
  title        = {{VisNavKit}: a composable toolkit for visual navigation policies},
  howpublished = {\url{https://github.com/VAIL-UCLA/visnavkit}},
  year         = {2026},
}
```

[`CITATION.cff`](CITATION.cff) carries the same entry for GitHub's *Cite this repository* button;
[`CITATION.bib`](CITATION.bib) has one entry per paper above — please also cite the work a recipe adapts.
