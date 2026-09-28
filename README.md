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

[Architecture](docs/architecture.md) · [Models & weights](docs/models.md) · [Data](docs/data.md) · [Benchmark](docs/benchmark.md) · [Roadmap](docs/roadmap.md) · [Credits](docs/references.md)

## Install

```bash
uv sync                    # CPU ONNX Runtime included; Linux resolves CUDA 13 torch wheels
uv sync --extra export     # + onnxslim for deployment graphs
uv sync --extra dali       # + NVIDIA DALI GPU video decoding (Linux)
export VISNAVKIT_DATA_ROOT=/data/nav_clips   # what common.data_root reads
```

Python 3.10+; the TorchCodec loader needs FFmpeg on the system (`apt install ffmpeg`).

## Quick start

No data, no downloads: composes a pipeline, prints its shapes, runs a training step, checks that
`act`, `forward` and `predict` agree, and with `--onnx` exports and checks ONNX Runtime parity.

```bash
uv run visnavkit-sanity-check model=s2e model/action_decoder=anchor_flow_dit --onnx
```

## Models

Pick one entry per group, or a recipe plus overrides:

| Group | Choices |
| --- | --- |
| `model/vision_encoder` | `cnn` or `vit` with any timm backbone; 22 presets from `fastvit_t8` to `dinov3_b` |
| `model/temporal_encoder` | `identity`, `causal`, `causal_4layer`, `bidirectional` |
| `model/goal_encoder` | `none`, `point`, `gps`, `image`, `route_image`, `instruction`; a list combines them |
| `model/modality_encoder` | `none`, `ego`, `camera`, `ego_camera`; `+model.modality_encoders.<name>=...` adds yours |
| `model/action_decoder` | `regression`, `mhp`, `anchor`, `{diffusion,flow}_{mlp,dit,unet}`, `anchor_{diffusion,flow}_dit` |

- **Paper recipes:** `gnm`, `vint`, `nomad`, `citywalker`, `mbra`, `navdp`, `s2e`, `socialnav`, `internvla_n1`,
  `mimic`, `flowpilot`, `flowpilot_dst`.
- **FlowPilot family:** FlowPilot, FlowPilot-DST, FlowPilot-DUNE-DST and FlowMatchingPolicy (step flow, S2E,
  FlowBridge).
- **Pretrained weights:** FlowPilot-DST, FastViT and DUNE versions, in the
  [model zoo](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints).

Recipe tables, weights and how each FlowPilot version is trained: [model catalog](docs/models.md).
Any checkpoint, old versions included, loads with its own config:

```python
from visnavkit.models.checkpoint import load_checkpoint

model, cfg = load_checkpoint("flowpilot_dst_fastvit_t12.ckpt")
```

## Data, training, deployment

```bash
uv run visnavkit-dataset command=preprocess                         # validate clips, write manifests
uv run visnavkit-dataset command=stats   dataset=torch name=city    # normalizer NPZ + plots
uv run visnavkit-dataset command=anchors dataset=torch num_anchors=64

uv run visnavkit-train dataset=torch model=vint \
  model/vision_encoder=dinov3_s model/goal_encoder=gps model/action_decoder=flow_dit
uv run visnavkit-train dataset=torch model=mimic ema=default
uv run visnavkit-train-route route=vae                      # route-patch AE/VAE from route_images.npy

uv run visnavkit-export checkpoint=logs/baseline/.../last.ckpt output=outputs/policy.onnx
uv run visnavkit-export-dst checkpoint=<ckpt> output=flowpilot_dst.onnx [streaming=true]
uv run python -m visnavkit.scripts.profile_dst checkpoint=<ckpt>
uv run visnavkit-benchmark command=export model=gnm output_dir=outputs/benchmark/gnm
```

A clip is a directory with `video.mp4` and four NumPy sidecars; a manifest lists `video_path label start end`
([data guide](docs/data.md)). Training records Git provenance (`strict_git=true` requires a clean tree); recipe
overrides live in [`configs/experiment/`](visnavkit/configs/experiment/), and `model/goal_encoder=route_image
model.goal_encoder.weights=<ckpt>` seeds the route goal encoder from a route checkpoint. `visnavkit-export` writes the
streaming `NavigationPolicy` graph and `visnavkit-export-dst` the window graph of the FlowPilot-DST family
([ONNX IO](docs/flowpilot_dst_onnx.md)); the benchmark runs the full window for latency and open-loop metrics
([benchmark](docs/benchmark.md)).

## Development

```bash
uv run pytest tests/ -q && uv run ruff check visnavkit tests
```

Use it from another project with `uv add --editable /path/to/visnavkit`; the configs ship in the
package as `visnavkit.configs`. Coding-agent contract: [AGENTS.md](AGENTS.md).

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
[`CITATION.bib`](CITATION.bib) has one entry per paper in [the references](docs/references.md) — please also cite the work a recipe adapts.
