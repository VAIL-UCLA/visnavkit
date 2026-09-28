# Credits and references

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

Navigation policies, oldest first; every one but ViKiNG has a recipe in [the model catalog](models.md#recipes).

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
