# AGENTS.md

VisNavKit composes visual navigation policies from interchangeable stages, trains them, exports
ONNX deployment graphs and benchmarks them open-loop. This file is the contract for coding
agents; `CLAUDE.md` carries the same golden rules.

## Setup and commands

```bash
uv sync                                   # deps; --extra export|dali|plot for the rest
echo 'export VISNAVKIT_DATA_ROOT=/data/nav_clips' >> ~/.bashrc && source ~/.bashrc

uv run pytest tests/ -q                   # full suite, no data or downloads needed
uv run ruff check visnavkit tests         # lint (ruff format is available but not enforced)
uv run visnavkit-sanity-check model=gnm   # architecture, shapes, a train step, buffer parity
uv run visnavkit-sanity-check model=gnm --onnx      # + export and ONNX Runtime parity
uv run visnavkit-dataset command=cache dataset=torch  # corpus tooling, see scripts/dataset/
uv run visnavkit-train dataset=torch model=mimic
uv run visnavkit-train-route route=vae    # route-patch AE/VAE; model.goal_encoder.weights=<ckpt> seeds route_image
uv run visnavkit-export checkpoint=<ckpt> output=outputs/policy.onnx precision=fp32  # fp32 | fp16; + .pth, .metadata.json, .inputs.npz
uv run visnavkit-build-engine onnx=outputs/policy.onnx precision=fp16      # TensorRT fp32 | fp16 | bf16 (uv pip install tensorrt)
uv run visnavkit-check-export checkpoint=<ckpt> onnx=outputs/policy.onnx engine=outputs/policy.fp16.engine
```

Everything runs through `uv run`. Training records Git provenance; `strict_git=true` requires a
clean tree, untracked files included.

## Architecture in one screen

`NavigationPolicy` takes `vision (B, F, 3, H, W)`, any number of modality inputs and any number
of goals, and connects its stages with tokens of width `feat_size`:

- `models/vision` one RGB frame -> `(N, Kv, D)` tokens (any timm backbone).
- `models/modality` any non-image input -> per-frame tokens. An encoder declares `input_names`,
  which are at once the dataset keys, the `forward`/`predict` keywords and the ONNX input names.
- `models/temporal` mixes `(B, F, K, D)` across time; `models/goal` produces goal tokens.
- `models/action` decodes trajectories; generative decoders are a denoiser x a scheduler.
- `models/normalization.py` one `Normalizer` (`none|meanstd|minmax|scale`), used separately by
  supervision targets and by each input encoder.

Hydra groups mirror that layout under `visnavkit/configs/model/`. Full contracts:
[docs/architecture.md](docs/architecture.md).

## Rules

- **Concise everywhere.** Docs and comments state the contract and the non-obvious reason,
  nothing else.
- **Nothing is fixed.** Vision is the only input the policy knows by name. A new input is a
  `BaseModalityEncoder` plus a config entry — never a new argument in `policy.py`.
- **Paper tricks are opt-in.** Model-specific auxiliary losses (the per-frame speed head today)
  stay per-recipe, never baked into a base class, a loss dict or the export graph.
- **Focus only on the asked task.** No scope creep, no drive-by refactors.
- **Minimal code**: smallest viable diff, reuse proven components over new abstractions.
- **Review means review**: when asked for analysis, do not change code until told to.
- **Verify before claiming done**: run the sanity check or the real command and report actual
  shapes and losses, never assume.
- **Follow the source.** Paper-named recipes cite the paper and match the published code where
  it is public; where it is not, say so in the recipe rather than inventing architecture.

## Conventions

- Installable library: configs ship in the wheel via `visnavkit.configs` package data, so every
  yaml lives under `visnavkit/configs/`.
- Adding a component: subclass the stage base, add a yaml to the matching config group with an
  explicit `_target_`, run `uv run visnavkit-sanity-check model/<group>=<name> --onnx`, then add
  it to the group test in `tests/models/test_model_configs.py`.
- Tests must pass without network access or a dataset; use the synthetic fixtures in `tests/`,
  or drop a tiny corpus into `assets/datasets/` and `tests/data/test_assets.py` picks it up.
- Every paper a recipe adapts has an entry in `CITATION.bib`; add one (Google Scholar's BibTeX
  export, not a transcription) when adding a recipe.
