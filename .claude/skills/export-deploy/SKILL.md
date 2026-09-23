---
name: export-deploy
description: Export a visnavkit checkpoint to ONNX and debug export failures. Use when producing a deployment model or when torch.onnx.export breaks.
---

# Export / deploy

```bash
uv run visnavkit-export checkpoint=<ckpt> output=<out.onnx> precision=fp32|fp16  # checkpoint=null: untrained pipeline check (parity reported, not enforced)
uv run visnavkit-build-engine onnx=<out.onnx>                # TensorRT engine at the graph's precision + metadata (uv pip install --python .venv tensorrt-cu12, matching the driver's CUDA)
uv run visnavkit-check-export checkpoint=<ckpt> pth=<out.pth> onnx=<out.onnx> engine=<engine>  # the traced inputs through every artifact
uv run visnavkit-export-smoke <overrides> [--precision fp16 --engine-precision bf16 --no-engine]  # random weights through the whole path
```

## What `export/policy.py` does (`scripts/export.py` is its Hydra main)
1. Restores the checkpoint's model/preprocessing config (`prepare_export_config`), flips `temporal_encoder.reduction` `none -> last`.
2. `LitModel.load_from_checkpoint(..., weights_only=False)`, then `reparameterize_model`: `.reparameterize()` (FastViT) + Linear->BatchNorm1d folding; `vision_encoder.prepare_for_export((h, w))` precomputes ViT position embeddings.
3. Builds example inputs from `policy.example_inputs(...)`; input names are presence-driven: `vision (1,3,h,w)`, `feature_buffer (1, seq_step*(seq_len-1), K*feat_size)`, then one `goal` input per goal encoder (`goal`, or `goal_0..n`), one input per key the modality encoders read (`ego (1,E)`, `intrinsics (1,3,3)`, `extrinsics (1,4,4)`, ...), and `noise (1, M, T, A)` for generative decoders. Outputs: `plan, feat_out`, then `speed` when `vision_encoder.speed_head=true`, then `*heads`.
4. Traces `policy.predict` with the MHA fastpath disabled, slims (onnxslim, optional extra), casts the weights to `precision` (io kept fp32), enforces output order, runs ONNX Runtime parity on the same nonzero inputs at that precision's tolerance (`export/artifacts.py`) and writes `.pth` (fp32 weights + config), `.metadata.json` and `.inputs.npz` beside the graph.

## Debugging
- Unsupported aten op: usually a training-only branch; confirm `eval()` and the reduction flip. New denoisers must avoid data-dependent control flow (fixed `sample_steps`).
- Shape mismatch at parity: `feature_idxs` gathering in `NavigationPolicy.predict` must match the buffer layout (newest last, width `K * feat_size`).
- Goal/noise missing in the graph: check `policy.export_input_names()`; None inputs are dropped by design.
- fp16 NaNs: rerun with `precision=fp32` to isolate; a `bf16` engine tolerates 2e-2 rel, check the endpoint delta.
- Engine build: TensorRT 11 is strongly typed, so an fp16 engine needs an fp16 export (TensorRT 10 can still cast with `precision=`); only the batch axis may be dynamic (`batch=[min,opt,max]`); an engine runs only on the GPU / TensorRT version that built it.
- Paste the EXPORT + SANITY CHECK blocks and the EXPORT CHECK table in the PR/report.
