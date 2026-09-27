# Export, engines and the artifact check

```bash
uv run visnavkit-export checkpoint=<ckpt> output=outputs/policy.onnx precision=fp16   # + batch_size=1
uv run visnavkit-build-engine onnx=outputs/policy.onnx          # -> outputs/policy.fp16.engine
uv run visnavkit-check-export checkpoint=<ckpt> onnx=outputs/policy.onnx engine=outputs/policy.fp16.engine
uv run visnavkit-export-smoke model=gnm --precision fp16        # all of it on random weights, no data
```

## Precision

`precision` names the dtype the graph stores and its engine computes in. Graph inputs and outputs
are fp32 at every precision, so the consumer's code does not change.

| precision | stored weights | checked by | tolerance (rtol / atol) |
| --- | --- | --- | --- |
| `fp32` | float32 | ONNX Runtime, engine | 2e-3 / 2e-4 |
| `fp16` | float16, half the file | ONNX Runtime, engine | 1e-2 / 1e-2 |
| `bf16` | bfloat16, half the file: fp32's range at 8 significant bits | engine | 5e-2 / 3e-2 |

An output passes when `|artifact - reference| <= atol + rtol * |reference|` holds on every element,
the reference being the fp32 PyTorch model, in the outputs' own units (m, rad, m/s, probability).
Parity beyond the tolerance fails an export of checkpoint weights after every file is written;
`strict=false` reports it only. ONNX Runtime has no bf16 kernels: a bf16 export checks
the fp32 graph before the cast, and the engine check covers the cast itself. Pick `bf16` when `fp16`
overflows; it is not the faster of the two.

## Files

| file | content |
| --- | --- |
| `<output>.onnx` | the graph: fixed shapes at `batch_size`, weights at `precision` |
| `<output>.pth` | fp32 weights and the resolved config; loads without Lightning |
| `<output>.metadata.json` | io names and shapes, precision, parity errors, the config, sha256 of checkpoint, `.pth` and graph |
| `<output>.inputs.npz` | the traced inputs, which the check replays |
| `<engine>.metadata.json` | precision, TensorRT version, GPU, io, sha256 of the engine and of its source graph |

## TensorRT

- Install beside the locked environment: `uv pip install --python .venv tensorrt-cu12`
  (`tensorrt-cu13` on a CUDA 13 driver). TensorRT 10 or newer; tested on 11.3.
- Engines are strongly typed: one computes at its graph's precision, so an fp16 engine comes from
  an fp16 export. There is no precision option on the build.
- An fp32 engine is exact fp32. `tf32=true` lets matmuls and convolutions round to 10 mantissa
  bits on tensor cores (TensorRT's own default); the check then holds it to the fp16 tolerance.
- An engine runs only on the GPU model and TensorRT version that built it.
- Engines run through `libcudart` buffers, so neither build nor check needs a CUDA-enabled PyTorch.

## The check

`visnavkit-check-export` runs every artifact it is given on the same inputs and compares it with
the checkpoint (or the `.pth` when no checkpoint is given). It prints, per artifact, precision,
size, hash and median latency; per output, the maximum absolute and relative error with PASS or
FAIL; the decision and its endpoint distance to the reference's; and whether each sidecar hash
matches its file. The exit status is 1 when an output leaves its tolerance, is not finite, or a
hash does not match.

Random weights, batch 1, RTX 5080, TensorRT 11.3 (`visnavkit-export-smoke`; CPU runs on 8 threads):

| model | PyTorch CPU | ONNX Runtime CPU | engine fp32 | engine fp16 | engine bf16 |
| --- | --- | --- | --- | --- | --- |
| `model=gnm` (6.0M) | 29 ms | 12 ms | 1.2 ms | 0.9 ms | 1.0 ms |
| `experiment=flowpilot_dst_clips1k` (21.4M, 20 frames) | 360 ms | 323 ms | 10.7 ms | 5.1 ms | 13.1 ms |

fp32 passes on every output. Random weights amplify float noise, so fp16 and bf16 leave their
tolerance on some outputs there; the smoke reports parity without enforcing it. Trained weights
are the test that counts: run the check on the checkpoint before deploying a low precision.

A ranked output can fail the elementwise comparison while the decision agrees: when the last
returned plan and the first dropped one are nearly tied, a lower precision returns the other of
the two. The `decision` line tells the two cases apart, since it compares the top plan only.

## A new model

Two methods on the model, nothing in `visnavkit/export/`:

- `export_graph(cfg, batch_size, **options)` returns `(wrapper, inputs, input_names, output_names)`:
  the module to trace, its example inputs and the graph's io names. `options` are the export
  config's `export_heads` and its free `graph` dict, so a model's own option needs no shared
  config key: `+graph.top_k=3`.
- `decision(outputs)` returns `(label, endpoint_xy, lines)` from NumPy outputs by name: what the
  summaries print and the check compares.
