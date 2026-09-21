# FlowPilot-DST ONNX: inputs, outputs, usage

One call = one decision. The graph takes the 20-slot observation window and returns the current
frame's top-`k` plans. Shapes are fixed (batch 1), so every slot must carry a frame and a route
patch; at startup repeat the oldest frame and patch.

```bash
uv run visnavkit-export-dst \
  checkpoint=logs/visnavkit/<run>/checkpoints/last.ckpt output=flowpilot_dst.onnx
```
The export writes `flowpilot_dst.onnx`, `.metadata.json` (shapes, anchor times, parity, checkpoint
hash) and `.inputs.npz` (the traced sample inputs, for a first smoke run). fp32, opset 17.

## Inputs

| name | shape | dtype | meaning |
| --- | --- | --- | --- |
| `vision` | (1, 20, 3, 216, 384) | float32 | the last 20 frames at 20 Hz, oldest first, RGB in [0, 1]. Slot 19 is now. The model normalises with the ImageNet statistics itself. |
| `route_patch` | (1, 20, 80, 80) | float32 | per slot, the route raster around the ego: class ids 0 background, 1 sidewalk, 2 crosswalk, held as floats. 0.5 m per pixel, 40 x 40 m; row 0 is 20 m ahead, column 0 is 20 m to the left, both in that slot's ego frame. |
| `goal` | (1, 20, 3) | float32 | per slot `[distance_m, cos, sin]` of the goal in that slot's ego frame, with the heading angle measured from straight ahead, positive to the left. |
| `ego` | (1, 20, 2) | float32 | per slot `[v_mps, w_radps]`, the measured speed and yaw rate. |
| `action_bounds` | (1, 2, 5) | float32 | `[[lo x 5], [hi x 5]]` of the per-step `[dx, dy, dyaw, v, w]`, the same row the recipe trained with. Use the corpus's entry in `action_bounds.json` (clips1k for this model), the same numbers for every call. |

Frames must be prepared exactly as in training:
1. **Resize** to 384 x 216 (the trained resolution), then scale to [0, 1].
2. **Calibration**, clips1k only: shift the frame so that `out(x, y) = in(x + dx, y + dy)` with the
   clip's `principal_point_delta`, bilinear, zeros outside. Skip this for a camera that is already
   centred; a shifted camera without this step degrades the plan.
3. **Ordering**: oldest to newest, exactly 50 ms apart. The encoder pairs each frame with the one
   before, so an irregular gap is seen as motion.

There is no embodiment input: this export is trained on clips1k alone with the embodiment token off.

## Outputs

| name | shape | meaning |
| --- | --- | --- |
| `modes` | (1, 6, 80, 5) | the top 6 plans for the current frame, ranked, each 80 steps of `[x_m, y_m, yaw_rad, v_mps, w_radps]` in the current ego frame: x forward, y left, yaw relative to now. Steps are 0.05 s apart, covering 0.05 s to 4.0 s. |
| `probs` | (1, 6) | softmax scores of those 6, descending. They sum to less than 1, since the other 58 anchors are dropped. |
| `speed` | (1, 1) | the auxiliary speed head on the current frame, m/s. Diagnostic; the planner does not read it. |

`modes[0, 0]` is the plan to drive. It is deterministic: decoding starts from zero noise, so the
same window always gives the same plan.

## Usage

```python
import numpy as np, onnxruntime as ort

session = ort.InferenceSession("flowpilot_dst.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
feeds = dict(np.load("flowpilot_dst.inputs.npz"))          # replace with live data
modes, probs, speed = session.run(["modes", "probs", "speed"], feeds)

plan = modes[0, 0]                                          # (80, 5) best plan
x, y, yaw, v, w = plan.T
print(f"p=%.2f  v[0]=%.2f m/s  w[0]=%.2f rad/s  end=(%.1f, %.1f) m" % (probs[0, 0], v[0], w[0], x[-1], y[-1]))
```

**Controller.** Take the command from the first steps, not the endpoint: `v[0]`, `w[0]` are the
target speed and yaw rate for the next 50 ms. A pure-pursuit controller can instead track the `x, y`
path directly.

**Multimodality.** The 6 plans are separate hypotheses, not a distribution to average. Averaging
two plans that pass an obstacle on opposite sides produces a plan through it. Pick one, or reject
plans that collide and take the highest remaining probability.

**Drawing a plan on the frame** (the same projection as the training panels, `utils/plan_viz.py`),
with the camera vector `[fx, fy, cx, cy, k1..k4, cam_height_m, cam_type]` from the clip's
`camera.json` scaled to 384 x 216:

```python
X, Y, Z = -y, np.full_like(x, cam_height_m), x            # ego ground point -> camera axes
xn, yn = X / Z, Y / Z                                      # Z > 0.05 only
r = np.hypot(xn, yn); th = np.arctan(r)                    # fisheye (cam_type 1)
s = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8) / np.maximum(r, 1e-9)
u, v_px = fx * xn * s + cx, fy * yn * s + cy
```
For a pinhole camera (`cam_type` 0) drop the `s` factor.

## Performance and checks

- **Parity:** the export asserts PyTorch/ONNX agreement on the traced window (rtol 2e-3, atol 2e-4)
  and records the maximum error per output in the metadata.
- **Cost:** one call encodes all 20 frames through FastViT-T12, so it is about 20 frame encodes plus
  4 flow steps: 1.35 s per call on this machine's CPU (ONNX Runtime, single session). Use the CUDA
  provider for anything interactive. To run at 20 Hz on a CPU, cache per-frame features instead;
  that needs a streaming graph, which this export does not provide.
- **On a real clips1k validation window** the graph matched PyTorch to 3.0e-6, and the top plan's
  ADE against the ground truth was 0.45 m over 4 s (the epoch-5 checkpoint).
- **A wrong `action_bounds` row silently rescales every output**, since the plan is decoded from
  normalised steps. If plans look too short or too fast, check that row first.
