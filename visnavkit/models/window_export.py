"""The window deployment graph of a model with ``deploy(vision, goal, route_patch, ego, action_bounds, k[, noise])``:
the ``seq_length`` window in, the current frame's plans out. Every slot holds a frame and a route patch; pad a
short history by repeating the oldest frame. ``docs/flowpilot_dst_onnx.md`` documents the contract."""

import numpy as np
import torch
import torch.nn as nn

OUTPUTS = ["modes", "probs", "speed"]


class _ExportWindow(nn.Module):
    """Positional ``deploy`` wrapper for the tracer."""

    def __init__(self, model, names, top_k):
        super().__init__()
        self.model, self.names, self.top_k = model, names, top_k

    def forward(self, *inputs):
        feeds = dict(zip(self.names, inputs))
        options = {name: feeds[name] for name in ("noise",) if name in feeds}
        if self.top_k:
            options["k"] = self.top_k
        return self.model.deploy(
            feeds["vision"], feeds.get("goal"), feeds["route_patch"], feeds["ego"], feeds["action_bounds"], **options
        )


class WindowExport:
    """``export_graph`` / ``decision`` of such a model (``visnavkit/export``); ``export_inputs`` names what its
    graph takes."""

    export_inputs = ("vision", "route_patch", "goal", "ego", "action_bounds")

    def export_graph(self, cfg, batch_size=1, top_k=None, noise="zero", num_samples=1, **_):
        """``(wrapper, inputs, input_names, output_names)``. ``top_k``: plans per call (default: ``deploy``'s).
        ``noise``: ``zero``, the deterministic path, or ``randn``, a ``noise`` input ``(B, num_samples, T, 5)``
        the caller fills with N(0, I) (models whose ``deploy`` takes ``noise``)."""
        if noise not in ("zero", "randn"):
            raise ValueError(f"noise must be zero or randn, got {noise!r}")
        b, t = batch_size, int(cfg.common.seq_length)
        w, h = (int(v // cfg.common.downscale_factor) for v in cfg.common.crop_wh)
        device, route = next(self.parameters()).device, self.route_encoder
        bounds = torch.tensor([[-0.0, -0.09, -0.05, 0.0, -0.85], [0.14, 0.09, 0.05, 2.76, 0.85]], device=device)
        examples = dict(
            vision=torch.rand(b, t, 3, h, w, device=device),
            route_patch=torch.randint(0, route.num_classes, (b, t, *route.hw), device=device).float(),
            goal=torch.tensor([8.0, 1.0, 0.0], device=device).expand(b, t, 3).contiguous(),
            ego=torch.tensor([1.0, 0.0], device=device).expand(b, t, 2).contiguous(),
            action_bounds=bounds.expand(b, 2, 5),
        )
        names = list(self.export_inputs)
        if noise == "randn":
            examples["noise"] = torch.randn(b, num_samples, self.action_decoder.num_pts, 5, device=device)
            names.append("noise")
        return _ExportWindow(self, names, top_k).eval(), tuple(examples[name] for name in names), names, OUTPUTS

    def decision(self, outputs):
        """The top plan of NumPy ``outputs`` (by name): a label, its endpoint (x, y) in metres and the SANITY
        CHECK lines."""
        modes, probs, speed = (np.asarray(outputs[name]) for name in OUTPUTS)
        lines = [
            f"probs: {np.round(probs[0], 3)}",
            f"best plan p0 [x y yaw v w]: {np.round(modes[0, 0, 0], 3)}",
            f"best plan pN [x y yaw v w]: {np.round(modes[0, 0, -1], 3)}",
            f"speed: {float(speed.reshape(-1)[0]):.4f}",
        ]
        return f"top plan p={probs[0, 0]:.3f}", modes[0, 0, -1, :2].astype(np.float64), lines
