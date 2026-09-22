"""Row selection that keeps fixed shapes under the ONNX tracer."""

import torch


def masked_rows(mask):
    """``(B, T)`` bool -> flat indices of the True entries; a static ``arange`` while tracing a full mask,
    so the export graph keeps fixed shapes."""
    flat = mask.reshape(-1)
    if torch.jit.is_tracing() and bool(flat.all()):
        return torch.arange(flat.numel(), device=mask.device)
    return flat.nonzero().squeeze(1)
