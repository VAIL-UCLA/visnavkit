"""Deployment artifacts: an ONNX graph at a precision with its ``.pth`` / metadata / inputs sidecars, a TensorRT
engine built from it, and the check that replays the traced inputs through all of them.

Any model with ``export_graph(cfg, batch_size, **options)`` and ``decision(outputs)`` exports through the same
path (``graph.py``); ``scripts/`` holds the Hydra entry points.
"""
