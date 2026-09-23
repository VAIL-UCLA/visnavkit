"""Deployment artifacts: an ONNX graph at a precision with its ``.pth`` / metadata / inputs sidecars, a TensorRT
engine built from it, and the check that replays the traced inputs through all of them.

Two export families share the machinery: ``policy`` (``NavigationPolicy.predict``, one frame + feature buffer)
and ``dst`` (``FlowPilotDST.deploy``, one window). Each exposes ``prepare_graph`` and ``decision``; ``scripts/``
holds the Hydra entry points.
"""
