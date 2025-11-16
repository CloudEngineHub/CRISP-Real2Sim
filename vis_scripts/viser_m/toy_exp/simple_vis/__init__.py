"""
Light‑weight stand‑in for the heavy NeuralPlane frontend.

Import `SimpleVis` from here when you want to avoid the Nerfstudio /
NeuralPlane dependency tree.  The class mimics the small slice of the API
that `visualizer_megasam.py` and friends rely on (callable that returns
plane predictions + depth/normal tensors).
"""

from .vis_normal import SimpleVis

__all__ = ["SimpleVis"]
