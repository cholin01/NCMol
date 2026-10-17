"""Small fallbacks for optional PyG compiled extensions.

The NVIDIA base image used for this experiment contains PyTorch Geometric but
not torch_scatter/torch_cluster.  LiTEN only needs scatter for the selected
non-periodic path; the implementations below keep the original source tree
untouched and use native PyTorch operations.
"""

from __future__ import annotations

import sys
import types

import torch


def _scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
    if dim < 0:
        dim += src.dim()
    if dim != 0:
        src = src.transpose(0, dim)
        result = _scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=reduce)
        return result.transpose(0, dim)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    shape = (dim_size,) + tuple(src.shape[1:])
    if reduce in {"sum", "add"}:
        result = src.new_zeros(shape) if out is None else out
        result.index_add_(0, index, src)
        return result
    if reduce == "mean":
        result = src.new_zeros(shape)
        result.index_add_(0, index, src)
        counts = src.new_zeros(dim_size)
        counts.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
        return result / counts.clamp_min(1).view((-1,) + (1,) * (src.dim() - 1))
    if reduce in {"min", "max"}:
        name = "amin" if reduce == "min" else "amax"
        fill = torch.inf if reduce == "min" else -torch.inf
        result = src.new_full(shape, fill)
        expanded = index.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
        result.scatter_reduce_(0, expanded, src, reduce=name, include_self=True)
        return result
    raise ValueError(f"Unsupported scatter reduction: {reduce}")


def install_fallbacks() -> None:
    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        module = types.ModuleType("torch_scatter")
        module.scatter = _scatter
        sys.modules["torch_scatter"] = module

    try:
        import torch_cluster  # noqa: F401
    except ImportError:
        module = types.ModuleType("torch_cluster")

        def radius_graph(*_args, **_kwargs):
            raise RuntimeError("radius_graph fallback was called; edges must be precomputed")

        module.radius_graph = radius_graph
        sys.modules["torch_cluster"] = module
