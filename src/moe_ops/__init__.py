"""Stable top-k indices for strided CPU float32 tensors.

``topk_experts(scores, k)`` matches stable descending argsort sliced to ``:k``
for nonnegative integer k. Leading dimensions are preserved; the output is int64.
NaNs precede numeric values, and equal scores keep their original index order.
Scalar inputs, unsupported devices/layouts/dtypes, and negative k raise RuntimeError
from the C++ extension. Selection produces integer indices and has no gradient.
"""

import torch as _torch  # Load the shared libraries needed by the extension.

from ._C import topk_experts

__all__ = ["top8_experts", "topk_experts"]


def top8_experts(scores: _torch.Tensor) -> _torch.Tensor:
    """Select up to eight experts using the general C++ top-k operation."""
    return topk_experts(scores, 8)
