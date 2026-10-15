"""Stable top-k expert selection for CPU float32 tensors."""

import torch as _torch  # Load the shared libraries needed by the extension.

from ._C import topk_experts

__all__ = ["topk_experts", "top8_experts"]


def top8_experts(scores: _torch.Tensor) -> _torch.Tensor:
    """Select up to eight experts using the general C++ top-k operation."""
    return topk_experts(scores, 8)
