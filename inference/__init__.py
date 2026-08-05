"""
nanoMoE inference package.

Provides NanoMoELayer — a drop-in replacement for HuggingFace MoE blocks
that routes tokens through our custom CUDA kernels (csrc/moe.cu).
"""

from .nano_moe_layer import NanoMoELayer

__all__ = ["NanoMoELayer"]
