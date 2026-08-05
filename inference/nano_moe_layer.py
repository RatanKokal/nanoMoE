"""
NanoMoELayer — hardware-accelerated MoE dispatch/gather layer.

Replaces the token routing + weighted-reduce path of a HuggingFace
MixtralSparseMoeBlock with our zero-atomic, float4-vectorised CUDA kernels
from csrc/moe.cu, while keeping all expert weight matrices untouched.

Architecture
────────────
HF forward:  router_logits = gate(x)   →  softmax topk  →  loop over experts
                                                              + weighted sum

NanoMoE:     router_logits = gate(x)   →  custom_moe_cuda.route_and_permute()
             [permuted_x per-expert]   →  grouped GEMM (HF expert nn.Modules)
             [expert_out]              →  custom_moe_cuda.unpermute()

Input contract
──────────────
• x shape  : (batch, seq_len, d_model)   — standard transformer hidden state
• dtype    : float16 (production) or float32 (test mode)
• The HF block passed to __init__ must expose:
    .gate         : nn.Linear(d_model, num_experts, bias=False)
    .experts      : nn.ModuleList of Expert modules, each with .forward(x)->x
    The expert modules must accept and return (N_routed, d_model) tensors.

Precision note
──────────────
Our CUDA kernels operate in float32 internally. When the model runs in fp16,
we cast to float32 before routing and cast back after the weighted reduce,
matching the behaviour of the reference HF implementation which also promotes
to float32 for the softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import custom_moe_cuda  # compiled via setup.py
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


class NanoMoELayer(nn.Module):
    """
    Drop-in wrapper around a HuggingFace MixtralSparseMoeBlock that routes
    tokens through the nanoMoE CUDA engine.

    Parameters
    ----------
    hf_moe_block : nn.Module
        A pre-instantiated HuggingFace MixtralSparseMoeBlock (or any module
        that exposes ``.gate`` and ``.experts`` attributes as described above).
    num_experts : int
        Total number of expert FFN layers (must match hf_moe_block.experts).
    top_k : int
        Number of experts each token is dispatched to (usually 2 for Mixtral).
    fallback_to_hf : bool
        If True and the CUDA extension is not available, transparently fall
        back to the original HF forward pass instead of raising an error.
        Useful for CPU-only debugging environments.
    """

    def __init__(
        self,
        hf_moe_block: nn.Module,
        num_experts: int,
        top_k: int,
        fallback_to_hf: bool = True,
    ) -> None:
        super().__init__()

        # ── Validate the HF block surface ────────────────────────────────────
        if not hasattr(hf_moe_block, "gate"):
            raise AttributeError(
                "hf_moe_block must expose a '.gate' nn.Linear routing module."
            )
        if not hasattr(hf_moe_block, "experts"):
            raise AttributeError(
                "hf_moe_block must expose an '.experts' nn.ModuleList."
            )
        if len(hf_moe_block.experts) != num_experts:
            raise ValueError(
                f"num_experts={num_experts} does not match "
                f"len(hf_moe_block.experts)={len(hf_moe_block.experts)}."
            )

        # ── Store sub-modules (registered so .parameters() / .to() work) ────
        self.gate = hf_moe_block.gate          # nn.Linear  (d_model → E)
        self.experts = hf_moe_block.experts    # nn.ModuleList[Expert]

        self.num_experts = num_experts
        self.top_k = top_k
        self.fallback_to_hf = fallback_to_hf
        self._hf_block = hf_moe_block          # kept for fallback path

        if not _CUDA_AVAILABLE:
            if fallback_to_hf:
                import warnings
                warnings.warn(
                    "custom_moe_cuda not found — NanoMoELayer will use the "
                    "HuggingFace forward pass as a fallback. "
                    "Run `pip install -e .` from the repo root to compile.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise ImportError(
                    "custom_moe_cuda extension is not compiled. "
                    "Run `pip install -e .` from the repo root."
                )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        hidden_states : Tensor, shape (batch, seq_len, d_model)

        Returns
        -------
        tuple of:
          • hidden_states : Tensor, shape (batch, seq_len, d_model)
          • router_logits : None  (only used for load-balancing loss during
                                   training; safe to omit at inference time)

        This two-element tuple matches the HuggingFace MixtralSparseMoeBlock
        API contract, making NanoMoELayer a transparent drop-in replacement.
        """
        # Fall back to HF if the CUDA extension is unavailable.
        if not _CUDA_AVAILABLE:
            return self._hf_block(hidden_states)  # already a (Tensor, Tensor) tuple

        return self._nano_forward(hidden_states)

    def _nano_forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, d_model = hidden_states.shape
        orig_dtype = hidden_states.dtype

        # ── 1. Flatten to 2-D token matrix ───────────────────────────────────
        # Our CUDA kernels operate on (N_tokens, d_model) tensors.
        x_2d = hidden_states.view(-1, d_model)          # (N, d_model)
        N = x_2d.shape[0]

        # ── 2. Routing gate ───────────────────────────────────────────────────
        # Gate runs in float32 for numerical stability (matches HF behaviour).
        x_f32 = x_2d.float()
        W_g   = self.gate.weight.T.float()              # (d_model, E)

        # ── 3. CUDA route_and_permute ─────────────────────────────────────────
        # Returns: (permuted_x, reverse_map, topk_weights, offsets)
        # • permuted_x  : (N*k, d_model) float32 — tokens sorted by expert ID
        # • reverse_map : (N*k,)  int32  — for zero-atomic unpermute gather
        # • topk_weights: (N, k)  float32 — softmax scores
        # • offsets     : (E,)    int32  — exclusive prefix-sum of histogram
        permuted_x, reverse_map, topk_weights, offsets = \
            custom_moe_cuda.route_and_permute(x_f32, W_g, self.top_k)

        # ── 4. Per-expert GEMM ────────────────────────────────────────────────
        # Compute the token counts per expert from adjacent offsets.
        total_routed = N * self.top_k
        expert_out = torch.empty_like(permuted_x)

        for e_id in range(self.num_experts):
            start = offsets[e_id].item()
            end   = (offsets[e_id + 1].item()
                     if e_id + 1 < self.num_experts
                     else total_routed)
            length = end - start
            if length == 0:
                continue

            chunk = permuted_x[start:end]               # (len_e, d_model)

            # Cast to the expert's dtype, run, cast back to float32.
            chunk_typed = chunk.to(orig_dtype)
            with torch.no_grad() if not self.training else torch.enable_grad():
                result = self.experts[e_id](chunk_typed)
            expert_out[start:end] = result.float()

        # ── 5. Zero-atomic unpermute + weighted reduce ────────────────────────
        # unpermute(expert_out, reverse_map, topk_weights, N)
        # → (N, d_model) float32, each token = Σ w_j · expert_j(x)
        final_out = custom_moe_cuda.unpermute(
            expert_out, reverse_map, topk_weights, N
        )

        # ── 6. Restore original shape and dtype ───────────────────────────────
        final_out = final_out.to(orig_dtype).view(batch, seq_len, d_model)

        # Return a tuple to satisfy the HuggingFace API contract:
        # (hidden_states, router_logits). Router logits are only needed for
        # auxiliary load-balancing loss during training — None is safe here.
        return final_out, None

    # ── Utility ──────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"cuda_available={_CUDA_AVAILABLE}"
        )
