import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import custom_moe_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False

# Requires transformers >= 4.51.
# MixtralSparseMoeBlock API (4.51+):
#   block.gate   → MixtralTopKRouter  (.weight: [num_experts, hidden])
#   block.experts → MixtralExperts   (.gate_up_proj: [E, 2*ffn, hidden],
#                                     .down_proj:     [E, hidden, ffn])
#   block.forward(x) → Tensor [B, S, D]  (single output, no router_logits)

class NanoMoELayer(nn.Module):
    def __init__(self, hf_moe_block: nn.Module, num_experts: int, top_k: int, fallback_to_hf: bool = True):
        super().__init__()
        self.gate    = hf_moe_block.gate
        self.experts = hf_moe_block.experts
        self.num_experts = num_experts
        self.top_k   = top_k
        self._hf_block   = hf_moe_block
        self._expert_fns = self._build_expert_fns(hf_moe_block)

    def _build_expert_fns(self, hf_moe_block: nn.Module) -> list:
        """
        Build a list of per-expert callables from the stacked MixtralExperts tensors.

        transformers >= 4.51 format:
          gate_up_proj : [E, 2*ffn, hidden]  — gate and up projections concatenated on dim 0
          down_proj    : [E, hidden, ffn]
          act_fn       : SiLU (or whatever config.hidden_act resolves to)
        """
        experts = hf_moe_block.experts

        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            raise RuntimeError(
                f"Expected transformers >= 4.51 MixtralExperts with gate_up_proj / down_proj, "
                f"but got class '{type(experts).__name__}' with attrs: "
                f"{[a for a in dir(experts) if not a.startswith('_')]}\n"
                f"Upgrade: pip install 'transformers>=4.51'"
            )

        act_fn = experts.act_fn
        fns = []
        for e in range(self.num_experts):
            w_gu = experts.gate_up_proj[e]   # [2*ffn, hidden]
            w_d  = experts.down_proj[e]      # [hidden, ffn]
            half = w_gu.shape[0] // 2
            fns.append(
                lambda x, _w_gu=w_gu, _w_d=w_d, _half=half:
                F.linear(
                    act_fn(F.linear(x, _w_gu[:_half])) * F.linear(x, _w_gu[_half:]),
                    _w_d,
                )
            )
        return fns

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not _CUDA_AVAILABLE:
            # transformers >= 4.51 returns a single 3-D tensor; wrap as (out, None).
            return self._hf_block(hidden_states), None
        return self._nano_forward(hidden_states)

    def _nano_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, d_model = hidden_states.shape
        orig_dtype = hidden_states.dtype

        x_2d = hidden_states.view(-1, d_model)
        N    = x_2d.shape[0]

        x_f32 = x_2d.float()
        # gate.weight is [num_experts, hidden] — same layout as nn.Linear.weight.
        W_g   = self.gate.weight.T.float()   # [hidden, num_experts]

        permuted_x, reverse_map, topk_weights, offsets = custom_moe_cuda.route_and_permute(
            x_f32, W_g, self.top_k
        )

        total_routed = N * self.top_k
        expert_out   = torch.empty_like(permuted_x)

        for e_id in range(self.num_experts):
            start = offsets[e_id].item()
            end   = offsets[e_id + 1].item() if e_id + 1 < self.num_experts else total_routed
            if end - start == 0:
                continue

            chunk = permuted_x[start:end].to(orig_dtype)
            with torch.no_grad():
                result = self._expert_fns[e_id](chunk)
            expert_out[start:end] = result.float()

        final_out = custom_moe_cuda.unpermute(expert_out, reverse_map, topk_weights, N)
        return final_out.to(orig_dtype).view(batch, seq_len, d_model), None
