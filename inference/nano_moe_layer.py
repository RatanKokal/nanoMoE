import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import custom_moe_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False

class NanoMoELayer(nn.Module):
    def __init__(self, hf_moe_block: nn.Module, num_experts: int, top_k: int, fallback_to_hf: bool = True):
        super().__init__()
        self.gate = hf_moe_block.gate
        self.experts = hf_moe_block.experts
        self.num_experts = num_experts
        self.top_k = top_k
        self._hf_block = hf_moe_block
        self._expert_fns = self._build_expert_fns(hf_moe_block)

    def _build_expert_fns(self, hf_moe_block: nn.Module) -> list:
        experts = hf_moe_block.experts
        # act_fn may live on the experts object (fused format) or on the block itself
        act_fn = getattr(experts, "act_fn", None) or getattr(hf_moe_block, "act_fn", nn.SiLU())

        # ── Format 1: nn.ModuleList / indexable list of individual expert modules ──
        # Each element is a callable MLP (e.g. MixtralBLockSparseTop2MLP pre-4.44)
        if isinstance(experts, (nn.ModuleList, list)):
            return [experts[i] for i in range(self.num_experts)]

        # ── Format 2: MixtralExperts (transformers 4.44–4.50) ─────────────────────
        # Stacked tensors: w1[E, ffn, hidden], w3[E, ffn, hidden], w2[E, hidden, ffn]
        # Mixtral convention: w1=gate, w3=up, w2=down
        if hasattr(experts, "w1") and hasattr(experts, "w2") and hasattr(experts, "w3"):
            fns = []
            for e in range(self.num_experts):
                fns.append(
                    lambda x, w1=experts.w1[e], w2=experts.w2[e], w3=experts.w3[e]:
                    F.linear(act_fn(F.linear(x, w1)) * F.linear(x, w3), w2)
                )
            return fns

        # ── Format 3: Fused gate_up_proj (transformers ≥ 4.51) ───────────────────
        # gate_up_proj: [E, 2*ffn, hidden]  (gate and up concatenated along dim 0)
        # down_proj:    [E, hidden, ffn]
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            fns = []
            for e in range(self.num_experts):
                fns.append(
                    lambda x, w_gu=experts.gate_up_proj[e], w_d=experts.down_proj[e]:
                    F.linear(
                        act_fn(F.linear(x, w_gu[:w_gu.shape[0] // 2]))
                        * F.linear(x, w_gu[w_gu.shape[0] // 2:]),
                        w_d,
                    )
                )
            return fns

        raise RuntimeError(
            f"Unknown experts format — cannot build expert callables.\n"
            f"Class: {type(experts).__name__}\n"
            f"Relevant attrs: gate_up_proj={hasattr(experts, 'gate_up_proj')}, "
            f"down_proj={hasattr(experts, 'down_proj')}, "
            f"w1={hasattr(experts, 'w1')}, w2={hasattr(experts, 'w2')}, w3={hasattr(experts, 'w3')}\n"
            f"Full dir: {[a for a in dir(experts) if not a.startswith('_')]}"
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not _CUDA_AVAILABLE:
            return self._hf_block(hidden_states)
        return self._nano_forward(hidden_states)

    def _nano_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, d_model = hidden_states.shape
        orig_dtype = hidden_states.dtype

        x_2d = hidden_states.view(-1, d_model)
        N = x_2d.shape[0]

        x_f32 = x_2d.float()
        W_g   = self.gate.weight.T.float()

        permuted_x, reverse_map, topk_weights, offsets = custom_moe_cuda.route_and_permute(x_f32, W_g, self.top_k)

        total_routed = N * self.top_k
        expert_out = torch.empty_like(permuted_x)

        for e_id in range(self.num_experts):
            start = offsets[e_id].item()
            end   = (offsets[e_id + 1].item() if e_id + 1 < self.num_experts else total_routed)
            if end - start == 0:
                continue

            chunk = permuted_x[start:end].to(orig_dtype)
            with torch.no_grad():
                result = self._expert_fns[e_id](chunk)
            expert_out[start:end] = result.float()

        final_out = custom_moe_cuda.unpermute(expert_out, reverse_map, topk_weights, N)
        return final_out.to(orig_dtype).view(batch, seq_len, d_model), None
