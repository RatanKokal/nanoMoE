import sys, os, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MixtralConfig, MixtralForCausalLM

# ── NanoMoELayer shim with the Tuple Fix ──────────────────────────────────────
class NanoMoELayerForBench(nn.Module):
    def __init__(self, hf_moe_block, num_experts: int, top_k: int, cuda_ext):
        super().__init__()
        self.gate       = hf_moe_block.gate
        self.experts    = hf_moe_block.experts
        self.num_experts = num_experts
        self.top_k      = top_k
        self._ext       = cuda_ext
        self._act       = hf_moe_block.experts.act_fn
        self._expert_fns = self._build_fns()

    def _build_fns(self):
        gu = self.experts.gate_up_proj
        d  = self.experts.down_proj
        act = self._act
        half = gu.shape[1] // 2
        fns = []
        for e in range(self.num_experts):
            fns.append(
                lambda x, _w=gu[e], _d=d[e], _h=half:
                F.linear(act(F.linear(x, _w[:_h])) * F.linear(x, _w[_h:]), _d)
            )
        return fns

    def forward(self, hidden_states):
        batch, seq_len, d_model = hidden_states.shape
        orig_dtype = hidden_states.dtype

        x_2d  = hidden_states.view(-1, d_model)
        N      = x_2d.shape[0]
        x_f32  = x_2d.float()
        W_g    = self.gate.weight.T.float()

        permuted_x, reverse_map, topk_weights, offsets = \
            self._ext.route_and_permute(x_f32, W_g, self.top_k)

        total_routed = N * self.top_k
        expert_out   = torch.empty_like(permuted_x)

        for e_id in range(self.num_experts):
            start = offsets[e_id].item()
            end   = offsets[e_id + 1].item() if e_id + 1 < self.num_experts else total_routed
            if end - start == 0:
                continue
            chunk = permuted_x[start:end].to(orig_dtype)
            with torch.no_grad():
                expert_out[start:end] = self._expert_fns[e_id](chunk).float()

        final = self._ext.unpermute(expert_out, reverse_map, topk_weights, N)
        return final.to(orig_dtype).view(batch, seq_len, d_model)

def patch_with_ext(hf_model, cuda_ext, num_experts, top_k):
    def _replace(module):
        for name, child in module.named_children():
            if child.__class__.__name__ == "MixtralSparseMoeBlock":
                setattr(module, name, NanoMoELayerForBench(child, num_experts, top_k, cuda_ext))
            else:
                _replace(child)
    _replace(hf_model)
    return hf_model

# ── Timed Prefill Region ──────────────────────────────────────────────────────
def measure_prefill_tps(model, batch_size, seq_len, warmup, runs):
    """Feeds a massive block of tokens through the network in a single pass."""
    # Generate a massive synthetic prompt matrix
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device="cuda")
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids)
            
    # Timed Measurement
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for _ in range(runs):
        with torch.no_grad():
            _ = model(input_ids)
            
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    total_tokens = batch_size * seq_len * runs
    return total_tokens / elapsed if elapsed > 0 else 0.0, total_tokens

# ── Main Benchmark ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=1024)
    args = parser.parse_args()

    DEVICE = "cuda"
    DTYPE  = torch.float16
    
    import custom_moe_cuda as ext_new
    import custom_moe_legacy as ext_legacy

    # Upscaling dimensions to match your C++ benchmark (N=4096, hidden=1024)
    config = MixtralConfig(
        vocab_size=32000, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
        num_local_experts=8, num_experts_per_tok=2
    )

    print("=" * 62)
    print("  NanoMoE — Prefill Throughput Benchmark (Massive Batches)")
    print("=" * 62)
    print(f"  Tokens per pass: {args.batch * args.seq} (Batch: {args.batch}, Seq: {args.seq})")
    
    results = {}

    print("\n[1/3] HF Baseline (PyTorch Native)...")
    model_hf = MixtralForCausalLM(config).to(device=DEVICE, dtype=DTYPE).eval()
    tps, tok = measure_prefill_tps(model_hf, args.batch, args.seq, warmup=2, runs=10)
    results["HF Baseline"] = tps
    print(f"  → {tps:,.1f} TPS")

    print("\n[2/3] Legacy CUDA (Atomic + Serial Prefix-Sum)...")
    model_legacy = MixtralForCausalLM(config).to(device=DEVICE, dtype=DTYPE).eval()
    model_legacy = patch_with_ext(model_legacy, ext_legacy, 8, 2)
    tps, tok = measure_prefill_tps(model_legacy, args.batch, args.seq, warmup=2, runs=10)
    results["Legacy CUDA"] = tps
    print(f"  → {tps:,.1f} TPS")

    print("\n[3/3] NanoMoE (CUB ExclusiveSum + Zero-Atomic float4)...")
    model_nano = MixtralForCausalLM(config).to(device=DEVICE, dtype=DTYPE).eval()
    model_nano = patch_with_ext(model_nano, ext_new, 8, 2)
    tps, tok = measure_prefill_tps(model_nano, args.batch, args.seq, warmup=2, runs=10)
    results["NanoMoE"] = tps
    print(f"  → {tps:,.1f} TPS")

    baseline = results["HF Baseline"]
    print("\n" + "=" * 62)
    print("  PREFILL RESULTS (Tokens/sec)  │  Speedup vs HF")
    print("─" * 62)
    for label, tps in results.items():
        print(f"  {label:<30}│  {tps:>10.1f}  ({tps/baseline:.2f}x)")
    print("=" * 62)

if __name__ == "__main__":
    main()