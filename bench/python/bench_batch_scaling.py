import sys, os, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MixtralConfig, MixtralForCausalLM

# ── NanoMoELayer shim ─────────────────────────────────────────────────────────
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

# ── Timed Batched Decoding ────────────────────────────────────────────────────
def measure_batched_decoding(model, batch_size, runs=50, warmup=10):
    """
    Simulates a Continuous Batching decoding step.
    Input shape is [Batch_Size, 1], meaning 1 token per user.
    """
    input_ids = torch.randint(0, 32000, (batch_size, 1), device="cuda")
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids)
            
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for _ in range(runs):
        with torch.no_grad():
            _ = model(input_ids)
            
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    total_tokens = batch_size * runs
    return total_tokens / elapsed if elapsed > 0 else 0.0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        import custom_moe_cuda as ext_new
    except ImportError:
        print("[ERROR] custom_moe_cuda not found.")
        sys.exit(1)

    DEVICE = "cuda"
    DTYPE  = torch.float16
    
    config = MixtralConfig(
        vocab_size=32000, hidden_size=512, intermediate_size=2048,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
        num_local_experts=8, num_experts_per_tok=2
    )

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    hf_tps = []
    nano_tps = []

    print("=" * 65)
    print("  NanoMoE — The Turning Point: Continuous Batching Scaling")
    print("=" * 65)

    # 1. Native PyTorch
    print("[1/2] Profiling HF Baseline (Native PyTorch)...")
    model_hf = MixtralForCausalLM(config).to(device=DEVICE, dtype=DTYPE).eval()
    for bs in batch_sizes:
        tps = measure_batched_decoding(model_hf, bs)
        hf_tps.append(tps)

    # 2. NanoMoE
    print("[2/2] Profiling NanoMoE (Zero-Atomic + CUB)...")
    model_nano = patch_with_ext(model_hf, ext_new, 8, 2)
    for bs in batch_sizes:
        tps = measure_batched_decoding(model_nano, bs)
        nano_tps.append(tps)

    # 3. Print Results
    print("\n" + "=" * 65)
    print(f" {'Batch Size':<12} | {'PyTorch TPS':<15} | {'NanoMoE TPS':<15} | {'Winner'}")
    print("-" * 65)
    
    for i, bs in enumerate(batch_sizes):
        pt = hf_tps[i]
        nano = nano_tps[i]
        winner = "PyTorch" if pt > nano else "NanoMoE"
        diff = max(pt, nano) / min(pt, nano)
        
        print(f" {bs:<12} | {pt:<15.1f} | {nano:<15.1f} | {winner} ({diff:.2f}x)")
    print("=" * 65)

    # 4. Plot (if matplotlib is installed)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, hf_tps, marker='o', label='Native PyTorch', color='red')
        plt.plot(batch_sizes, nano_tps, marker='o', label='NanoMoE', color='blue')
        plt.title('Decoding Throughput vs. Batch Size (The Turning Point)')
        plt.xlabel('Batch Size (Simultaneous Users)')
        plt.ylabel('Throughput (Tokens / Second)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.xscale('log', base=2)
        
        # Define x-ticks explicitly
        plt.xticks(batch_sizes, labels=[str(b) for b in batch_sizes])
        
        plt.tight_layout()
        plt.savefig('batch_scaling.png')
        print("\n[Success] Graph saved as 'batch_scaling.png'!")
    except ImportError:
        print("\n[Tip] Install matplotlib (`pip install matplotlib`) to automatically generate a graph of this crossover.")

if __name__ == "__main__":
    main()