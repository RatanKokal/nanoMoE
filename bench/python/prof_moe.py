import torch
import custom_moe_cuda
import custom_moe_legacy

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Native Reference Implementation (Vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def pytorch_reference_route_and_permute(x, W_g, k):
    N, d_model = x.shape
    E = W_g.shape[1]
    
    logits = torch.matmul(x, W_g)
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
    
    max_logits = torch.max(topk_logits, dim=-1, keepdim=True).values
    exp_logits = torch.exp(topk_logits - max_logits)
    topk_weights = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    flat_indices = topk_indices.view(-1)
    sort_idx = torch.argsort(flat_indices)
    
    token_ids = torch.arange(N, device=x.device).unsqueeze(1).repeat(1, k).view(-1)
    coo_indices = token_ids[sort_idx]
    coo_weights = topk_weights.view(-1)[sort_idx]
    permuted_x = x[coo_indices]
    
    histogram = torch.bincount(flat_indices, minlength=E).int()
    offsets = torch.zeros(E, dtype=torch.int32, device=x.device)
    offsets[1:] = torch.cumsum(histogram[:-1], dim=0).int()
    
    return permuted_x, coo_indices, coo_weights, offsets

def pytorch_reference_unpermute(expert_out, coo_indices, coo_weights, N):
    d_model = expert_out.size(1)
    weighted = expert_out * coo_weights.unsqueeze(1)
    final_out = torch.zeros(N, d_model, device=expert_out.device, dtype=torch.float32)
    final_out.index_add_(0, coo_indices, weighted)
    return final_out

# ─────────────────────────────────────────────────────────────────────────────
# Profiling Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_pytorch(x, W_g, k, N, iters=50, warmup=10):
    for _ in range(warmup):
        p_x, c_idx, c_wt, _ = pytorch_reference_route_and_permute(x, W_g, k)
        pytorch_reference_unpermute(p_x, c_idx, c_wt, N)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push("PyTorch_Native_Reference")
    start_evt.record()
    for _ in range(iters):
        p_x, c_idx, c_wt, _ = pytorch_reference_route_and_permute(x, W_g, k)
        pytorch_reference_unpermute(p_x, c_idx, c_wt, N)
    end_evt.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    total_ms = start_evt.elapsed_time(end_evt)
    avg_us = (total_ms / iters) * 1000.0
    avg_ms = total_ms / iters
    return avg_us, avg_ms

def benchmark_engine(engine, engine_name, x, W_g, k, N, iters=50, warmup=10):
    for _ in range(warmup):
        res = engine.route_and_permute(x, W_g, k)
        engine.unpermute(res[0], res[1], res[2], N)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push(f"{engine_name}_Routing_Unpermute")
    start_evt.record()
    for _ in range(iters):
        res = engine.route_and_permute(x, W_g, k)
        engine.unpermute(res[0], res[1], res[2], N)
    end_evt.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    total_ms = start_evt.elapsed_time(end_evt)
    avg_us = (total_ms / iters) * 1000.0
    avg_ms = total_ms / iters
    return avg_us, avg_ms

if __name__ == "__main__":
    # Setup production-scale dimensions
    N = 4096         # Massive batch of tokens
    d_model = 1024   # Standard small LLM dimension
    E = 8            # 8 Experts
    k = 2            # Top-2 Routing
    iters = 50
    
    print("=" * 70)
    print("NanoMoE 3-Way Comparative Profiler: PyTorch vs. Legacy vs. Optimized CUDA")
    print(f"Dimensions -> Tokens (N): {N}, d_model: {d_model}, Experts (E): {E}, k: {k}")
    print("=" * 70)
    
    device = torch.device('cuda')
    x = torch.randn(N, d_model, dtype=torch.float32, device=device)
    W_g = torch.randn(d_model, E, dtype=torch.float32, device=device)

    # 1. Profile PyTorch Native Reference
    print("\nProfiling 1/3: PyTorch Native Reference...")
    pt_us, pt_ms = benchmark_pytorch(x, W_g, k, N, iters=iters)
    print(f" -> PyTorch Native:         {pt_us:9.2f} µs ({pt_ms:6.3f} ms)")

    # 2. Profile Legacy CUDA Engine (csrc/moe_legacy.cu)
    print("\nProfiling 2/3: Legacy CUDA Engine (custom_moe_legacy)...")
    legacy_us, legacy_ms = benchmark_engine(custom_moe_legacy, "Legacy_CUDA_Engine", x, W_g, k, N, iters=iters)
    print(f" -> Legacy CUDA Engine:    {legacy_us:9.2f} µs ({legacy_ms:6.3f} ms)")

    # 3. Profile Optimized CUDA Engine (csrc/moe.cu)
    print("\nProfiling 3/3: Optimized CUDA Engine (custom_moe_cuda)...")
    opt_us, opt_ms = benchmark_engine(custom_moe_cuda, "Optimized_CUDA_Engine", x, W_g, k, N, iters=iters)
    print(f" -> Optimized CUDA Engine: {opt_us:9.2f} µs ({opt_ms:6.3f} ms)")

    # 4. Print Performance Summary & Speedup Comparisons
    speedup_vs_legacy = legacy_us / opt_us if opt_us > 0 else 0.0
    speedup_vs_pytorch = pt_us / opt_us if opt_us > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON SUMMARY:")
    print(f"  1. PyTorch Native Reference:        {pt_us:9.2f} µs ({pt_ms:6.3f} ms) | [Baseline]")
    print(f"  2. Legacy Engine (custom_moe_legacy):  {legacy_us:9.2f} µs ({legacy_ms:6.3f} ms) | {legacy_us/pt_us:6.2f}x vs PyTorch")
    print(f"  3. Optimized Engine (custom_moe_cuda):   {opt_us:9.2f} µs ({opt_ms:6.3f} ms) | {speedup_vs_legacy:6.2f}x vs Legacy | {speedup_vs_pytorch:6.2f}x vs PyTorch")
    print("=" * 70)