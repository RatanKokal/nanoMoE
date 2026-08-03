import torch
import custom_moe_cuda

def pytorch_reference_routing(x, W_g, k):
    N, d_model = x.shape
    E = W_g.shape[1]
    
    # 1. Logits and Top-K
    logits = torch.matmul(x, W_g)
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
    
    # 2. Softmax
    max_logits = torch.max(topk_logits, dim=-1, keepdim=True).values
    exp_logits = torch.exp(topk_logits - max_logits)
    topk_weights = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    # 3. Histogram and Prefix Sum
    histogram = torch.bincount(topk_indices.view(-1), minlength=E).int()
    offsets = torch.zeros(E, dtype=torch.int32, device=x.device)
    offsets[1:] = torch.cumsum(histogram[:-1], dim=0).int()
    
    # 4. Permute (The PyTorch bottleneck)
    write_pointers = offsets.clone()
    permuted_x = torch.zeros(N * k, d_model, device=x.device, dtype=torch.float32)
    coo_indices = torch.zeros(N * k, dtype=torch.int32, device=x.device)
    coo_weights = torch.zeros(N * k, device=x.device, dtype=torch.float32)

    for i in range(N):
        for j in range(k):
            expert_id = topk_indices[i, j].item()
            pos = write_pointers[expert_id].item()
            permuted_x[pos] = x[i]
            coo_indices[pos] = i
            coo_weights[pos] = topk_weights[i, j]
            write_pointers[expert_id] += 1
            
    return permuted_x, coo_indices, coo_weights, offsets


if __name__ == "__main__":
    # Setup production-scale dimensions
    N = 4096         # Massive batch of tokens
    d_model = 1024   # Standard small LLM dimension
    E = 8            # 8 Experts
    k = 2            # Top-2 Routing
    
    print(f"Initializing Benchmark -> Tokens: {N}, d_model: {d_model}, Experts: {E}")
    
    device = torch.device('cuda')
    x = torch.randn(N, d_model, dtype=torch.float32, device=device)
    W_g = torch.randn(d_model, E, dtype=torch.float32, device=device)

    # --- 1. GPU WARMUP ---
    # The first few CUDA calls are always slow as the GPU clocks wake up. 
    # We run dummy iterations to stabilize the hardware state.
    print("Warming up GPU clocks...")
    for _ in range(10):
        pytorch_reference_routing(x, W_g, k)
        custom_moe_cuda.route_and_permute(x, W_g, k)
        
    torch.cuda.synchronize()
    
    iters = 50
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # --- 2. PROFILE PYTORCH ---
    print(f"Running PyTorch baseline ({iters} iterations)...")
    torch.cuda.nvtx.range_push("PyTorch_Routing")
    start_event.record()
    
    for _ in range(iters):
        pytorch_reference_routing(x, W_g, k)
        
    end_event.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    
    pt_time = start_event.elapsed_time(end_event) / iters * 1000  # ms to microseconds

    # --- 3. PROFILE CUSTOM CUDA ENGINE ---
    print(f"Running CUDA Engine ({iters} iterations)...")
    torch.cuda.nvtx.range_push("CUDA_Engine_Routing")
    start_event.record()
    
    for _ in range(iters):
        custom_moe_cuda.route_and_permute(x, W_g, k)
        
    end_event.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    
    cu_time = start_event.elapsed_time(end_event) / iters * 1000  # ms to microseconds

    # --- 4. RESULTS ---
    print("\n" + "="*40)
    print("         BENCHMARK RESULTS")
    print("="*40)
    print(f"PyTorch Latency:     {pt_time:.2f} µs")
    print(f"CUDA Engine Latency: {cu_time:.2f} µs")
    print("-" * 40)
    print(f"Hardware Speedup:    {pt_time / cu_time:.2f}x")
    print("="*40)