import torch
import custom_moe_cuda

if __name__ == "__main__":
    # Setup production-scale dimensions
    N = 4096         # Massive batch of tokens
    d_model = 1024   # Standard small LLM dimension
    E = 8            # 8 Experts
    k = 2            # Top-2 Routing
    
    print(f"Initializing Clean Profiler -> Tokens: {N}, d_model: {d_model}, Experts: {E}")
    
    device = torch.device('cuda')
    x = torch.randn(N, d_model, dtype=torch.float32, device=device)
    W_g = torch.randn(d_model, E, dtype=torch.float32, device=device)

    # --- 1. GPU WARMUP ---
    print("Warming up GPU clocks...")
    for _ in range(10):
        custom_moe_cuda.route_and_permute(x, W_g, k)
        
    torch.cuda.synchronize()
    
    iters = 50
    
    # --- 2. PROFILE CUSTOM CUDA ENGINE ---
    print(f"Executing Custom CUDA Engine ({iters} iterations) for Nsight Trace...")
    
    torch.cuda.nvtx.range_push("CUDA_Engine_Routing")
    
    for _ in range(iters):
        custom_moe_cuda.route_and_permute(x, W_g, k)
        
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    
    print("Profiling loop complete. Check your .nsys-rep file.")