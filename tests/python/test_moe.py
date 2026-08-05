"""
nanoMoE CUDA Routing & Permutation Test Suite.

Verifies mathematical parity between custom CUDA kernels (topk_softmax, histogram,
permute, unpermute) and native PyTorch reference routing. Measures hardware speedup.
"""

import torch
import torch.nn as nn
import custom_moe_cuda

class ExpertLayer(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

def run_grouped_gemm(permuted_x, offsets, experts, N, k):
    E = len(experts)
    total_routed = N * k
    
    counts = torch.zeros(E, dtype=torch.int32, device=permuted_x.device)
    counts[:-1] = offsets[1:] - offsets[:-1]
    counts[-1] = total_routed - offsets[-1]
    
    expert_out = torch.empty_like(permuted_x)
    
    for i in range(E):
        start = offsets[i].item()
        length = counts[i].item()
        
        if length == 0:
            continue
            
        end = start + length
        chunk = permuted_x[start:end]
        expert_out[start:end] = experts[i](chunk)
        
    return expert_out

def pytorch_reference_routing(x, W_g, k):
    N, d_model = x.shape
    E = W_g.shape[1]
    
    logits = torch.matmul(x, W_g)
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
    
    max_logits = torch.max(topk_logits, dim=-1, keepdim=True).values
    exp_logits = torch.exp(topk_logits - max_logits)
    topk_weights = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    histogram = torch.bincount(topk_indices.view(-1), minlength=E).int()
    offsets = torch.zeros(E, dtype=torch.int32, device=x.device)
    offsets[1:] = torch.cumsum(histogram[:-1], dim=0).int()
    
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
    torch.manual_seed(42)
    device = torch.device('cuda')
    
    # 1. CORRECTNESS TEST
    N_test, d_model, E, k = 8, 128, 4, 2 
    x_test = torch.randn(N_test, d_model, dtype=torch.float32, device=device)
    W_g_test = torch.randn(d_model, E, dtype=torch.float32, device=device)
    experts = nn.ModuleList([ExpertLayer(d_model=d_model, d_hidden=512).to(device) for _ in range(E)])
    
    print("--- Phase 1: Correctness Verification ---")
    py_perm, py_coo_idx, py_coo_wt, py_offsets = pytorch_reference_routing(x_test, W_g_test, k)
    cu_perm, cu_coo_idx, cu_coo_wt, cu_offsets = custom_moe_cuda.route_and_permute(x_test, W_g_test, k)
    
    py_expert_out = run_grouped_gemm(py_perm, py_offsets, experts, N_test, k)
    cu_expert_out = run_grouped_gemm(cu_perm, cu_offsets, experts, N_test, k)
    
    py_final = torch.zeros(N_test, d_model, device=device, dtype=torch.float32)
    for pos in range(N_test * k):
        orig_seq_idx = py_coo_idx[pos].item()
        weight = py_coo_wt[pos].item()
        py_final[orig_seq_idx] += py_expert_out[pos] * weight

    cu_final = custom_moe_cuda.unpermute(cu_expert_out, cu_coo_idx, cu_coo_wt, N_test)
    
    print(f"Final output shape restored: {cu_final.shape == x_test.shape}")
    print(f"Final outputs match perfectly: {torch.allclose(py_final, cu_final, atol=1e-4)}\n")

    # 2. SPEEDUP BENCHMARK
    N_bench, d_model_bench, E_bench, k_bench = 4096, 1024, 8, 2
    x_bench = torch.randn(N_bench, d_model_bench, dtype=torch.float32, device=device)
    W_g_bench = torch.randn(d_model_bench, E_bench, dtype=torch.float32, device=device)

    print("--- Phase 2: Speedup Benchmark ---")
    print("Warming up GPU clocks...")
    for _ in range(10):
        pytorch_reference_routing(x_bench, W_g_bench, k_bench)
        custom_moe_cuda.route_and_permute(x_bench, W_g_bench, k_bench)
    torch.cuda.synchronize()
    
    iters = 50
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        pytorch_reference_routing(x_bench, W_g_bench, k_bench)
    end_event.record()
    torch.cuda.synchronize()
    pt_time = start_event.elapsed_time(end_event) / iters * 1000

    start_event.record()
    for _ in range(iters):
        custom_moe_cuda.route_and_permute(x_bench, W_g_bench, k_bench)
    end_event.record()
    torch.cuda.synchronize()
    cu_time = start_event.elapsed_time(end_event) / iters * 1000

    print("="*40)
    print(f"PyTorch Latency:     {pt_time:.2f} us")
    print(f"CUDA Engine Latency: {cu_time:.2f} us")
    print("-" * 40)
    print(f"Hardware Speedup:    {pt_time / cu_time:.2f}x")
    print("="*40)