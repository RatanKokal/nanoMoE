import torch
import torch.nn as nn
import custom_moe_cuda

class ExpertLayer(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        # Standard Feed-Forward Network for an expert
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

def run_grouped_gemm(permuted_x, offsets, experts, N, k):
    """
    Slices the permuted buffer and routes each chunk to the correct expert.
    """
    E = len(experts)
    total_routed = N * k
    
    # We need the token counts per expert to know where to slice.
    counts = torch.zeros(E, dtype=torch.int32, device=permuted_x.device)
    counts[:-1] = offsets[1:] - offsets[:-1]
    counts[-1] = total_routed - offsets[-1]
    
    # Pre-allocate the output buffer
    expert_out = torch.empty_like(permuted_x)
    
    for i in range(E):
        start = offsets[i].item()
        length = counts[i].item()
        
        if length == 0:
            continue
            
        end = start + length
        
        # View the contiguous chunk (zero-copy) and pass through the expert
        chunk = permuted_x[start:end]
        expert_out[start:end] = experts[i](chunk)
        
    return expert_out

def pytorch_reference(x, W_g, k):
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
    
    N, d_model, E, k = 8, 128, 4, 2  # d_model MUST be a multiple of 4
    
    x = torch.randn(N, d_model, dtype=torch.float32, device=device)
    W_g = torch.randn(d_model, E, dtype=torch.float32, device=device)
    
    # Initialize the real neural network experts
    experts = nn.ModuleList([ExpertLayer(d_model=d_model, d_hidden=512).to(device) for _ in range(E)])
    
    print("Running PyTorch Reference Routing...")
    py_perm, py_coo_idx, py_coo_wt, py_offsets = pytorch_reference(x, W_g, k)
    
    print("Running CUDA Engine Routing...")
    cu_perm, cu_coo_idx, cu_coo_wt, cu_offsets = custom_moe_cuda.route_and_permute(x, W_g, k)
    
    print("\n--- Intermediate Memory Maps ---")
    print("Note: These return False due to parallel atomicAdd non-determinism (Expected)")
    print(f"Permuted X exact match: {torch.allclose(py_perm, cu_perm, atol=1e-5)}")
    print(f"COO Indices exact match: {torch.equal(py_coo_idx, cu_coo_idx)}")
    
    print("\nRunning Grouped GEMM (Real Experts)...")
    # Pass both buffers through the exact same neural networks
    py_expert_out = run_grouped_gemm(py_perm, py_offsets, experts, N, k)
    cu_expert_out = run_grouped_gemm(cu_perm, cu_offsets, experts, N, k)
    
    print("Running Unpermute Sequences...")
    # 1. PyTorch Unpermute Sequence
    py_final = torch.zeros(N, d_model, device=device, dtype=torch.float32)
    for pos in range(N * k):
        orig_seq_idx = py_coo_idx[pos].item()
        weight = py_coo_wt[pos].item()
        py_final[orig_seq_idx] += py_expert_out[pos] * weight

    # 2. CUDA Unpermute Sequence
    cu_final = custom_moe_cuda.unpermute(cu_expert_out, cu_coo_idx, cu_coo_wt, N)
    
    print("\n--- Final Unpermute Check (The True Test) ---")
    print(f"Final output shape restored: {cu_final.shape == x.shape}")
    print(f"Final outputs match perfectly: {torch.allclose(py_final, cu_final, atol=1e-4)}")