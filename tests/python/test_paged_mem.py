import torch
import custom_moe_cuda 

def test_paged_attention_fetch():
    print("[Test] Initializing Scatter-Gather Integrity Test...")
    
    # 1. Define hardware constraints
    block_size = 16
    head_dim = 64
    batch_size = 2
    seq_len = 32 # 2 logical blocks per sequence
    max_blocks_per_seq = seq_len // block_size
    num_physical_blocks = 10 # A tiny simulated VRAM pool
    
    # 2. Allocate the Physical VRAM Pool
    # We fill it with sequential numbers so we can exactly track where data came from.
    total_elements = num_physical_blocks * block_size * head_dim
    physical_kv_cache = torch.arange(
        total_elements, dtype=torch.float32, device="cuda"
    ).reshape(num_physical_blocks, block_size, head_dim)
    
    # 3. Simulate memory fragmentation (The Block Table)
    # Sequence 0 is fragmented across physical block 5 and physical block 2
    # Sequence 1 is fragmented across physical block 8 and physical block 0
    block_tables_cpu = torch.tensor([
        [5, 2], 
        [8, 0]
    ], dtype=torch.int32)
    
    # Send the flattened translation table to the GPU
    block_tables_gpu = block_tables_cpu.to("cuda")
    
    # 4. Generate the Ground Truth (Brute Force PyTorch)
    # We manually stitch the physical blocks together using standard PyTorch indexing
    seq_0_truth = torch.cat([physical_kv_cache[5], physical_kv_cache[2]], dim=0)
    seq_1_truth = torch.cat([physical_kv_cache[8], physical_kv_cache[0]], dim=0)
    ground_truth = torch.stack([seq_0_truth, seq_1_truth]) # Shape: [2, 32, 64]
    
    print("[Test] Launching Custom CUDA Paged Fetch Kernel...")
    # 5. Run your custom C++/CUDA engine
    custom_output = custom_moe_cuda.paged_kv_fetch(
        physical_kv_cache,
        block_tables_gpu,
        batch_size,
        seq_len,
        block_size,
        head_dim
    )
    
    # 6. Verify Mathematical Equivalence
    # We use allclose to ensure the pointers fetched the exact correct floats
    is_correct = torch.allclose(custom_output, ground_truth)
    
    if is_correct:
        print("\n[SUCCESS] Memory Translation Verified!")
        print(f"The GPU successfully reconstructed the fragmented blocks.")
        print(f"Output Tensor Shape: {custom_output.shape}")
    else:
        print("\n[FAILED] Kernel output does not match the PyTorch brute-force calculation.")
        print("Check your index math inside paged_memory_fetch_kernel.")

if __name__ == "__main__":
    # Ensure CUDA is available before running
    assert torch.cuda.is_available(), "This test requires a GPU."
    test_paged_attention_fetch()