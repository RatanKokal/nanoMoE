import torch
import custom_moe_cuda
import time

def profile_fetch():
    print("[Profile] Initializing Production-Scale Workload...")
    
    # 1. Scale parameters to saturation limits
    batch_size = 256
    seq_len = 2048
    block_size = 16
    head_dim = 64
    max_blocks_per_seq = seq_len // block_size  # 128 blocks per request
    
    # Simulate a massive pre-allocated VRAM pool (~3.2 million tokens)
    num_physical_blocks = 50000 
    
    # Allocate dummy data (using empty to save initialization time)
    physical_kv_cache = torch.empty(
        (num_physical_blocks, block_size, head_dim), 
        dtype=torch.float32, 
        device="cuda"
    )
    
    # 2. Simulate Extreme Fragmentation
    # We assign completely random physical block IDs to every single logical block
    block_tables_cpu = torch.randint(
        0, num_physical_blocks, 
        (batch_size, max_blocks_per_seq), 
        dtype=torch.int32
    )
    block_tables_gpu = block_tables_cpu.to("cuda")
    
    print("[Profile] Warming up CUDA runtime...")
    for _ in range(10):
        _ = custom_moe_cuda.paged_kv_fetch(
            physical_kv_cache, block_tables_gpu, batch_size, seq_len, block_size, head_dim
        )
    torch.cuda.synchronize()
    
    print("[Profile] Running Benchmark Loop...")
    # Start the Nsight Systems profiler capture window
    torch.cuda.cudart().cudaProfilerStart()
    
    start_time = time.perf_counter()
    iterations = 100
    
    for _ in range(iterations):
        _ = custom_moe_cuda.paged_kv_fetch(
            physical_kv_cache, block_tables_gpu, batch_size, seq_len, block_size, head_dim
        )
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Stop the profiler
    torch.cuda.cudart().cudaProfilerStop()
    
    avg_time_us = ((end_time - start_time) / iterations) * 1_000_000
    print(f"\n[Profile Result] Average Paged Fetch Latency: {avg_time_us:.2f} microseconds")

if __name__ == "__main__":
    assert torch.cuda.is_available()
    profile_fetch()