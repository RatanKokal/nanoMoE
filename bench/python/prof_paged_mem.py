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
    
    print("[Profile] Running Benchmark Loop with CUDA Events...")
    torch.cuda.cudart().cudaProfilerStart()
    
    iterations = 1000  # Increased iterations for better percentile accuracy
    events = []
    
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _ = custom_moe_cuda.paged_kv_fetch(
            physical_kv_cache, block_tables_gpu, batch_size, seq_len, block_size, head_dim
        )
        end_event.record()
        
        events.append((start_event, end_event))
        
    # Wait for all queued kernels to finish
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    
    # Extract times (elapsed_time returns milliseconds)
    latencies_us = [s.elapsed_time(e) * 1000.0 for s, e in events]
    
    # Sort to calculate percentiles
    latencies_us.sort()
    
    avg_us = sum(latencies_us) / len(latencies_us)
    p95_us = latencies_us[int(len(latencies_us) * 0.95)]
    p99_us = latencies_us[int(len(latencies_us) * 0.99)]
    
    print(f"\n[Profile Results over {iterations} iterations]")
    print(f"Average Latency: {avg_us:.2f} microseconds")
    print(f"p95 Latency:     {p95_us:.2f} microseconds")
    print(f"p99 Latency:     {p99_us:.2f} microseconds")

if __name__ == "__main__":
    assert torch.cuda.is_available()
    profile_fetch()