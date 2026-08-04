#include "memory_manager.h"
#include <cassert>
#include <iostream>

void run_tests() {
    std::cout << "[Test] Initializing PagedAttention Allocator..." << std::endl;
    int total_vram_blocks = 1000;
    BlockAllocator allocator(total_vram_blocks);
    
    assert(allocator.get_free_count() == total_vram_blocks);
    
    // Create simulated traffic
    BlockTable req_1(101);
    BlockTable req_2(102);
    BlockTable req_3(103);

    std::cout << "[Test] Simulating Continuous Batching Traffic..." << std::endl;
    // req_1 generates 10 tokens (needs 1 block)
    for(int i=0; i<10; i++) allocator.append_token(req_1);
    
    // req_2 generates 20 tokens (needs 2 blocks)
    for(int i=0; i<20; i++) allocator.append_token(req_2);
    
    // req_3 generates 35 tokens (needs 3 blocks)
    for(int i=0; i<35; i++) allocator.append_token(req_3);

    // Assert the VRAM pool correctly distributed the scattered blocks
    int expected_free = total_vram_blocks - (1 + 2 + 3);
    assert(allocator.get_free_count() == expected_free);
    std::cout << "[Test] Fragmentation bypassed. Blocks correctly allocated." << std::endl;

    std::cout << "[Test] Simulating Request Completion..." << std::endl;
    allocator.free_sequence(req_1);
    allocator.free_sequence(req_2);
    allocator.free_sequence(req_3);

    // Assert Zero Memory Leaks
    assert(allocator.get_free_count() == total_vram_blocks);
    std::cout << "[Test] PASSED: Zero memory leaks. VRAM fully recycled." << std::endl;
}

int main() {
    run_tests();
    return 0;
}