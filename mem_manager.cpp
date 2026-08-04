#include "mem_manager.h"

// 1. Initialize the Pool
BlockAllocator::BlockAllocator(int num_blocks) : total_blocks(num_blocks) {
    physical_memory_pool.reserve(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        physical_memory_pool.emplace_back(i);
        free_list.push(&physical_memory_pool.back());
    }
}

// 2. Handle Allocation
PhysicalBlock* BlockAllocator::allocate_block() {
    if (free_list.empty()) {
        throw std::runtime_error("OOM: KV Cache Pool Depleted.");
    }
    
    PhysicalBlock* block = free_list.front();
    free_list.pop();
    
    block->ref_count = 1;
    block->num_tokens = 0;
    
    return block;
}

// 3. Handle Deallocation (Recycling)
void BlockAllocator::free_block(PhysicalBlock* block) {
    block->ref_count--;
    if (block->ref_count == 0) {
        block->num_tokens = 0;
        free_list.push(block);
    }
}

// 4. State Machine: Append Token
void BlockAllocator::append_token(BlockTable& table) {
    PhysicalBlock* active_block = table.get_append_block();
    
    // Page Fault: Request a new block from the OS
    if (active_block == nullptr) {
        active_block = allocate_block();
        table.blocks.push_back(active_block);
    }
    
    active_block->num_tokens++;
    table.logical_length++;
}

// 5. State Machine: Evict Sequence
void BlockAllocator::free_sequence(BlockTable& table) {
    for (PhysicalBlock* block : table.blocks) {
        free_block(block);
    }
    table.blocks.clear();
    table.logical_length = 0;
}

// Diagnostics
int BlockAllocator::get_free_count() const { return free_list.size(); }
int BlockAllocator::get_total_blocks() const { return total_blocks; }