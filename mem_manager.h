#pragma once
#include <vector>
#include <queue>
#include <stdexcept>
#include <iostream>

constexpr int BLOCK_SIZE = 16; 

// --- Physical Frame ---
struct PhysicalBlock {
    int physical_block_id; 
    int ref_count;         
    int num_tokens;        

    PhysicalBlock(int id) : physical_block_id(id), ref_count(0), num_tokens(0) {}

    bool is_full() const { return num_tokens == BLOCK_SIZE; }
    bool is_empty() const { return num_tokens == 0; }
};

// --- Page Table ---
struct BlockTable {
    int sequence_id;
    int logical_length; 
    std::vector<PhysicalBlock*> blocks; 

    BlockTable(int seq_id) : sequence_id(seq_id), logical_length(0) {}

    PhysicalBlock* get_append_block() const {
        if (blocks.empty() || blocks.back()->is_full()) {
            return nullptr; 
        }
        return blocks.back();
    }
};

// --- Memory Controller ---
class BlockAllocator {
private:
    std::vector<PhysicalBlock> physical_memory_pool;
    std::queue<PhysicalBlock*> free_list;
    int total_blocks;

public:
    BlockAllocator(int num_blocks);
    
    PhysicalBlock* allocate_block();
    void free_block(PhysicalBlock* block);
    
    void append_token(BlockTable& table);
    void free_sequence(BlockTable& table);
    
    int get_free_count() const;
    int get_total_blocks() const;
};