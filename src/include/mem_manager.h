#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

/// Standard physical KV-cache block size (number of token positions per block).
constexpr int BLOCK_SIZE = 16;

/**
 * @brief Represents a single fixed-size physical memory block in the KV cache.
 *
 * Managed within a pre-allocated contiguous array (physical_memory_pool).
 * Contains atomic reference counter for prefix-cache block sharing and
 * relaxed token usage counter.
 */
struct PhysicalBlock {
    int physical_block_id;
    std::atomic<int> ref_count{0};
    std::atomic<int> num_tokens{0};

    /// Default constructor: initializes ID to sentinel (-1). Overwritten during pool creation.
    PhysicalBlock() : physical_block_id(-1) {}
    explicit PhysicalBlock(int id) : physical_block_id(id) {}

    // Non-copyable & non-movable to guarantee physical pointer stability.
    PhysicalBlock(const PhysicalBlock&) = delete;
    PhysicalBlock& operator=(const PhysicalBlock&) = delete;

    [[nodiscard]] bool is_full() const {
        return num_tokens.load(std::memory_order_relaxed) == BLOCK_SIZE;
    }
    [[nodiscard]] bool is_empty() const {
        return num_tokens.load(std::memory_order_relaxed) == 0;
    }
};

/**
 * @brief Logical sequence mapping table (PagedAttention Page Table entry).
 *
 * Maps a sequence ID to its allocated non-contiguous physical blocks.
 */
struct BlockTable {
    int sequence_id;
    int logical_length;
    std::vector<PhysicalBlock*> blocks;

    explicit BlockTable(int seq_id) : sequence_id(seq_id), logical_length(0) {}

    /// Returns the active physical block accepting token appends, or nullptr if full/empty.
    [[nodiscard]] PhysicalBlock* get_append_block() const {
        if (blocks.empty() || blocks.back()->is_full()) {
            return nullptr;
        }
        return blocks.back();
    }
};

/**
 * @brief Lock-Free Multi-Producer Multi-Consumer (MPMC) Bounded Queue (Vyukov Ring).
 *
 * Stores uint32_t physical block indices (not raw pointers) to eliminate pointer ABA hazards.
 * Monotonically increasing sequence tags differentiate individual push/pop operations across slot reuses.
 */
struct FreeListRing {
    struct Slot {
        std::atomic<uint32_t> sequence;  /// Generation tag; avoids wrap-around aliasing.
        uint32_t block_id;               /// Index into physical_memory_pool.
    };

    std::unique_ptr<Slot[]> slots;  /// Power-of-2 capacity buffer.
    uint32_t capacity;              /// Ring capacity (next power of 2 >= num_blocks).
    uint32_t mask;                  /// Fast bitwise modulo mask (capacity - 1).

    // Cache-line aligned cursors to avoid false sharing between producers and consumers.
    alignas(64) std::atomic<uint32_t> head{0};  /// Consumer index (allocations).
    alignas(64) std::atomic<uint32_t> tail{0};  /// Producer index (deallocations).

    explicit FreeListRing(uint32_t num_blocks);

    /**
     * @brief Enqueues a block index (lock-free producer path).
     * @param block_id Physical block index to return to the pool.
     * @return true on success.
     */
    bool push(uint32_t block_id);

    /**
     * @brief Dequeues a block index (lock-free consumer path).
     * @param[out] out_block_id Receives the allocated physical block index.
     * @return true if allocated, false if ring is empty (OOM).
     */
    bool pop(uint32_t& out_block_id);

    /// Approximate count of free blocks (advisory snapshot under concurrent access).
    [[nodiscard]] uint32_t size_approx() const {
        return tail.load(std::memory_order_relaxed) -
               head.load(std::memory_order_relaxed);
    }
};

/**
 * @brief Paged KV-Cache Memory Allocator and Controller.
 *
 * Combines thread-local arena (TLA) wallets with a global Vyukov MPMC ring buffer
 * to provide lock-free O(1) allocation and deallocation with minimal atomic contention.
 */
class BlockAllocator {
private:
    std::unique_ptr<PhysicalBlock[]> physical_memory_pool;
    FreeListRing free_ring;
    int total_blocks;

public:
    explicit BlockAllocator(int num_blocks);

    /// Allocates a physical block from the thread-local wallet or global ring buffer.
    PhysicalBlock* allocate_block();

    /// Decrements ref_count (acq_rel) and recycles block when ref_count reaches 0.
    void free_block(PhysicalBlock* block);

    /// Appends a token to the sequence block table, allocating a new page on page fault.
    void append_token(BlockTable& table);

    /// Frees all physical blocks assigned to a sequence block table.
    void free_sequence(BlockTable& table);

    /// Drains thread-local wallet blocks back into the global ring buffer.
    void flush_wallet();

    [[nodiscard]] int get_free_count() const;
    [[nodiscard]] int get_total_blocks() const;
};