#pragma once
#include <vector>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <cstdint>

constexpr int BLOCK_SIZE = 16;

// --- Physical Frame ---
struct PhysicalBlock {
    int                physical_block_id;
    std::atomic<int>   ref_count{0};
    std::atomic<int>   num_tokens{0};

    // Default constructor: id initialised to -1 (sentinel); overwritten by
    // BlockAllocator's constructor loop immediately after pool allocation.
    PhysicalBlock() : physical_block_id(-1) {}
    explicit PhysicalBlock(int id) : physical_block_id(id) {}

    // std::atomic members are neither copy- nor move-constructible.
    // Disable to prevent accidental vector reallocation after reserve().
    PhysicalBlock(const PhysicalBlock&)            = delete;
    PhysicalBlock& operator=(const PhysicalBlock&) = delete;

    bool is_full()  const { return num_tokens.load(std::memory_order_relaxed) == BLOCK_SIZE; }
    bool is_empty() const { return num_tokens.load(std::memory_order_relaxed) == 0; }
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

// --- Lock-Free MPMC Ring Buffer (Vyukov) ---
//
// Stores uint32_t block *indices* into physical_memory_pool — no raw pointers
// in the hot path.  Integer indices cannot alias across reuse cycles, so the
// pointer-aliasing form of the ABA hazard is structurally impossible here.
//
// Each Slot carries a monotonically increasing sequence number.  A producer
// claims a tail slot by CAS-ing tail; a consumer claims a head slot by
// CAS-ing head.  The sequence number distinguishes every individual
// push/pop from all prior and future ones on the same slot, even if the same
// block_id cycles through it many times.
struct FreeListRing {
    struct Slot {
        std::atomic<uint32_t> sequence;   // generation tag — never wraps in practice
        uint32_t              block_id;   // payload: index into physical_memory_pool
    };

    // Fixed-size heap array — avoids std::vector's copy/move requirement on
    // resize/reserve, which would be ill-formed because std::atomic is non-copyable.
    std::unique_ptr<Slot[]> slots;
    uint32_t                capacity;  // always a power of 2 (set in constructor)
    uint32_t                mask;      // capacity - 1  (fast modulo via bitwise AND)

    // Pad head and tail onto separate cache lines to eliminate false sharing
    // between the allocate (consumer) and free (producer) paths.
    alignas(64) std::atomic<uint32_t> head{0};   // consumer cursor (allocate)
    alignas(64) std::atomic<uint32_t> tail{0};   // producer cursor (free)

    explicit FreeListRing(uint32_t num_blocks);

    // Enqueue a block index (called by free_block).
    // Returns false only if the ring is full — pool over-committed, should never occur.
    bool push(uint32_t block_id);

    // Dequeue a block index (called by allocate_block).
    // Returns false if the ring is empty — OOM condition.
    bool pop(uint32_t& out_block_id);

    // Advisory snapshot of the number of free blocks.
    // May be transiently inaccurate under heavy concurrent access.
    [[nodiscard]] uint32_t size_approx() const {
        return tail.load(std::memory_order_relaxed) -
               head.load(std::memory_order_relaxed);
    }
};

// --- Memory Controller ---
class BlockAllocator {
private:
    // Fixed-size heap array — same rationale as FreeListRing::slots above.
    std::unique_ptr<PhysicalBlock[]> physical_memory_pool;
    FreeListRing                     free_ring;
    int                        total_blocks;

public:
    explicit BlockAllocator(int num_blocks);

    PhysicalBlock* allocate_block();
    void           free_block(PhysicalBlock* block);

    void append_token(BlockTable& table);
    void free_sequence(BlockTable& table);

    // Drain this calling thread's TLA wallet back into the global ring.
    // Must be called explicitly (e.g. in tests before get_free_count(), or via
    // a RAII WalletFlusher at thread exit) — the wallet has no automatic
    // destructor path back to a specific allocator instance.
    void flush_wallet();

    [[nodiscard]] int get_free_count()   const;
    [[nodiscard]] int get_total_blocks() const;
};