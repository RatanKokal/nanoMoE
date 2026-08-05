#include "include/mem_manager.h"

// Round n up to the next power of 2.
static uint32_t next_pow2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >>  1;
    n |= n >>  2;
    n |= n >>  4;
    n |= n >>  8;
    n |= n >> 16;
    return n + 1;
}

// ============================================================
// FreeListRing - Vyukov MPMC Bounded Queue
// ============================================================

FreeListRing::FreeListRing(uint32_t num_blocks) {
    capacity = next_pow2(num_blocks);
    mask     = capacity - 1;
    slots    = std::make_unique<Slot[]>(capacity);
    for (uint32_t i = 0; i < capacity; i++) {
        slots[i].sequence.store(i, std::memory_order_relaxed);
        slots[i].block_id = 0;
    }
}

// Lock-free MPMC push via CAS on tail. Sequence store (release) publishes block_id to consumers.
bool FreeListRing::push(uint32_t block_id) {
    uint32_t pos = tail.load(std::memory_order_relaxed);
    Slot* slot;

    while (true) {
        slot          = &slots[pos & mask];
        uint32_t seq  = slot->sequence.load(std::memory_order_acquire);
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

        if (diff == 0) {
            if (tail.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                break;
            }
        } else {
            pos = tail.load(std::memory_order_relaxed);
        }
    }

    slot->block_id = block_id;
    slot->sequence.store(pos + 1, std::memory_order_release);
    return true;
}

// Lock-free MPMC pop via CAS on head. Sequence store (release) recycles slot for future producers.
bool FreeListRing::pop(uint32_t& out_block_id) {
    uint32_t pos = head.load(std::memory_order_relaxed);
    Slot* slot;

    while (true) {
        slot          = &slots[pos & mask];
        uint32_t seq  = slot->sequence.load(std::memory_order_acquire);
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

        if (diff == 0) {
            if (head.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                break;
            }
        } else if (diff < 0) {
            return false;
        } else {
            pos = head.load(std::memory_order_relaxed);
        }
    }

    out_block_id = slot->block_id;
    slot->sequence.store(pos + capacity, std::memory_order_release);
    return true;
}

// ============================================================
// Thread-Local Arena (TLA) Wallet
// ============================================================

struct ThreadLocalWallet {
    static constexpr int BATCH_SIZE   = 8;
    static constexpr int MAX_CAPACITY = 16;

    std::vector<uint32_t> free_ids;

    ThreadLocalWallet() { free_ids.reserve(MAX_CAPACITY); }
};

thread_local ThreadLocalWallet t_wallet;

// ============================================================
// BlockAllocator
// ============================================================

BlockAllocator::BlockAllocator(int num_blocks)
    : free_ring(static_cast<uint32_t>(num_blocks)), total_blocks(num_blocks) {
    t_wallet.free_ids.clear();
    physical_memory_pool = std::make_unique<PhysicalBlock[]>(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        physical_memory_pool[i].physical_block_id = i;
        free_ring.push(static_cast<uint32_t>(i));
    }
}

PhysicalBlock* BlockAllocator::allocate_block() {
    if (t_wallet.free_ids.empty()) {
        for (int i = 0; i < ThreadLocalWallet::BATCH_SIZE; ++i) {
            uint32_t fetched_id;
            if (free_ring.pop(fetched_id)) {
                t_wallet.free_ids.push_back(fetched_id);
            } else {
                break;
            }
        }
        if (t_wallet.free_ids.empty()) {
            throw std::runtime_error("OOM: KV Cache Pool Depleted.");
        }
    }

    uint32_t id = t_wallet.free_ids.back();
    t_wallet.free_ids.pop_back();

    PhysicalBlock* block = &physical_memory_pool[id];
    block->ref_count.store(1, std::memory_order_release);
    block->num_tokens.store(0, std::memory_order_relaxed);
    return block;
}

void BlockAllocator::free_block(PhysicalBlock* block) {
    if (block->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        block->num_tokens.store(0, std::memory_order_relaxed);
        t_wallet.free_ids.push_back(block->physical_block_id);

        if (static_cast<int>(t_wallet.free_ids.size()) >= ThreadLocalWallet::MAX_CAPACITY) {
            int to_flush = static_cast<int>(t_wallet.free_ids.size()) / 2;
            for (int i = 0; i < to_flush; ++i) {
                free_ring.push(t_wallet.free_ids.back());
                t_wallet.free_ids.pop_back();
            }
        }
    }
}

void BlockAllocator::flush_wallet() {
    for (uint32_t id : t_wallet.free_ids) {
        free_ring.push(id);
    }
    t_wallet.free_ids.clear();
}

void BlockAllocator::append_token(BlockTable& table) {
    PhysicalBlock* active_block = table.get_append_block();
    if (active_block == nullptr) {
        active_block = allocate_block();
        table.blocks.push_back(active_block);
    }
    active_block->num_tokens.fetch_add(1, std::memory_order_relaxed);
    table.logical_length++;
}

void BlockAllocator::free_sequence(BlockTable& table) {
    for (PhysicalBlock* block : table.blocks) {
        free_block(block);
    }
    table.blocks.clear();
    table.logical_length = 0;
}

int BlockAllocator::get_free_count()   const { return static_cast<int>(free_ring.size_approx()); }
int BlockAllocator::get_total_blocks() const { return total_blocks; }