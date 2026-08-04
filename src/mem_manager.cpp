#include "include/mem_manager.h"

// ============================================================
// Internal Helper
// ============================================================

// Round n up to the next power of 2 (required for the bitmask trick).
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
// FreeListRing — Vyukov MPMC Bounded Queue
// ============================================================

FreeListRing::FreeListRing(uint32_t num_blocks) {
    capacity = next_pow2(num_blocks);
    mask     = capacity - 1;
    // Allocate a fixed-size array of Slots — no copy/move of std::atomic needed.
    slots = std::make_unique<Slot[]>(capacity);
    // Each slot's sequence is initialised to its own index,
    // signalling it is ready to be written by the first producer.
    for (uint32_t i = 0; i < capacity; i++) {
        slots[i].sequence.store(i, std::memory_order_relaxed);
        slots[i].block_id = 0;
    }
}

// Enqueue (free path) — lock-free MPMC push via CAS on tail.
//
// diff < 0 does NOT mean "ring full" in our allocator context. It means the
// consumer popped this slot position but has not yet written the recycle
// sequence (pos + capacity). We spin until it does. True OOM is detected
// only on pop() — push() is always called after a successful pop(), so the
// slot will become available once the consumer finishes its store.
//
// Memory ordering:
//   tail CAS        : relaxed  — ordering is provided by the sequence stores.
//   seq load        : acquire  — observe the slot's current generation.
//   seq store(pos+1): release  — publish block_id to any future consumer.
bool FreeListRing::push(uint32_t block_id) {
    uint32_t pos = tail.load(std::memory_order_relaxed);
    Slot*    slot;

    while (true) {
        slot          = &slots[pos & mask];
        uint32_t seq  = slot->sequence.load(std::memory_order_acquire);
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

        if (diff == 0) {
            // Slot is ready to accept a write; race to claim this tail position.
            if (tail.compare_exchange_weak(pos, pos + 1,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
                break;  // We own this slot.
            }
            // Another producer claimed it; CAS updated pos — retry immediately.
        } else {
            // diff != 0: either the consumer hasn't recycled the slot yet (diff < 0)
            // or another producer is writing ahead of us (diff > 0). Either way,
            // refresh our view of tail and retry.
            pos = tail.load(std::memory_order_relaxed);
        }
    }

    slot->block_id = block_id;
    // Publish: sequence = pos+1 signals "slot has data" to any consumer waiting
    // for exactly this generation.  The release pairs with the acquire in pop()
    slot->sequence.store(pos + 1, std::memory_order_release);
    return true;
}

// Dequeue (allocate path) — lock-free MPMC pop via CAS on head.
//
// Memory ordering:
//   head CAS          : relaxed  — ordering via sequence.
//   seq load          : acquire  — synchronise with push()'s release store.
//   seq store(pos+cap): release  — recycle slot; signals "ready for next producer".
bool FreeListRing::pop(uint32_t& out_block_id) {
    uint32_t pos = head.load(std::memory_order_relaxed);
    Slot*    slot;

    while (true) {
        slot          = &slots[pos & mask];
        uint32_t seq  = slot->sequence.load(std::memory_order_acquire);
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

        if (diff == 0) {
            // Slot has data for this generation; race to claim this head position.
            if (head.compare_exchange_weak(pos, pos + 1,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
                break;  // We own this slot.
            }
            // Another consumer claimed it first; pos was updated by CAS — retry.
        } else if (diff < 0) {
            // sequence < pos+1: ring is empty — OOM.
            return false;
        } else {
            // sequence > pos+1: producer hasn't committed yet (transient). Retry.
            pos = head.load(std::memory_order_relaxed);
        }
    }

    out_block_id = slot->block_id;
    // Recycle: sequence = pos+capacity signals the slot is writable again for
    // the producer that will wrap around to this index.  Pairs with push()'s
    // acquire load of the same slot.
    slot->sequence.store(pos + capacity, std::memory_order_release);
    return true;
}

// ============================================================
// Thread-Local Arena (TLA) Wallet
//
// Each hardware thread owns a private slab of pre-fetched block IDs.  The
// global FreeListRing is only touched for bulk refill (BATCH_SIZE pops) and
// bulk flush (half of MAX_CAPACITY pushes), reducing atomic CAS pressure by
// ~90% on the hot alloc/free path.
//
// Lifecycle note: The wallet deliberately has NO automatic destructor path
// back to a specific allocator instance — doing so would require a global
// singleton pointer and would break multi-GPU / multi-pool designs.  Instead,
// callers are responsible for draining the wallet explicitly via one of:
//   1. BlockAllocator::flush_wallet()  — called in single-threaded tests
//      before get_free_count() assertions.
//   2. The RAII WalletFlusher guard  — placed at the top of worker lambdas
//      so that blocks are returned to the correct allocator on thread exit,
//      even if an exception is thrown.
// ============================================================

struct ThreadLocalWallet {
    // BATCH_SIZE: number of block IDs pulled from the global ring per refill.
    // MAX_CAPACITY: once the wallet reaches this size, flush half back.
    //
    // These limits are deliberately small.  Unlike CPU allocators (tcmalloc
    // uses 1 024+ item batches), each block here represents physical GPU VRAM.
    // Over-hoarding hides VRAM from the system and can cause false OOM errors
    // even when hundreds of blocks sit idle in other threads' wallets.
    static constexpr int BATCH_SIZE   = 8;
    static constexpr int MAX_CAPACITY = 16;

    std::vector<uint32_t> free_ids;

    ThreadLocalWallet() { free_ids.reserve(MAX_CAPACITY); }
};

thread_local ThreadLocalWallet t_wallet;

// ============================================================
// BlockAllocator
// ============================================================

// 1. Initialize the Pool
BlockAllocator::BlockAllocator(int num_blocks)
    : free_ring(static_cast<uint32_t>(num_blocks)), total_blocks(num_blocks) {
    // Allocate the pool as a fixed-size heap array. PhysicalBlock contains
    // std::atomic members that are non-copyable/non-movable, so std::vector
    // cannot be used (it would attempt a copy/move on reallocation).
    physical_memory_pool = std::make_unique<PhysicalBlock[]>(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        physical_memory_pool[i].physical_block_id = i;  // fix sentinel id
        free_ring.push(static_cast<uint32_t>(i));
    }
}

// 2. Handle Allocation — fast path via TLA wallet; slow path bulk-refills from ring.
//
// Hot path (wallet non-empty): zero atomics, L1 cache hit on t_wallet.free_ids.
// Cold path (wallet empty): BATCH_SIZE atomic pops from free_ring in one go,
// amortising CAS cost across the next BATCH_SIZE allocations.
PhysicalBlock* BlockAllocator::allocate_block() {
    // Cold path: wallet is empty — go to the global Bank and bulk-fetch.
    if (t_wallet.free_ids.empty()) {
        for (int i = 0; i < ThreadLocalWallet::BATCH_SIZE; ++i) {
            uint32_t fetched_id;
            if (free_ring.pop(fetched_id)) {
                t_wallet.free_ids.push_back(fetched_id);
            } else {
                break;  // Global pool exhausted; stop fetching.
            }
        }

        // Still empty after trying the global ring → truly OOM.
        if (t_wallet.free_ids.empty()) {
            throw std::runtime_error("OOM: KV Cache Pool Depleted.");
        }
    }

    // Hot path: pop instantly from local wallet.  Zero atomics, zero locks.
    uint32_t id = t_wallet.free_ids.back();
    t_wallet.free_ids.pop_back();

    PhysicalBlock* block = &physical_memory_pool[id];
    // Release: publish the initialised state to any concurrent thread that may
    // later call fetch_sub (acq_rel) on ref_count.
    block->ref_count.store(1, std::memory_order_release);
    block->num_tokens.store(0, std::memory_order_relaxed);
    return block;
}

// 3. Handle Deallocation (Recycling) — deposit into wallet; flush half when full.
//
// fetch_sub with acq_rel:
//   Acquire: all reads/writes by any other thread that held a reference are
//            ordered before this decrement — safe to inspect the block.
//   Release: this thread's writes complete before the block re-enters the pool
//            and potentially gets handed to a new owner.
//
// Checking the *return value* (the old count) == 1 is the correct pattern;
// checking ref_count == 0 *after* the decrement introduces a TOCTOU race.
void BlockAllocator::free_block(PhysicalBlock* block) {
    if (block->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        block->num_tokens.store(0, std::memory_order_relaxed);

        // Hot path: deposit into local wallet.  Zero atomics, zero locks.
        t_wallet.free_ids.push_back(block->physical_block_id);

        // If the wallet is getting too fat, flush half back to the global ring.
        // Flushing exactly half (rather than all) leaves the wallet warm for
        // the next allocation burst, avoiding a double round-trip.
        if (static_cast<int>(t_wallet.free_ids.size()) >= ThreadLocalWallet::MAX_CAPACITY) {
            int to_flush = static_cast<int>(t_wallet.free_ids.size()) / 2;
            for (int i = 0; i < to_flush; ++i) {
                free_ring.push(t_wallet.free_ids.back());
                t_wallet.free_ids.pop_back();
            }
        }
    }
}

// 4. flush_wallet — drain this thread's entire wallet back to the global ring.
//
// Call sites:
//   • Tests: before any get_free_count() assertion to restore exact accounting.
//   • RAII WalletFlusher: at thread exit to prevent block leaks when a worker
//     thread terminates while holding IDs in its wallet.
//
// Thread safety: only touches t_wallet (TLS — no sharing) and free_ring
// (lock-free MPMC — safe from any thread).
void BlockAllocator::flush_wallet() {
    for (uint32_t id : t_wallet.free_ids) {
        free_ring.push(id);
    }
    t_wallet.free_ids.clear();
}

// 5. State Machine: Append Token
void BlockAllocator::append_token(BlockTable& table) {
    PhysicalBlock* active_block = table.get_append_block();

    // Page Fault: request a new block from the pool.
    if (active_block == nullptr) {
        active_block = allocate_block();
        table.blocks.push_back(active_block);
    }

    // Relaxed: num_tokens is only written by the single thread owning this
    // BlockTable; no cross-thread synchronisation needed here.
    active_block->num_tokens.fetch_add(1, std::memory_order_relaxed);
    table.logical_length++;
}

// 6. State Machine: Evict Sequence
void BlockAllocator::free_sequence(BlockTable& table) {
    for (PhysicalBlock* block : table.blocks) {
        free_block(block);
    }
    table.blocks.clear();
    table.logical_length = 0;
}

// Diagnostics
int BlockAllocator::get_free_count()   const { return static_cast<int>(free_ring.size_approx()); }
int BlockAllocator::get_total_blocks() const { return total_blocks; }