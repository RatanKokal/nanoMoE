/**
 * @file test_allocator.cpp
 * @brief Comprehensive test suite for the lock-free BlockAllocator.
 *
 * Tests are grouped into three tiers:
 *   Tier 1  -  Correctness  : single-threaded invariant checks.
 *   Tier 2  -  Concurrency  : multi-threaded stress under contention.
 *   Tier 3  -  Edge Cases   : OOM, pool reuse, ref-count sharing.
 *
 * Build:
 *   make -C tests/cpp          (from repo root)
 *   make -C tests/cpp tsan     (ThreadSanitizer build)
 */

#include "../../src/include/mem_manager.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// -------------------------------------------------------------
// RAII wallet flush guard
//
// Place one of these at the top of every worker lambda to ensure the
// thread's TLA wallet is drained back into the correct allocator when
// the thread exits  -  even if it exits via an exception.  Binding to a
// specific BlockAllocator& keeps the design free of global singletons
// and multi-pool / multi-GPU safe.
// -------------------------------------------------------------

struct WalletFlusher {
    BlockAllocator& alloc;
    explicit WalletFlusher(BlockAllocator& a) : alloc(a) {}
    ~WalletFlusher() { alloc.flush_wallet(); }
    // Non-copyable, non-movable  -  must be a named local in the lambda.
    WalletFlusher(const WalletFlusher&)            = delete;
    WalletFlusher& operator=(const WalletFlusher&) = delete;
};

// -------------------------------------------------------------
// Minimal test harness
// -------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;

/**
 * @brief Execute a single named test case and record pass/fail.
 *
 * Catches any std::exception and reports it as a failure so that
 * one bad test never silences the rest.
 */
static void run_test(const std::string& name, void (*fn)()) {
    try {
        fn();
        std::cout << "  \033[32m\033[0m  " << name << "\n";
        ++g_passed;
    } catch (const std::exception& e) {
        std::cout << "  \033[31m\033[0m  " << name
                  << "  ->  " << e.what() << "\n";
        ++g_failed;
    } catch (...) {
        std::cout << "  \033[31m\033[0m  " << name
                  << "  ->  unknown exception\n";
        ++g_failed;
    }
}

/** @brief Throw a descriptive runtime_error when an assertion fails. */
#define ASSERT(cond, msg)                                   \
    do {                                                    \
        if (!(cond))                                        \
            throw std::runtime_error(                       \
                std::string("ASSERT FAILED: ") + (msg) +   \
                " [" #cond "] at line " +                   \
                std::to_string(__LINE__));                   \
    } while (0)

// -------------------------------------------------------------
// Tier 1  -  Correctness Tests (single-threaded)
// -------------------------------------------------------------

/** Pool reports exactly num_blocks free entries after construction. */
static void test_pool_init() {
    BlockAllocator alloc(256);
    alloc.flush_wallet();  // drain any TLA residue before exact count check
    ASSERT(alloc.get_free_count() == 256, "free count should equal pool size");
    ASSERT(alloc.get_total_blocks() == 256, "total_blocks should equal pool size");
}

/** A single allocate/free cycle returns the block and restores count. */
static void test_single_alloc_free() {
    BlockAllocator alloc(64);
    PhysicalBlock* b = alloc.allocate_block();

    ASSERT(b != nullptr, "allocated block must not be null");
    ASSERT(b->ref_count.load() == 1, "fresh block ref_count must be 1");
    ASSERT(b->num_tokens.load() == 0, "fresh block num_tokens must be 0");
    alloc.flush_wallet();  // drain speculative prefetch before exact count check
    ASSERT(alloc.get_free_count() == 63, "free count should decrease by 1");

    alloc.free_block(b);
    alloc.flush_wallet();  // drain wallet deposit before exact count check
    ASSERT(alloc.get_free_count() == 64, "free count should be restored");
}

/** Exhausting the pool throws OOM; freeing one block resolves it. */
static void test_full_pool_oom() {
    constexpr int N = 16;
    BlockAllocator alloc(N);

    std::vector<PhysicalBlock*> held(N);
    for (int i = 0; i < N; i++) {
        held[i] = alloc.allocate_block();
    }
    alloc.flush_wallet();  // drain speculative prefetch before exact count check
    ASSERT(alloc.get_free_count() == 0, "pool must be empty");

    bool threw = false;
    try {
        alloc.allocate_block();  // must throw
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ASSERT(threw, "OOM must throw std::runtime_error");

    // Free one block; the pool recovers.
    alloc.free_block(held[0]);
    alloc.flush_wallet();  // drain wallet deposit before exact count check
    ASSERT(alloc.get_free_count() == 1, "one freed block must reappear");
    PhysicalBlock* recovered = alloc.allocate_block();
    ASSERT(recovered != nullptr, "recovered block must be valid");

    // Clean up.
    for (int i = 1; i < N; i++) alloc.free_block(held[i]);
}

/**
 * @brief Ref-count sharing: simulates two sequences sharing one block
 *        (as in prefix-cache / copy-on-write).
 *
 * The block must not return to the pool until the last reference is
 * released, mirroring the behaviour expected for prompt-prefix sharing.
 */
static void test_refcount_sharing() {
    BlockAllocator alloc(8);
    PhysicalBlock* shared = alloc.allocate_block();
    ASSERT(shared->ref_count.load() == 1, "initial ref_count must be 1");

    // A second sequence "borrows" the block  -  bump ref_count manually.
    shared->ref_count.fetch_add(1, std::memory_order_relaxed);
    ASSERT(shared->ref_count.load() == 2, "ref_count must be 2 after borrow");

    // First sequence is done  -  decrement but block stays alive.
    alloc.free_block(shared);
    alloc.flush_wallet();  // drain wallet before exact count check
    ASSERT(alloc.get_free_count() == 7, "block must NOT be recycled yet");
    ASSERT(shared->ref_count.load() == 1, "ref_count must drop to 1");

    // Second sequence is done  -  now the block is truly free.
    alloc.free_block(shared);
    alloc.flush_wallet();  // drain wallet before exact count check
    ASSERT(alloc.get_free_count() == 8, "block must be recycled when ref_count hits 0");
}

/**
 * @brief append_token drives the page-fault path: a new PhysicalBlock
 *        is allocated and linked when the current one is full.
 *
 * With BLOCK_SIZE=16: 33 tokens require ceil(33/16) = 3 blocks.
 */
static void test_append_token_state_machine() {
    BlockAllocator alloc(32);
    BlockTable req(42);

    for (int i = 0; i < 33; i++) {
        alloc.append_token(req);
    }

    ASSERT(req.logical_length == 33,  "logical_length must equal tokens appended");
    ASSERT((int)req.blocks.size() == 3, "3 physical blocks needed for 33 tokens");
    alloc.flush_wallet();  // drain speculative prefetch before exact count check
    ASSERT(alloc.get_free_count() == 29, "32 - 3 = 29 free blocks must remain");
}

/** free_sequence recycles every block; pool is fully restored. */
static void test_free_sequence_recycles_all() {
    BlockAllocator alloc(32);
    BlockTable req(7);

    for (int i = 0; i < 48; i++) alloc.append_token(req);  // 3 blocks
    alloc.free_sequence(req);

    ASSERT(req.blocks.empty(),        "BlockTable must be empty after free");
    ASSERT(req.logical_length == 0,   "logical_length must reset to 0");
    alloc.flush_wallet();  // drain wallet deposits before exact count check
    ASSERT(alloc.get_free_count() == 32, "all blocks must be recycled");
}

/**
 * @brief Pool reuse: allocate all -> free all -> allocate all again.
 *
 * This exercises the ring buffer's wraparound path: after N pops followed
 * by N pushes, the sequence numbers advance by one full rotation and the
 * next N pops must see consistent, valid block IDs.
 */
static void test_pool_reuse_wraparound() {
    constexpr int N = 64;
    BlockAllocator alloc(N);

    // Round 1  -  drain pool.
    std::vector<PhysicalBlock*> held(N);
    for (int i = 0; i < N; i++) held[i] = alloc.allocate_block();
    alloc.flush_wallet();  // drain speculative prefetch before exact count check
    ASSERT(alloc.get_free_count() == 0, "pool must be drained");

    // Round 1  -  refill.
    for (auto* b : held) alloc.free_block(b);
    alloc.flush_wallet();  // drain wallet deposits before exact count check
    ASSERT(alloc.get_free_count() == N, "pool must be fully refilled");

    // Round 2  -  drain again (ring has wrapped).
    std::vector<PhysicalBlock*> held2(N);
    for (int i = 0; i < N; i++) {
        held2[i] = alloc.allocate_block();
        ASSERT(held2[i] != nullptr, "post-wraparound alloc must succeed");
    }
    for (auto* b : held2) alloc.free_block(b);
    alloc.flush_wallet();  // drain wallet deposits before exact count check
    ASSERT(alloc.get_free_count() == N, "pool must be fully refilled after round 2");
}

/** Multiple independent sequences can coexist without interfering. */
static void test_multiple_sequences_independent() {
    BlockAllocator alloc(128);
    BlockTable req_a(1), req_b(2), req_c(3);

    for (int i = 0; i < 20; i++) alloc.append_token(req_a);
    for (int i = 0; i < 35; i++) alloc.append_token(req_b);
    for (int i = 0; i < 10; i++) alloc.append_token(req_c);

    // 20->2 blocks, 35->3 blocks, 10->1 block = 6 total
    int expected_free = 128 - (2 + 3 + 1);
    alloc.flush_wallet();  // drain speculative prefetch before exact count check
    ASSERT(alloc.get_free_count() == expected_free,
           "free count must reflect all three concurrent sequences");

    alloc.free_sequence(req_a);
    alloc.free_sequence(req_b);
    alloc.free_sequence(req_c);
    alloc.flush_wallet();  // drain wallet deposits before exact count check
    ASSERT(alloc.get_free_count() == 128, "all blocks must return after eviction");
}

/** Block IDs returned from the ring are always valid pool indices. */
static void test_block_ids_are_valid() {
    constexpr int N = 50;
    BlockAllocator alloc(N);
    std::vector<PhysicalBlock*> ptrs(N);
    for (int i = 0; i < N; i++) ptrs[i] = alloc.allocate_block();
    for (auto* b : ptrs) {
        ASSERT(b->physical_block_id >= 0 && b->physical_block_id < N,
               "physical_block_id must be in [0, N)");
    }
    for (auto* b : ptrs) alloc.free_block(b);
}

// -------------------------------------------------------------
// Tier 2  -  Concurrency Tests (multi-threaded)
// -------------------------------------------------------------

/**
 * @brief Standard 4-thread stress: each thread allocates one block,
 *        does a tiny amount of work, then frees it.  Runs 10 000
 *        iterations per thread.  Pool must be fully recovered at the end.
 */
static void test_concurrent_4thread_stress() {
    constexpr int POOL    = 512;
    constexpr int THREADS = 4;
    constexpr int ITERS   = 10'000;

    BlockAllocator alloc(POOL);
    std::atomic<int> oom_count{0};

    auto worker = [&]() {
        // RAII guard: flushes this thread's TLA wallet back to alloc on exit,
        // even if the thread exits via exception.  Prevents block leaks without
        // a global singleton pointer.
        WalletFlusher flusher(alloc);

        for (int i = 0; i < ITERS; i++) {
            try {
                PhysicalBlock* b = alloc.allocate_block();
                // Simulate a brief hold (prevents trivial lock-free degeneracy).
                b->num_tokens.fetch_add(1, std::memory_order_relaxed);
                b->num_tokens.fetch_sub(1, std::memory_order_relaxed);
                alloc.free_block(b);
            } catch (const std::runtime_error&) {
                oom_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads(THREADS);
    for (auto& t : threads) t = std::thread(worker);
    for (auto& t : threads) t.join();

    // Main thread: flush its own wallet too (it may have been used by
    // construction-time ring pushes in BlockAllocator's constructor).
    alloc.flush_wallet();
    ASSERT(alloc.get_free_count() == POOL,
           "all blocks must be recovered after concurrent stress");
}

/**
 * @brief High-contention scenario: 8 threads competing over a small pool
 *        (128 blocks), intentionally causing transient OOM events to test
 *        the ring's empty-detection path under pressure.
 */
static void test_concurrent_high_contention() {
    constexpr int POOL    = 128;
    constexpr int THREADS = 8;
    constexpr int ITERS   = 5'000;

    BlockAllocator alloc(POOL);
    std::atomic<int> oom_count{0};
    std::atomic<int> success_count{0};

    auto worker = [&]() {
        WalletFlusher flusher(alloc);  // flush on thread exit

        for (int i = 0; i < ITERS; i++) {
            try {
                PhysicalBlock* b = alloc.allocate_block();
                success_count.fetch_add(1, std::memory_order_relaxed);
                alloc.free_block(b);
            } catch (const std::runtime_error&) {
                oom_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads(THREADS);
    for (auto& t : threads) t = std::thread(worker);
    for (auto& t : threads) t.join();

    alloc.flush_wallet();  // flush main thread wallet
    ASSERT(alloc.get_free_count() == POOL,
           "pool must be fully recovered after high-contention stress");
    // Sanity: there must have been at least some successful allocations.
    ASSERT(success_count.load() > 0, "at least one alloc must have succeeded");
}

/**
 * @brief Producer/consumer split: 4 threads allocate, 4 threads free.
 *        This is the closest model to real inference batching, where the
 *        scheduler thread frees blocks for completed sequences while the
 *        prefill thread allocates for new ones simultaneously.
 */
static void test_concurrent_producer_consumer() {
    constexpr int POOL      = 256;
    constexpr int PRODUCERS = 4;
    constexpr int CONSUMERS = 4;
    constexpr int ITERS     = 2'000;

    BlockAllocator alloc(POOL);

    // A simple lock-free hand-off queue using an atomic array.
    // Producers push allocated pointers; consumers pull and free them.
    constexpr int Q = POOL * 2;
    std::atomic<PhysicalBlock*> queue[Q]{};  // initialised to nullptr
    std::atomic<int> write_cursor{0};
    std::atomic<int> read_cursor{0};
    std::atomic<bool> producers_done{false};

    auto producer = [&]() {
        WalletFlusher flusher(alloc);  // flush on thread exit

        for (int i = 0; i < ITERS; i++) {
            try {
                PhysicalBlock* b = alloc.allocate_block();
                // Spin-insert into the hand-off queue.
                int slot = write_cursor.fetch_add(1, std::memory_order_relaxed) & (Q - 1);
                PhysicalBlock* expected = nullptr;
                while (!queue[slot].compare_exchange_weak(
                            expected, b,
                            std::memory_order_release,
                            std::memory_order_relaxed)) {
                    expected = nullptr;
                }
            } catch (const std::runtime_error&) { /* pool busy  -  skip */ }
        }
    };

    auto consumer = [&]() {
        WalletFlusher flusher(alloc);  // flush on thread exit

        while (!producers_done.load(std::memory_order_acquire)) {
            int slot = read_cursor.fetch_add(1, std::memory_order_relaxed) & (Q - 1);
            PhysicalBlock* b = queue[slot].exchange(nullptr, std::memory_order_acquire);
            if (b) alloc.free_block(b);
        }
        // Drain any remaining hand-off items.
        for (int s = 0; s < Q; s++) {
            PhysicalBlock* b = queue[s].exchange(nullptr, std::memory_order_acquire);
            if (b) alloc.free_block(b);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < PRODUCERS; i++) threads.emplace_back(producer);
    for (int i = 0; i < CONSUMERS; i++) threads.emplace_back(consumer);

    for (int i = 0; i < PRODUCERS; i++) threads[i].join();
    producers_done.store(true, std::memory_order_release);
    for (int i = PRODUCERS; i < PRODUCERS + CONSUMERS; i++) threads[i].join();

    alloc.flush_wallet();  // flush main thread wallet
    ASSERT(alloc.get_free_count() == POOL,
           "producer/consumer pattern must leave pool fully recovered");
}

// -------------------------------------------------------------
// Tier 3  -  Edge Cases
// -------------------------------------------------------------

/** A pool of size 1 must still satisfy alloc -> free -> alloc correctly. */
static void test_pool_size_one() {
    BlockAllocator alloc(1);
    PhysicalBlock* b = alloc.allocate_block();
    ASSERT(b != nullptr, "single-slot pool must yield a block");
    alloc.flush_wallet();  // drain speculative prefetch before exact count check
    ASSERT(alloc.get_free_count() == 0, "pool must be empty");
    alloc.free_block(b);
    alloc.flush_wallet();  // drain wallet deposit before exact count check
    ASSERT(alloc.get_free_count() == 1, "pool must recover");

    // Must be allocatable again.
    PhysicalBlock* b2 = alloc.allocate_block();
    ASSERT(b2 != nullptr, "re-alloc from size-1 pool must succeed");
    alloc.free_block(b2);
}

/** append_token correctly handles the boundary where a block is exactly full. */
static void test_block_boundary_exact_fill() {
    BlockAllocator alloc(8);
    BlockTable req(99);

    // Fill exactly one block (BLOCK_SIZE = 16 tokens).
    for (int i = 0; i < BLOCK_SIZE; i++) alloc.append_token(req);
    ASSERT((int)req.blocks.size() == 1, "one full block must be used");
    ASSERT(req.blocks[0]->is_full(), "first block must report is_full()");

    // One more token must trigger a page fault -> second block.
    alloc.append_token(req);
    ASSERT((int)req.blocks.size() == 2, "page fault must allocate second block");
    ASSERT(!req.blocks[1]->is_full(), "second block must not yet be full");

    alloc.free_sequence(req);
    alloc.flush_wallet();  // drain wallet deposits before exact count check
    ASSERT(alloc.get_free_count() == 8, "full recovery after boundary test");
}

// -------------------------------------------------------------
// Main
// -------------------------------------------------------------

int main() {
    using std::chrono::steady_clock;
    auto t0 = steady_clock::now();

    std::cout << "\n\033[1;34m\033[0m\n";
    std::cout << "\033[1;34m  nanoMoE :: BlockAllocator Test Suite\033[0m\n";
    std::cout << "\033[1;34m\033[0m\n\n";

    // -- Tier 1: Correctness ----------------------------------
    std::cout << "\033[1mTier 1  -  Correctness\033[0m\n";
    run_test("Pool initialisation",              test_pool_init);
    run_test("Single allocate / free cycle",     test_single_alloc_free);
    run_test("Full-pool OOM + recovery",         test_full_pool_oom);
    run_test("Ref-count sharing (prefix cache)", test_refcount_sharing);
    run_test("append_token state machine",       test_append_token_state_machine);
    run_test("free_sequence recycles all",       test_free_sequence_recycles_all);
    run_test("Pool reuse / ring wraparound",     test_pool_reuse_wraparound);
    run_test("Multiple independent sequences",   test_multiple_sequences_independent);
    run_test("Block IDs in valid range",         test_block_ids_are_valid);
    run_test("Block boundary exact fill",        test_block_boundary_exact_fill);
    run_test("Pool size = 1 (edge case)",        test_pool_size_one);

    // -- Tier 2: Concurrency ----------------------------------
    std::cout << "\n\033[1mTier 2  -  Concurrency\033[0m\n";
    run_test("4-thread stress (10k iters each)", test_concurrent_4thread_stress);
    run_test("8-thread high contention",         test_concurrent_high_contention);
    run_test("Producer / consumer split",        test_concurrent_producer_consumer);

    // -- Summary ----------------------------------------------
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        steady_clock::now() - t0).count();

    int total = g_passed + g_failed;
    std::cout << "\n\033[1;34m\033[0m\n";
    if (g_failed == 0) {
        std::cout << "\033[1;32m  PASSED " << g_passed << " / " << total
                  << " tests  (" << elapsed_ms << " ms)\033[0m\n";
    } else {
        std::cout << "\033[1;31m  FAILED " << g_failed << " / " << total
                  << "  (" << g_passed << " passed, " << elapsed_ms << " ms)\033[0m\n";
    }
    std::cout << "\033[1;34m\033[0m\n\n";

    return g_failed > 0 ? 1 : 0;
}
