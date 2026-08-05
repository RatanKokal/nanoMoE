/**
 * @file bench_legacy.cpp
 * @brief Three-way benchmark: unsafe queue vs. mutex-locked queue vs. TLA.
 *
 * Allocator variants:
 *   A. LegacyBlockAllocator      -  original std::queue, NO synchronisation.
 *                                  Fast on 1 thread; undefined behaviour on 2+.
 *   B. MutexBlockAllocator       -  same std::queue wrapped in std::mutex.
 *                                  Thread-safe; the fair baseline for comparison.
 *
 * The TLA (Vyukov + wallet) results come from running `make -C bench/cpp run`
 * separately.  This file deliberately does not link mem_manager.cpp so it
 * stays self-contained and can be compiled independently.
 *
 * Build:
 *   make -C bench/cpp legacy        (from repo root)
 * Compare all three:
 *   make -C bench/cpp compare
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// -------------------------------------------------------------
// Shared block type (plain ints  -  no atomics)
// -------------------------------------------------------------

constexpr int BLOCK_SIZE = 16;

struct PhysicalBlock {
    int physical_block_id;
    int ref_count;
    int num_tokens;

    explicit PhysicalBlock(int id) : physical_block_id(id), ref_count(0), num_tokens(0) {}
    bool is_full()  const { return num_tokens == BLOCK_SIZE; }
    bool is_empty() const { return num_tokens == 0; }
};

// -------------------------------------------------------------
// Allocator A  -  original, NO synchronisation
// -------------------------------------------------------------

class LegacyBlockAllocator {
    std::vector<PhysicalBlock> physical_memory_pool;
    std::queue<PhysicalBlock*> free_list;
    int total_blocks;
public:
    explicit LegacyBlockAllocator(int num_blocks) : total_blocks(num_blocks) {
        physical_memory_pool.reserve(num_blocks);
        for (int i = 0; i < num_blocks; i++) {
            physical_memory_pool.emplace_back(i);
            free_list.push(&physical_memory_pool.back());
        }
    }
    PhysicalBlock* allocate_block() {
        if (free_list.empty()) throw std::runtime_error("OOM");
        PhysicalBlock* b = free_list.front(); free_list.pop();
        b->ref_count = 1; b->num_tokens = 0;
        return b;
    }
    void free_block(PhysicalBlock* b) {
        if (--b->ref_count == 0) { b->num_tokens = 0; free_list.push(b); }
    }
};

// -------------------------------------------------------------
// Allocator B  -  same queue, wrapped in std::mutex
// -------------------------------------------------------------

class MutexBlockAllocator {
    std::vector<PhysicalBlock> physical_memory_pool;
    std::queue<PhysicalBlock*> free_list;
    mutable std::mutex mtx;
    int total_blocks;
public:
    explicit MutexBlockAllocator(int num_blocks) : total_blocks(num_blocks) {
        physical_memory_pool.reserve(num_blocks);
        for (int i = 0; i < num_blocks; i++) {
            physical_memory_pool.emplace_back(i);
            free_list.push(&physical_memory_pool.back());
        }
    }
    PhysicalBlock* allocate_block() {
        std::lock_guard<std::mutex> lk(mtx);
        if (free_list.empty()) throw std::runtime_error("OOM");
        PhysicalBlock* b = free_list.front(); free_list.pop();
        b->ref_count = 1; b->num_tokens = 0;
        return b;
    }
    void free_block(PhysicalBlock* b) {
        std::lock_guard<std::mutex> lk(mtx);
        if (--b->ref_count == 0) { b->num_tokens = 0; free_list.push(b); }
    }
};

// -------------------------------------------------------------
// Timing
// -------------------------------------------------------------

static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
}

// -------------------------------------------------------------
// Generic benchmark helpers  -  work on any allocator type via template
// -------------------------------------------------------------

template <typename Alloc>
static double run_single_thread(int pool_size, int iters) {
    Alloc alloc(pool_size);
    // Warm-up
    { std::vector<PhysicalBlock*> t(pool_size);
      for (auto& p : t) p = alloc.allocate_block();
      for (auto* p : t) alloc.free_block(p); }

    uint64_t t0 = now_ns();
    for (int i = 0; i < iters; i++) {
        PhysicalBlock* b = alloc.allocate_block();
        alloc.free_block(b);
    }
    uint64_t t1 = now_ns();
    return (iters * 2.0) / ((t1 - t0) / 1e9) / 1e6;  // Mops/s
}

template <typename Alloc>
static double run_multi_thread(int pool_size, int iters_per_thread, int nthreads) {
    Alloc alloc(pool_size);
    std::atomic<int> oom{0};

    auto worker = [&]() {
        for (int i = 0; i < iters_per_thread; i++) {
            try {
                PhysicalBlock* b = alloc.allocate_block();
                alloc.free_block(b);
            } catch (...) { oom.fetch_add(1, std::memory_order_relaxed); }
        }
    };

    uint64_t t0 = now_ns();
    std::vector<std::thread> threads(nthreads);
    for (auto& t : threads) t = std::thread(worker);
    for (auto& t : threads) t.join();
    uint64_t t1 = now_ns();

    long total_ops = static_cast<long>(nthreads) * iters_per_thread * 2;
    return total_ops / ((t1 - t0) / 1e9) / 1e6;  // Mops/s
}

template <typename Alloc>
static std::vector<uint64_t> run_latency_samples(int pool_size, int samples) {
    Alloc alloc(pool_size);
    for (int i = 0; i < 5000; i++) {
        PhysicalBlock* b = alloc.allocate_block(); alloc.free_block(b);
    }
    std::vector<uint64_t> v(samples);
    for (int i = 0; i < samples; i++) {
        uint64_t t0 = now_ns();
        PhysicalBlock* b = alloc.allocate_block(); alloc.free_block(b);
        v[i] = now_ns() - t0;
    }
    std::sort(v.begin(), v.end());
    return v;
}

// -------------------------------------------------------------
// Print helpers
// -------------------------------------------------------------

static void print_header(const std::string& title, const std::string& colour) {
    std::cout << "\n" << colour
              << "\033[0m\n"
              << colour << "  " << title << "\033[0m\n"
              << colour
              << "\033[0m\n";
}

static void print_scaling_row(int nthreads, double mops, double baseline, bool first) {
    double scale = mops / baseline;
    std::cout << "  " << std::setw(8) << nthreads
              << std::setw(14) << std::fixed << std::setprecision(2) << mops
              << std::setw(14) << std::setprecision(1) << (1000.0 / mops)
              << "   (" << (first ? "baseline" :
                            std::to_string(static_cast<int>(scale * 100) / 100.0)
                            .substr(0, 4) + "x") << ")\n";
}

// -------------------------------------------------------------
// Main
// -------------------------------------------------------------

int main() {
    constexpr int POOL    = 4096;
    constexpr int ITERS   = 2'000'000;
    constexpr int SAMPLES = 200'000;
    constexpr int MT_ITER = 500'000;
    const int hw = static_cast<int>(std::thread::hardware_concurrency());

    // -- Section A: Unsafe Legacy -----------------------------
    print_header("nanoMoE :: UNSAFE Legacy  (std::queue, no locks)",
                 "\033[1;31m");

    {
        double mops = run_single_thread<LegacyBlockAllocator>(POOL, ITERS);
        std::cout << "\nBenchmark 1  -  Single-threaded throughput\n";
        std::cout << "  Throughput   : \033[1;32m" << std::fixed
                  << std::setprecision(2) << mops << " Mops/s\033[0m\n";
        std::cout << "  Latency/op   : \033[1;32m" << std::setprecision(1)
                  << 1000.0 / mops << " ns\033[0m\n";

        auto v = run_latency_samples<LegacyBlockAllocator>(POOL, SAMPLES);
        std::cout << "\nBenchmark 2  -  Latency percentiles\n";
        std::cout << "  p50: " << v[SAMPLES/2] << " ns  "
                  << "p99: " << v[static_cast<int>(SAMPLES*0.99)] << " ns  "
                  << "max: " << v.back() << " ns\n";

        std::cout << "\nBenchmark 3  -  Multi-thread  \033[1;31m[UNSAFE  -  DATA RACES]\033[0m\n";
        std::cout << "  Skipped to avoid silent heap corruption.\n";
        std::cout << "  (Running this crashes or produces garbage results.)\n";
    }

    // -- Section B: Mutex-Locked (Fair Baseline) --------------
    print_header("nanoMoE :: MUTEX Legacy   (std::queue + std::mutex)",
                 "\033[1;33m");

    {
        double mops1 = run_single_thread<MutexBlockAllocator>(POOL, ITERS);
        std::cout << "\nBenchmark 1  -  Single-threaded throughput\n";
        std::cout << "  Throughput   : \033[1;32m" << std::fixed
                  << std::setprecision(2) << mops1 << " Mops/s\033[0m\n";
        std::cout << "  Latency/op   : \033[1;32m" << std::setprecision(1)
                  << 1000.0 / mops1 << " ns\033[0m\n";

        auto v = run_latency_samples<MutexBlockAllocator>(POOL, SAMPLES);
        std::cout << "\nBenchmark 2  -  Latency percentiles\n";
        std::cout << "  p50: " << v[SAMPLES/2] << " ns  "
                  << "p99: " << v[static_cast<int>(SAMPLES*0.99)] << " ns  "
                  << "max: " << v.back() << " ns\n";

        std::cout << "\nBenchmark 3  -  Multi-thread throughput scaling\n";
        std::cout << "  Pool: " << POOL << " blocks  |  Iters/thread: " << MT_ITER
                  << "  |  HW threads: " << hw << "\n\n";
        std::cout << std::setw(10) << "Threads"
                  << std::setw(16) << "Mops/s (total)"
                  << std::setw(16) << "ns/op (avg)" << "\n";
        std::cout << "  " << std::string(40, '-') << "\n";

        double baseline = 0;
        for (int n : {1, 2, 4, 8, 16}) {
            if (n > hw * 2) break;
            double m = run_multi_thread<MutexBlockAllocator>(POOL, MT_ITER, n);
            if (n == 1) baseline = m;
            std::cout << "  " << std::setw(8) << n
                      << std::setw(14) << std::fixed << std::setprecision(2) << m
                      << std::setw(14) << std::setprecision(1) << (1000.0 / m)
                      << "   (" << std::setprecision(2) << m / baseline << "x)\n";
        }
    }

    std::cout << "\n\033[1;33m\033[0m\n\n";
    return 0;
}
