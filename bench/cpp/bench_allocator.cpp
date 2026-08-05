/**
 * @file bench_allocator.cpp
 * @brief CPU-side micro-benchmark for the lock-free BlockAllocator.
 *
 * Measures:
 *   1. Single-threaded throughput  (Mops/s, no contention baseline)
 *   2. Latency percentiles         (p50 / p95 / p99 / p99.9 in nanoseconds)
 *   3. N-thread throughput scaling (1 -> 2 -> 4 -> 8 threads)
 *
 * All timings use CLOCK_MONOTONIC_RAW to avoid NTP adjustments.
 *
 * Build:
 *   make -C bench/cpp          (from repo root)
 */

#include "../../src/include/mem_manager.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

// -------------------------------------------------------------
// Timing utilities
// -------------------------------------------------------------

/** Return monotonic nanoseconds using the highest-resolution clock. */
static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
}

// -------------------------------------------------------------
// Benchmark 1  -  Single-threaded throughput
// -------------------------------------------------------------

/**
 * @brief Measure raw allocate + free throughput on a single thread.
 *
 * A large pool ensures OOM never occurs, isolating the ring buffer's
 * CAS path from any contention or backpressure.
 */
static void bench_single_threaded_throughput() {
    constexpr int POOL  = 4096;
    constexpr int ITERS = 2'000'000;

    BlockAllocator alloc(POOL);

    // Warm-up: touch every slot to pull the pool into L1/L2.
    {
        std::vector<PhysicalBlock*> tmp(POOL);
        for (auto& p : tmp) p = alloc.allocate_block();
        for (auto* p : tmp) alloc.free_block(p);
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++) {
        PhysicalBlock* b = alloc.allocate_block();
        alloc.free_block(b);
    }
    uint64_t t1 = now_ns();

    double elapsed_s  = (t1 - t0) / 1e9;
    double mops       = (ITERS * 2.0) / elapsed_s / 1e6;  // alloc + free = 2 ops
    double ns_per_op  = (t1 - t0) / static_cast<double>(ITERS * 2);

    std::cout << "\n\033[1mBenchmark 1  -  Single-threaded throughput\033[0m\n";
    std::cout << "  Iterations   : " << ITERS << "\n";
    std::cout << "  Elapsed      : " << std::fixed << std::setprecision(3)
              << elapsed_s * 1000.0 << " ms\n";
    std::cout << "  Throughput   : \033[1;32m" << std::setprecision(2)
              << mops << " Mops/s\033[0m\n";
    std::cout << "  Latency/op   : \033[1;32m" << std::setprecision(1)
              << ns_per_op << " ns\033[0m\n";
}

// -------------------------------------------------------------
// Benchmark 2  -  Latency percentiles
// -------------------------------------------------------------

/**
 * @brief Collect per-operation latency samples and report percentiles.
 *
 * Each sample measures one allocate_block() + free_block() pair.
 * Collecting individual samples is intentionally minimal to avoid
 * observer effect  -  clock_gettime overhead is ~25 ns on modern Linux.
 */
static void bench_latency_percentiles() {
    constexpr int POOL    = 4096;
    constexpr int SAMPLES = 200'000;

    BlockAllocator alloc(POOL);

    // Warm-up.
    for (int i = 0; i < 5000; i++) {
        PhysicalBlock* b = alloc.allocate_block();
        alloc.free_block(b);
    }

    std::vector<uint64_t> samples(SAMPLES);
    for (int i = 0; i < SAMPLES; i++) {
        uint64_t t0 = now_ns();
        PhysicalBlock* b = alloc.allocate_block();
        alloc.free_block(b);
        uint64_t t1 = now_ns();
        samples[i]  = t1 - t0;
    }

    std::sort(samples.begin(), samples.end());
    auto pct = [&](double p) -> uint64_t {
        return samples[static_cast<size_t>(p * SAMPLES)];
    };

    std::cout << "\n\033[1mBenchmark 2  -  Latency percentiles (alloc + free)\033[0m\n";
    std::cout << "  Samples      : " << SAMPLES << "\n";
    std::cout << "  p50          : " << pct(0.50) << " ns\n";
    std::cout << "  p95          : " << pct(0.95) << " ns\n";
    std::cout << "  p99          : " << pct(0.99) << " ns\n";
    std::cout << "  p99.9        : " << pct(0.999) << " ns\n";
    std::cout << "  max          : " << samples.back() << " ns\n";
}

// -------------------------------------------------------------
// Benchmark 3  -  Multi-thread throughput scaling
// -------------------------------------------------------------

/**
 * @brief Measure aggregate throughput as thread count scales from 1 to N.
 *
 * Each thread runs a tight alloc+free loop.  Total ops/s across all
 * threads reveals how well the lock-free design scales under contention
 * versus a sequential baseline.
 */
static void bench_throughput_scaling() {
    constexpr int POOL       = 4096;
    constexpr int ITERS      = 500'000;
    const int     hw_threads = static_cast<int>(std::thread::hardware_concurrency());

    std::cout << "\n\033[1mBenchmark 3  -  Multi-thread throughput scaling\033[0m\n";
    std::cout << "  Pool size    : " << POOL << " blocks\n";
    std::cout << "  Iters/thread : " << ITERS << "\n";
    std::cout << "  HW threads   : " << hw_threads << "\n\n";
    std::cout << std::setw(10) << "Threads"
              << std::setw(16) << "Mops/s (total)"
              << std::setw(16) << "ns/op (avg)"
              << "\n";
    std::cout << "  " << std::string(40, '-') << "\n";

    double baseline_mops = 0.0;

    for (int nthreads : {1, 2, 4, 8, 16}) {
        if (nthreads > hw_threads * 2) break;

        BlockAllocator alloc(POOL);
        std::atomic<int> oom_count{0};

        auto worker = [&]() {
            for (int i = 0; i < ITERS; i++) {
                try {
                    PhysicalBlock* b = alloc.allocate_block();
                    alloc.free_block(b);
                } catch (...) {
                    oom_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        };

        // Time the parallel section only.
        uint64_t t0 = now_ns();
        std::vector<std::thread> threads(nthreads);
        for (auto& t : threads) t = std::thread(worker);
        for (auto& t : threads) t.join();
        uint64_t t1 = now_ns();

        double elapsed_s = (t1 - t0) / 1e9;
        long   total_ops = static_cast<long>(nthreads) * ITERS * 2;
        double mops      = total_ops / elapsed_s / 1e6;
        double ns_per_op = (t1 - t0) / static_cast<double>(total_ops);
        if (nthreads == 1) baseline_mops = mops;

        double scale = mops / baseline_mops;
        std::cout << "  " << std::setw(8) << nthreads
                  << std::setw(14) << std::fixed << std::setprecision(2) << mops
                  << std::setw(14) << std::setprecision(1) << ns_per_op
                  << "   (" << std::setprecision(2) << scale << "x scaling)\n";
    }
}

// -------------------------------------------------------------
// Main
// -------------------------------------------------------------

int main() {
    std::cout << "\n\033[1;34m\033[0m\n";
    std::cout << "\033[1;34m  nanoMoE :: BlockAllocator CPU Benchmark\033[0m\n";
    std::cout << "\033[1;34m\033[0m\n";

    bench_single_threaded_throughput();
    bench_latency_percentiles();
    bench_throughput_scaling();

    std::cout << "\n\033[1;34m\033[0m\n\n";
    return 0;
}
