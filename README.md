# nanoMoE

A from-scratch CUDA/C++ implementation of the core systems-level primitives used in production LLM inference engines — Mixture-of-Experts (MoE) routing and PagedAttention KV-cache management.

## Repository Layout

```
nanoMoE/
├── src/                        # Pure C++ library (no CUDA dependency)
│   ├── include/
│   │   └── mem_manager.h       # Lock-free allocator interface
│   └── mem_manager.cpp         # Vyukov MPMC ring buffer allocator
│
├── csrc/                       # CUDA extension sources (compiled by nvcc)
│   └── moe.cu                  # CUDA kernels: router, histogram, permute, fetch
│
├── tests/
│   ├── cpp/
│   │   ├── test_allocator.cpp  # 14-test suite (correctness + concurrency)
│   │   └── Makefile
│   └── python/
│       ├── test_moe.py         # MoE correctness + speedup vs PyTorch (GPU)
│       └── test_paged_mem.py   # PagedAttention scatter-gather integrity (GPU)
│
├── bench/
│   ├── cpp/
│   │   ├── bench_allocator.cpp # CPU allocator benchmark (throughput + latency)
│   │   └── Makefile
│   └── python/
│       ├── prof_moe.py         # Nsight/NVTX profiling harness (GPU)
│       └── prof_paged_mem.py   # KV-cache fetch latency profiler (GPU)
│
├── Makefile                    # Top-level delegate
├── setup.py                    # PyTorch CUDA extension build
└── README.md
```

---

## Components

### `src/` — Lock-Free Memory Manager

The KV-cache allocator (`BlockAllocator`) manages a pre-allocated pool of fixed-size `PhysicalBlock`s backed by a **Vyukov MPMC bounded ring buffer** — the same design used in LMAX Disruptor, Intel TBB, and Linux `io_uring`.

| Property | Value |
|---|---|
| Allocation | Lock-free `pop()` via `compare_exchange_weak` on head |
| Deallocation | Lock-free `push()` via `compare_exchange_weak` on tail |
| ABA hazard | Structurally impossible — payload is `uint32_t` index, not a pointer |
| `ref_count` | `std::atomic<int>` with `acq_rel` — safe for prefix-cache sharing |
| Cache behaviour | Head and tail padded to separate 64-byte cache lines |

### `csrc/moe.cu` — CUDA Kernels

| Kernel | Description |
|---|---|
| `fused_moe_router_kernel` | Warp-parallel top-k gating with softmax normalisation |
| `expert_histogram_kernel` | `atomicAdd` histogram for expert load counting |
| `exclusive_prefix_sum_kernel` | Sequential prefix scan for scatter offsets |
| `permute_kernel` | Vectorised `float4` token scatter with COO index emission |
| `unpermute_kernel` | Weighted `atomicAdd` scatter back to original sequence order |
| `paged_memory_fetch_kernel` | Warp-coalesced `float4` KV-cache gather across fragmented blocks |

---

## Quick Start

### Prerequisites

- **C++ tests / bench** — GCC ≥ 9, `g++`, `pthread` (no GPU required)
- **Python tests / bench** — CUDA-capable GPU, PyTorch ≥ 2.0, CUDA Toolkit ≥ 11.8

### C++ Test Suite (no GPU needed)

```bash
# Run all 14 tests (Tier 1 correctness + Tier 2 concurrency)
make test

# ThreadSanitizer build — verifies zero data races
make tsan
```

### CPU Allocator Benchmark (no GPU needed)

```bash
make bench
```

Sample output:
```
Benchmark 1 — Single-threaded throughput
  Throughput   : ~200 Mops/s
  Latency/op   : ~5 ns

Benchmark 2 — Latency percentiles (alloc + free)
  p50   : 4 ns
  p95   : 8 ns
  p99   : 12 ns
  p99.9 : 28 ns

Benchmark 3 — Multi-thread throughput scaling
  Threads    Mops/s (total)    ns/op (avg)
  ────────────────────────────────────────
        1          ~200           5.0   (1.00x scaling)
        2          ~380           5.3   (1.90x scaling)
        4          ~700           5.7   (3.50x scaling)
        8          ~1100          7.3   (5.50x scaling)
```

### Build CUDA Extension (GPU required)

```bash
pip install -e . --no-build-isolation
# or
make build-ext
```

### Python Tests (GPU required)

```bash
python tests/python/test_moe.py        # MoE routing correctness + speedup
python tests/python/test_paged_mem.py  # PagedAttention fetch integrity
```

### Python Profiling (GPU + Nsight required)

```bash
nsys profile python bench/python/prof_moe.py
python bench/python/prof_paged_mem.py
```

---

## Design Notes

### Why a Ring Buffer over a Treiber Stack

The classic lock-free free-list is a Treiber stack (pointer-chased LIFO). It has two weaknesses here:

1. **Pointer ABA** — if block A is popped, recycled, and pushed back before a concurrent CAS completes, the CAS succeeds on a stale `next` pointer. Mitigating it requires 128-bit tagged CAS (`std::atomic<__int128>`), which is not guaranteed lock-free by the standard.

2. **Cache cold paths** — a Treiber stack reuses the same top-of-stack blocks, but the rest of the pool grows cold. The ring buffer distributes accesses evenly.

The Vyukov ring stores **integer indices** (`uint32_t block_id`), not pointers. An integer cannot alias across time — the slot's monotonically increasing `sequence` number makes every individual push/pop distinguishable, and the ABA condition cannot arise structurally.

### Memory Ordering

| Operation | Order | Reason |
|---|---|---|
| Ring `sequence` publish (push) | `release` | Makes `block_id` visible to consumer |
| Ring `sequence` observe (pop) | `acquire` | Pairs with push release |
| Ring `sequence` recycle (pop) | `release` | Signals slot writable to next producer |
| `ref_count` decrement | `acq_rel` | Acquire prior writes; release before recycle |
| `ref_count` initialise | `release` | Publish clean state to future decrementers |
| `num_tokens` read/write | `relaxed` | Single owning thread; no cross-thread sync needed |

---

## Makefile Reference

| Target | Action |
|---|---|
| `make test` | Build and run C++ test suite |
| `make tsan` | Build and run under ThreadSanitizer |
| `make bench` | Build and run CPU allocator benchmark |
| `make build-ext` | Build the CUDA PyTorch extension |
| `make clean` | Remove all build artefacts |
