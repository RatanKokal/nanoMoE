#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <math.h>

#include "../src/include/mem_manager.h"

#define MAX_K 8
#define MAX_EXPERTS_PER_LANE 16  // Supports up to 512 experts (16 * 32 lanes)

// ─────────────────────────────────────────────────────────────────────────────
// Warp Primitives
// ─────────────────────────────────────────────────────────────────────────────

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernels (Identical logic, zero atomics, float4 vectorization)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void topk_softmax_kernel(
    const float* __restrict__ logits, 
    int*         __restrict__ topk_indices, 
    float*       __restrict__ topk_weights, 
    int N, int E, int k
) {
    int token_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id  = threadIdx.x % 32;
    if (token_id >= N) return;

    int num_per_lane = min((E + 31) / 32, MAX_EXPERTS_PER_LANE);

    float local_logits[MAX_EXPERTS_PER_LANE];
    int   local_expert_ids[MAX_EXPERTS_PER_LANE];
    for (int i = 0; i < MAX_EXPERTS_PER_LANE; i++) { local_logits[i] = -1e9f; local_expert_ids[i] = -1; }
    for (int i = 0; i < num_per_lane; i++) {
        int eid = lane_id + i * 32;
        if (eid >= E) break;
        local_logits[i]     = logits[token_id * E + eid];
        local_expert_ids[i] = eid;
    }

    int   final_topk_ids[MAX_K];
    float final_topk_logits[MAX_K];

    for (int step = 0; step < k; step++) {
        float thread_max         = -1e9f;
        int   thread_best_expert = 0;   // default to expert 0; guards against NaN logits
                                        // producing index -1 which causes illegal GPU memory access
        int   best_local_idx     = -1;

        for (int i = 0; i < num_per_lane; i++) {
            if (local_logits[i] > thread_max) {
                thread_max         = local_logits[i];
                thread_best_expert = local_expert_ids[i];
                best_local_idx     = i;
            }
        }

        float warp_max = warp_reduce_max(thread_max);
        warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);

        unsigned int winner_mask = __ballot_sync(0xFFFFFFFF, thread_max == warp_max);
        int winner_lane          = __ffs(winner_mask) - 1;
        int winning_expert       = __shfl_sync(0xFFFFFFFF, thread_best_expert, winner_lane);

        final_topk_logits[step] = warp_max;
        final_topk_ids[step]    = winning_expert;

        if (lane_id == winner_lane)
            local_logits[best_local_idx] = -1e9f;
    }

    if (lane_id == 0) {
        float max_logit = final_topk_logits[0];
        float exp_vals[MAX_K];
        float sum_exp = 0.0f;
        for (int step = 0; step < k; step++) {
            exp_vals[step] = __expf(final_topk_logits[step] - max_logit);
            sum_exp += exp_vals[step];
        }
        float inv_sum = __fdividef(1.0f, sum_exp);
        for (int step = 0; step < k; step++) {
            topk_indices[token_id * k + step] = final_topk_ids[step];
            topk_weights[token_id * k + step] = exp_vals[step] * inv_sum;
        }
    }
}

__global__ void expert_histogram_kernel(
    const int* __restrict__ topk_indices,
    int*       __restrict__ histogram,
    int total_routed,
    int E
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_routed) {
        int e = topk_indices[idx];
        if (e >= 0 && e < E)              // guard: NaN logits can write e=-1
            atomicAdd(&histogram[e], 1);
    }
}

__global__ void permute_kernel(
    const float* __restrict__ input_tokens,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    int*         __restrict__ write_pointers,
    float*       __restrict__ permuted_x,
    int*         __restrict__ reverse_map,
    int d_model, int k, int total_routed
) {
    int row = blockIdx.x;
    if (row >= total_routed) return;

    __shared__ int dest_row;
    if (threadIdx.x == 0) {
        dest_row         = atomicAdd(&write_pointers[topk_indices[row]], 1);
        reverse_map[row] = dest_row; // Needed for zero-atomic unpermute gather
    }
    __syncthreads();

    int orig_id = row / k;
    int vec_dim = d_model / 4;
    const float4* vec_in  = reinterpret_cast<const float4*>(input_tokens + orig_id * d_model);
    float4*       vec_out = reinterpret_cast<float4*>(permuted_x + dest_row * d_model);
    for (int col = threadIdx.x; col < vec_dim; col += blockDim.x)
        vec_out[col] = vec_in[col];
}

__global__ void unpermute_kernel(
    const float* __restrict__ expert_out,    
    const float* __restrict__ topk_weights,  
    const int*   __restrict__ reverse_map,   
    float*       __restrict__ final_out,     
    int d_model, int k
) {
    int token   = blockIdx.x;
    int vec_dim = d_model / 4;

    for (int col = threadIdx.x; col < vec_dim; col += blockDim.x) {
        float4 acc = {0.f, 0.f, 0.f, 0.f};

        for (int j = 0; j < k; j++) {
            int   dest_row = reverse_map[token * k + j];
            float w        = topk_weights[token * k + j];

            const float4* src = reinterpret_cast<const float4*>(expert_out + dest_row * d_model);
            float4 v = src[col];

            acc.x = __fmaf_rn(w, v.x, acc.x);
            acc.y = __fmaf_rn(w, v.y, acc.y);
            acc.z = __fmaf_rn(w, v.z, acc.z);
            acc.w = __fmaf_rn(w, v.w, acc.w);
        }
        reinterpret_cast<float4*>(final_out + token * d_model)[col] = acc;
    }
}

__global__ void paged_memory_fetch_kernel(
    const float* __restrict__ physical_kv_cache,
    const int*   __restrict__ block_tables,
    float*       __restrict__ output_tokens,
    int max_blocks_per_seq, int block_size, int head_dim,
    int total_tokens_to_fetch
) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id        = threadIdx.x % 32;
    if (global_warp_id >= total_tokens_to_fetch) return;

    int seq_len      = max_blocks_per_seq * block_size;
    int seq_idx      = global_warp_id / seq_len;
    int token_in_seq = global_warp_id % seq_len;

    int logical_block = token_in_seq / block_size;
    int token_offset  = token_in_seq % block_size;

    int physical_block_id = block_tables[seq_idx * max_blocks_per_seq + logical_block];
    physical_block_id     = __shfl_sync(0xFFFFFFFF, physical_block_id, 0);
    if (physical_block_id < 0) return;

    int memory_base = physical_block_id * block_size * head_dim + token_offset * head_dim;
    int out_base    = global_warp_id * head_dim;
    int vec_dim     = head_dim / 4;

    const float4* vec_in  = reinterpret_cast<const float4*>(physical_kv_cache + memory_base);
    float4*       vec_out = reinterpret_cast<float4*>(output_tokens + out_base);
    if (lane_id < vec_dim) vec_out[lane_id] = vec_in[lane_id];
}

// ─────────────────────────────────────────────────────────────────────────────
// Stable Persistent Buffer Cache (No Raw CUDA Graphs)
// ─────────────────────────────────────────────────────────────────────────────

struct MoEBufferCache {
    bool  ready   = false;
    int   N = -1, d_model = -1, E = -1, k = -1;

    torch::Tensor x, logits, topk_indices, topk_weights, histogram;
    torch::Tensor offsets, write_pointers, permuted_x, reverse_map, cub_temp;
    size_t        cub_temp_bytes = 0;
};

static MoEBufferCache g_cache;

static void allocate_buffers(const torch::Tensor& x, const torch::Tensor& W_g, int k, cudaStream_t stream) {
    int N = x.size(0);
    int d_model = x.size(1);
    int E = W_g.size(1);
    int total = N * k;
    
    auto f32 = x.options();
    auto i32 = x.options().dtype(torch::kInt32);
    auto u8  = x.options().dtype(torch::kUInt8);

    g_cache.x              = torch::empty({N, d_model}, f32);
    g_cache.logits         = torch::empty({N, E}, f32);
    g_cache.topk_indices   = torch::empty({N, k}, i32);
    g_cache.topk_weights   = torch::empty({N, k}, f32);
    g_cache.histogram      = torch::empty({E}, i32);
    g_cache.offsets        = torch::empty({E}, i32);
    g_cache.write_pointers = torch::empty({E}, i32);
    g_cache.permuted_x     = torch::empty({total, d_model}, f32);
    g_cache.reverse_map    = torch::empty({total}, i32);

    g_cache.cub_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, g_cache.cub_temp_bytes,
        g_cache.histogram.data_ptr<int>(),
        g_cache.offsets.data_ptr<int>(),
        E, stream
    );
    g_cache.cub_temp = torch::empty({(int64_t)g_cache.cub_temp_bytes}, u8);

    g_cache.N = N; g_cache.d_model = d_model;
    g_cache.E = E; g_cache.k = k;
    g_cache.ready = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// PyTorch C++ Bindings
// ─────────────────────────────────────────────────────────────────────────────

torch::Tensor prepare_block_tables_for_gpu(const std::vector<BlockTable>& active_requests, int max_blocks_per_seq) {
    int batch_size = active_requests.size();
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto tensor = torch::zeros({batch_size, max_blocks_per_seq}, opts);
    int* flat = tensor.data_ptr<int>();
    for (int b = 0; b < batch_size; b++) {
        const auto& table = active_requests[b];
        for (int i = 0; i < (int)table.blocks.size(); i++)
            flat[b * max_blocks_per_seq + i] = table.blocks[i]->physical_block_id;
    }
    return tensor.to(torch::kCUDA, /*non_blocking=*/true);
}

torch::Tensor paged_kv_fetch(torch::Tensor physical_kv_cache, torch::Tensor block_tables, int batch_size, int seq_len, int block_size, int head_dim) {
    int max_blocks_per_seq = block_tables.size(1);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(physical_kv_cache.device());
    auto output = torch::empty({batch_size, seq_len, head_dim}, opts);
    int total = batch_size * seq_len;
    int threads = 256;
    int blocks = (total * 32 + threads - 1) / threads;
    paged_memory_fetch_kernel<<<blocks, threads>>>(physical_kv_cache.data_ptr<float>(), block_tables.data_ptr<int>(), output.data_ptr<float>(), max_blocks_per_seq, block_size, head_dim, total);
    return output;
}

std::vector<torch::Tensor> route_and_permute(torch::Tensor x, torch::Tensor W_g, int k) {
    int N = x.size(0);
    int d_model = x.size(1);
    int E = W_g.size(1);
    int total = N * k;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (!g_cache.ready || g_cache.N != N || g_cache.d_model != d_model || g_cache.E != E || g_cache.k != k) {
        allocate_buffers(x, W_g, k, stream);
    }

    g_cache.x.copy_(x, /*non_blocking=*/true);
    torch::mm_out(g_cache.logits, g_cache.x, W_g);

    cudaMemsetAsync(g_cache.histogram.data_ptr<int>(), 0, E * sizeof(int), stream);

    int wpb = 4;
    topk_softmax_kernel<<<(N + wpb - 1) / wpb, wpb * 32, 0, stream>>>(
        g_cache.logits.data_ptr<float>(), g_cache.topk_indices.data_ptr<int>(), g_cache.topk_weights.data_ptr<float>(), N, E, k
    );

    expert_histogram_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        g_cache.topk_indices.data_ptr<int>(), g_cache.histogram.data_ptr<int>(), total, E
    );

    cub::DeviceScan::ExclusiveSum(
        g_cache.cub_temp.data_ptr(), g_cache.cub_temp_bytes, g_cache.histogram.data_ptr<int>(), g_cache.offsets.data_ptr<int>(), E, stream
    );

    cudaMemcpyAsync(g_cache.write_pointers.data_ptr<int>(), g_cache.offsets.data_ptr<int>(), E * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    permute_kernel<<<total, 256, 0, stream>>>(
        g_cache.x.data_ptr<float>(), g_cache.topk_indices.data_ptr<int>(), g_cache.topk_weights.data_ptr<float>(),
        g_cache.write_pointers.data_ptr<int>(), g_cache.permuted_x.data_ptr<float>(), g_cache.reverse_map.data_ptr<int>(), d_model, k, total
    );

    // Returns exactly 4 tensors to map transparently into Python's expectations:
    // cu_perm, cu_coo_idx, cu_coo_wt, cu_offsets
    return {
        g_cache.permuted_x,
        g_cache.reverse_map,   // Substitutes coo_indices to drive unpermute gather
        g_cache.topk_weights,  // Substitutes coo_weights
        g_cache.offsets
    };
}

// Signature maps exactly to Python call: unpermute(expert_out, cu_coo_idx, cu_coo_wt, N)
torch::Tensor unpermute(torch::Tensor expert_out, torch::Tensor reverse_map, torch::Tensor topk_weights, int N) {
    const int d_model = expert_out.size(1);
    const int k       = topk_weights.size(1);
    auto final_out = torch::empty({N, d_model}, expert_out.options());

    unpermute_kernel<<<N, 256>>>(
        expert_out.data_ptr<float>(), topk_weights.data_ptr<float>(), reverse_map.data_ptr<int>(), final_out.data_ptr<float>(), d_model, k
    );
    return final_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("route_and_permute", &route_and_permute, "Fused route + permute (returns 4 tensors to match legacy unpack)");
    m.def("unpermute", &unpermute, "Gather unpermute — 1 block per token, no atomics, float4 stores");
    m.def("paged_kv_fetch", &paged_kv_fetch, "Warp-coalesced paged KV cache fetch");
    m.def("prepare_block_tables_for_gpu", &prepare_block_tables_for_gpu, "CPU BlockTable vector → GPU int32 tensor");
}
