#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_K 8

// --- 1. Warp Primitives ---
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// --- 2. Router Kernel ---
__global__ void fused_moe_router_kernel(
    const float* __restrict__ input_tokens, const float* __restrict__ gating_weights,
    int* __restrict__ topk_indices, float* __restrict__ topk_weights,
    int N, int d_model, int E, int k
) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int token_id = global_thread_id / 32;
    int lane_id = threadIdx.x % 32;

    if (token_id >= N) return;

    float local_logits[MAX_K] = {-1e9f}; 
    int local_expert_ids[MAX_K] = {-1};
    int num_experts_per_thread = (E + 31) / 32;

    for (int i = 0; i < num_experts_per_thread; i++) {
        int expert_id = lane_id + (i * 32);
        if (expert_id >= E) continue;
        
        float logit = 0.0f;
        for (int d = 0; d < d_model; d++) {
            logit += input_tokens[token_id * d_model + d] * gating_weights[d * E + expert_id];
        }
        local_logits[i] = logit;
        local_expert_ids[i] = expert_id;
    }

    int final_topk_ids[MAX_K];
    float final_topk_logits[MAX_K];

    for (int step = 0; step < k; step++) {
        float thread_max = -1e9f;
        int thread_best_expert = -1;
        int best_local_idx = -1;
        
        for (int i = 0; i < num_experts_per_thread; i++) {
            if (local_logits[i] > thread_max) {
                thread_max = local_logits[i];
                thread_best_expert = local_expert_ids[i];
                best_local_idx = i;
            }
        }

        float warp_max = warp_reduce_max(thread_max);
        warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);

        int is_winner = (thread_max == warp_max) ? 1 : 0;
        unsigned int winner_mask = __ballot_sync(0xFFFFFFFF, is_winner);
        int winner_lane = __ffs(winner_mask) - 1;
        int winning_expert = __shfl_sync(0xFFFFFFFF, thread_best_expert, winner_lane);
        
        final_topk_logits[step] = warp_max;
        final_topk_ids[step] = winning_expert;

        if (lane_id == winner_lane) {
            local_logits[best_local_idx] = -1e9f;
        }
    }

    if (lane_id == 0) {
        float max_logit = final_topk_logits[0];
        float sum_exp = 0.0f;
        for (int step = 0; step < k; step++) sum_exp += expf(final_topk_logits[step] - max_logit);
        for (int step = 0; step < k; step++) {
            topk_indices[token_id * k + step] = final_topk_ids[step];
            topk_weights[token_id * k + step] = expf(final_topk_logits[step] - max_logit) / sum_exp;
        }
    }
}

// --- 3. Histogram Kernel ---
__global__ void expert_histogram_kernel(const int* __restrict__ topk_indices, int* __restrict__ histogram, int total_routed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_routed) atomicAdd(&histogram[topk_indices[idx]], 1);
}

// --- 4. Prefix Sum Kernel ---
__global__ void exclusive_prefix_sum_kernel(const int* __restrict__ histogram, int* __restrict__ offsets, int E) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < E; i++) {
            offsets[i] = sum;
            sum += histogram[i];
        }
    }
}

// --- Updated Permute Kernel ---
__global__ void permute_kernel(
    const float* __restrict__ input_tokens, const int* __restrict__ topk_indices, const float* __restrict__ topk_weights,
    int* __restrict__ write_pointers, float* __restrict__ permuted_x, int* __restrict__ coo_indices, float* __restrict__ coo_weights,
    int d_model, int k, int total_routed
) {
    // 1 Block = Exactly 1 Token Routing
    int row = blockIdx.x; 
    if (row >= total_routed) return;

    __shared__ int dest_row;
    
    // Only Thread 0 does the atomic addition
    if (threadIdx.x == 0) {
        dest_row = atomicAdd(&write_pointers[topk_indices[row]], 1);
        coo_indices[dest_row] = row / k; 
        coo_weights[dest_row] = topk_weights[row];
    }
    
    // Ensure all threads wait for dest_row to be set
    __syncthreads(); 

    // Vectorized Block-Stride Loop
    int cols_per_thread = d_model / 4;
    int orig_id = row / k;
    
    const float4* vec_in = reinterpret_cast<const float4*>(input_tokens + orig_id * d_model);
    float4* vec_out = reinterpret_cast<float4*>(permuted_x + dest_row * d_model);
    
    // Threads leapfrog over each other until all columns are copied
    for (int col = threadIdx.x; col < cols_per_thread; col += blockDim.x) {
        vec_out[col] = vec_in[col];
    }
}

// --- Updated Unpermute Kernel ---
__global__ void unpermute_kernel(
    const float* __restrict__ expert_out, const int* __restrict__ coo_indices, const float* __restrict__ coo_weights,
    float* __restrict__ final_out, int d_model, int total_routed
) {
    int row = blockIdx.x; 
    if (row >= total_routed) return;

    int orig_seq_idx = coo_indices[row];
    float weight = coo_weights[row];

    // Standard Float Block-Stride Loop
    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        float result = expert_out[row * d_model + col] * weight;
        atomicAdd(&final_out[orig_seq_idx * d_model + col], result);
    }
}

// ==========================================
// PyTorch C++ Bindings
// ==========================================

std::vector<torch::Tensor> route_and_permute(torch::Tensor x, torch::Tensor W_g, int k) {
    int N = x.size(0);
    int d_model = x.size(1);
    int E = W_g.size(1);
    int total = N * k;

    auto opt_int = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto opt_flt = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    auto topk_indices = torch::empty({N, k}, opt_int);
    auto topk_weights = torch::empty({N, k}, opt_flt);
    auto histogram = torch::zeros({E}, opt_int);
    auto offsets = torch::empty({E}, opt_int);
    
    auto permuted_x = torch::empty({total, d_model}, opt_flt);
    auto coo_indices = torch::empty({total}, opt_int);
    auto coo_weights = torch::empty({total}, opt_flt);

    int wpb = 4;
    fused_moe_router_kernel<<<(N + wpb - 1)/wpb, wpb * 32>>>(
        x.data_ptr<float>(), W_g.data_ptr<float>(), topk_indices.data_ptr<int>(), topk_weights.data_ptr<float>(), N, d_model, E, k
    );

    expert_histogram_kernel<<<(total + 255)/256, 256>>>(topk_indices.data_ptr<int>(), histogram.data_ptr<int>(), total);
    exclusive_prefix_sum_kernel<<<1, 1>>>(histogram.data_ptr<int>(), offsets.data_ptr<int>(), E);

    auto write_pointers = offsets.clone();
    
    permute_kernel<<<total, 256>>>(
        x.data_ptr<float>(), topk_indices.data_ptr<int>(), topk_weights.data_ptr<float>(),
        write_pointers.data_ptr<int>(), permuted_x.data_ptr<float>(), coo_indices.data_ptr<int>(), coo_weights.data_ptr<float>(),
        d_model, k, total
    );

    return {permuted_x, coo_indices, coo_weights, offsets};
}

torch::Tensor unpermute(torch::Tensor expert_out, torch::Tensor coo_indices, torch::Tensor coo_weights, int N) {
    int total = expert_out.size(0);
    int d_model = expert_out.size(1);
    auto final_out = torch::zeros({N, d_model}, expert_out.options());

    unpermute_kernel<<<total, 256>>>(expert_out.data_ptr<float>(), coo_indices.data_ptr<int>(), coo_weights.data_ptr<float>(), final_out.data_ptr<float>(), d_model, total);
    
    return final_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("route_and_permute", &route_and_permute, "Forward routing and permute");
    m.def("unpermute", &unpermute, "Unpermute output");
}
