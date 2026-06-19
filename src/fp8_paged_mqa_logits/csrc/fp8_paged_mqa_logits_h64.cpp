#pragma once
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "utils/utils.h"

// ============================================================================
// fp8_paged_mqa_logits_h64 — decode-phase indexer kernel, q_heads = 64.
//
// This is a standalone copy of the v10 kernel from
// csrc/fp8_paged_mqa_logits.cpp specialized for NUM_HEADS = 64 (instead of 32).
//
// Why only NUM_HEADS changes:
//   The MFMA tile is 16x16x32. Each warp lane owns 4 output rows (heads):
//       head_row = 4*(laneId/16) + j,  j in [0,4)
//   so the four lane-groups (laneId/16 in {0,1,2,3}) each cover a disjoint set
//   of H_LOOPS = NUM_HEADS/16 heads. The post-MFMA per-lane accumulation sums
//   over h in [0,H_LOOPS) and j in [0,4), and the two warp shuffles
//       total += __shfl_down(total, 32);   // fold lane-group 2,3 -> 0,1
//       total += __shfl_down(total, 16);   // fold lane-group 1   -> 0
//   fold all four lane-groups together. The result in lanes 0..15 is therefore
//   the full sum over all NUM_HEADS heads for any NUM_HEADS = H_LOOPS*16.
//   For NUM_HEADS=64, H_LOOPS=4 — the reduction code is byte-for-byte identical
//   to the 32-head kernel; only the loop trip count and the SMEM/W-load width
//   grow. The W cooperative load is widened from 8 to NUM_HEADS/4 threads.
//
// Everything else (paged addressing, KV_BLOCK_SIZE template, cross-chunk
// double-buffered KV + block-table prefetch) is unchanged from v10.
// Uses V_MFMA_F32_16X16X32_FP8_FP8.
// ============================================================================

namespace h64 {

using fp8 = __hip_fp8_storage_t;

constexpr int WARP_SIZE = 64;
constexpr int NUM_HEADS = 64;   // <-- the only conceptual change vs v10 (was 32)
constexpr int HEAD_SIZE = 128;

#define H64_CDIV(a, b) (((a) + (b) - 1) / (b))

// ----------------------------------------------------------------------------
// v10-style kernel, NUM_HEADS = 64.
//   KV_BLOCK_SIZE is a compile-time template constant (1 or 64).
//   smem_bt is indexed by LOGICAL block (BLOCKS_PER_CHUNK entries).
// ----------------------------------------------------------------------------
template <int NUM_WARPS, int CHUNK_K, int KV_BLOCK_SIZE>
__global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void fp8_paged_mqa_logits_kernel(
    const fp8*   __restrict__ Q_ptr,           // [batch, next_n, 64, 128]     fp8
    const fp8*   __restrict__ kv_cache_ptr,    // [num_blocks, block_size, 1, index_dim] raw
    const float* __restrict__ weights_ptr,     // [batch*next_n, 64]           fp32
    const int*   __restrict__ context_lens,    // [batch]                      int32
    const int*   __restrict__ block_tables,    // [batch, max_blocks_per_seq]  int32
    float*       __restrict__ logits_ptr,      // [batch*next_n, max_model_len] fp32
    int batch_size, int next_n,
    int max_blocks_per_seq, int max_model_len, int index_dim,
    int SplitKV
){
    constexpr int MFMA_MN = 16;
    constexpr int MFMA_K  = 32;
    constexpr int GPRs_AB = 2;
    constexpr int GPRs_C  = 4;
    constexpr int numInputElementMFMA  = GPRs_AB * sizeof(float) / sizeof(fp8);  // 8
    constexpr int numOutputElementMFMA = GPRs_C;                                 // 4

    using VecInMFMA  = __attribute__((__vector_size__(GPRs_AB * sizeof(float)))) fp8;
    using VecOutMFMA = __attribute__((__vector_size__(GPRs_C  * sizeof(float)))) float;

    constexpr int BLOCK_THREADS    = NUM_WARPS * WARP_SIZE;
    constexpr int H_LOOPS          = NUM_HEADS / MFMA_MN;   // 4 (was 2 for 32 heads)
    constexpr int D_LOOPS          = HEAD_SIZE / MFMA_K;    // 4
    constexpr int TILES_PER_CHUNK  = CHUNK_K / MFMA_MN;
    constexpr int PAD              = 8;
    constexpr int KV_ROW           = HEAD_SIZE + PAD;
    constexpr int VEC_LEN          = sizeof(float4) / sizeof(fp8);   // 16
    constexpr int NB_LOAD_KV       = CHUNK_K * HEAD_SIZE / (BLOCK_THREADS * VEC_LEN);
    constexpr int NB_LOAD_SCALE    = H64_CDIV(CHUNK_K, BLOCK_THREADS);
    // Distinct physical blocks touched by one chunk (==CHUNK_K when KB==1).
    constexpr int BLOCKS_PER_CHUNK = CHUNK_K / KV_BLOCK_SIZE;
    constexpr int NB_LOAD_BT       = H64_CDIV(BLOCKS_PER_CHUNK, BLOCK_THREADS);
    // Threads cooperating on the W[NUM_HEADS] load (4 floats each).
    constexpr int W_LOAD_THREADS   = NUM_HEADS / 4;          // 16 (was 8 for 32 heads)

    static_assert(NUM_HEADS % MFMA_MN == 0, "NUM_HEADS must be a multiple of 16");
    static_assert(NUM_HEADS % 4 == 0, "NUM_HEADS must be a multiple of 4 for the W load");
    static_assert(CHUNK_K * HEAD_SIZE % (BLOCK_THREADS * VEC_LEN) == 0, "CHUNK_K * HEAD_SIZE must be divisible by BLOCK_THREADS * VEC_LEN");
    static_assert(NUM_HEADS * HEAD_SIZE % (BLOCK_THREADS * VEC_LEN) == 0, "NUM_HEADS * HEAD_SIZE must be divisible by BLOCK_THREADS * VEC_LEN");
    static_assert(CHUNK_K % MFMA_MN == 0, "CHUNK_K must be multiple of MFMA_MN=16");
    static_assert(CHUNK_K % KV_BLOCK_SIZE == 0, "CHUNK_K must be a multiple of KV_BLOCK_SIZE");

    // Per-physical-block byte stride and value-region size (scales follow values).
    const int64_t block_stride = (int64_t)KV_BLOCK_SIZE * index_dim;
    constexpr int  scale_region = KV_BLOCK_SIZE * HEAD_SIZE;

    const int pid_batch    = blockIdx.x;
    const int pid_next_n   = blockIdx.y;
    const int pid_split_kv = blockIdx.z;
    if (pid_batch >= batch_size) return;

    const int ctx_len      = context_lens[pid_batch];
    const int ctx_chunks   = H64_CDIV(ctx_len, CHUNK_K);
    const int split_chunks = H64_CDIV(ctx_chunks, SplitKV);
    const int split_start  = pid_split_kv * split_chunks * CHUNK_K;
    const int split_end    = min(ctx_len, split_start + split_chunks * CHUNK_K);
    if (split_start >= ctx_len) return;

    const int tid        = threadIdx.x;
    const int warpId     = tid / WARP_SIZE;
    const int laneId     = tid % WARP_SIZE;
    const int mfmaInRow  = laneId % MFMA_MN;
    const int mfmaInCol  = numInputElementMFMA * (laneId / MFMA_MN);
    const int mfmaOutRow = numOutputElementMFMA * (laneId / MFMA_MN);
    const int mfmaOutCol = laneId % MFMA_MN;

    __shared__ fp8   smem_Q    [NUM_HEADS * HEAD_SIZE];   // 64*128 = 8 KB
    __shared__ float smem_W    [NUM_HEADS];               // 256 B
    __shared__ fp8   smem_KV   [CHUNK_K][KV_ROW];
    __shared__ float smem_scale[CHUNK_K];
    __shared__ int   smem_bt   [BLOCKS_PER_CHUNK];        // resolved phys id, per LOGICAL block

    VecInMFMA q_reg[D_LOOPS][H_LOOPS];
    float     w_reg[H_LOOPS][numOutputElementMFMA];

    float4 pf_kv[NB_LOAD_KV];
    float  pf_scale[NB_LOAD_SCALE];
    int    pf_bt[NB_LOAD_BT];

    const int  q_row        = pid_batch * next_n + pid_next_n;
    const int* bt           = block_tables + (int64_t)pid_batch * max_blocks_per_seq;
    float*     out_base     = logits_ptr + (int64_t)q_row * max_model_len;
    const int  causal_limit = ctx_len - next_n + pid_next_n;

    auto load_qw_global = [&]() {
        const int q_base = q_row * NUM_HEADS * HEAD_SIZE;
        for (int i = tid * VEC_LEN; i < NUM_HEADS * HEAD_SIZE; i += BLOCK_THREADS * VEC_LEN) {
            *reinterpret_cast<float4*>(&smem_Q[i]) =
                *reinterpret_cast<const float4*>(&Q_ptr[q_base + i]);
        }
        const int w_base = q_row * NUM_HEADS;
        if (tid < W_LOAD_THREADS) {   // NUM_HEADS / 4 float4 loads
            *reinterpret_cast<float4*>(&smem_W[tid * 4]) =
                *reinterpret_cast<const float4*>(&weights_ptr[w_base + tid * 4]);
        }
    };

    auto load_qw_to_regs = [&]() {
        #pragma unroll
        for (int d = 0; d < D_LOOPS; ++d) {
            #pragma unroll
            for (int h = 0; h < H_LOOPS; ++h) {
                q_reg[d][h] = *reinterpret_cast<const VecInMFMA*>(
                    &smem_Q[(h * MFMA_MN + mfmaInRow) * HEAD_SIZE + d * MFMA_K + mfmaInCol]);
            }
        }
        #pragma unroll
        for (int h = 0; h < H_LOOPS; ++h) {
            #pragma unroll
            for (int j = 0; j < numOutputElementMFMA; ++j) {
                w_reg[h][j] = smem_W[h * MFMA_MN + mfmaOutRow + j];
            }
        }
    };

    // value byte offset for (phys, within, d) = phys*block_stride + within*HEAD_SIZE + d
    // scale (float) byte offset            = phys*block_stride + scale_region + within*4
    // Initial tile: phys resolved directly via bt (smem_bt not yet populated).
    auto load_kv_to_smem = [&](int kv_start, int kv_valid) {
        for (int i = tid * VEC_LEN; i < CHUNK_K * HEAD_SIZE; i += BLOCK_THREADS * VEC_LEN) {
            int k = i / HEAD_SIZE;
            int d = i % HEAD_SIZE;
            if (k < kv_valid) {
                int phys   = bt[(kv_start + k) / KV_BLOCK_SIZE];
                int within = (kv_start + k) % KV_BLOCK_SIZE;
                *reinterpret_cast<float4*>(&smem_KV[k][d]) =
                    *reinterpret_cast<const float4*>(
                        kv_cache_ptr + (int64_t)phys * block_stride + within * HEAD_SIZE + d);
            } else {
                *reinterpret_cast<float4*>(&smem_KV[k][d]) = make_float4(0, 0, 0, 0);
            }
        }
        for (int i = tid; i < CHUNK_K; i += BLOCK_THREADS) {
            if (i < kv_valid) {
                int phys   = bt[(kv_start + i) / KV_BLOCK_SIZE];
                int within = (kv_start + i) % KV_BLOCK_SIZE;
                smem_scale[i] = *reinterpret_cast<const float*>(
                    kv_cache_ptr + (int64_t)phys * block_stride + scale_region + within * 4);
            } else {
                smem_scale[i] = 0.0f;
            }
        }
    };

    // smem_bt holds the resolved phys id for each LOGICAL block of the chunk.
    auto load_bt_to_smem = [&](int kv_start, int kv_valid) {
        const int base_lb  = kv_start / KV_BLOCK_SIZE;
        const int valid_lb = H64_CDIV(kv_valid, KV_BLOCK_SIZE);
        for (int b = tid; b < BLOCKS_PER_CHUNK; b += BLOCK_THREADS) {
            smem_bt[b] = (b < valid_lb) ? bt[base_lb + b] : 0;
        }
    };

    // phys read from smem_bt (no chained HBM dep); within recomputed from kv_start.
    auto prefetch_kv = [&](int kv_start, int kv_valid) {
        #pragma unroll
        for (int i = 0; i < NB_LOAD_KV; ++i) {
            int idx = tid * VEC_LEN + i * BLOCK_THREADS * VEC_LEN;
            int k = idx / HEAD_SIZE;
            int d = idx % HEAD_SIZE;
            if (k < kv_valid) {
                int phys   = smem_bt[k / KV_BLOCK_SIZE];
                int within = (kv_start + k) % KV_BLOCK_SIZE;
                pf_kv[i] = *reinterpret_cast<const float4*>(
                    kv_cache_ptr + (int64_t)phys * block_stride + within * HEAD_SIZE + d);
            } else {
                pf_kv[i] = make_float4(0, 0, 0, 0);
            }
        }
        #pragma unroll
        for (int i = 0; i < NB_LOAD_SCALE; ++i) {
            int idx = tid + i * BLOCK_THREADS;
            if (idx < kv_valid) {
                int phys   = smem_bt[idx / KV_BLOCK_SIZE];
                int within = (kv_start + idx) % KV_BLOCK_SIZE;
                pf_scale[i] = *reinterpret_cast<const float*>(
                    kv_cache_ptr + (int64_t)phys * block_stride + scale_region + within * 4);
            } else {
                pf_scale[i] = 0.0f;
            }
        }
    };

    auto prefetch_bt = [&](int kv_start, int kv_valid) {
        const int base_lb  = kv_start / KV_BLOCK_SIZE;
        const int valid_lb = H64_CDIV(kv_valid, KV_BLOCK_SIZE);
        #pragma unroll
        for (int i = 0; i < NB_LOAD_BT; ++i) {
            int b = tid + i * BLOCK_THREADS;
            pf_bt[i] = (b < valid_lb) ? bt[base_lb + b] : 0;
        }
    };

    auto flush_kv_prefetch = [&]() {
        #pragma unroll
        for (int i = 0; i < NB_LOAD_KV; ++i) {
            int idx = tid * VEC_LEN + i * BLOCK_THREADS * VEC_LEN;
            int k = idx / HEAD_SIZE;
            int d = idx % HEAD_SIZE;
            *reinterpret_cast<float4*>(&smem_KV[k][d]) = pf_kv[i];
        }
        #pragma unroll
        for (int i = 0; i < NB_LOAD_SCALE; ++i) {
            int idx = tid + i * BLOCK_THREADS;
            if (idx < CHUNK_K) smem_scale[idx] = pf_scale[i];
        }
    };

    auto flush_bt_prefetch = [&]() {
        #pragma unroll
        for (int i = 0; i < NB_LOAD_BT; ++i) {
            int b = tid + i * BLOCK_THREADS;
            if (b < BLOCKS_PER_CHUNK) smem_bt[b] = pf_bt[i];
        }
    };

    auto compute_and_store = [&](int kv_start, int kv_valid) {
        for (int bk = warpId; bk < TILES_PER_CHUNK; bk += NUM_WARPS) {
            const int k = bk * MFMA_MN;
            const float kv_scale = (k + mfmaOutCol < kv_valid) ? smem_scale[k + mfmaOutCol] : 0.0f;

            VecOutMFMA vC[H_LOOPS] = {};

            #pragma unroll
            for (int d = 0; d < D_LOOPS; ++d) {
                VecInMFMA vB = *reinterpret_cast<const VecInMFMA*>(
                    &smem_KV[k + mfmaInRow][d * MFMA_K + mfmaInCol]);
                #pragma unroll
                for (int h = 0; h < H_LOOPS; ++h) {
                    vC[h] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                        (long)q_reg[d][h], (long)vB, vC[h], 0, 0, 0);
                }
            }

            float total_score = 0.0f;
            #pragma unroll
            for (int h = 0; h < H_LOOPS; ++h) {
                #pragma unroll
                for (int j = 0; j < numOutputElementMFMA; ++j) {
                    total_score += fmaxf(vC[h][j] * kv_scale, 0.0f) * w_reg[h][j];
                }
            }
            // Fold the four lane-groups (each owns H_LOOPS=4 disjoint heads) -> lanes 0..15.
            total_score += __shfl_down(total_score, 32);
            total_score += __shfl_down(total_score, 16);

            if (laneId < 16) {
                const int abs_pos = kv_start + k + laneId;
                if (abs_pos < ctx_len && abs_pos <= causal_limit && abs_pos < max_model_len) {
                    out_base[abs_pos] = total_score;
                }
            }
        }
    };

    load_qw_global();
    __syncthreads();
    load_qw_to_regs();

    const int first_valid = min(CHUNK_K, split_end - split_start);
    load_kv_to_smem(split_start, first_valid);
    if (split_start + CHUNK_K < split_end) {
        const int next_valid = min(CHUNK_K, split_end - split_start - CHUNK_K);
        load_bt_to_smem(split_start + CHUNK_K, next_valid);
    }
    __syncthreads();

    for (int kv_start = split_start; kv_start < split_end; kv_start += CHUNK_K) {
        const int  kv_valid      = min(CHUNK_K, split_end - kv_start);
        const bool has_next      = (kv_start +     CHUNK_K < split_end);
        const bool has_next_next = (kv_start + 2 * CHUNK_K < split_end);

        if (has_next) {
            const int next_valid = min(CHUNK_K, split_end - kv_start - CHUNK_K);
            prefetch_kv(kv_start + CHUNK_K, next_valid);
        }
        if (has_next_next) {
            const int next_next_valid = min(CHUNK_K, split_end - kv_start - 2 * CHUNK_K);
            prefetch_bt(kv_start + 2 * CHUNK_K, next_next_valid);
        }

        compute_and_store(kv_start, kv_valid);

        __syncthreads();

        if (has_next)      flush_kv_prefetch();
        if (has_next_next) flush_bt_prefetch();

        __syncthreads();
    }
}

// ============================================================================
// Host dispatch (q_heads = 64)
// ============================================================================
void launch_fp8_paged_mqa_logits_h64(
    const fp8* d_q, const fp8* d_kv, const float* d_w,
    const int* d_ctx, const int* d_bt, float* d_out,
    int batch_size, int next_n,
    int max_blocks_per_seq, int max_model_len, int index_dim,
    int ChunkK, int SplitKV, int num_warps, cudaStream_t stream,
    int block_size = 1
) {
    const dim3 grid = dim3(batch_size, next_n, SplitKV);

    #define H64_LAUNCH(NW, CK, KB)                                                          \
        fp8_paged_mqa_logits_kernel<NW, CK, KB><<<grid, NW * WARP_SIZE, 0, stream>>>(       \
            d_q, d_kv, d_w, d_ctx, d_bt, d_out,                                             \
            batch_size, next_n, max_blocks_per_seq, max_model_len, index_dim, SplitKV);

    #define H64_DISPATCH_BS(NW, CK)                                                         \
        switch (block_size) {                                                              \
            case 1:  { H64_LAUNCH(NW, CK, 1)  break; }                                     \
            case 64: { H64_LAUNCH(NW, CK, 64) break; }                                     \
            default:                                                                       \
                throw std::runtime_error("h64 unsupported block_size="                     \
                    + std::to_string(block_size) + ". Supported: 1, 64");                  \
        }

    #define H64_DISPATCH_CK(NW)                                                            \
        switch (ChunkK) {                                                                  \
            case 64:  { H64_DISPATCH_BS(NW, 64)  break; }                                  \
            case 128: { H64_DISPATCH_BS(NW, 128) break; }                                  \
            case 256: { H64_DISPATCH_BS(NW, 256) break; }                                  \
            default:                                                                       \
                throw std::runtime_error("Unsupported ChunkK=" + std::to_string(ChunkK)    \
                    + ". Supported: 64, 128, 256");                                         \
        }

    switch (num_warps) {
        case 2: { H64_DISPATCH_CK(2) break; }
        case 4: { H64_DISPATCH_CK(4) break; }
        case 8: { H64_DISPATCH_CK(8) break; }
        default:
            throw std::runtime_error("Unsupported num_warps=" + std::to_string(num_warps)
                + ". Supported: 2, 4, 8");
    }

    #undef H64_DISPATCH_CK
    #undef H64_DISPATCH_BS
    #undef H64_LAUNCH
}

}  // namespace h64

// ============================================================================
// Python-facing entry point (q_heads = 64)
// ============================================================================
torch::Tensor fp8_paged_mqa_logits_h64(
    torch::Tensor q_fp8,            // [batch, next_n, 64, 128]
    torch::Tensor kv_cache_fp8,     // [num_blocks, block_size, 1, index_dim]
    torch::Tensor weights,          // [batch*next_n, 64]
    torch::Tensor context_lens,     // [batch]
    torch::Tensor block_tables,     // [batch, max_blocks_per_seq]
    int max_model_len,
    int ChunkK    = 256,
    int SplitKV   = -1,             // -1 = auto
    int num_warps = 4,
    int TotalCuCount = 304,
    int version   = 10,             // accepted for interface parity; only v10 (h64) exists
    std::optional<torch::Tensor> out_logits_opt = std::nullopt
) {
    // `fp8` (== __hip_fp8_storage_t) comes from utils.h at global scope.
    const int batch_size         = q_fp8.size(0);
    const int next_n             = q_fp8.size(1);
    const int n_heads            = q_fp8.size(2);
    const int head_dim           = q_fp8.size(3);
    const int block_size         = kv_cache_fp8.size(1);
    const int index_dim          = kv_cache_fp8.size(3);
    const int max_blocks_per_seq = block_tables.size(1);

    TORCH_CHECK(n_heads == h64::NUM_HEADS && head_dim == h64::HEAD_SIZE,
                "fp8_paged_mqa_logits_h64 requires n_heads=64, head_dim=128, got n_heads=",
                n_heads, " head_dim=", head_dim);
    TORCH_CHECK(block_size == 1 || block_size == 64,
                "Only kv_cache block_size in {1, 64} supported, got ", block_size);
    TORCH_CHECK(version == 10,
                "fp8_paged_mqa_logits_h64 only implements the v10 kernel (got version=",
                version, ")");

    // ---- auto SplitKV (same heuristic as the 32-head kernel) ----
    if (SplitKV <= 0) {
        constexpr int WavePerEU = 2;
        const int tiles = batch_size * next_n;
        SplitKV = std::max(1, ((std::max(1, TotalCuCount / tiles) + 4) / 5) * 5 * WavePerEU);
    }

    torch::Tensor out_logits = out_logits_opt.has_value()
        ? out_logits_opt.value()
        : torch::full(
              {batch_size * next_n, max_model_len},
              -std::numeric_limits<float>::infinity(),
              torch::dtype(torch::kFloat32).device(q_fp8.device()));

    const at::cuda::OptionalCUDAGuard guard(device_of(out_logits));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    h64::launch_fp8_paged_mqa_logits_h64(
        static_cast<fp8*>(q_fp8.data_ptr()),
        static_cast<fp8*>(kv_cache_fp8.data_ptr()),
        static_cast<float*>(weights.data_ptr()),
        static_cast<int*>(context_lens.data_ptr()),
        static_cast<int*>(block_tables.data_ptr()),
        out_logits.data_ptr<float>(),
        batch_size, next_n, max_blocks_per_seq, max_model_len, index_dim,
        ChunkK, SplitKV, num_warps, stream, block_size);

    return out_logits;
}

// ============================================================================
// pybind module (self-contained — built as moreh_fp8_paged_mqa_logits_h64)
// ============================================================================
PYBIND11_MODULE(moreh_fp8_paged_mqa_logits_h64, m) {
    m.def("fp8_paged_mqa_logits_h64", &fp8_paged_mqa_logits_h64,
          "FP8 Paged MQA Logits HIP kernel for q_heads=64 (v10-based).",
          pybind11::arg("q_fp8"),
          pybind11::arg("kv_cache_fp8"),
          pybind11::arg("weights"),
          pybind11::arg("context_lens"),
          pybind11::arg("block_tables"),
          pybind11::arg("max_model_len"),
          pybind11::arg("ChunkK")       = 256,
          pybind11::arg("SplitKV")      = -1,
          pybind11::arg("num_warps")    = 4,
          pybind11::arg("TotalCuCount") = 304,
          pybind11::arg("version")      = 10,
          pybind11::arg("out_logits")   = std::optional<torch::Tensor>{});
}
