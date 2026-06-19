// ============================================================================
//  OPTIMIZATION TARGET  --  this is the ONLY file the loop is allowed to edit.
// ============================================================================
//
//  Problem:  C[M,N] = A[M,K] * B[K,N]   (row-major, float32)
//
//  Contract you MUST keep:
//    * Keep the exact signature of `solve` below (extern "C", device pointers).
//    * dA, dB, dC are DEVICE pointers already allocated/filled by the harness.
//    * Do NOT allocate/free dA/dB/dC, do NOT copy to host, do NOT add timing.
//    * You MAY add helper kernels / __device__ code / shared memory / tiling
//      / launch-config changes ABOVE solve, as long as solve produces C=A*B.
// ============================================================================

#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// MFMA (matrix-core) SGEMM for CDNA3 (gfx942), full FP32.
//   Instruction: v_mfma_f32_16x16x4_f32  (M=N=16, K=4).
//   Block tile  : BM x BN = 64 x 64, staged in LDS in steps of BK = 16.
//   Threads     : 256 = 4 wavefronts. Waves in a 2x2 grid, each wave owns a
//                 32 x 32 output region = a 2 x 2 grid of 16x16 MFMA tiles.
//   Smaller tiles -> 4096 wavefronts -> fills the 304-CU GPU (occupancy fix).
// Assumes M,N,K multiples of 64 / 16 (harness runs 2048^3).
// ---------------------------------------------------------------------------
#define BM 64
#define BN 64
#define BK 16
#define NTHREADS 256
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 4
#define WTILES 2          // 2x2 MFMA tiles per wave -> 32x32

typedef float f4 __attribute__((ext_vector_type(4)));

__global__ void __launch_bounds__(NTHREADS)
matmul_mfma(const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            int M, int N, int K) {
    // Padded LDS to break shared-memory bank conflicts on the MFMA fragment
    // reads: As gets an odd row stride (17), Bs a stride of 80 (=16 mod 32),
    // both reaching the 2-way conflict floor for a 64-lane fragment read.
    __shared__ float As[BK][BM + 16];  // As[k][m], k-major, stride 80
    __shared__ float Bs[BK][BN + 16];  // Bs[k][n], stride 80

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid  = threadIdx.x;
    const int wave = tid / 64;
    const int lane = tid % 64;
    const int waveRow = wave / 2;          // 0..1
    const int waveCol = wave % 2;          // 0..1

    const int laneRow = lane % 16;
    const int laneK   = lane / 16;         // 0..3

    f4 acc[WTILES][WTILES];
    #pragma unroll
    for (int i = 0; i < WTILES; ++i)
        #pragma unroll
        for (int j = 0; j < WTILES; ++j)
            acc[i][j] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kk = 0; kk < K; kk += BK) {
        #pragma unroll
        for (int i = tid; i < (BM * BK) / 4; i += NTHREADS) {
            int r  = i / (BK / 4);
            int c4 = (i % (BK / 4)) * 4;
            float4 a4 = *reinterpret_cast<const float4*>(
                &A[(size_t)(blockRow + r) * K + (kk + c4)]);
            As[c4 + 0][r] = a4.x; As[c4 + 1][r] = a4.y;
            As[c4 + 2][r] = a4.z; As[c4 + 3][r] = a4.w;
        }
        #pragma unroll
        for (int i = tid; i < (BK * BN) / 4; i += NTHREADS) {
            int r  = i / (BN / 4);
            int c4 = (i % (BN / 4)) * 4;
            float4 b4 = *reinterpret_cast<const float4*>(
                &B[(size_t)(kk + r) * N + (blockCol + c4)]);
            *reinterpret_cast<float4*>(&Bs[r][c4]) = b4;
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < BK; ks += WMMA_K) {
            float aFrag[WTILES];
            float bFrag[WTILES];
            #pragma unroll
            for (int mi = 0; mi < WTILES; ++mi)
                aFrag[mi] = As[ks + laneK][waveRow * 32 + mi * WMMA_M + laneRow];
            #pragma unroll
            for (int mj = 0; mj < WTILES; ++mj)
                bFrag[mj] = Bs[ks + laneK][waveCol * 32 + mj * WMMA_N + laneRow];
            #pragma unroll
            for (int mi = 0; mi < WTILES; ++mi)
                #pragma unroll
                for (int mj = 0; mj < WTILES; ++mj)
                    acc[mi][mj] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                        aFrag[mi], bFrag[mj], acc[mi][mj], 0, 0, 0);
        }
        __syncthreads();
    }

    const int outRowBase = (lane / 16) * 4;
    const int outCol     = lane % 16;
    #pragma unroll
    for (int mi = 0; mi < WTILES; ++mi) {
        #pragma unroll
        for (int mj = 0; mj < WTILES; ++mj) {
            int colG = blockCol + waveCol * 32 + mj * WMMA_N + outCol;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int rowG = blockRow + waveRow * 32 + mi * WMMA_M + outRowBase + r;
                C[(size_t)rowG * N + colG] = acc[mi][mj][r];
            }
        }
    }
}

extern "C" void solve(const float* dA, const float* dB, float* dC,
                      int M, int N, int K) {
    dim3 block(NTHREADS);
    dim3 grid(N / BN, M / BM);
    matmul_mfma<<<grid, block>>>(dA, dB, dC, M, N, K);
}
