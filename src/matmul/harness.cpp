// ============================================================================
//  MEASUREMENT HARNESS  --  off-limits to the optimization loop. DO NOT EDIT.
// ============================================================================
//
//  Owns the ground truth (naive CPU matmul, ikj loop order) and the hipBLAS
//  performance baseline + all GPU timing. The kernel under test only provides
//  `solve` (declared extern below); everything that decides correctness or
//  speed lives here so the loop cannot game the metric.
//
//  Usage (one JSON object per invocation, printed to stdout):
//    ./matmul_harness check M N K
//    ./matmul_harness bench M N K ITERS
// ============================================================================

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Provided by matmul_kernel.cpp (the file under optimization).
extern "C" void solve(const float* dA, const float* dB, float* dC,
                      int M, int N, int K);

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
    std::exit(2); } } while (0)

#define BLAS_CHECK(x) do { hipblasStatus_t _s = (x); if (_s != HIPBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "hipBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, (int)_s); \
    std::exit(2); } } while (0)

static void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// Ground-truth CPU matmul, ikj order (cache-friendly), row-major.
static void cpu_reference(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<float>& C, int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < M; ++i) {
        const float* arow = &A[(size_t)i * K];
        float* crow = &C[(size_t)i * N];
        for (int k = 0; k < K; ++k) {
            float a = arow[k];
            const float* brow = &B[(size_t)k * N];
            for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
        }
    }
}

static int run_check(int M, int N, int K) {
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N), hRef((size_t)M * N);
    fill_random(hA, 1234);
    fill_random(hB, 5678);
    cpu_reference(hA, hB, hRef, M, N, K);

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, hC.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, hC.size() * sizeof(float)));

    solve(dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost));

    // Relative error normalized by the reference magnitude scale.
    double max_abs_err = 0.0, ref_scale = 1e-8;
    for (size_t i = 0; i < hRef.size(); ++i) {
        ref_scale = std::max(ref_scale, (double)std::fabs(hRef[i]));
        max_abs_err = std::max(max_abs_err, (double)std::fabs(hC[i] - hRef[i]));
    }
    double max_rel_err = max_abs_err / ref_scale;

    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    printf("{\"mode\":\"check\",\"M\":%d,\"N\":%d,\"K\":%d,\"max_rel_err\":%.6e}\n",
           M, N, K, max_rel_err);
    return 0;
}

static float time_ms(void (*body)(void*), void* ctx, int iters) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < iters; ++i) body(ctx);
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms / iters;
}

struct KernelCtx { const float *dA, *dB; float *dC; int M, N, K; };
static void kernel_body(void* p) {
    KernelCtx* c = (KernelCtx*)p;
    solve(c->dA, c->dB, c->dC, c->M, c->N, c->K);
}

struct BlasCtx { hipblasHandle_t h; const float *dA, *dB; float *dC; int M, N, K; float alpha, beta; };
static void blas_body(void* p) {
    BlasCtx* c = (BlasCtx*)p;
    // Row-major C[M,N]=A*B via column-major sgemm: compute C^T = B^T*A^T.
    hipblasSgemm(c->h, HIPBLAS_OP_N, HIPBLAS_OP_N,
                 c->N, c->M, c->K,
                 &c->alpha, c->dB, c->N, c->dA, c->K,
                 &c->beta, c->dC, c->N);
}

static int run_bench(int M, int N, int K, int iters) {
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    fill_random(hA, 1234);
    fill_random(hB, 5678);

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice));

    const int warmup = 5;

    KernelCtx kctx{dA, dB, dC, M, N, K};
    for (int i = 0; i < warmup; ++i) kernel_body(&kctx);
    HIP_CHECK(hipGetLastError());
    float kernel_ms = time_ms(kernel_body, &kctx, iters);

    hipblasHandle_t handle;
    BLAS_CHECK(hipblasCreate(&handle));
    BlasCtx bctx{handle, dA, dB, dC, M, N, K, 1.0f, 0.0f};
    for (int i = 0; i < warmup; ++i) blas_body(&bctx);
    HIP_CHECK(hipDeviceSynchronize());
    float blas_ms = time_ms(blas_body, &bctx, iters);
    BLAS_CHECK(hipblasDestroy(handle));

    double flops = 2.0 * M * N * K;
    double kernel_tflops = flops / (kernel_ms * 1e-3) / 1e12;
    double blas_tflops   = flops / (blas_ms   * 1e-3) / 1e12;

    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    printf("{\"mode\":\"bench\",\"M\":%d,\"N\":%d,\"K\":%d,"
           "\"kernel_ms\":%.6f,\"blas_ms\":%.6f,"
           "\"kernel_tflops\":%.4f,\"blas_tflops\":%.4f}\n",
           M, N, K, kernel_ms, blas_ms, kernel_tflops, blas_tflops);
    return 0;
}

// Profile mode: run ONLY the kernel under test, repeatedly, so an external
// profiler (rocprofv2) attributes all counters to it. No hipBLAS, no copy-back,
// no correctness — keep the profiled region a clean stream of `solve` launches.
static int run_profile(int M, int N, int K, int iters) {
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    fill_random(hA, 1234);
    fill_random(hB, 5678);

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice));

    for (int i = 0; i < iters; ++i) solve(dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    printf("{\"mode\":\"profile\",\"M\":%d,\"N\":%d,\"K\":%d,\"iters\":%d}\n", M, N, K, iters);
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 5 && std::strcmp(argv[1], "check") == 0) {
        return run_check(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
    }
    if (argc >= 6 && std::strcmp(argv[1], "bench") == 0) {
        return run_bench(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));
    }
    if (argc >= 6 && std::strcmp(argv[1], "profile") == 0) {
        return run_profile(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));
    }
    fprintf(stderr, "usage: %s check M N K | bench M N K ITERS | profile M N K ITERS\n", argv[0]);
    return 1;
}
