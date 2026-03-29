#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dot_kernel.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include <omp.h>
#include <cstdint>
#include <cstddef>

HWY_BEFORE_NAMESPACE();

namespace forway {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Vector-Vector dot product: a · b → scalar (Multi-threaded)
float DotVVImpl(const float* __restrict__ a, const float* __restrict__ b, std::size_t n, int max_threads) {
    float global_sum = 0.0f;

    #pragma omp parallel num_threads(max_threads) reduction(+:global_sum)
    {
        float local_sum = 0.0f;

        int tid = omp_get_thread_num();
        int active_threads = omp_get_num_threads();
        std::size_t chunk = n / active_threads;
        std::size_t start = tid * chunk;
        std::size_t end = (tid == active_threads - 1) ? n : start + chunk;

        const hn::ScalableTag<float> d;
        const std::size_t N = hn::Lanes(d);

        auto v0 = hn::Zero(d), v1 = hn::Zero(d), v2 = hn::Zero(d), v3 = hn::Zero(d);

        std::size_t j = start;
        for (; j + 4 * N <= end; j += 4 * N) {
            v0 = hn::MulAdd(hn::LoadU(d, a + j + 0 * N), hn::LoadU(d, b + j + 0 * N), v0);
            v1 = hn::MulAdd(hn::LoadU(d, a + j + 1 * N), hn::LoadU(d, b + j + 1 * N), v1);
            v2 = hn::MulAdd(hn::LoadU(d, a + j + 2 * N), hn::LoadU(d, b + j + 2 * N), v2);
            v3 = hn::MulAdd(hn::LoadU(d, a + j + 3 * N), hn::LoadU(d, b + j + 3 * N), v3);
        }

        auto v_sum = hn::Add(hn::Add(v0, v1), hn::Add(v2, v3));
        for (; j + N <= end; j += N) {
            v_sum = hn::MulAdd(hn::LoadU(d, a + j), hn::LoadU(d, b + j), v_sum);
        }

        local_sum += hn::ReduceSum(d, v_sum);
        for (; j < end; ++j) {
            local_sum += a[j] * b[j];
        }

        global_sum += local_sum;
    }
    return global_sum;
}

// Matrix-Vector dot product: A[M×K] · x[K] → y[M]
// Each row is an independent dot product — perfect for OpenMP
void DotMVImpl(const float* __restrict__ A, const float* __restrict__ x,
               float* __restrict__ y, std::size_t M, std::size_t K, int max_threads) {
    const hn::ScalableTag<float> d;
    const std::size_t N = hn::Lanes(d);

    #pragma omp parallel for num_threads(max_threads)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(M); ++i) {
        const float* __restrict__ row = A + i * K;

        auto v0 = hn::Zero(d), v1 = hn::Zero(d), v2 = hn::Zero(d), v3 = hn::Zero(d);

        std::size_t j = 0;
        for (; j + 4 * N <= K; j += 4 * N) {
            v0 = hn::MulAdd(hn::LoadU(d, row + j + 0 * N), hn::LoadU(d, x + j + 0 * N), v0);
            v1 = hn::MulAdd(hn::LoadU(d, row + j + 1 * N), hn::LoadU(d, x + j + 1 * N), v1);
            v2 = hn::MulAdd(hn::LoadU(d, row + j + 2 * N), hn::LoadU(d, x + j + 2 * N), v2);
            v3 = hn::MulAdd(hn::LoadU(d, row + j + 3 * N), hn::LoadU(d, x + j + 3 * N), v3);
        }

        auto v_sum = hn::Add(hn::Add(v0, v1), hn::Add(v2, v3));
        for (; j + N <= K; j += N) {
            v_sum = hn::MulAdd(hn::LoadU(d, row + j), hn::LoadU(d, x + j), v_sum);
        }

        float row_sum = hn::ReduceSum(d, v_sum);
        for (; j < K; ++j) {
            row_sum += row[j] * x[j];
        }
        y[i] = row_sum;
    }
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace forway {
HWY_EXPORT(DotVVImpl);
HWY_EXPORT(DotMVImpl);

extern "C" {
    float forway_dot_vv(const float* a, const float* b, std::size_t n, int max_threads) noexcept {
        return HWY_STATIC_DISPATCH(DotVVImpl)(a, b, n, max_threads);
    }
    void forway_dot_mv(const float* A, const float* x, float* y,
                       std::size_t M, std::size_t K, int max_threads) noexcept {
        HWY_STATIC_DISPATCH(DotMVImpl)(A, x, y, M, K, max_threads);
    }
}
}

#endif
