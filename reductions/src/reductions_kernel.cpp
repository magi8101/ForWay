#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "reductions_kernel.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include <omp.h>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>

HWY_BEFORE_NAMESPACE();

namespace forway {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

float SumImpl(const float* __restrict__ input, std::size_t num_elements, int max_threads) {
    float global_sum = 0.0f;

    #pragma omp parallel num_threads(max_threads) reduction(+:global_sum)
    {
        float local_sum = 0.0f; // Thread-local accumulator decoupling the OpenMP silent Race Conditions safely!
        
        int tid = omp_get_thread_num();
        int active_threads = omp_get_num_threads();
        std::size_t chunk = num_elements / active_threads;
        std::size_t start = tid * chunk;
        std::size_t end = (tid == active_threads - 1) ? num_elements : start + chunk;

        const hn::ScalableTag<float> d;
        const std::size_t N = hn::Lanes(d);
        
        auto v_sum0 = hn::Zero(d), v_sum1 = hn::Zero(d), v_sum2 = hn::Zero(d), v_sum3 = hn::Zero(d);
        
        std::size_t j = start;
        for (; j + 4 * N <= end; j += 4 * N) {
            v_sum0 = hn::Add(v_sum0, hn::LoadU(d, input + j + 0 * N));
            v_sum1 = hn::Add(v_sum1, hn::LoadU(d, input + j + 1 * N));
            v_sum2 = hn::Add(v_sum2, hn::LoadU(d, input + j + 2 * N));
            v_sum3 = hn::Add(v_sum3, hn::LoadU(d, input + j + 3 * N));
        }
        
        auto v_sum = hn::Add(hn::Add(v_sum0, v_sum1), hn::Add(v_sum2, v_sum3));
        for (; j + N <= end; j += N) {
            v_sum = hn::Add(v_sum, hn::LoadU(d, input + j));
        }
        
        local_sum += hn::ReduceSum(d, v_sum);
        
        for (; j < end; ++j) {
            local_sum += input[j];
        }
        
        global_sum += local_sum; // Safely bounded by OpenMP structural boundaries implicitly cleanly securely!
    }
    return global_sum;
}

float MaxImpl(const float* __restrict__ input, std::size_t num_elements, int max_threads) {
    float global_max = -std::numeric_limits<float>::infinity();

    #pragma omp parallel num_threads(max_threads) reduction(max:global_max)
    {
        float local_max = -std::numeric_limits<float>::infinity();
        
        int tid = omp_get_thread_num();
        int active_threads = omp_get_num_threads();
        std::size_t chunk = num_elements / active_threads;
        std::size_t start = tid * chunk;
        std::size_t end = (tid == active_threads - 1) ? num_elements : start + chunk;

        const hn::ScalableTag<float> d;
        const std::size_t N = hn::Lanes(d);
        
        auto v_max0 = hn::Set(d, local_max);
        auto v_max1 = hn::Set(d, local_max);
        auto v_max2 = hn::Set(d, local_max);
        auto v_max3 = hn::Set(d, local_max);
        
        std::size_t j = start;
        for (; j + 4 * N <= end; j += 4 * N) {
            v_max0 = hn::Max(v_max0, hn::LoadU(d, input + j + 0 * N));
            v_max1 = hn::Max(v_max1, hn::LoadU(d, input + j + 1 * N));
            v_max2 = hn::Max(v_max2, hn::LoadU(d, input + j + 2 * N));
            v_max3 = hn::Max(v_max3, hn::LoadU(d, input + j + 3 * N));
        }
        
        auto v_max = hn::Max(hn::Max(v_max0, v_max1), hn::Max(v_max2, v_max3));
        for (; j + N <= end; j += N) {
            v_max = hn::Max(v_max, hn::LoadU(d, input + j));
        }
        
        float lane_max = hn::ReduceMax(d, v_max);
        if (lane_max > local_max) local_max = lane_max;
        
        for (; j < end; ++j) {
            if (input[j] > local_max) local_max = input[j];
        }
        
        if (local_max > global_max) global_max = local_max;
    }
    return global_max;
}

struct MaxInfo {
    float val;
    std::int64_t idx;
};

std::int64_t ArgmaxImpl(const float* __restrict__ input, std::size_t num_elements, int max_threads) {
    if (num_elements == 0) return -1;
    
    // Decoupling `std::vector` to absolutely enforce C-array strict Thread-Safety constraints gracefully.
    MaxInfo* thread_results = new MaxInfo[max_threads];
    for(int t = 0; t < max_threads; ++t) {
        thread_results[t] = {-std::numeric_limits<float>::infinity(), -1};
    }

    #pragma omp parallel num_threads(max_threads)
    {
        int tid = omp_get_thread_num();
        int active_threads = omp_get_num_threads();
        std::size_t chunk = num_elements / active_threads;
        std::size_t start = tid * chunk;
        std::size_t end = (tid == active_threads - 1) ? num_elements : start + chunk;

        const hn::ScalableTag<float> d;
        const hn::ScalableTag<uint32_t> du32;
        const std::size_t N = hn::Lanes(d);
        
        auto v_max = hn::Set(d, -std::numeric_limits<float>::infinity());
        auto v_idx = hn::Zero(du32);
        auto v_curr_idx = hn::Iota(du32, static_cast<uint32_t>(start));
        auto v_stride = hn::Set(du32, static_cast<uint32_t>(N));
        
        std::size_t j = start;
        for (; j + N <= end; j += N) {
            auto vals = hn::LoadU(d, input + j);
            auto mask = hn::Gt(vals, v_max);
            v_max = hn::IfThenElse(mask, vals, v_max);
            v_idx = hn::IfThenElse(hn::RebindMask(du32, mask), v_curr_idx, v_idx);
            v_curr_idx = hn::Add(v_curr_idx, v_stride);
        }
        
        alignas(64) float tmp_max[32];
        alignas(64) uint32_t tmp_idx[32];
        hn::StoreU(v_max, d, tmp_max);
        hn::StoreU(v_idx, du32, tmp_idx);

        float best_val = -std::numeric_limits<float>::infinity();
        std::int64_t best_idx = -1;
        for (std::size_t lane = 0; lane < N; ++lane) {
            if (tmp_max[lane] > best_val) {
                best_val = tmp_max[lane];
                best_idx = static_cast<std::int64_t>(tmp_idx[lane]);
            }
        }
        
        for (; j < end; ++j) {
            if (input[j] > best_val) {
                best_val = input[j];
                best_idx = static_cast<std::int64_t>(j);
            }
        }
        
        thread_results[tid] = {best_val, best_idx};
    }

    float global_best_val = -std::numeric_limits<float>::infinity();
    std::int64_t global_best_idx = -1;
    for (int t = 0; t < max_threads; ++t) {
        if (thread_results[t].val > global_best_val) {
            global_best_val = thread_results[t].val;
            global_best_idx = thread_results[t].idx;
        } else if (thread_results[t].val == global_best_val && thread_results[t].idx < global_best_idx) {
            global_best_idx = thread_results[t].idx; 
        }
    }
    
    delete[] thread_results; // Relocated safely conditionally natively
    
    return global_best_idx;
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace forway {
HWY_EXPORT(SumImpl);
HWY_EXPORT(MaxImpl);
HWY_EXPORT(ArgmaxImpl);

extern "C" {
    float forway_sum(const float* input, std::size_t num_elements, int max_threads) noexcept {
        return HWY_STATIC_DISPATCH(SumImpl)(input, num_elements, max_threads);
    }
    float forway_max(const float* input, std::size_t num_elements, int max_threads) noexcept {
        return HWY_STATIC_DISPATCH(MaxImpl)(input, num_elements, max_threads);
    }
    std::int64_t forway_argmax(const float* input, std::size_t num_elements, int max_threads) noexcept {
        return HWY_STATIC_DISPATCH(ArgmaxImpl)(input, num_elements, max_threads);
    }
}
}

#endif
