#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "activations_kernel.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

#include <omp.h>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

HWY_BEFORE_NAMESPACE();

namespace forway {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void ExpImpl(const float* __restrict__ input, float* __restrict__ output, std::size_t num_elements, int max_threads) {
    const hn::ScalableTag<float> d;
    const std::size_t N = hn::Lanes(d);

    #pragma omp parallel for num_threads(max_threads)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(num_elements); i += N) {
        if (i + N <= static_cast<std::int64_t>(num_elements)) {
            auto v = hn::LoadU(d, input + i);
            auto v_exp = hn::Exp(d, v);
            hn::StoreU(v_exp, d, output + i);
        } else {
            // Scalar fallback tail
            for (std::size_t j = i; j < num_elements; ++j) {
                output[j] = std::exp(input[j]);
            }
        }
    }
}

void TanhImpl(const float* __restrict__ input, float* __restrict__ output, std::size_t num_elements, int max_threads) {
    const hn::ScalableTag<float> d;
    const std::size_t N = hn::Lanes(d);

    #pragma omp parallel for num_threads(max_threads)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(num_elements); i += N) {
        if (i + N <= static_cast<std::int64_t>(num_elements)) {
            auto v = hn::LoadU(d, input + i);
            auto v_tanh = hn::Tanh(d, v);
            hn::StoreU(v_tanh, d, output + i);
        } else {
            for (std::size_t j = i; j < num_elements; ++j) {
                output[j] = std::tanh(input[j]);
            }
        }
    }
}

void SoftmaxImpl(const float* __restrict__ input, float* __restrict__ output, std::size_t num_rows, std::size_t num_cols, int max_threads) {
    const hn::ScalableTag<float> d;
    const std::size_t N = hn::Lanes(d);

    #pragma omp parallel for num_threads(max_threads)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(num_rows); ++i) {
        const float* __restrict__ in_row = input + i * num_cols;
        float* __restrict__ out_row = output + i * num_cols;

        // 1. Array Max pass (To prevent Exp overflow bounds natively checking dynamically)
        auto v_max = hn::Set(d, -std::numeric_limits<float>::infinity());
        std::size_t j = 0;
        for (; j + N <= num_cols; j += N) {
            v_max = hn::Max(v_max, hn::LoadU(d, in_row + j));
        }
        float row_max = hn::ReduceMax(d, v_max);
        for (; j < num_cols; ++j) {
            if (in_row[j] > row_max) row_max = in_row[j];
        }

        // 2. Compute Exp differences explicitly tracking accumulators organically dynamically
        auto v_row_max = hn::Set(d, row_max);
        auto v_sum = hn::Zero(d);
        j = 0;
        
        // Pipelining 2x explicitly to hide Polynomial Execution latency paths natively checking
        for (; j + 2 * N <= num_cols; j += 2 * N) {
            auto v0 = hn::LoadU(d, in_row + j + 0 * N);
            auto v1 = hn::LoadU(d, in_row + j + 1 * N);
            
            auto exp0 = hn::Exp(d, hn::Sub(v0, v_row_max));
            auto exp1 = hn::Exp(d, hn::Sub(v1, v_row_max));
            
            hn::StoreU(exp0, d, out_row + j + 0 * N);
            hn::StoreU(exp1, d, out_row + j + 1 * N);
            
            v_sum = hn::Add(v_sum, hn::Add(exp0, exp1));
        }
        for (; j + N <= num_cols; j += N) {
            auto v = hn::LoadU(d, in_row + j);
            auto exp_v = hn::Exp(d, hn::Sub(v, v_row_max));
            hn::StoreU(exp_v, d, out_row + j);
            v_sum = hn::Add(v_sum, exp_v);
        }
        
        float row_sum = hn::ReduceSum(d, v_sum);
        for (; j < num_cols; ++j) {
            float exp_val = std::exp(in_row[j] - row_max);
            out_row[j] = exp_val;
            row_sum += exp_val;
        }

        // 3. Normalize limits linearly avoiding complex division cycles purely explicitly
        const float inv_sum = 1.0f / row_sum;
        auto v_inv_sum = hn::Set(d, inv_sum);
        j = 0;
        for (; j + N <= num_cols; j += N) {
            auto v_exp = hn::LoadU(d, out_row + j);
            hn::StoreU(hn::Mul(v_exp, v_inv_sum), d, out_row + j);
        }
        for (; j < num_cols; ++j) {
            out_row[j] *= inv_sum;
        }
    }
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace forway {
HWY_EXPORT(ExpImpl);
HWY_EXPORT(TanhImpl);
HWY_EXPORT(SoftmaxImpl);

extern "C" {
    void forway_exp(const float* input, float* output, std::size_t num_elements, int max_threads) noexcept {
        HWY_STATIC_DISPATCH(ExpImpl)(input, output, num_elements, max_threads);
    }
    void forway_tanh(const float* input, float* output, std::size_t num_elements, int max_threads) noexcept {
        HWY_STATIC_DISPATCH(TanhImpl)(input, output, num_elements, max_threads);
    }
    void forway_softmax(const float* input, float* output, std::size_t num_rows, std::size_t num_cols, int max_threads) noexcept {
        HWY_STATIC_DISPATCH(SoftmaxImpl)(input, output, num_rows, num_cols, max_threads);
    }
}
}

#endif
