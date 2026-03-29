#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "metrics_kernel.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include <omp.h>
#include <cmath>
#include <cstdint>
#include <cstddef>

HWY_BEFORE_NAMESPACE();

namespace forway {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void CosineSimilarityImpl(
    const float* __restrict__ query,
    const float* __restrict__ db_matrix,
    float* __restrict__ output,
    const std::size_t num_vectors,
    const std::size_t num_dims,
    const int max_threads
) {
    const hn::ScalableTag<float> d;
    const std::size_t N = hn::Lanes(d);

    // Pre-calculate the L2 norm of the single 1D query vector purely sequentially
    // (As it only happens exactly once, scaling SIMD here is structurally optional but we will do it natively)
    float query_norm_sq = 0.0f;
    {
        auto v_norm_q = hn::Zero(d);
        std::size_t j = 0;
        for (; j + N <= num_dims; j += N) {
            auto q = hn::LoadU(d, query + j);
            v_norm_q = hn::MulAdd(q, q, v_norm_q);
        }
        query_norm_sq = hn::ReduceSum(d, v_norm_q);
        for (; j < num_dims; ++j) {
            query_norm_sq += query[j] * query[j];
        }
    }
    
    // Mathematically catch zero-length vectors purely mapping natively
    const float query_norm = (query_norm_sq > 0.0f) ? std::sqrt(query_norm_sq) : 1.0f;
    const float inv_query_norm = 1.0f / query_norm;

    // Evaluate structural DB Vectors independently explicitly breaking constraints parallel
    #pragma omp parallel for num_threads(max_threads)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(num_vectors); ++i) {
        
        const float* __restrict__ db_row = db_matrix + (i * num_dims);
        
        auto v_dot0 = hn::Zero(d), v_dot1 = hn::Zero(d), v_dot2 = hn::Zero(d), v_dot3 = hn::Zero(d);
        auto v_norm0 = hn::Zero(d), v_norm1 = hn::Zero(d), v_norm2 = hn::Zero(d), v_norm3 = hn::Zero(d);
        
        std::size_t j = 0;
        
        // 1. Pipelined Loop: Unrolled by 4 to cleanly hide physical hardware FMA stall limitations
        for (; j + 4 * N <= num_dims; j += 4 * N) {
            auto q0 = hn::LoadU(d, query + j + 0 * N);
            auto v0 = hn::LoadU(d, db_row + j + 0 * N);
            v_dot0 = hn::MulAdd(q0, v0, v_dot0);
            v_norm0 = hn::MulAdd(v0, v0, v_norm0);

            auto q1 = hn::LoadU(d, query + j + 1 * N);
            auto v1 = hn::LoadU(d, db_row + j + 1 * N);
            v_dot1 = hn::MulAdd(q1, v1, v_dot1);
            v_norm1 = hn::MulAdd(v1, v1, v_norm1);

            auto q2 = hn::LoadU(d, query + j + 2 * N);
            auto v2 = hn::LoadU(d, db_row + j + 2 * N);
            v_dot2 = hn::MulAdd(q2, v2, v_dot2);
            v_norm2 = hn::MulAdd(v2, v2, v_norm2);

            auto q3 = hn::LoadU(d, query + j + 3 * N);
            auto v3 = hn::LoadU(d, db_row + j + 3 * N);
            v_dot3 = hn::MulAdd(q3, v3, v_dot3);
            v_norm3 = hn::MulAdd(v3, v3, v_norm3);
        }
        
        // 2. Collapse the pipelined accumulators recursively structurally
        auto v_dot = hn::Add(hn::Add(v_dot0, v_dot1), hn::Add(v_dot2, v_dot3));
        auto v_db_norm_sq = hn::Add(hn::Add(v_norm0, v_norm1), hn::Add(v_norm2, v_norm3));
        
        // 3. Vector Remainder Sequence Loop (Catching unrolled vectors)
        for (; j + N <= num_dims; j += N) {
            auto q = hn::LoadU(d, query + j);
            auto v = hn::LoadU(d, db_row + j);
            v_dot = hn::MulAdd(q, v, v_dot);
            v_db_norm_sq = hn::MulAdd(v, v, v_db_norm_sq);
        }
        
        float dot_sum = hn::ReduceSum(d, v_dot);
        float db_norm_sq_sum = hn::ReduceSum(d, v_db_norm_sq);
        
        // Scalar fallback tail execution
        for (; j < num_dims; ++j) {
            float q = query[j];
            float v = db_row[j];
            dot_sum += q * v;
            db_norm_sq_sum += v * v;
        }
        
        float float_db_norm = (db_norm_sq_sum > 0.0f) ? std::sqrt(db_norm_sq_sum) : 1.0f;
        
        output[i] = dot_sum * inv_query_norm * (1.0f / float_db_norm);
    }
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace forway {
HWY_EXPORT(CosineSimilarityImpl);

extern "C" {
    void forway_cosine_similarity(
        const float* query,
        const float* db_matrix,
        float* output,
        std::size_t num_vectors,
        std::size_t num_dims,
        int max_threads
    ) noexcept {
        HWY_STATIC_DISPATCH(CosineSimilarityImpl)(query, db_matrix, output, num_vectors, num_dims, max_threads);
    }
}
}

#endif
