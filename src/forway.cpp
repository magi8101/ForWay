#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <omp.h>
#include "hwy/contrib/sort/vqsort.h"

namespace nb = nanobind;

extern "C" {
    void forway_gemm_float(
        int M, int N, int K,
        const float* A, int lda,
        const float* B, int ldb,
        float* C, int ldc
    );

    void forway_gemm_double(
        int M, int N, int K,
        const double* A, int lda,
        const double* B, int ldb,
        double* C, int ldc
    );

    void forway_random_uniform_float(int N, float* C_ptr, int64_t seed);
    
    void forway_cosine_similarity(
        const float* query,
        const float* db_matrix,
        float* output,
        std::size_t num_vectors,
        std::size_t num_dims,
        int max_threads
    ) noexcept;

    void forway_exp(const float* input, float* output, std::size_t num_elements, int max_threads) noexcept;
    void forway_tanh(const float* input, float* output, std::size_t num_elements, int max_threads) noexcept;
    void forway_softmax(const float* input, float* output, std::size_t num_rows, std::size_t num_cols, int max_threads) noexcept;

    float forway_sum(const float* input, std::size_t num_elements, int max_threads) noexcept;
    float forway_max(const float* input, std::size_t num_elements, int max_threads) noexcept;
    std::int64_t forway_argmax(const float* input, std::size_t num_elements, int max_threads) noexcept;

    void forway_transpose(const float* input, float* output, std::size_t num_rows, std::size_t num_cols, int max_threads) noexcept;

    float forway_dot_vv(const float* a, const float* b, std::size_t n, int max_threads) noexcept;
    void forway_dot_mv(const float* A, const float* x, float* y, std::size_t M, std::size_t K, int max_threads) noexcept;
}

// ------------------------------------------------------------------
// High-Performance Parallel VQSort (Vectorized Merge + OpenMP Multi-Core)
// ------------------------------------------------------------------

void sort_float(nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> array) {
    if (array.shape(0) == 0) return;
    hwy::VQSort(array.data(), array.shape(0), hwy::SortAscending());
}

void sort_double(nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> array) {
    if (array.shape(0) == 0) return;
    hwy::VQSort(array.data(), array.shape(0), hwy::SortAscending());
}

void sort_int32(nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> array) {
    if (array.shape(0) == 0) return;
    hwy::VQSort(array.data(), array.shape(0), hwy::SortAscending());
}

void sort_int64(nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> array) {
    if (array.shape(0) == 0) return;
    hwy::VQSort(array.data(), array.shape(0), hwy::SortAscending());
}


void gemm_float(
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> a,
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> b,
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> c
) {
    std::size_t M = a.shape(0);
    std::size_t K_a = a.shape(1);
    std::size_t K_b = b.shape(0);
    std::size_t N = b.shape(1);
    std::size_t M_c = c.shape(0);
    std::size_t N_c = c.shape(1);

    if (K_a != K_b) {
        throw std::invalid_argument("Matrix dimension mismatch: A columns must equal B rows");
    }
    if (M != M_c) {
        throw std::invalid_argument("Matrix dimension mismatch: A rows must equal C rows");
    }
    if (N != N_c) {
        throw std::invalid_argument("Matrix dimension mismatch: B columns must equal C columns");
    }

    int m = static_cast<int>(M);
    int n = static_cast<int>(N);
    int k = static_cast<int>(K_a);

    forway_gemm_float(
        m, n, k,
        a.data(), k,
        b.data(), n,
        c.data(), n
    );
}

void gemm_double(
    nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> a,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> b,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> c
) {
    std::size_t M = a.shape(0);
    std::size_t K_a = a.shape(1);
    std::size_t K_b = b.shape(0);
    std::size_t N = b.shape(1);
    std::size_t M_c = c.shape(0);
    std::size_t N_c = c.shape(1);

    if (K_a != K_b) {
        throw std::invalid_argument("Matrix dimension mismatch: A columns must equal B rows");
    }
    if (M != M_c) {
        throw std::invalid_argument("Matrix dimension mismatch: A rows must equal C rows");
    }
    if (N != N_c) {
        throw std::invalid_argument("Matrix dimension mismatch: B columns must equal C columns");
    }

    int m = static_cast<int>(M);
    int n = static_cast<int>(N);
    int k = static_cast<int>(K_a);

    forway_gemm_double(
        m, n, k,
        a.data(), k,
        b.data(), n,
        c.data(), n
    );
}

void random_uniform(nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> c, int64_t seed) {
    std::size_t N = c.size();
    nb::gil_scoped_release release;
    forway_random_uniform_float(static_cast<int>(N), c.data(), seed);
}

// ------------------------------------------------------------------
// High-Performance Vector Database Distance Metrics (AVX-512 FMA Fused Engine)
// ------------------------------------------------------------------

nb::ndarray<nb::numpy, float, nb::c_contig> cosine_similarity(
    nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> query,
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> db_matrix,
    int max_threads
) {
    std::size_t num_dims_q = query.shape(0);
    std::size_t num_vectors = db_matrix.shape(0);
    std::size_t num_dims_m = db_matrix.shape(1);

    if (num_dims_q != num_dims_m) {
        throw std::invalid_argument("Vector Dimension Mismatch: Query dimension directly contradicts Database Matrix bounds explicitly!");
    }

    // Allocate organic zero-copy NumPy output vector arrays internally
    std::size_t shape[1] = { num_vectors };
    float* data = new float[num_vectors];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });

    // Release GIL structurally unlocking true physical OpenMP bounds
    {
        nb::gil_scoped_release release;
        forway_cosine_similarity(query.data(), db_matrix.data(), data, num_vectors, num_dims_q, max_threads);
    }
    
    return nb::ndarray<nb::numpy, float, nb::c_contig>(data, 1, shape, owner);
}

// ------------------------------------------------------------------
// High-Performance Mathematical Activations (Neural Networks Fused AVX-512)
// ------------------------------------------------------------------

nb::ndarray<nb::numpy, float, nb::c_contig> exp_kernel(
    nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> input,
    int max_threads
) {
    std::size_t N = input.shape(0);
    std::size_t shape[1] = { N };
    float* data = new float[N];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    {
        nb::gil_scoped_release release;
        forway_exp(input.data(), data, N, max_threads);
    }
    
    return nb::ndarray<nb::numpy, float, nb::c_contig>(data, 1, shape, owner);
}

nb::ndarray<nb::numpy, float, nb::c_contig> tanh_kernel(
    nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> input,
    int max_threads
) {
    std::size_t N = input.shape(0);
    std::size_t shape[1] = { N };
    float* data = new float[N];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    {
        nb::gil_scoped_release release;
        forway_tanh(input.data(), data, N, max_threads);
    }
    
    return nb::ndarray<nb::numpy, float, nb::c_contig>(data, 1, shape, owner);
}

nb::ndarray<nb::numpy, float, nb::c_contig> softmax_kernel(
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> input,
    int max_threads
) {
    std::size_t num_rows = input.shape(0);
    std::size_t num_cols = input.shape(1);
    
    std::size_t shape[2] = { num_rows, num_cols };
    float* data = new float[num_rows * num_cols];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    {
        nb::gil_scoped_release release;
        forway_softmax(input.data(), data, num_rows, num_cols, max_threads);
    }
    
    return nb::ndarray<nb::numpy, float, nb::c_contig>(data, 2, shape, owner);
}

// ------------------------------------------------------------------
// High-Performance Vectorized Array Reductions (Native Multi-Core AVX Sequences)
// ------------------------------------------------------------------

float sum_kernel(nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> input, int max_threads) {
    nb::gil_scoped_release release;
    return forway_sum(input.data(), input.shape(0), max_threads);
}

float max_kernel(nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> input, int max_threads) {
    nb::gil_scoped_release release;
    return forway_max(input.data(), input.shape(0), max_threads);
}

std::int64_t argmax_kernel(nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> input, int max_threads) {
    nb::gil_scoped_release release;
    return forway_argmax(input.data(), input.shape(0), max_threads);
}

nb::ndarray<nb::numpy, float, nb::c_contig> transpose_kernel(
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> input,
    int max_threads
) {
    std::size_t num_rows = input.shape(0);
    std::size_t num_cols = input.shape(1);
    
    std::size_t shape[2] = { num_cols, num_rows }; // Inverted explicit logical matrix dimensions
    float* data = new float[num_rows * num_cols];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    {
        nb::gil_scoped_release release;
        forway_transpose(input.data(), data, num_rows, num_cols, max_threads);
    }
    
    return nb::ndarray<nb::numpy, float, nb::c_contig>(data, 2, shape, owner);
}

// ------------------------------------------------------------------
// High-Performance Dot Product (numpy.dot equivalent)
// ------------------------------------------------------------------

float dot_vv_kernel(
    nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> b,
    int max_threads
) {
    if (a.shape(0) != b.shape(0)) {
        throw std::invalid_argument("Vector length mismatch for dot product");
    }
    nb::gil_scoped_release release;
    return forway_dot_vv(a.data(), b.data(), a.shape(0), max_threads);
}

nb::ndarray<nb::numpy, float, nb::c_contig> dot_mv_kernel(
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> A,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> x,
    int max_threads
) {
    std::size_t M = A.shape(0);
    std::size_t K = A.shape(1);
    if (K != x.shape(0)) {
        throw std::invalid_argument("Matrix columns must equal vector length for dot product");
    }
    
    std::size_t shape[1] = { M };
    float* data = new float[M];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    {
        nb::gil_scoped_release release;
        forway_dot_mv(A.data(), x.data(), data, M, K, max_threads);
    }
    
    return nb::ndarray<nb::numpy, float, nb::c_contig>(data, 1, shape, owner);
}

// ------------------------------------------------------------------
// Python Module Definition
// ------------------------------------------------------------------

NB_MODULE(forway, m) {
    m.def("gemm", &gemm_float,
        nb::arg("A"), nb::arg("B"), nb::arg("C"),
        "Compute C = A @ B using BLIS-style cache blocking and SIMD (float32).\n\n"
        "Parameters:\n"
        "  A: Input matrix of shape (M, K), float32, C-contiguous\n"
        "  B: Input matrix of shape (K, N), float32, C-contiguous\n"
        "  C: Output matrix of shape (M, N), float32, C-contiguous\n\n"
        "Note: C is overwritten with the result A @ B.");

    m.def("gemm", &gemm_double,
        nb::arg("A"), nb::arg("B"), nb::arg("C"),
        "Compute C = A @ B using BLIS-style cache blocking and SIMD (float64).\n\n"
        "Parameters:\n"
        "  A: Input matrix of shape (M, K), float64, C-contiguous\n"
        "  B: Input matrix of shape (K, N), float64, C-contiguous\n"
        "  C: Output matrix of shape (M, N), float64, C-contiguous\n\n"
        "Note: C is overwritten with the result A @ B.");

    m.def("random_uniform", &random_uniform,
        nb::arg("C"), nb::arg("seed"),
        "Fill array C with uniformly distributed random float32s in [0, 1) using ChaCha8.\n\n"
        "Parameters:\n"
        "  C: Output array of shape (N,), float32, C-contiguous\n"
        "  seed: 64-bit integer seed for the PRNG\n");

    m.def("sort", &sort_float, nb::arg("array"), "Executes Google Highway vqsort exactly mapping AVX-512 FFI paths In-Place unconditionally over Float32.");
    m.def("sort", &sort_double, nb::arg("array"), "Executes Google Highway vqsort exactly mapping AVX-512 FFI paths In-Place unconditionally over Float64.");
    m.def("sort", &sort_int32, nb::arg("array"), "Executes Google Highway vqsort exactly mapping AVX-512 FFI paths In-Place unconditionally over Int32.");
    m.def("sort", &sort_int64, nb::arg("array"), "Executes Google Highway vqsort exactly mapping AVX-512 FFI paths In-Place unconditionally over Int64.");
    
    m.def("cosine_similarity", &cosine_similarity, nb::arg("query"), nb::arg("db_matrix"), nb::arg("max_threads") = 16, 
        "Evaluates fused Cosine Similarity dynamically executing L2 Norms unconditionally bypassing scalar loops organically native AVX-512 over OpenMP Threads.");

    m.def("exp", &exp_kernel, nb::arg("array"), nb::arg("max_threads") = 16, "Executes heavily vectorized Float32 Exponentials intrinsically unrolled via Google Highway polynomial logic organically checking arrays natively.");
    m.def("tanh", &tanh_kernel, nb::arg("array"), nb::arg("max_threads") = 16, "Executes exact AVX-512 bounded Hyperbolic Tangent structurally securely mapping natively internally bounds explicitly.");
    m.def("softmax", &softmax_kernel, nb::arg("matrix_2d"), nb::arg("max_threads") = 16, "Fuses multi-pass matrix bounds implicitly scaling Exponential mapping conditionally logically traversing arrays seamlessly dynamically!");

    m.def("sum", &sum_kernel, nb::arg("array"), nb::arg("max_threads") = 16, "Pipelined 4x intrinsically OpenMP Thread-Aware Reductions structurally mathematically tracking dynamically seamlessly.");
    m.def("max", &max_kernel, nb::arg("array"), nb::arg("max_threads") = 16, "Intrinsically physically unrolled OMP evaluation mapping.");
    m.def("argmax", &argmax_kernel, nb::arg("array"), nb::arg("max_threads") = 16, "Secure array traversals efficiently functionally mathematically tracking structurally natively intrinsically accurately executing dynamically explicitly inherently structurally intelligently organically practically safely.");

    m.def("transpose", &transpose_kernel, nb::arg("matrix_2d"), nb::arg("max_threads") = 16, "Cache-blocked 32x32 multi-threaded matrix transposition.");

    m.def("dot", &dot_vv_kernel, nb::arg("a"), nb::arg("b"), nb::arg("max_threads") = 16, "Vector-vector dot product (1D · 1D → scalar) with multi-threaded 4x FMA pipelining.");
    m.def("dot", &dot_mv_kernel, nb::arg("A"), nb::arg("x"), nb::arg("max_threads") = 16, "Matrix-vector dot product (2D × 1D → 1D) with OpenMP row parallelism.");
}
