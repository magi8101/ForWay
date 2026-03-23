#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

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
}
