#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "micro_kernel.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/cache_control.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>
#endif

HWY_BEFORE_NAMESPACE();

namespace forway {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_ATTR void MicroKernelImpl(
    const T* HWY_RESTRICT packed_a,
    const T* HWY_RESTRICT packed_b,
    T* HWY_RESTRICT c_block,
    std::size_t mr,
    std::size_t nr,
    std::size_t kc,
    std::size_t ldc,
    int accumulate
) {
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;

    V c0, c1, c2, c3, c4, c5, c6, c7;
    if (accumulate) {
        c0 = hn::LoadU(d, c_block + 0 * ldc);
        c1 = hn::LoadU(d, c_block + 1 * ldc);
        c2 = hn::LoadU(d, c_block + 2 * ldc);
        c3 = hn::LoadU(d, c_block + 3 * ldc);
        c4 = hn::LoadU(d, c_block + 4 * ldc);
        c5 = hn::LoadU(d, c_block + 5 * ldc);
        c6 = hn::LoadU(d, c_block + 6 * ldc);
        c7 = hn::LoadU(d, c_block + 7 * ldc);
    } else {
        c0 = hn::Zero(d);
        c1 = hn::Zero(d);
        c2 = hn::Zero(d);
        c3 = hn::Zero(d);
        c4 = hn::Zero(d);
        c5 = hn::Zero(d);
        c6 = hn::Zero(d);
        c7 = hn::Zero(d);
    }

    for (std::size_t p = 0; p < kc; ++p) {
        V a = hn::Load(d, packed_a + p * mr);

        c0 = hn::MulAdd(a, hn::Set(d, packed_b[p * nr + 0]), c0);
        c1 = hn::MulAdd(a, hn::Set(d, packed_b[p * nr + 1]), c1);
        c2 = hn::MulAdd(a, hn::Set(d, packed_b[p * nr + 2]), c2);
        c3 = hn::MulAdd(a, hn::Set(d, packed_b[p * nr + 3]), c3);
        c4 = hn::MulAdd(a, hn::Set(d, packed_b[p * nr + 4]), c4);
        c5 = hn::MulAdd(a, hn::Set(d, packed_b[p * nr + 5]), c5);
        c6 = hn::MulAdd(a, hn::Set(d, packed_b[p * nr + 6]), c6);
        c7 = hn::MulAdd(a, hn::Set(d, packed_b[p * nr + 7]), c7);
    }

    hn::StoreU(c0, d, c_block + 0 * ldc);
    hn::StoreU(c1, d, c_block + 1 * ldc);
    hn::StoreU(c2, d, c_block + 2 * ldc);
    hn::StoreU(c3, d, c_block + 3 * ldc);
    hn::StoreU(c4, d, c_block + 4 * ldc);
    hn::StoreU(c5, d, c_block + 5 * ldc);
    hn::StoreU(c6, d, c_block + 6 * ldc);
    hn::StoreU(c7, d, c_block + 7 * ldc);
}

void MicroKernelFloat(
    const float* packed_a,
    const float* packed_b,
    float* c_block,
    std::size_t mr,
    std::size_t nr,
    std::size_t kc,
    std::size_t ldc,
    int accumulate
) {
    MicroKernelImpl<float>(packed_a, packed_b, c_block, mr, nr, kc, ldc, accumulate);
}

void MicroKernelDouble(
    const double* packed_a,
    const double* packed_b,
    double* c_block,
    std::size_t mr,
    std::size_t nr,
    std::size_t kc,
    std::size_t ldc,
    int accumulate
) {
    MicroKernelImpl<double>(packed_a, packed_b, c_block, mr, nr, kc, ldc, accumulate);
}

std::size_t GetLanesFloat() {
    const hn::ScalableTag<float> d;
    return hn::Lanes(d);
}

std::size_t GetLanesDouble() {
    const hn::ScalableTag<double> d;
    return hn::Lanes(d);
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace forway {

HWY_EXPORT(MicroKernelFloat);
HWY_EXPORT(MicroKernelDouble);
HWY_EXPORT(GetLanesFloat);
HWY_EXPORT(GetLanesDouble);

extern "C" {

void* forway_aligned_malloc(std::size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, 64);
#else
    void* ptr;
    if (posix_memalign(&ptr, 64, size) != 0) return nullptr;
    return ptr;
#endif
}

void forway_aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void forway_micro_kernel_float(
    const float* packed_a,
    const float* packed_b,
    float* c_block,
    std::size_t mr,
    std::size_t nr,
    std::size_t kc,
    std::size_t ldc,
    int accumulate
) noexcept {
    HWY_STATIC_DISPATCH(MicroKernelFloat)(packed_a, packed_b, c_block, mr, nr, kc, ldc, accumulate);
}

void forway_micro_kernel_double(
    const double* packed_a,
    const double* packed_b,
    double* c_block,
    std::size_t mr,
    std::size_t nr,
    std::size_t kc,
    std::size_t ldc,
    int accumulate
) noexcept {
    HWY_STATIC_DISPATCH(MicroKernelDouble)(packed_a, packed_b, c_block, mr, nr, kc, ldc, accumulate);
}

std::size_t forway_get_lanes_float() noexcept {
    return HWY_STATIC_DISPATCH(GetLanesFloat)();
}

std::size_t forway_get_lanes_double() noexcept {
    return HWY_STATIC_DISPATCH(GetLanesDouble)();
}

}

}

#endif
