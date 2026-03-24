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
    const std::size_t vec_len = hn::Lanes(d);

    V c00, c01, c02, c03, c04, c05, c06, c07;
    V c10, c11, c12, c13, c14, c15, c16, c17;
    
    if (accumulate) {
        c00 = hn::LoadU(d, c_block + 0 * ldc + 0 * vec_len); c10 = hn::LoadU(d, c_block + 0 * ldc + 1 * vec_len);
        c01 = hn::LoadU(d, c_block + 1 * ldc + 0 * vec_len); c11 = hn::LoadU(d, c_block + 1 * ldc + 1 * vec_len);
        c02 = hn::LoadU(d, c_block + 2 * ldc + 0 * vec_len); c12 = hn::LoadU(d, c_block + 2 * ldc + 1 * vec_len);
        c03 = hn::LoadU(d, c_block + 3 * ldc + 0 * vec_len); c13 = hn::LoadU(d, c_block + 3 * ldc + 1 * vec_len);
        c04 = hn::LoadU(d, c_block + 4 * ldc + 0 * vec_len); c14 = hn::LoadU(d, c_block + 4 * ldc + 1 * vec_len);
        c05 = hn::LoadU(d, c_block + 5 * ldc + 0 * vec_len); c15 = hn::LoadU(d, c_block + 5 * ldc + 1 * vec_len);
        c06 = hn::LoadU(d, c_block + 6 * ldc + 0 * vec_len); c16 = hn::LoadU(d, c_block + 6 * ldc + 1 * vec_len);
        c07 = hn::LoadU(d, c_block + 7 * ldc + 0 * vec_len); c17 = hn::LoadU(d, c_block + 7 * ldc + 1 * vec_len);
    } else {
        c00 = hn::Zero(d); c10 = hn::Zero(d);
        c01 = hn::Zero(d); c11 = hn::Zero(d);
        c02 = hn::Zero(d); c12 = hn::Zero(d);
        c03 = hn::Zero(d); c13 = hn::Zero(d);
        c04 = hn::Zero(d); c14 = hn::Zero(d);
        c05 = hn::Zero(d); c15 = hn::Zero(d);
        c06 = hn::Zero(d); c16 = hn::Zero(d);
        c07 = hn::Zero(d); c17 = hn::Zero(d);
    }

    std::size_t p = 0;
    
    // Main pipelined loop (Unrolled by 2 to hide memory latency)
    for (; p + 1 < kc; p += 2) {
        // --- TICK 0 ---
        V a00 = hn::Load(d, packed_a + p * mr + 0 * vec_len); 
        V a01 = hn::Load(d, packed_a + p * mr + 1 * vec_len);

        V b0_0 = hn::Set(d, packed_b[p * nr + 0]);
        c00 = hn::MulAdd(a00, b0_0, c00); c10 = hn::MulAdd(a01, b0_0, c10);
        
        V b0_1 = hn::Set(d, packed_b[p * nr + 1]);
        c01 = hn::MulAdd(a00, b0_1, c01); c11 = hn::MulAdd(a01, b0_1, c11);
        
        V b0_2 = hn::Set(d, packed_b[p * nr + 2]);
        c02 = hn::MulAdd(a00, b0_2, c02); c12 = hn::MulAdd(a01, b0_2, c12);
        
        V b0_3 = hn::Set(d, packed_b[p * nr + 3]);
        c03 = hn::MulAdd(a00, b0_3, c03); c13 = hn::MulAdd(a01, b0_3, c13);
        
        V b0_4 = hn::Set(d, packed_b[p * nr + 4]);
        c04 = hn::MulAdd(a00, b0_4, c04); c14 = hn::MulAdd(a01, b0_4, c14);
        
        V b0_5 = hn::Set(d, packed_b[p * nr + 5]);
        c05 = hn::MulAdd(a00, b0_5, c05); c15 = hn::MulAdd(a01, b0_5, c15);
        
        V b0_6 = hn::Set(d, packed_b[p * nr + 6]);
        c06 = hn::MulAdd(a00, b0_6, c06); c16 = hn::MulAdd(a01, b0_6, c16);
        
        V b0_7 = hn::Set(d, packed_b[p * nr + 7]);
        c07 = hn::MulAdd(a00, b0_7, c07); c17 = hn::MulAdd(a01, b0_7, c17);

        // --- TICK 1 ---
        V a10 = hn::Load(d, packed_a + (p + 1) * mr + 0 * vec_len); 
        V a11 = hn::Load(d, packed_a + (p + 1) * mr + 1 * vec_len);

        V b1_0 = hn::Set(d, packed_b[(p + 1) * nr + 0]);
        c00 = hn::MulAdd(a10, b1_0, c00); c10 = hn::MulAdd(a11, b1_0, c10);
        
        V b1_1 = hn::Set(d, packed_b[(p + 1) * nr + 1]);
        c01 = hn::MulAdd(a10, b1_1, c01); c11 = hn::MulAdd(a11, b1_1, c11);
        
        V b1_2 = hn::Set(d, packed_b[(p + 1) * nr + 2]);
        c02 = hn::MulAdd(a10, b1_2, c02); c12 = hn::MulAdd(a11, b1_2, c12);
        
        V b1_3 = hn::Set(d, packed_b[(p + 1) * nr + 3]);
        c03 = hn::MulAdd(a10, b1_3, c03); c13 = hn::MulAdd(a11, b1_3, c13);
        
        V b1_4 = hn::Set(d, packed_b[(p + 1) * nr + 4]);
        c04 = hn::MulAdd(a10, b1_4, c04); c14 = hn::MulAdd(a11, b1_4, c14);
        
        V b1_5 = hn::Set(d, packed_b[(p + 1) * nr + 5]);
        c05 = hn::MulAdd(a10, b1_5, c05); c15 = hn::MulAdd(a11, b1_5, c15);
        
        V b1_6 = hn::Set(d, packed_b[(p + 1) * nr + 6]);
        c06 = hn::MulAdd(a10, b1_6, c06); c16 = hn::MulAdd(a11, b1_6, c16);
        
        V b1_7 = hn::Set(d, packed_b[(p + 1) * nr + 7]);
        c07 = hn::MulAdd(a10, b1_7, c07); c17 = hn::MulAdd(a11, b1_7, c17);
    }

    // Safe remainder loop to prevent segfaults on odd matrices (e.g. 7x13x11)
    for (; p < kc; ++p) {
        V a00 = hn::Load(d, packed_a + p * mr + 0 * vec_len); 
        V a01 = hn::Load(d, packed_a + p * mr + 1 * vec_len);

        V b0_0 = hn::Set(d, packed_b[p * nr + 0]);
        c00 = hn::MulAdd(a00, b0_0, c00); c10 = hn::MulAdd(a01, b0_0, c10);
        
        V b0_1 = hn::Set(d, packed_b[p * nr + 1]);
        c01 = hn::MulAdd(a00, b0_1, c01); c11 = hn::MulAdd(a01, b0_1, c11);
        
        V b0_2 = hn::Set(d, packed_b[p * nr + 2]);
        c02 = hn::MulAdd(a00, b0_2, c02); c12 = hn::MulAdd(a01, b0_2, c12);
        
        V b0_3 = hn::Set(d, packed_b[p * nr + 3]);
        c03 = hn::MulAdd(a00, b0_3, c03); c13 = hn::MulAdd(a01, b0_3, c13);
        
        V b0_4 = hn::Set(d, packed_b[p * nr + 4]);
        c04 = hn::MulAdd(a00, b0_4, c04); c14 = hn::MulAdd(a01, b0_4, c14);
        
        V b0_5 = hn::Set(d, packed_b[p * nr + 5]);
        c05 = hn::MulAdd(a00, b0_5, c05); c15 = hn::MulAdd(a01, b0_5, c15);
        
        V b0_6 = hn::Set(d, packed_b[p * nr + 6]);
        c06 = hn::MulAdd(a00, b0_6, c06); c16 = hn::MulAdd(a01, b0_6, c16);
        
        V b0_7 = hn::Set(d, packed_b[p * nr + 7]);
        c07 = hn::MulAdd(a00, b0_7, c07); c17 = hn::MulAdd(a01, b0_7, c17);
    }

    hn::StoreU(c00, d, c_block + 0 * ldc + 0 * vec_len); hn::StoreU(c10, d, c_block + 0 * ldc + 1 * vec_len);
    hn::StoreU(c01, d, c_block + 1 * ldc + 0 * vec_len); hn::StoreU(c11, d, c_block + 1 * ldc + 1 * vec_len);
    hn::StoreU(c02, d, c_block + 2 * ldc + 0 * vec_len); hn::StoreU(c12, d, c_block + 2 * ldc + 1 * vec_len);
    hn::StoreU(c03, d, c_block + 3 * ldc + 0 * vec_len); hn::StoreU(c13, d, c_block + 3 * ldc + 1 * vec_len);
    hn::StoreU(c04, d, c_block + 4 * ldc + 0 * vec_len); hn::StoreU(c14, d, c_block + 4 * ldc + 1 * vec_len);
    hn::StoreU(c05, d, c_block + 5 * ldc + 0 * vec_len); hn::StoreU(c15, d, c_block + 5 * ldc + 1 * vec_len);
    hn::StoreU(c06, d, c_block + 6 * ldc + 0 * vec_len); hn::StoreU(c16, d, c_block + 6 * ldc + 1 * vec_len);
    hn::StoreU(c07, d, c_block + 7 * ldc + 0 * vec_len); hn::StoreU(c17, d, c_block + 7 * ldc + 1 * vec_len);
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
    return hn::Lanes(d) * 2;
}

std::size_t GetLanesDouble() {
    const hn::ScalableTag<double> d;
    return hn::Lanes(d) * 2;
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
