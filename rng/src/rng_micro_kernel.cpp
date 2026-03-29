#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rng_micro_kernel.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

HWY_BEFORE_NAMESPACE();

namespace forway {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <class V>
HWY_INLINE void QuarterRound(V& a, V& b, V& c, V& d) {
    a = hn::Add(a, b); d = hn::Xor(d, a); d = hn::RotateRight<16>(d);
    c = hn::Add(c, d); b = hn::Xor(b, c); b = hn::RotateRight<20>(b);
    a = hn::Add(a, b); d = hn::Xor(d, a); d = hn::RotateRight<24>(d);
    c = hn::Add(c, d); b = hn::Xor(b, c); b = hn::RotateRight<25>(b);
}

HWY_ATTR void ChaCha8MicroKernelImpl(
    float* HWY_RESTRICT output_array, 
    std::size_t num_elements, 
    uint64_t seed, 
    uint64_t start_counter
) {
    const hn::ScalableTag<uint32_t> d32;
    const hn::ScalableTag<float> d_f32;
    using V32 = hn::Vec<decltype(d32)>;
    using VF32 = hn::Vec<decltype(d_f32)>;
    
    // Explicit Dynamic Hardware Fetch
    const std::size_t lanes = hn::Lanes(d32);
    const std::size_t output_stride = 16 * lanes; 

    uint32_t seed_low = static_cast<uint32_t>(seed);
    uint32_t seed_high = static_cast<uint32_t>(seed >> 32);

    std::size_t p = 0;
    
    // Track Exact Global 64-bit bounds to safely execute over TB of generated limits.
    uint64_t current_counter = start_counter;

    while (p < num_elements) {
        // Broadcast constants INSIDE the loop. 
        // Bypassing exterior ZMM initialization completely drops Register Pressure!
        V32 s0 = hn::Set(d32, 0x61707865); V32 s1 = hn::Set(d32, 0x3320646e);
        V32 s2 = hn::Set(d32, 0x79622d32); V32 s3 = hn::Set(d32, 0x6b206574);
        
        V32 s4 = hn::Set(d32, seed_low);  V32 s5 = hn::Set(d32, seed_high);
        V32 s6 = hn::Set(d32, seed_low);  V32 s7 = hn::Set(d32, seed_high);
        V32 s8 = hn::Set(d32, seed_low);  V32 s9 = hn::Set(d32, seed_high);
        V32 s10 = hn::Set(d32, seed_low); V32 s11 = hn::Set(d32, seed_high);

        // Pre-calculating precise mathematical 12 / 13 bounds safely bypassing Vector 32-bit Carry loss.
        alignas(64) uint32_t counter_low[16];
        alignas(64) uint32_t counter_high[16];
        for (std::size_t i=0; i < lanes; i++) {
            uint64_t exact = current_counter + i;
            counter_low[i] = static_cast<uint32_t>(exact);
            counter_high[i] = static_cast<uint32_t>(exact >> 32);
        }
        
        V32 s12 = hn::LoadU(d32, counter_low); 
        V32 s13 = hn::LoadU(d32, counter_high); 
        
        V32 s14 = hn::Zero(d32); V32 s15 = hn::Zero(d32);

        // Standard ChaCha math bounds.
        for (int i = 0; i < 4; i++) {
            QuarterRound(s0, s4, s8, s12); QuarterRound(s1, s5, s9, s13);
            QuarterRound(s2, s6, s10, s14); QuarterRound(s3, s7, s11, s15);
            QuarterRound(s0, s5, s10, s15); QuarterRound(s1, s6, s11, s12);
            QuarterRound(s2, s7, s8, s13); QuarterRound(s3, s4, s9, s14);
        }

        s0 = hn::Add(s0, hn::Set(d32, 0x61707865)); s1 = hn::Add(s1, hn::Set(d32, 0x3320646e));
        s2 = hn::Add(s2, hn::Set(d32, 0x79622d32)); s3 = hn::Add(s3, hn::Set(d32, 0x6b206574));
        s4 = hn::Add(s4, hn::Set(d32, seed_low)); s5 = hn::Add(s5, hn::Set(d32, seed_high));
        s6 = hn::Add(s6, hn::Set(d32, seed_low)); s7 = hn::Add(s7, hn::Set(d32, seed_high));
        s8 = hn::Add(s8, hn::Set(d32, seed_low)); s9 = hn::Add(s9, hn::Set(d32, seed_high));
        s10 = hn::Add(s10, hn::Set(d32, seed_low)); s11 = hn::Add(s11, hn::Set(d32, seed_high));
        
        s12 = hn::Add(s12, hn::LoadU(d32, counter_low)); 
        s13 = hn::Add(s13, hn::LoadU(d32, counter_high));

        V32 float_one = hn::BitCast(d32, hn::Set(d_f32, 1.0f));
        
        // ARM SVE Fix: Vectors are 'sizeless' and cannot be elements of a standard C++ array.
        // We must unroll the conversion and store operations for s0...s15 individually.
        auto process = [&](auto s, std::size_t k, float* buf) {
             V32 mantissa = hn::ShiftRight<9>(s);
             V32 combined = hn::Or(float_one, mantissa);
             VF32 f = hn::Sub(hn::BitCast(d_f32, combined), hn::Set(d_f32, 1.0f));
             if (p + output_stride <= num_elements) {
                 hn::Stream(f, d_f32, output_array + p + k * lanes);
             } else {
                 hn::StoreU(f, d_f32, buf + k * lanes);
             }
        };

        alignas(64) float tmp_buffer[16 * 64]; 
        float* b = (p + output_stride <= num_elements) ? nullptr : tmp_buffer;

        process(s0, 0, b);   process(s1, 1, b);   process(s2, 2, b);   process(s3, 3, b);
        process(s4, 4, b);   process(s5, 5, b);   process(s6, 6, b);   process(s7, 7, b);
        process(s8, 8, b);   process(s9, 9, b);   process(s10, 10, b); process(s11, 11, b);
        process(s12, 12, b); process(s13, 13, b); process(s14, 14, b); process(s15, 15, b);

        if (p + output_stride > num_elements) {
            std::memcpy(output_array + p, tmp_buffer, (num_elements - p) * sizeof(float));
        }

        current_counter += lanes; // Advance the global counter by explicit Hardware Lanes calculated.
        p += output_stride;
    }
}

std::size_t RngGetLanesFloat() {
    const hn::ScalableTag<uint32_t> d32;
    return hn::Lanes(d32);
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace forway {
HWY_EXPORT(ChaCha8MicroKernelImpl);
HWY_EXPORT(RngGetLanesFloat);

extern "C" {
    void forway_chacha8_micro_kernel_float(
        float* output_array, 
        std::size_t num_elements, 
        uint64_t seed, 
        uint64_t start_counter
    ) noexcept {
        HWY_STATIC_DISPATCH(ChaCha8MicroKernelImpl)(output_array, num_elements, seed, start_counter);
    }

    std::size_t forway_rng_get_lanes_float() noexcept {
        return HWY_STATIC_DISPATCH(RngGetLanesFloat)();
    }
}
}
#endif
