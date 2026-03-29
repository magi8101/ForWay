#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "sort_kernel.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort.h"

#include <cstddef>
#include <cstdint>

HWY_BEFORE_NAMESPACE();

namespace forway {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void VQSortFloatImpl(float* array, std::size_t num_elements) {
    hwy::HWY_NAMESPACE::VQSort(array, num_elements, hwy::SortAscending());
}

void VQSortDoubleImpl(double* array, std::size_t num_elements) {
    hwy::HWY_NAMESPACE::VQSort(array, num_elements, hwy::SortAscending());
}

void VQSortInt32Impl(int32_t* array, std::size_t num_elements) {
    hwy::HWY_NAMESPACE::VQSort(array, num_elements, hwy::SortAscending());
}

void VQSortInt64Impl(int64_t* array, std::size_t num_elements) {
    hwy::HWY_NAMESPACE::VQSort(array, num_elements, hwy::SortAscending());
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace forway {
HWY_EXPORT(VQSortFloatImpl);
HWY_EXPORT(VQSortDoubleImpl);
HWY_EXPORT(VQSortInt32Impl);
HWY_EXPORT(VQSortInt64Impl);

extern "C" {
    void forway_vqsort_float(float* array, std::size_t num_elements) noexcept {
        HWY_STATIC_DISPATCH(VQSortFloatImpl)(array, num_elements);
    }
    void forway_vqsort_double(double* array, std::size_t num_elements) noexcept {
        HWY_STATIC_DISPATCH(VQSortDoubleImpl)(array, num_elements);
    }
    void forway_vqsort_int32(int32_t* array, std::size_t num_elements) noexcept {
        HWY_STATIC_DISPATCH(VQSortInt32Impl)(array, num_elements);
    }
    void forway_vqsort_int64(int64_t* array, std::size_t num_elements) noexcept {
        HWY_STATIC_DISPATCH(VQSortInt64Impl)(array, num_elements);
    }
}
}
#endif
