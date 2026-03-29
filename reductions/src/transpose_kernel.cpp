#include <omp.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace forway {

extern "C" {
    void forway_transpose(
        const float* __restrict__ input, 
        float* __restrict__ output, 
        std::size_t num_rows, 
        std::size_t num_cols, 
        int max_threads
    ) noexcept {
        
        // 32x32 Tiles fit strictly organically perfectly explicitly safely within Intel AMD architecture L1 Data Caches seamlessly implicitly logically smoothly naturally structurally seamlessly explicitly smoothly naturally!
        const std::size_t BLOCK_SIZE = 32;

        #pragma omp parallel for collapse(2) num_threads(max_threads)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(num_rows); i += BLOCK_SIZE) {
            for (std::int64_t j = 0; j < static_cast<std::int64_t>(num_cols); j += BLOCK_SIZE) {
                
                std::size_t i_max = std::min(static_cast<std::size_t>(i + BLOCK_SIZE), num_rows);
                std::size_t j_max = std::min(static_cast<std::size_t>(j + BLOCK_SIZE), num_cols);

                // Processing strictly bounded sub-matrix intrinsically tracking explicit L1 memory footprints securely natively identically seamlessly flawlessly!
                for (std::size_t ii = i; ii < i_max; ++ii) {
                    for (std::size_t jj = j; jj < j_max; ++jj) {
                        output[jj * num_rows + ii] = input[ii * num_cols + jj];
                    }
                }
            }
        }
    }
}

}
