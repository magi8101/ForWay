# ForWay

ForWay is a matrix multiplication library written in Python, Fortran, and C++. It uses nanobind for zero-copy Python interoperability, Fortran for cache blocking via OpenMP, and Google Highway for SIMD micro-kernel execution.

## Architecture

The execution pipeline crosses three language boundaries.

### 1. Runtime call stack

User space (Python) -> FFI boundary (nanobind) -> Macro layer (Fortran) -> Micro layer (C++) -> CPU.

*   **main.py:** Imports the module and calls `forway.gemm(A, B, C)`.
*   **forway.cpp:** Extracts raw pointers from NumPy arrays. Enforces `nb::c_contig` for zero-copy semantics before calling the Fortran ABI.
*   **macro_kernel.f90:** Receives pointers via `iso_c_binding`. Spawns threads using OpenMP. Executes BLIS loops 1 through 5, performing cache blocking and packing sub-panels for L3, L2, and L1 caches. Calls the C++ micro-kernel.
*   **micro_kernel.cpp:** A single-threaded module that executes BLIS loop 6. It dynamically dispatches to the highest available SIMD instruction set at runtime. Loads L1 blocks into SIMD vectors via `hwy::N` and executes fused multiply-add operations.

### 2. Repository structure

*   `CMakeLists.txt`: Links C++ and Fortran. Manages Highway and nanobind dependencies.
*   `src/forway.cpp`: Python nanobind wrapper and ABI translation.
*   `src/macro_kernel.f90`: Fortran OpenMP cache-blocking.
*   `src/micro_kernel.cpp`: C++ Google Highway SIMD math.
*   `tests/test_gemm.py`: Tests against `np.matmul`.

### 3. Compiler pipeline

CMake links the binaries across languages and platforms.

1.  `g++` compiles `micro_kernel.cpp` via Google Highway into a static object. Standard `-O3` is applied. It uses Highway's dynamic dispatch instead of `-march=native` to support multiple architectures natively.
2.  `gfortran` compiles `macro_kernel.f90` with `-O3` and OpenMP support via CMake's `FindOpenMP` module.
3.  `g++` compiles `forway.cpp` into a shared module.
4.  CMake links the objects with `libstdc++` and OpenMP runtimes to resolve cross-language calls so Fortran can invoke the C++ micro-kernel symbol.

## Cross-platform details

*   **Memory alignment:** L1 blocks passed from Fortran to C++ must be explicitly aligned (e.g., 32/64 bytes) to prevent Highway SIMD faults. You need C interconnects or Fortran alignment directives.
*   **Strides:** The macro-kernel logic accounts for row-major input from NumPy arrays.
*   **SIMD portability:** Do not compile the micro-kernel with static architecture flags. Google Highway handles runtime detection. A single build scales across AVX2, AVX-512, or NEON.
*   **FFI mapping:** Wrap the micro-kernel in `extern "C"`. The Fortran interface uses `bind(C, name="...")` to prevent compiler name mangling discrepancies.
