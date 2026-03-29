# ForWay

**High-performance numerical computing engine for Python.** Built on [Google Highway](https://github.com/google/highway) SIMD and OpenMP multi-threading, ForWay delivers vectorized operations that consistently outperform NumPy — achieving up to **14.9× speedups** on real workloads.

```python
import ForWay as fw

a = fw.rand(100_000_000, seed=42)     # ChaCha8 PRNG at 40+ GB/s
b = fw.rand(100_000_000, seed=7)

result = fw.dot(a, b)                  # Pipelined FMA dot product
fw.sort(a)                             # In-place vectorized quicksort

M = fw.randn(10_000, 512)
s = fw.softmax(M)                      # Fused 3-pass softmax
T = fw.transpose(M)                    # Cache-blocked transposition
```

## Benchmarks

All benchmarks measured on an 8-core (16-thread) system with DDR4 RAM.

| Operation | ForWay | NumPy | Speedup |
|---|---|---|---|
| Cosine Similarity (10K×512 DB) | 1.01 ms | 15.09 ms | **14.9×** |
| Softmax (10K×512 matrix) | 0.78 ms | 11.4 ms | **14.6×** |
| GEMM (1024×1024) | 32.8 ms | 289 ms | **8.8×** |
| Transpose (20K×10K) | 193 ms | 1308 ms | **6.76×** |
| Exp (100M elements) | 12.9 ms | 56.8 ms | **4.4×** |
| Sum (100M elements) | 9.8 ms | 31 ms | **3.2×** |
| Argmax (100M elements) | 9.6 ms | 24 ms | **2.5×** |
| Random Gen (100M floats) | 9.5 ms | 190 ms | **20×** |
| Dot V·V (100M elements) | 19.6 ms | 24.4 ms | **1.24×** |
| Sort (50M float32) | 1099 ms | 323 ms | 0.29× |

## API Reference

### Array Creation
```python
fw.array([1, 2, 3])              # From list → float32
fw.zeros((M, N))                 # Zero-filled
fw.ones(N)                       # Ones-filled
fw.empty((M, N))                 # Uninitialized
fw.rand(M, N, seed=42)           # Uniform [0,1) via ChaCha8
fw.randn(M, N)                   # Normal distribution
fw.arange(0, 100)                # Range
fw.linspace(0, 1, 1000)          # Linspace
```

### Linear Algebra
```python
fw.dot(a, b)                     # 1D·1D → scalar | 2D×1D → 1D | 2D×2D → 2D
fw.matmul(A, B)                  # Matrix multiply (BLIS-style tiled GEMM)
fw.transpose(M)                  # Cache-blocked parallel transposition
```

### Activations
```python
fw.exp(arr)                      # Vectorized exponential
fw.tanh(arr)                     # Vectorized hyperbolic tangent  
fw.softmax(logits_2d)            # Fused row-wise softmax
```

### Reductions
```python
fw.sum(arr)                      # Multi-threaded sum
fw.max(arr)                      # Multi-threaded max
fw.argmax(arr)                   # Multi-threaded argmax
```

### Distance Metrics
```python
fw.cosine_similarity(query, db)  # 1 vs N fused cosine similarity
```

### Sorting & Random
```python
fw.sort(arr)                     # In-place vectorized quicksort (vqsort)
fw.random.rand(N, seed=42)       # Namespace-style PRNG
```

### Configuration
```python
fw.set_num_threads(8)            # Set OpenMP thread count
fw.get_num_threads()             # Query current thread count
```

All functions default to `float32` and automatically handle dtype conversion and C-contiguity. Thread count defaults to `os.cpu_count()`.

## Architecture

```
Python (NumPy arrays)
  │
  ▼
nanobind FFI (zero-copy, nb::nogil)
  │
  ├──► Fortran macro-kernel (OpenMP cache-blocking, BLIS loops)
  │       └──► C++ micro-kernel (Google Highway SIMD, FMA)
  │
  ├──► C++ metrics kernel (fused dot/norm, software-pipelined FMA)
  ├──► C++ activations kernel (Highway polynomial math)
  ├──► C++ reductions kernel (OpenMP + SIMD reductions)
  └──► C++ transpose kernel (32×32 cache-blocked tiling)
```

**Key Design Decisions:**
- **Software Pipelining:** 4× accumulator unrolling hides FMA latency (4 cycles), keeping execution ports 100% saturated.
- **Fused Kernels:** Cosine similarity computes dot product + L2 norm in a single pass — one memory read instead of three.
- **OpenMP + nogil:** Python's GIL is released before entering parallel regions, enabling true multi-core execution.
- **Highway Dynamic Dispatch:** A single binary runs optimally on AVX2, AVX-512, and ARM NEON — no recompilation needed.

## Building from Source

### Requirements
- CMake ≥ 3.18
- C++17 compiler (GCC, Clang, or MSVC)
- Fortran compiler (gfortran)
- Python ≥ 3.9 with NumPy

### Build

```bash
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Run

```bash
export PYTHONPATH=build:$PYTHONPATH   # Linux/macOS
set PYTHONPATH=build;%PYTHONPATH%     # Windows

python -c "import ForWay as fw; print(fw.dot(fw.rand(100), fw.rand(100)))"
```

## Installing from PyPI

```bash
pip install forway
```

Pre-built wheels are available for:
- **Linux:** x86_64
- **macOS:** x86_64, arm64 (Apple Silicon)
- **Windows:** AMD64

## Project Structure

```
ForWay/
├── __init__.py                    # NumPy-style Python interface
├── src/
│   ├── forway.cpp                 # nanobind FFI bindings
│   ├── micro_kernel.cpp           # Highway SIMD GEMM micro-kernel
│   └── macro_kernel.f90           # Fortran OpenMP cache-blocking
├── metrics/src/
│   ├── metrics_kernel.cpp         # Fused cosine similarity
│   └── dot_kernel.cpp             # Pipelined dot product
├── activations/src/
│   └── activations_kernel.cpp     # Exp, Tanh, Softmax
├── reductions/src/
│   ├── reductions_kernel.cpp      # Sum, Max, Argmax
│   └── transpose_kernel.cpp       # Cache-blocked transposition
├── rng/src/
│   ├── rng_micro_kernel.cpp       # ChaCha8 Highway PRNG
│   └── rng_macro_kernel.f90       # OpenMP parallel RNG
├── CMakeLists.txt                 # Cross-platform build system
├── pyproject.toml                 # Python packaging (scikit-build-core)
└── .github/workflows/
    └── build_wheels.yml           # CI: multi-arch wheel builds
```

## License

MIT
