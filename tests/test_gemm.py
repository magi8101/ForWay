import os
import sys
import time
import numpy as np

if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    mingw_paths = [
        'C:/ProgramData/mingw64/mingw64/bin',
        'C:/msys64/mingw64/bin',
        'C:/mingw64/bin',
    ]
    for path in mingw_paths:
        if os.path.isdir(path):
            os.add_dll_directory(path)
            break

try:
    import forway
except ImportError as e:
    print(f"Failed to import forway: {e}")
    print("Make sure you have built the module and it is in your Python path.")
    sys.exit(1)


def test_correctness_float32():
    print("Testing correctness (float32)...")

    sizes = [(4, 4, 4), (16, 16, 16), (64, 64, 64), (128, 256, 64), (7, 13, 11)]

    for M, N, K in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        C_expected = A @ B

        forway.gemm(A, B, C)

        if not np.allclose(C, C_expected, rtol=1e-4, atol=1e-4):
            max_diff = np.max(np.abs(C - C_expected))
            print(f"  FAIL: {M}x{N}x{K}, max diff = {max_diff}")
            return False
        print(f"  PASS: {M}x{N}x{K}")

    return True


def test_correctness_float64():
    print("Testing correctness (float64)...")

    sizes = [(4, 4, 4), (16, 16, 16), (64, 64, 64), (128, 256, 64), (7, 13, 11)]

    for M, N, K in sizes:
        A = np.random.randn(M, K).astype(np.float64)
        B = np.random.randn(K, N).astype(np.float64)
        C = np.zeros((M, N), dtype=np.float64)

        C_expected = A @ B

        forway.gemm(A, B, C)

        if not np.allclose(C, C_expected, rtol=1e-10, atol=1e-10):
            max_diff = np.max(np.abs(C - C_expected))
            print(f"  FAIL: {M}x{N}x{K}, max diff = {max_diff}")
            return False
        print(f"  PASS: {M}x{N}x{K}")

    return True


def benchmark(size, dtype, num_warmup=3, num_runs=10):
    M, N, K = size, size, size
    A = np.random.randn(M, K).astype(dtype)
    B = np.random.randn(K, N).astype(dtype)
    C = np.zeros((M, N), dtype=dtype)

    for _ in range(num_warmup):
        forway.gemm(A, B, C)
        _ = A @ B

    forway_times = []
    for _ in range(num_runs):
        C.fill(0)
        start = time.perf_counter()
        forway.gemm(A, B, C)
        end = time.perf_counter()
        forway_times.append(end - start)

    numpy_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = A @ B
        end = time.perf_counter()
        numpy_times.append(end - start)

    forway_avg = sum(forway_times) / len(forway_times)
    numpy_avg = sum(numpy_times) / len(numpy_times)

    flops = 2.0 * M * N * K
    forway_gflops = flops / forway_avg / 1e9
    numpy_gflops = flops / numpy_avg / 1e9

    return forway_avg, numpy_avg, forway_gflops, numpy_gflops


def run_benchmarks():
    print("\nBenchmarking against np.matmul...")
    print("-" * 70)

    sizes = [128, 256, 512, 1024, 2048, 4096]
    dtypes = [np.float32, np.float64]

    for dtype in dtypes:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n{dtype_name}:")
        print(f"{'Size':>8} | {'ForWay (ms)':>12} | {'NumPy (ms)':>12} | {'ForWay GFLOPS':>14} | {'NumPy GFLOPS':>14}")
        print("-" * 70)

        for size in sizes:
            try:
                fw_time, np_time, fw_gflops, np_gflops = benchmark(size, dtype)
                print(f"{size:>8} | {fw_time*1000:>12.3f} | {np_time*1000:>12.3f} | {fw_gflops:>14.2f} | {np_gflops:>14.2f}")
            except Exception as e:
                print(f"{size:>8} | ERROR: {e}")


def main():
    print("ForWay GEMM Test Suite")
    print("=" * 70)

    passed = True

    if not test_correctness_float32():
        passed = False
    if not test_correctness_float64():
        passed = False

    if passed:
        run_benchmarks()
        print("\nAll tests passed.")
    else:
        print("\nSome tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
