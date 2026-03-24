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
    sys.exit(1)

def test_uniformity():
    print("Testing Statistical Uniformity (Float32)...")
    N = 10_000_000
    C = np.zeros(N, dtype=np.float32)
    forway.random_uniform(C, seed=42)
    
    mean = np.mean(C)
    var = np.var(C)
    
    # Expected mean = 0.5, variance = 1/12 (approx 0.08333333)
    print(f"  Mean:     {mean:.6f} (Expected: 0.500000)")
    print(f"  Variance: {var:.6f} (Expected: 0.083333)")
    
    if abs(mean - 0.5) > 1e-3 or abs(var - 1/12) > 1e-3:
        print("  FAIL: Distribution is not uniform.")
        return False
    print("  PASS: Distribution metrics align formally.")
    return True

def benchmark_rng(N, num_warmup=3, num_runs=10):
    C = np.zeros(N, dtype=np.float32)
    seed = 12345
    rng = np.random.default_rng(seed)

    for _ in range(num_warmup):
        forway.random_uniform(C, seed)
        _ = rng.random(N, dtype=np.float32)

    forway_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        forway.random_uniform(C, seed)
        end = time.perf_counter()
        forway_times.append(end - start)

    numpy_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = rng.random(N, dtype=np.float32)
        end = time.perf_counter()
        numpy_times.append(end - start)

    forway_avg = sum(forway_times) / len(forway_times)
    numpy_avg = sum(numpy_times) / len(numpy_times)

    # Calculate Gigabytes Per Second (GB/s)
    # N elements * 4 bytes per float32 = total bytes
    bytes_generated = N * 4
    forway_gbs = (bytes_generated / forway_avg) / 1e9
    numpy_gbs = (bytes_generated / numpy_avg) / 1e9

    return forway_avg, numpy_avg, forway_gbs, numpy_gbs

def run_benchmarks():
    print("\nBenchmarking PRNG execution bandwidth against np.random...")
    print("-" * 75)
    
    sizes = [1_000_000, 10_000_000, 50_000_000, 200_000_000] # Up to 800 MB Generation
    
    print(f"{'Elements':>12} | {'ForWay (ms)':>12} | {'NumPy (ms)':>12} | {'ForWay GB/s':>12} | {'NumPy GB/s':>12}")
    print("-" * 75)

    for size in sizes:
        try:
            fw_time, np_time, fw_gbs, np_gbs = benchmark_rng(size)
            print(f"{size:>12} | {fw_time*1000:>12.2f} | {np_time*1000:>12.2f} | {fw_gbs:>12.2f} | {np_gbs:>12.2f}")
        except Exception as e:
            print(f"{size:>12} | ERROR: {e}")

def main():
    print("ForWay ChaCha8 PRNG Integration Protocol")
    print("=" * 75)

    if test_uniformity():
        run_benchmarks()

if __name__ == "__main__":
    main()
