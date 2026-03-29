import time
import numpy as np

try:
    import forway
except ImportError:
    print("FATAL: forway not found. Ensure PYTHONPATH includes the build directory natively.")
    exit(1)

def test_sort_dtype(dtype, elements=10_000_000):
    print(f"\nEvaluating In-Place Vectorized Quicksort AVX-512 Natively over {dtype.__name__}...")
    
    # Generate arrays based on types
    if np.issubdtype(dtype, np.floating):
        arr_forway = np.random.rand(elements).astype(dtype)
    else:
        arr_forway = np.random.randint(-1000000, 1000000, size=elements, dtype=dtype)
        
    arr_numpy = arr_forway.copy()

    # Benchmark ForWay vqsort AVX-512
    t0 = time.perf_counter_ns()
    forway.sort(arr_forway)
    t1 = time.perf_counter_ns()
    time_forway_ms = (t1 - t0) / 1e6

    # Benchmark NumPy sort (usually std::sort or introsort)
    t0 = time.perf_counter_ns()
    # In-place sort exactly mirroring the boundaries for fair tracking
    arr_numpy.sort() 
    t1 = time.perf_counter_ns()
    time_numpy_ms = (t1 - t0) / 1e6

    # Validation correctness structurally
    assert np.all(arr_forway == arr_numpy), f"[FAIL] ForWay sort output does not formally match NumPy natively over {dtype.__name__} bounds!"

    # Statistics natively
    elements_per_sec_fw = elements / (time_forway_ms / 1000.0)
    elements_per_sec_np = elements / (time_numpy_ms / 1000.0)

    print(f"  {elements:,} Elements | ForWay: {time_forway_ms:>6.2f} ms | NumPy: {time_numpy_ms:>6.2f} ms | Speedup: {time_numpy_ms / time_forway_ms:.2f}x")
    print(f"  Throughput         | ForWay: {elements_per_sec_fw/1e6:>6.2f} M/s | NumPy: {elements_per_sec_np/1e6:>6.2f} M/s")

if __name__ == "__main__":
    print("===========================================================================")
    print("ForWay Google Highway VQSort Evaluation Matrix")
    print("===========================================================================")
    
    types_to_test = [np.float32, np.float64, np.int32, np.int64]
    
    for dt in types_to_test:
        test_sort_dtype(dt, elements=50_000_000)
        
    print("\nAll correctness bounds passed successfully Native Pipeline Complete.")
