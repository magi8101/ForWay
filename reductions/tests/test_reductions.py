import time
import numpy as np

try:
    import forway
except ImportError:
    print("FATAL: forway native library missing. Provide PYTHONPATH to build tree limits securely!")
    exit(1)

def evaluate_reductions():
    N_1D = 100_000_000
    print("===========================================================================")
    print("ForWay Vector Reduction Evaluation (Native AVX-512 FMA Bounds)")
    print("===========================================================================")
    print(f"Dataset 1D:   {N_1D:,} elements (Float32)")
    print("---------------------------------------------------------------------------")

    arr_1d = np.random.randn(N_1D).astype(np.float32)
    threads = 8
    
    # --- SUM BENCHMARK ---
    _ = forway.sum(arr_1d[:10], max_threads=threads)
    
    t0 = time.perf_counter_ns()
    out_fw_sum = float(forway.sum(arr_1d, max_threads=threads))
    t1 = time.perf_counter_ns()
    time_fw_sum = (t1 - t0) / 1e6
    
    t0 = time.perf_counter_ns()
    out_np_sum = float(np.sum(arr_1d))
    t1 = time.perf_counter_ns()
    time_np_sum = (t1 - t0) / 1e6
    
    np.testing.assert_allclose(out_fw_sum, out_np_sum, rtol=1e-4, atol=1e-3)
    print(f"Sum    | ForWay: {time_fw_sum:>6.2f} ms | NumPy: {time_np_sum:>6.2f} ms | Speedup: {time_np_sum / time_fw_sum:.2f}x")

    # --- MAX BENCHMARK ---
    _ = forway.max(arr_1d[:10], max_threads=threads)
    
    t0 = time.perf_counter_ns()
    out_fw_max = float(forway.max(arr_1d, max_threads=threads))
    t1 = time.perf_counter_ns()
    time_fw_max = (t1 - t0) / 1e6
    
    t0 = time.perf_counter_ns()
    out_np_max = float(np.max(arr_1d))
    t1 = time.perf_counter_ns()
    time_np_max = (t1 - t0) / 1e6
    
    np.testing.assert_allclose(out_fw_max, out_np_max, rtol=1e-4, atol=1e-5)
    print(f"Max    | ForWay: {time_fw_max:>6.2f} ms | NumPy: {time_np_max:>6.2f} ms | Speedup: {time_np_max / time_fw_max:.2f}x")

    # --- ARGMAX BENCHMARK ---
    _ = forway.argmax(arr_1d[:10], max_threads=threads)
    
    t0 = time.perf_counter_ns()
    out_fw_idx = int(forway.argmax(arr_1d, max_threads=threads))
    t1 = time.perf_counter_ns()
    time_fw_idx = (t1 - t0) / 1e6
    
    t0 = time.perf_counter_ns()
    out_np_idx = int(np.argmax(arr_1d))
    t1 = time.perf_counter_ns()
    time_np_idx = (t1 - t0) / 1e6
    
    assert out_fw_idx == out_np_idx, f"Argmax mismatch! ForWay: {out_fw_idx}, NumPy: {out_np_idx}"
    print(f"Argmax | ForWay: {time_fw_idx:>6.2f} ms | NumPy: {time_np_idx:>6.2f} ms | Speedup: {time_np_idx / time_fw_idx:.2f}x")

    print("---------------------------------------------------------------------------")
    print("[SUCCESS] Pure Highway Matrix Reductions mathematically execution functionally identical!")

if __name__ == "__main__":
    evaluate_reductions()
