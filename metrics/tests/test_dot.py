import time
import numpy as np

try:
    import forway
except ImportError:
    print("FATAL: forway native library missing.")
    exit(1)

def evaluate_dot():
    threads = 8

    # --- Vector-Vector Dot Product ---
    N_VV = 100_000_000
    print("===========================================================================")
    print("ForWay numpy.dot Evaluation (Native AVX-512 FMA Pipelined)")
    print("===========================================================================")
    print(f"Vector-Vector:  {N_VV:,} elements (Float32)")
    print("---------------------------------------------------------------------------")

    a = np.random.randn(N_VV).astype(np.float32)
    b = np.random.randn(N_VV).astype(np.float32)

    # Warmup
    _ = forway.dot(a[:1000], b[:1000])

    t0 = time.perf_counter_ns()
    fw_vv = forway.dot(a, b)
    t1 = time.perf_counter_ns()
    time_fw_vv = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    np_vv = np.dot(a, b)
    t1 = time.perf_counter_ns()
    time_np_vv = (t1 - t0) / 1e6

    np.testing.assert_allclose(fw_vv, np_vv, rtol=1e-3, atol=1e-2)
    print(f"V·V    | ForWay: {time_fw_vv:>6.2f} ms | NumPy: {time_np_vv:>6.2f} ms | Speedup: {time_np_vv / time_fw_vv:.2f}x")

    # --- Matrix-Vector Dot Product ---
    ROWS = 50_000
    COLS = 2048
    print(f"Matrix-Vector:  {ROWS:,} x {COLS:,} @ {COLS:,} (Float32)")
    print("---------------------------------------------------------------------------")

    A = np.random.randn(ROWS, COLS).astype(np.float32)
    x = np.random.randn(COLS).astype(np.float32)

    # Warmup
    _ = forway.dot(A[:100, :], x, max_threads=threads)

    t0 = time.perf_counter_ns()
    fw_mv = forway.dot(A, x, max_threads=threads)
    t1 = time.perf_counter_ns()
    time_fw_mv = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    np_mv = np.dot(A, x)
    t1 = time.perf_counter_ns()
    time_np_mv = (t1 - t0) / 1e6

    np.testing.assert_allclose(fw_mv, np_mv, rtol=1e-3, atol=1e-1)
    print(f"M×V    | ForWay: {time_fw_mv:>6.2f} ms | NumPy: {time_np_mv:>6.2f} ms | Speedup: {time_np_mv / time_fw_mv:.2f}x")

    print("---------------------------------------------------------------------------")
    print("[SUCCESS] Dot product outputs validated against NumPy baselines.")

if __name__ == "__main__":
    evaluate_dot()
