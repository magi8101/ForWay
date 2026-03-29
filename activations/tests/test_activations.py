import time
import numpy as np

try:
    from scipy.special import softmax as scipy_softmax
except ImportError:
    # Organic fallback perfectly emulating SciPy internals identically functionally!
    def scipy_softmax(x, axis=None):
        if axis is None:
            x = x.flatten()
            axis = 0
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

try:
    import forway
except ImportError:
    print("FATAL: forway native library directly missing. Provide PYTHONPATH to build tree limits securely!")
    exit(1)

def evaluate_activations():
    N_1D = 100_000_000
    print("===========================================================================")
    print("ForWay Neural Network Math Evaluation (Native AVX-512 FMA Bounds)")
    print("===========================================================================")
    print(f"Dataset 1D:   {N_1D:,} elements (Float32)")
    print("---------------------------------------------------------------------------")

    arr_1d = np.random.randn(N_1D).astype(np.float32)
    
    # --- EXP BENCHMARK ---
    # Warmup
    _ = forway.exp(arr_1d[:10], max_threads=8)
    
    t0 = time.perf_counter_ns()
    out_fw_exp = forway.exp(arr_1d, max_threads=8)
    t1 = time.perf_counter_ns()
    time_fw_exp = (t1 - t0) / 1e6
    
    t0 = time.perf_counter_ns()
    out_np_exp = np.exp(arr_1d)
    t1 = time.perf_counter_ns()
    time_np_exp = (t1 - t0) / 1e6
    
    np.testing.assert_allclose(out_fw_exp, out_np_exp, rtol=1e-4, atol=1e-5)
    print(f"Exp   | ForWay: {time_fw_exp:>6.2f} ms | NumPy: {time_np_exp:>6.2f} ms | Speedup: {time_np_exp / time_fw_exp:.2f}x")

    # --- TANH BENCHMARK ---
    t0 = time.perf_counter_ns()
    out_fw_tanh = forway.tanh(arr_1d, max_threads=8)
    t1 = time.perf_counter_ns()
    time_fw_tanh = (t1 - t0) / 1e6
    
    t0 = time.perf_counter_ns()
    out_np_tanh = np.tanh(arr_1d)
    t1 = time.perf_counter_ns()
    time_np_tanh = (t1 - t0) / 1e6
    
    np.testing.assert_allclose(out_fw_tanh, out_np_tanh, rtol=1e-4, atol=1e-5)
    print(f"Tanh  | ForWay: {time_fw_tanh:>6.2f} ms | NumPy: {time_np_tanh:>6.2f} ms | Speedup: {time_np_tanh / time_fw_tanh:.2f}x")

    # --- SOFTMAX BENCHMARK ---
    ROWS = 100_000
    COLS = 512
    print("---------------------------------------------------------------------------")
    print(f"Softmax 2D:   {ROWS:,} x {COLS} (Float32)")
    arr_2d = np.random.randn(ROWS, COLS).astype(np.float32)

    # Warmup
    _ = forway.softmax(arr_2d[:10, :], max_threads=8)

    t0 = time.perf_counter_ns()
    out_fw_sm = forway.softmax(arr_2d, max_threads=8)
    t1 = time.perf_counter_ns()
    time_fw_sm = (t1 - t0) / 1e6
    
    t0 = time.perf_counter_ns()
    out_np_sm = scipy_softmax(arr_2d, axis=1)
    t1 = time.perf_counter_ns()
    time_np_sm = (t1 - t0) / 1e6

    np.testing.assert_allclose(out_fw_sm, out_np_sm, rtol=1e-4, atol=1e-5)
    print(f"Sftmx | ForWay: {time_fw_sm:>6.2f} ms | NumPy: {time_np_sm:>6.2f} ms | Speedup: {time_np_sm / time_fw_sm:.2f}x")

    print("---------------------------------------------------------------------------")
    print("[SUCCESS] Pure Highway Mathematics execution completely functionally identical!")

if __name__ == "__main__":
    evaluate_activations()
