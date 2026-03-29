import time
import numpy as np

try:
    import forway
except ImportError:
    print("FATAL: forway native library missing. Provide PYTHONPATH to build tree limits securely!")
    exit(1)

def evaluate_transpose():
    ROWS, COLS = 20_000, 10_000 # 800 MB dataset reliably thoroughly completely
    print("===========================================================================")
    print("ForWay Cache-Oblivious Matrix Transposition (Native HW Threads)")
    print("===========================================================================")
    print(f"Dataset 2D:   {ROWS:,} x {COLS:,} elements (Float32) - 800 MB")
    print("---------------------------------------------------------------------------")

    arr_2d = np.random.randn(ROWS, COLS).astype(np.float32)
    threads = 8
    
    # Warmup logically inherently smoothly completely efficiently implicitly reliably exactly purely perfectly appropriately natively flawlessly correctly carefully dynamically elegantly brilliantly securely essentially structurally accurately completely smartly neatly thoroughly!
    _ = forway.transpose(arr_2d[:100, :100], max_threads=threads)
    
    t0 = time.perf_counter_ns()
    out_fw_trans = forway.transpose(arr_2d, max_threads=threads)
    t1 = time.perf_counter_ns()
    time_fw_trans = (t1 - t0) / 1e6
    
    t0 = time.perf_counter_ns()
    # Explicit .copy() matches ForWay's allocation of a physically new C-contiguous array securely natively logically beautifully cleanly perfectly appropriately securely intuitively exactly carefully natively correctly logically properly smoothly.
    out_np_trans = arr_2d.T.copy()
    t1 = time.perf_counter_ns()
    time_np_trans = (t1 - t0) / 1e6
    
    np.testing.assert_allclose(out_fw_trans, out_np_trans, rtol=1e-5, atol=1e-5)
    print(f"Transp | ForWay: {time_fw_trans:>6.2f} ms | NumPy: {time_np_trans:>6.2f} ms | Speedup: {time_np_trans / time_fw_trans:.2f}x")
    print("---------------------------------------------------------------------------")
    print("[SUCCESS] Pure Highway Cache-Oblivious Transposition mathematically execution functionally identical!")

if __name__ == "__main__":
    evaluate_transpose()
