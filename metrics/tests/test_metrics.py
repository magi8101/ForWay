import time
import numpy as np

try:
    import forway
except ImportError:
    print("FATAL: forway native library directly missing. Provide PYTHONPATH to build tree limits securely!")
    exit(1)

def evaluate_cosine():
    # 1 Million Vectors, 512 dimensions (Simulating typical Machine Learning Embedding tables organically)
    N = 1_000_000
    D = 512
    # Setting explicitly to 8 cores as standard baseline limits natively
    threads = 8 

    print("===========================================================================")
    print("ForWay Vector DB Metric Engine Evaluation (Native AVX-512 FMA Bounds)")
    print("===========================================================================")
    print(f"Dataset:      {N:,} vectors")
    print(f"Dimensions:   {D} (Float32)")
    print(f"Threads (OMP): {threads}")
    print(f"Memory Footprint: {((N * D * 4) / 1024**2):.2f} MB")
    print("---------------------------------------------------------------------------")

    # Generate test limits
    query = np.random.rand(D).astype(np.float32)
    db_matrix = np.random.rand(N, D).astype(np.float32)

    # 1. Warm-up explicitly forcing page-fault bounds logic sequentially
    _ = forway.cosine_similarity(query, db_matrix[:10], max_threads=threads)

    # 2. Benchmark Native Fused AVX-512 Pipeline natively
    t0 = time.perf_counter_ns()
    out_forway = forway.cosine_similarity(query, db_matrix, max_threads=threads)
    t1 = time.perf_counter_ns()
    time_fw = (t1 - t0) / 1e6

    # 3. Benchmark NumPy heavily optimized generic algebraic routines linearly
    t0 = time.perf_counter_ns()
    query_norm = np.linalg.norm(query).astype(np.float32)
    db_norms = np.linalg.norm(db_matrix, axis=1).astype(np.float32)
    # Enforce safe 0 checks mapping
    db_norms = np.where(db_norms == 0, 1.0, db_norms)
    out_numpy = db_matrix.dot(query) / (query_norm * db_norms)
    t1 = time.perf_counter_ns()
    time_scipy = (t1 - t0) / 1e6

    # Assertion of Arithmetic Accuracy limits spanning across dynamically aggregated floats natively
    np.testing.assert_allclose(out_forway, out_numpy, rtol=1e-4, atol=1e-5)

    print(f"ForWay Fused Multi-Thread: {time_fw:>8.2f} ms")
    print(f"NumPy Multi-Pass Algebra:  {time_scipy:>8.2f} ms")
    print("---------------------------------------------------------------------------")
    print(f"Speedup:                   {time_scipy / time_fw:>8.2f}x")
    print("---------------------------------------------------------------------------")
    print("\n[SUCCESS] Native Algorithmic Correctness Validated Identically Against NumPy Baselines explicitly.")

if __name__ == "__main__":
    evaluate_cosine()
