import os
import sys
import time
import numpy as np
import torch
from scipy import linalg
import importlib.util

# ---------------------------------------------------------------------------
# Load ForWay from build directory
# ---------------------------------------------------------------------------
def load_forway():
    """Load ForWay, either from an installed package (CI/Wheels) or the local build directory."""
    # 1. Try a standard import first (for CI/Installed wheels)
    try:
        import forway as fw
        # If it imports, check if we can actually call a function (triggers DLL loading)
        _ = fw.get_num_threads()
        return fw
    except (ImportError, AttributeError):
        pass

    # 2. Fallback to manual discovery (for local developer builds)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    build_dir = os.path.join(project_root, "build")
    
    # On Windows, Python 3.8+ restricts DLL search paths.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        added_paths = set()
        for search_root in [build_dir, project_root]:
            if os.path.isdir(search_root):
                for root, _, files in os.walk(search_root):
                    if any(f.lower().endswith(".dll") for f in files):
                        if root not in added_paths:
                            try:
                                os.add_dll_directory(root)
                                added_paths.add(root)
                            except: pass
        
        # Add common MinGW paths as a last resort
        for _mingw_path in [r"C:\ProgramData\mingw64\mingw64\bin", r"C:\mingw64\bin"]:
            if os.path.isdir(_mingw_path) and _mingw_path not in added_paths:
                try:
                    os.add_dll_directory(_mingw_path)
                    added_paths.add(_mingw_path)
                except: pass
    
    # Add project root parent to sys.path so we can import 'ForWay' or 'forway'
    parent_dir = os.path.dirname(project_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    if build_dir not in sys.path:
        sys.path.append(build_dir)
    
    try:
        import ForWay as fw
    except ImportError:
        import forway as fw
    return fw

def benchmark(name, func, *args, iterations=10, warmup=2):
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    
    # Synchronize if using torch (though we are on CPU, it's good practice)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start = time.perf_counter()
    for _ in range(iterations):
        _ = func(*args)
    end = time.perf_counter()
    
    avg_ms = ((end - start) / iterations) * 1000
    return avg_ms

def run_benchmarks():
    fw = load_forway()
    print(f"ForWay vs NumPy vs PyTorch vs SciPy")
    print("-" * 100)
    header = f"{'Operation':<25} | {'NumPy (ms)':>12} | {'ForWay (ms)':>12} | {'Torch CPU (ms)':>15} | {'SciPy (ms)':>12}"
    print(header)
    print("-" * 100)

    # 1. GEMM (1024 x 1024)
    N = 1024
    A_np = np.random.randn(N, N).astype(np.float32)
    B_np = np.random.randn(N, N).astype(np.float32)
    A_pt = torch.from_numpy(A_np)
    B_pt = torch.from_numpy(B_np)
    
    t_np = benchmark("GEMM", np.matmul, A_np, B_np)
    t_fw = benchmark("GEMM", fw.matmul, A_np, B_np)
    t_pt = benchmark("GEMM", torch.matmul, A_pt, B_pt)
    t_sp = benchmark("GEMM", linalg.blas.sgemm, 1.0, A_np, B_np) # Use BLAS directly for SciPy
    
    print(f"{'GEMM (1024x1024)':<25} | {t_np:>12.2f} | {t_fw:>12.2f} | {t_pt:>15.2f} | {t_sp:>12.2f}")

    # 2. Transpose (10000 x 5000)
    M, K = 10000, 5000
    M_np = np.random.randn(M, K).astype(np.float32)
    M_pt = torch.from_numpy(M_np)
    
    t_np = benchmark("Transpose", lambda x: x.T.copy(), M_np)
    t_fw = benchmark("Transpose", fw.transpose, M_np)
    t_pt = benchmark("Transpose", lambda x: x.t().contiguous(), M_pt)
    t_sp = benchmark("Transpose", lambda x: x.T.copy(), M_np) # SciPy uses NumPy for this usually
    
    print(f"{'Transpose (10k x 5k)':<25} | {t_np:>12.2f} | {t_fw:>12.2f} | {t_pt:>15.2f} | {t_sp:>12.2f}")

    # 3. Dot Product (100M elements)
    V_size = 100_000_000
    v1_np = np.random.randn(V_size).astype(np.float32)
    v2_np = np.random.randn(V_size).astype(np.float32)
    v1_pt = torch.from_numpy(v1_np)
    v2_pt = torch.from_numpy(v2_np)
    
    t_np = benchmark("Dot", np.dot, v1_np, v2_np)
    t_fw = benchmark("Dot", fw.dot, v1_np, v2_np)
    t_pt = benchmark("Dot", torch.dot, v1_pt, v2_pt)
    t_sp = benchmark("Dot", linalg.blas.sdot, v1_np, v2_np)
    
    print(f"{'Dot (100M)':<25} | {t_np:>12.2f} | {t_fw:>12.2f} | {t_pt:>15.2f} | {t_sp:>12.2f}")

    # 4. Softmax (10000 x 512)
    S_rows, S_cols = 10000, 512
    S_np = np.random.randn(S_rows, S_cols).astype(np.float32)
    S_pt = torch.from_numpy(S_np)
    
    def np_softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    t_np = benchmark("Softmax", np_softmax, S_np)
    t_fw = benchmark("Softmax", lambda x: fw.softmax(x, max_threads=16), S_np)
    t_pt = benchmark("Softmax", lambda x: torch.softmax(x, dim=1), S_pt)
    
    print(f"{'Softmax (10k x 512)':<25} | {t_np:>12.2f} | {t_fw:>12.2f} | {t_pt:>15.2f} | {'N/A':>12}")

    # 5. Cosine Similarity (1 vs 100,000)
    C_dim = 512
    C_num = 100_000
    query = np.random.randn(C_dim).astype(np.float32)
    db = np.random.randn(C_num, C_dim).astype(np.float32)
    query_pt = torch.from_numpy(query)
    db_pt = torch.from_numpy(db)

    def np_cosine(q, d):
        q_norm = np.linalg.norm(q)
        d_norm = np.linalg.norm(d, axis=1)
        return np.dot(d, q) / (q_norm * d_norm)

    t_np = benchmark("Cosine Sim", np_cosine, query, db)
    t_fw = benchmark("Cosine Sim", fw.cosine_similarity, query, db)
    t_pt = benchmark("Cosine Sim", lambda q, d: torch.nn.functional.cosine_similarity(q.unsqueeze(0), d), query_pt, db_pt)
    
    print(f"{'Cosine Sim (100k)':<25} | {t_np:>12.2f} | {t_fw:>12.2f} | {t_pt:>15.2f} | {'N/A':>12}")

    # 6. Vectorized Sort (50M elements)
    Sort_size = 50_000_000
    arr_sort_np = np.random.rand(Sort_size).astype(np.float32)
    arr_sort_fw = arr_sort_np.copy()
    arr_sort_pt = torch.from_numpy(arr_sort_np.copy())

    t_np = benchmark("Sort", np.sort, arr_sort_np)
    t_fw = benchmark("Sort", fw.sort, arr_sort_fw) # In-place
    t_pt = benchmark("Sort", torch.sort, arr_sort_pt)
    
    print(f"{'Sort (50M)':<25} | {t_np:>12.2f} | {t_fw:>12.2f} | {t_pt:>15.2f} | {'N/A':>12}")

    # 7. Big Reductions (500M elements)
    Big_V = 500_000_000
    arr_big = np.random.rand(Big_V).astype(np.float32)
    
    t_np = benchmark("Sum", np.sum, arr_big)
    t_fw = benchmark("Sum", fw.sum, arr_big)
    t_pt = benchmark("Sum", torch.sum, torch.from_numpy(arr_big))
    
    print(f"{'Sum (500M)':<25} | {t_np:>12.2f} | {t_fw:>12.2f} | {t_pt:>15.2f} | {'N/A':>12}")

    # 8. Huge GEMM (2048x2048) - 4096 might be too slow for iterations
    Huge_N = 2048
    H_A = np.random.randn(Huge_N, Huge_N).astype(np.float32)
    H_B = np.random.randn(Huge_N, Huge_N).astype(np.float32)
    
    t_np = benchmark("Huge GEMM", np.matmul, H_A, H_B)
    t_fw = benchmark("Huge GEMM", fw.matmul, H_A, H_B)
    t_pt = benchmark("Huge GEMM", torch.matmul, torch.from_numpy(H_A), torch.from_numpy(H_B))
    
    print(f"{'GEMM (2048x2048)':<25} | {t_np:>12.2f} | {t_fw:>12.2f} | {t_pt:>15.2f} | {'N/A':>12}")

    print("-" * 100)
    print("\nThread Scaling Analysis (ForWay Sum 100M)")
    print("-" * 40)
    print(f"{'Threads':<10} | {'Time (ms)':>10} | {'Speedup':>10}")
    print("-" * 40)
    
    V_scale = 100_000_000
    v_scale = np.random.randn(V_scale).astype(np.float32)
    base_time = benchmark("Scale", fw.sum, v_scale, iterations=5, warmup=1) # 1 thread default? No, need to pass
    
    t1 = benchmark("T1", lambda x: fw.sum(x, max_threads=1), v_scale, iterations=5)
    print(f"{'1':<10} | {t1:>10.2f} | {1.00:>10.2f}x")
    
    for th in [2, 4, 8, 16]:
        t = benchmark(f"T{th}", lambda x: fw.sum(x, max_threads=th), v_scale, iterations=5)
        print(f"{th:<10} | {t:>10.2f} | {t1/t:>10.2f}x")
    
    print("-" * 40)

if __name__ == "__main__":
    run_benchmarks()
