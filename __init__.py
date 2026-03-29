"""
ForWay — High-Performance Numerical Computing Engine
=====================================================

Drop-in accelerated replacements for common NumPy operations,
powered by Google Highway SIMD and OpenMP multi-threading.

Usage::

    import forway as fw
    
    a = fw.rand(1_000_000)          # ChaCha8 PRNG
    b = fw.rand(1_000_000)
    
    result = fw.dot(a, b)           # Pipelined FMA dot product
    fw.sort(a)                      # In-place vectorized quicksort
    
    M = fw.randn(1000, 512)
    s = fw.softmax(M)               # Fused row-wise softmax
    t = fw.transpose(M)             # Cache-blocked transposition
"""

import numpy as np
import importlib
import importlib.util
import os
import sys
import glob

# Find and load the compiled native module (.pyd / .so)
_this_dir = os.path.dirname(os.path.abspath(__file__))

def _find_native():
    """Find and return the path to the compiled native engine module."""
    # 1. Start with the directory containing __init__.py
    search_dirs = [_this_dir]
    
    # 2. Add build folder relative to repo root (for local development)
    repo_root = os.path.dirname(_this_dir) # If __init__.py is at root
    # Wait, e:\magi\ForWay\__init__.py means the repo root is _this_dir
    _build_path = os.path.join(_this_dir, "build")
    if os.path.isdir(_build_path):
        search_dirs.append(_build_path)
    
    # 3. Fallback to full sys.path
    search_dirs.extend([p for p in sys.path if os.path.isdir(p)])

    # Search recursively in all designated directories
    for d in search_dirs:
        for root, _, _ in os.walk(d):
            # Limit depth for sys.path to avoid massive slow-down
            if d != _this_dir and d != _build_path:
                if root.count(os.sep) - d.count(os.sep) > 1:
                    continue
            
            for ext in ('.pyd', '.so'):
                matches = glob.glob(os.path.join(root, f'forway*{ext}'))
                if matches:
                    return os.path.abspath(matches[0])
                    
    raise ImportError("Cannot find compiled ForWay native module (.pyd/.so)")

_native_path = _find_native()
_native_dir = os.path.dirname(_native_path)

# On Windows, Python 3.8+ restricts DLL search paths.
# We must explicitly add directories containing runtime DLLs (OpenMP, GCC, etc.)
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # 1. Recursively find and add any folders containing .dll in the build/pkg dirs
    _added = set()
    _search_roots = [_native_dir, _this_dir]
    if os.path.isdir(os.path.join(_this_dir, "build")):
        _search_roots.append(os.path.join(_this_dir, "build"))
        
    for _root_dir in _search_roots:
        for _root, _, _files in os.walk(_root_dir):
            if any(f.lower().endswith(".dll") for f in _files):
                if _root not in _added:
                    try:
                        os.add_dll_directory(_root)
                        _added.add(_root)
                    except: pass

    # 2. Fallback for common MinGW bin directories (for standalone dev builds)
    for _mingw_path in [
        r"C:\ProgramData\mingw64\mingw64\bin",
        r"C:\mingw64\bin",
        os.path.join(os.environ.get("MSYSTEM_PREFIX", ""), "bin"),
    ]:
        if os.path.isdir(_mingw_path) and _mingw_path not in _added:
            try:
                os.add_dll_directory(_mingw_path)
                _added.add(_mingw_path)
            except: pass

_spec = importlib.util.spec_from_file_location("forway", _native_path)
assert _spec is not None and _spec.loader is not None
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_THREADS = os.cpu_count()


def set_num_threads(n: int):
    """Set the default number of OpenMP threads for all ForWay operations."""
    global _DEFAULT_THREADS
    _DEFAULT_THREADS = max(1, min(n, 256))


def get_num_threads() -> int:
    """Get the current default thread count."""
    return _DEFAULT_THREADS


# ---------------------------------------------------------------------------
# Array Creation (numpy-style)
# ---------------------------------------------------------------------------

def array(data, dtype=np.float32) -> np.ndarray:
    """Create a C-contiguous NumPy array (ForWay-compatible).
    
    >>> fw.array([1, 2, 3])
    array([1., 2., 3.], dtype=float32)
    """
    return np.ascontiguousarray(data, dtype=dtype)


def zeros(shape, dtype=np.float32) -> np.ndarray:
    """Return a new array of given shape filled with zeros.
    
    >>> fw.zeros((3, 4))
    """
    return np.zeros(shape, dtype=dtype, order='C')


def ones(shape, dtype=np.float32) -> np.ndarray:
    """Return a new array of given shape filled with ones.
    
    >>> fw.ones(5)
    """
    return np.ones(shape, dtype=dtype, order='C')


def empty(shape, dtype=np.float32) -> np.ndarray:
    """Return a new uninitialized array of given shape.
    
    >>> fw.empty((1000, 512))
    """
    return np.empty(shape, dtype=dtype, order='C')


def rand(*shape, seed: int = 42) -> np.ndarray:
    """Uniform random floats in [0, 1) using ChaCha8 PRNG.
    
    >>> a = fw.rand(1_000_000)           # 1D
    >>> M = fw.rand(1000, 512)           # 2D (flattened then reshaped)
    """
    total = 1
    for s in shape:
        total *= s
    out = np.empty(total, dtype=np.float32)
    _native.random_uniform(out, seed)
    if len(shape) > 1:
        return out.reshape(shape)
    return out


def randn(*shape) -> np.ndarray:
    """Standard normal random floats (via NumPy, float32).
    
    >>> M = fw.randn(1000, 512)
    """
    return np.random.randn(*shape).astype(np.float32)


def arange(start, stop=None, step=1, dtype=np.float32) -> np.ndarray:
    """Return evenly spaced values within a given interval.
    
    >>> fw.arange(10)
    """
    return np.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, dtype=np.float32) -> np.ndarray:
    """Return evenly spaced numbers over a specified interval.
    
    >>> fw.linspace(0, 1, 100)
    """
    return np.linspace(start, stop, num, dtype=dtype)


# ---------------------------------------------------------------------------
# Ensure input is float32 C-contiguous
# ---------------------------------------------------------------------------

def _prep(x: np.ndarray) -> np.ndarray:
    """Ensure array is float32 and C-contiguous for the native backend."""
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    return x


# ---------------------------------------------------------------------------
# Linear Algebra
# ---------------------------------------------------------------------------

def dot(a, b, max_threads: int = None):
    """Dot product — works like ``numpy.dot``.
    
    - 1D · 1D → scalar  (pipelined FMA, multi-threaded)
    - 2D × 1D → 1D      (OpenMP row-parallel)
    - 2D × 2D → 2D      (BLIS-style tiled GEMM)
    
    >>> fw.dot(a, b)
    """
    t = max_threads or _DEFAULT_THREADS
    a, b = _prep(a), _prep(b)
    
    if a.ndim == 1 and b.ndim == 1:
        return _native.dot(a, b, max_threads=t)
    elif a.ndim == 2 and b.ndim == 1:
        return _native.dot(a, b, max_threads=t)
    elif a.ndim == 2 and b.ndim == 2:
        C = zeros((a.shape[0], b.shape[1]))
        _native.gemm(a, b, C)
        return C
    else:
        raise ValueError(f"Unsupported shapes for dot: {a.shape} @ {b.shape}")


def matmul(A, B):
    """Matrix multiplication — equivalent to ``A @ B``.
    
    >>> C = fw.matmul(A, B)
    """
    A, B = _prep(A), _prep(B)
    C = zeros((A.shape[0], B.shape[1]))
    _native.gemm(A, B, C)
    return C


def transpose(a, max_threads: int = None) -> np.ndarray:
    """Cache-blocked parallel matrix transposition.
    
    Unlike ``numpy.T`` (which returns a view), this returns a
    fully materialized C-contiguous copy — 6.7× faster than
    ``arr.T.copy()``.
    
    >>> T = fw.transpose(M)
    """
    t = max_threads or _DEFAULT_THREADS
    a = _prep(a)
    return _native.transpose(a, max_threads=t)


# ---------------------------------------------------------------------------
# Activations (element-wise math)
# ---------------------------------------------------------------------------

def exp(x, max_threads: int = None) -> np.ndarray:
    """Vectorized exponential using Highway polynomial approximation.
    
    >>> fw.exp(arr)
    """
    t = max_threads or _DEFAULT_THREADS
    x = _prep(x).ravel()
    return _native.exp(x, max_threads=t)


def tanh(x, max_threads: int = None) -> np.ndarray:
    """Vectorized hyperbolic tangent.
    
    >>> fw.tanh(arr)
    """
    t = max_threads or _DEFAULT_THREADS
    x = _prep(x).ravel()
    return _native.tanh(x, max_threads=t)


def softmax(x, max_threads: int = None) -> np.ndarray:
    """Fused row-wise softmax: ``exp(x - max) / sum(exp)``.
    
    Input must be 2D. Each row is independently normalized.
    
    >>> probs = fw.softmax(logits)
    """
    t = max_threads or _DEFAULT_THREADS
    x = _prep(x)
    if x.ndim != 2:
        raise ValueError(f"softmax requires a 2D array, got {x.ndim}D")
    return _native.softmax(x, max_threads=t)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def sum(x, max_threads: int = None) -> float:
    """Multi-threaded pipelined summation (44 GB/s throughput).
    
    >>> fw.sum(arr)
    """
    t = max_threads or _DEFAULT_THREADS
    x = _prep(x).ravel()
    return _native.sum(x, max_threads=t)


def max(x, max_threads: int = None) -> float:
    """Multi-threaded pipelined maximum.
    
    >>> fw.max(arr)
    """
    t = max_threads or _DEFAULT_THREADS
    x = _prep(x).ravel()
    return _native.max(x, max_threads=t)


def argmax(x, max_threads: int = None) -> int:
    """Multi-threaded index of maximum element.
    
    >>> idx = fw.argmax(arr)
    """
    t = max_threads or _DEFAULT_THREADS
    x = _prep(x).ravel()
    return _native.argmax(x, max_threads=t)


# ---------------------------------------------------------------------------
# Distance Metrics
# ---------------------------------------------------------------------------

def cosine_similarity(query, database, max_threads: int = None) -> np.ndarray:
    """Fused cosine similarity: one query vector vs N database vectors.
    
    Computes dot(q,v) / (||q|| * ||v||) in a single pass per row.
    
    Args:
        query:    1D array of shape (D,)
        database: 2D array of shape (N, D)
    
    Returns:
        1D array of shape (N,) with similarity scores
    
    >>> scores = fw.cosine_similarity(query, db_matrix)
    """
    t = max_threads or _DEFAULT_THREADS
    query, database = _prep(query), _prep(database)
    return _native.cosine_similarity(query, database, max_threads=t)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort(x) -> None:
    """In-place vectorized quicksort (Google Highway vqsort).
    
    Supports float32, float64, int32, int64.
    
    >>> fw.sort(arr)  # modifies arr in-place
    """
    if not x.flags['C_CONTIGUOUS']:
        raise ValueError("sort requires a C-contiguous array")
    _native.sort(x)


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

class _Random:
    """ForWay random number generation namespace."""
    
    @staticmethod
    def rand(*shape, seed: int = 42) -> np.ndarray:
        """Uniform random floats in [0, 1) via ChaCha8 PRNG (>40 GB/s).
        
        >>> fw.random.rand(1_000_000)
        """
        return rand(*shape, seed=seed)
    
    @staticmethod
    def randn(*shape) -> np.ndarray:
        """Standard normal random floats (float32).
        
        >>> fw.random.randn(1000, 512)
        """
        return randn(*shape)


random = _Random()


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

__version__ = "0.1.0"
__all__ = [
    # creation
    "array", "zeros", "ones", "empty", "rand", "randn", "arange", "linspace",
    # linear algebra
    "dot", "matmul", "transpose",
    # activations
    "exp", "tanh", "softmax",
    # reductions
    "sum", "max", "argmax",
    # distance
    "cosine_similarity",
    # sorting
    "sort",
    # random
    "random",
    # config
    "set_num_threads", "get_num_threads",
]
