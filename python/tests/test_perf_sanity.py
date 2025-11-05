# python/tests/test_perf_sanity.py
import time
from python.circuit import Circuit

def build_chain(n, depth):
    c = Circuit.empty(n)
    for _ in range(depth):
        for k in range(n):
            c.h(k)
        for k in range(0, n-1, 2):
            c.cnot(k, k+1)
    return c

def test_bench_runs_and_times():
    n, depth = 16, 5     # ~moderate but quick in CI/local
    c = build_chain(n, depth)
    t0 = time.perf_counter()
    s1 = c.run(backend="serial")
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    s2 = c.run(backend="numba", num_threads=8)
    t2 = time.perf_counter() - t0

    # correctness
    import numpy as np
    assert np.allclose(s1.as_numpy(), s2.as_numpy(), atol=1e-5, rtol=0)
    # sanity: both timings are positive
    assert t1 > 0 and t2 > 0
    # don't hard-assert speedup (machines vary); just ensure it isn't catastrophically slower
    assert t2 < 5.0 * t1

