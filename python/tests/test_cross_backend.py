# python/tests/test_cross_backend.py
import numpy as np
from python.circuit import Circuit

def max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))

def test_serial_vs_numba_small():
    # 3-qubit mixed circuit
    c = Circuit.empty(3).h(0).x(1).cnot(1,2).h(2).cnot(0,1).x(2)
    st_s = c.run(backend="serial", dtype=np.complex64)
    st_n = c.run(backend="numba", dtype=np.complex64, num_threads=4)
    d = max_abs_diff(st_s.as_numpy(), st_n.as_numpy())
    assert d < 1e-5

def test_random_circuits_match():
    rng = np.random.default_rng(123)
    n = 4
    for depth in (5, 10, 20):
        c = Circuit.empty(n)
        for _ in range(depth):
            g = rng.integers(0, 3)  # 0:H,1:X,2:CNOT
            if g == 0:
                c.h(int(rng.integers(0, n)))
            elif g == 1:
                c.x(int(rng.integers(0, n)))
            else:
                c1 = int(rng.integers(0, n))
                c2 = c1
                while c2 == c1:
                    c2 = int(rng.integers(0, n))
                c.cnot(c1, c2)
        s = c.run(backend="serial", dtype=np.complex64)
        t = c.run(backend="numba", dtype=np.complex64, num_threads=8)
        assert np.allclose(s.as_numpy(), t.as_numpy(), atol=1e-5, rtol=0)
