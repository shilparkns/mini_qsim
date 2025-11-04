# python/tests/test_correctness_small.py
import numpy as np
from python.circuit import Circuit

def almost(p, q, tol=1e-6):
    return np.allclose(p, q, atol=tol, rtol=0)

def probs(psi):
    return np.abs(psi)**2

def test_h_on_zero():
    c = Circuit.empty(1).h(0)
    st = c.run(dtype=np.complex64)
    p = probs(st.as_numpy())
    assert almost(p, np.array([0.5, 0.5], dtype=np.float32))

def test_x_flips():
    # |0> -> X -> |1>
    c = Circuit.empty(1).x(0)
    st = c.run(dtype=np.complex64)
    p = probs(st.as_numpy())
    assert almost(p, np.array([0.0, 1.0], dtype=np.float32))

def test_cnot_control_off_noop():
    # |00> --(CNOT c=1,t=0)--> stays |00>
    c = Circuit.empty(2).cnot(1,0)
    st = c.run(dtype=np.complex64)
    p = probs(st.as_numpy())
    expect = np.zeros(4, dtype=np.float32); expect[0]=1.0
    assert almost(p, expect)

def test_cnot_control_on_flips():
    # Prepare |10> by X on qubit 1 (control), then CNOT(1->0): |10> -> |11>
    c = Circuit.empty(2).x(1).cnot(1,0)
    st = c.run(dtype=np.complex64)
    p = probs(st.as_numpy())
    expect = np.zeros(4, dtype=np.float32); expect[3]=1.0
    assert almost(p, expect)

def test_normalization():
    c = Circuit.empty(2).h(0).h(1).cnot(1,0)
    st = c.run(dtype=np.complex64)
    n2 = float((st.as_numpy().conj()*st.as_numpy()).sum().real)
    assert abs(1.0 - n2) < 1e-6

