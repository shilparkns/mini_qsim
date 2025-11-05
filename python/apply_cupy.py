# python/apply_cupy.py
# import cupy as cp
import numpy as np
from .state import State

# ----------------------------- utils -----------------------------

def to_device(state: State) -> cp.ndarray:
    """Ensure state.psi lives on the GPU (CuPy ndarray) and return it."""
    if isinstance(state.psi, cp.ndarray):
        return state.psi
    state.psi = cp.asarray(state.psi)  # in-place swap to device
    return state.psi

def to_host(state: State):
    """Bring state.psi back to host as NumPy ndarray (used by circuit.run)."""
    if isinstance(state.psi, cp.ndarray):
        state.psi = cp.asnumpy(state.psi)

def _H(dtype):
    """2x2 Hadamard on device with correct dtype."""
    f = cp.float32 if dtype == cp.complex64 else cp.float64
    s = f(1.0 / np.sqrt(2.0))
    return cp.array([[s, s], [s, -s]], dtype=dtype)

def _X(dtype):
    """2x2 Pauli-X on device with correct dtype."""
    return cp.array([[0, 1], [1, 0]], dtype=dtype)

# -------------------------- core kernels --------------------------

def apply_single_qubit(state: State, U2_host, k: int):
    """
    Apply a single-qubit 2x2 gate U2 to qubit k (little-endian, k=0 is LSB).
    Reshape-based kernel (no giant index arrays), in-place on GPU.
    """
    psi = to_device(state)
    U2 = cp.asarray(U2_host, dtype=psi.dtype)

    n = state.n
    # View psi as (right, 2, left) where the middle axis is qubit k
    left  = 1 << k
    right = 1 << (n - k - 1)
    psi3 = psi.reshape(right, 2, left)

    # out[r, a, l] = sum_b U2[a,b] * psi3[r, b, l]
    psi3[:] = cp.einsum('ab,rbl->ral', U2, psi3, optimize=True)

def apply_two_qubit_4x4(state: State, U4_host, k: int, l: int):
    """
    Apply a two-qubit 4x4 gate U4 to qubits k,l (little-endian).
    Basis order for U4 is |00>,|01>,|10>,|11|.
    Reshape-based kernel, in-place on GPU.
    """
    if k == l:
        raise ValueError("k and l must differ")
    if k > l:
        k, l = l, k

    psi = to_device(state)
    U4 = cp.asarray(U4_host, dtype=psi.dtype)

    n = state.n
    # Reshape psi into (outer, 2, mid, 2, inner) where the two '2' axes are k and l
    inner = 1 << k
    mid   = 1 << (l - k - 1)
    outer = 1 << (n - l - 1)
    psi5 = psi.reshape(outer, 2, mid, 2, inner)

    # Reshape U4 (4x4) -> (2,2,2,2): [ab,cd] with (a,b) output axes, (c,d) input axes
    U = U4.reshape(2, 2, 2, 2)

    # out[o, a, m, b, i] = sum_{c,d} U[a,b,c,d] * psi[o, c, m, d, i]
    psi5[:] = cp.einsum('abcd,ocmdi->oabmi', U, psi5, optimize=True)

# -------------------------- convenience --------------------------

def apply_H(state: State, k: int):
    """Hadamard on qubit k."""
    dtype = to_device(state).dtype
    apply_single_qubit(state, _H(dtype), k)

def apply_X(state: State, k: int):
    """Pauli-X on qubit k."""
    dtype = to_device(state).dtype
    apply_single_qubit(state, _X(dtype), k)

def apply_CNOT(state: State, control: int, target: int):
    """
    CNOT(control -> target) via a 4x4 matrix on the two target axes.
    Basis order: |00|01|10|11>, flip target when control=1.
    """
    psi = to_device(state)
    U4 = cp.eye(4, dtype=psi.dtype)
    # swap 10 <-> 11 rows/cols
    U4[2, 2] = 0; U4[3, 3] = 0
    U4[2, 3] = 1; U4[3, 2] = 1
    apply_two_qubit_4x4(state, U4, control, target)
