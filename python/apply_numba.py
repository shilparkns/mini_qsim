# python/apply_numba.py
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
from .state import State

# ---------- low-level kernels (Numba JIT) ----------

@njit(parallel=True, fastmath=True)
def _single_qubit_kernel(psi, U2, k):
    N = psi.shape[0]
    step = 1 << k
    block = step << 1
    nblocks = N // block
    for b in prange(nblocks):
        base = b * block
        for off in range(step):
            i0 = base + off
            i1 = i0 + step
            a0 = psi[i0]
            a1 = psi[i1]
            psi[i0] = U2[0,0]*a0 + U2[0,1]*a1
            psi[i1] = U2[1,0]*a0 + U2[1,1]*a1

@njit(parallel=True, fastmath=True)
def _two_qubit_4x4_kernel(psi, U4, k, l):
    if k > l:
        k, l = l, k
    N = psi.shape[0]
    mk = 1 << k
    ml = 1 << l
    # Iterate only bases where bits k and l are 0 → disjoint quads.
    for i00 in prange(N):
        if (i00 & mk) == 0 and (i00 & ml) == 0:
            i01 = i00 | mk
            i10 = i00 | ml
            i11 = i00 | mk | ml
            a00 = psi[i00]; a01 = psi[i01]; a10 = psi[i10]; a11 = psi[i11]
            psi[i00] = U4[0,0]*a00 + U4[0,1]*a01 + U4[0,2]*a10 + U4[0,3]*a11
            psi[i01] = U4[1,0]*a00 + U4[1,1]*a01 + U4[1,2]*a10 + U4[1,3]*a11
            psi[i10] = U4[2,0]*a00 + U4[2,1]*a01 + U4[2,2]*a10 + U4[2,3]*a11
            psi[i11] = U4[3,0]*a00 + U4[3,1]*a01 + U4[3,2]*a10 + U4[3,3]*a11

@njit(parallel=True, fastmath=True)
def _cnot_kernel(psi, control, target):
    N = psi.shape[0]
    mc = 1 << control
    mt = 1 << target
    for base in prange(N):
        if (base & mc) == 0 and (base & mt) == 0:
            i10 = base | mc          # control=1, target=0
            i11 = i10 | mt           # control=1, target=1
            a10 = psi[i10]
            psi[i10] = psi[i11]
            psi[i11] = a10

# ---------- user-facing apply helpers ----------

def set_threads(n: int):
    set_num_threads(n)

def get_threads() -> int:
    return get_num_threads()

def apply_single_qubit(state: State, U2: np.ndarray, k: int):
    _single_qubit_kernel(state.psi, U2.astype(state.dtype), k)

def apply_two_qubit_4x4(state: State, U4: np.ndarray, k: int, l: int):
    _two_qubit_4x4_kernel(state.psi, U4.astype(state.dtype), k, l)

def apply_H(state: State, k: int):
    s = np.sqrt(0.5).astype(np.float32 if state.dtype==np.complex64 else np.float64)
    U = np.array([[s, s],[s, -s]], dtype=state.dtype)
    _single_qubit_kernel(state.psi, U, k)

def apply_X(state: State, k: int):
    U = np.array([[0,1],[1,0]], dtype=state.dtype)
    _single_qubit_kernel(state.psi, U, k)

def apply_CNOT(state: State, control: int, target: int):
    _cnot_kernel(state.psi, control, target)
