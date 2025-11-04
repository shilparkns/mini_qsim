# python/apply_serial.py
import numpy as np
from .state import State

def apply_single_qubit(state: State, U2: np.ndarray, k: int):
    """Apply 2x2 gate U2 to qubit k (little-endian: bit k)."""
    n = state.n
    psi = state.psi
    assert U2.shape == (2,2)
    N = psi.shape[0]
    step = 1 << k
    block = step << 1
    # iterate blocks of size 2^(k+1), update pairs (i0, i1=i0+step)
    for base in range(0, N, block):
        for off in range(step):
            i0 = base + off
            i1 = i0 + step
            a0 = psi[i0]
            a1 = psi[i1]
            psi[i0] = U2[0,0]*a0 + U2[0,1]*a1
            psi[i1] = U2[1,0]*a0 + U2[1,1]*a1

def apply_two_qubit_4x4(state: State, U4: np.ndarray, k: int, l: int):
    """Apply 4x4 gate U4 to qubits k<l (little-endian)."""
    if k == l:
        raise ValueError("k and l must differ")
    if k > l:
        k, l = l, k
    assert U4.shape == (4,4)

    psi = state.psi
    N = psi.shape[0]
    mk = 1 << k
    ml = 1 << l
    # iterate over all basis indices and group into quads by zeroing k and l bits
    # i00 = base, i01 = base|mk, i10 = base|ml, i11 = base|mk|ml
    # loop over indices where bits k and l are 0
    # pattern repeats every 2^(l+1); within that, we scan pairs around mk and ml.
    for base in range(0, N, 1 << (l+1)):
        # iterate lower chunk [0, 2^l) in steps of 2^(k+1), updating pairs inside
        for chunk in range(0, 1 << l, 1 << (k+1)):
            for off in range(1 << k):
                i00 = base + chunk + off
                i01 = i00 | mk
                i10 = i00 | ml
                i11 = i00 | mk | ml
                a00, a01, a10, a11 = psi[i00], psi[i01], psi[i10], psi[i11]
                # 4x4 multiply
                psi[i00] = U4[0,0]*a00 + U4[0,1]*a01 + U4[0,2]*a10 + U4[0,3]*a11
                psi[i01] = U4[1,0]*a00 + U4[1,1]*a01 + U4[1,2]*a10 + U4[1,3]*a11
                psi[i10] = U4[2,0]*a00 + U4[2,1]*a01 + U4[2,2]*a10 + U4[2,3]*a11
                psi[i11] = U4[3,0]*a00 + U4[3,1]*a01 + U4[3,2]*a10 + U4[3,3]*a11

def apply_X(state: State, k: int):
    U = np.array([[0,1],[1,0]], dtype=state.dtype)
    apply_single_qubit(state, U, k)

def apply_H(state: State, k: int):
    s = np.sqrt(0.5).astype(np.float32 if state.dtype==np.complex64 else np.float64)
    U = np.array([[s,s],[s,-s]], dtype=state.dtype)
    apply_single_qubit(state, U, k)

def apply_CNOT(state: State, control: int, target: int):
    if control == target:
        raise ValueError("control and target must differ")
    # Only acts when control=1 → we can either use a 4x4 or a conditional flip.
    # Using conditional flip for speed in serial baseline:
    psi = state.psi
    N = psi.shape[0]
    mc = 1 << control
    mt = 1 << target
    if control < target:
        # iterate quads by zeroing control and target bits
        step_c = 1 << control
        step_t = 1 << target
        block_t = step_t << 1
        for base in range(0, N, block_t):
            # within each block, walk chunks that zero target bit
            for sub in range(0, 1 << target, step_c << 1):
                for off in range(step_c):
                    i00 = base + sub + off           # c=0,t=0
                    i01 = i00 | mt                    # c=0,t=1
                    i10 = i00 | mc                    # c=1,t=0
                    i11 = i10 | mt                    # c=1,t=1
                    # swap target when control=1 → swap i10 <-> i11
                    a10 = psi[i10]
                    psi[i10] = psi[i11]
                    psi[i11] = a10
    else:
        # symmetric; reuse general 4x4 for clarity
        U4 = np.eye(4, dtype=state.dtype)
        U4[2,2]=0; U4[3,3]=0; U4[2,3]=1; U4[3,2]=1
        apply_two_qubit_4x4(state, U4, target, control)  # ensure k<l inside
