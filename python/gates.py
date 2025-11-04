# python/gates.py
import numpy as np

def H(dtype=np.complex64) -> np.ndarray:
    s = np.sqrt(0.5).astype(dtype if dtype==np.float32 else np.float64)
    return np.array([[s, s],
                     [s, -s]], dtype=dtype)

def X(dtype=np.complex64) -> np.ndarray:
    return np.array([[0, 1],
                     [1, 0]], dtype=dtype)

def CNOT(dtype=np.complex64) -> np.ndarray:
    # 4x4 in order 00,01,10,11 (target is LSB if you use k=target in apply)
    mat = np.eye(4, dtype=dtype)
    # swap |10> <-> |11>
    mat[2,2] = 0; mat[3,3] = 0
    mat[2,3] = 1; mat[3,2] = 1
    return mat

# Optional later
def RZ(theta: float, dtype=np.complex64) -> np.ndarray:
    return np.array([[np.exp(-0.5j*theta), 0],
                     [0, np.exp(+0.5j*theta)]], dtype=dtype)

def RX(theta: float, dtype=np.complex64) -> np.ndarray:
    c = np.cos(theta/2.0)
    s = -1j*np.sin(theta/2.0)
    return np.array([[c, s],
                     [s, c]], dtype=dtype)
