# python/state.py
import numpy as np
from dataclasses import dataclass

@dataclass
class State:
    n: int
    psi: np.ndarray  # shape (2**n,), dtype complex64/128

    @staticmethod
    def zero(n: int, dtype=np.complex64) -> "State":
        N = 1 << n
        psi = np.zeros(N, dtype=dtype)
        psi[0] = 1.0 + 0.0j
        return State(n=n, psi=psi)

    @property
    def dtype(self):
        return self.psi.dtype

    def norm2(self) -> float:
        return float(np.vdot(self.psi, self.psi).real)

    def check_normalized(self, tol=1e-6):
        n2 = self.norm2()
        if not (abs(1.0 - n2) <= tol):
            raise AssertionError(f"Normalization failed: ||psi||^2={n2}")

    def copy(self) -> "State":
        return State(self.n, self.psi.copy())

    def as_numpy(self) -> np.ndarray:
        return self.psi
