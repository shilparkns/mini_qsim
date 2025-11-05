# python/circuit.py
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import numpy as np
from .state import State
from .apply_serial import apply_H, apply_X, apply_CNOT, apply_single_qubit, apply_two_qubit_4x4
from . import gates as G

Op = Tuple[str, Tuple]  # e.g., ("H",(k,)) or ("CNOT",(c,t)) or ("RZ",(k,theta))

@dataclass
class Circuit:
    n: int
    ops: List[Op]

    @staticmethod
    def empty(n:int) -> "Circuit":
        return Circuit(n, [])

    def h(self, k:int): self.ops.append(("H",(k,))); return self
    def x(self, k:int): self.ops.append(("X",(k,))); return self
    def cnot(self, c:int, t:int): self.ops.append(("CNOT",(c,t))); return self
    def rz(self, k:int, theta:float): self.ops.append(("RZ",(k,theta))); return self  # optional
    def rx(self, k:int, theta:float): self.ops.append(("RX",(k,theta))); return self  # optional

    def run(self, backend:str="serial", dtype=np.complex64, check_norm=True, num_threads=None, check_norm_tol=None) -> State:
        st = State.zero(self.n, dtype=dtype)

        if backend == "serial":
            from .apply_serial import apply_H, apply_X, apply_CNOT, apply_single_qubit
            import numpy as np
            from . import gates as G
            ap_H, ap_X, ap_CNOT, ap_1q = apply_H, apply_X, apply_CNOT, apply_single_qubit
            RZ = lambda k,th: ap_1q(st, G.RZ(th, dtype=st.dtype), k)
            RX = lambda k,th: ap_1q(st, G.RX(th, dtype=st.dtype), k)

        elif backend == "numba":
            try:
                from .apply_numba import apply_H, apply_X, apply_CNOT, apply_single_qubit, set_threads
                import numpy as np
                from . import gates as G
            except Exception as e:
                raise RuntimeError("Numba backend not available. Did you `pip install numba`?") from e
            if num_threads is not None:
                set_threads(int(num_threads))
            ap_H, ap_X, ap_CNOT, ap_1q = apply_H, apply_X, apply_CNOT, apply_single_qubit
            RZ = lambda k,th: ap_1q(st, G.RZ(th, dtype=st.dtype), k)
            RX = lambda k,th: ap_1q(st, G.RX(th, dtype=st.dtype), k)

        else:
            raise NotImplementedError(f"Unknown backend: {backend}")

        for name, args in self.ops:
            if name == "H":
                (k,) = args; ap_H(st, k)
            elif name == "X":
                (k,) = args; ap_X(st, k)
            elif name == "CNOT":
                c,t = args; ap_CNOT(st, c, t)
            elif name == "RZ":
                k,theta = args; RZ(k, theta)
            elif name == "RX":
                k,theta = args; RX(k, theta)
            else:
                raise ValueError(f"Unknown gate {name}")

        if check_norm:
            st.check_normalized(tol=check_norm_tol)
        return st

