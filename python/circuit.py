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

    def run(self, backend:str="serial", dtype=np.complex64, check_norm=True) -> State:
        if backend != "serial":
            raise NotImplementedError("Only serial backend implemented at Step 1")
        st = State.zero(self.n, dtype=dtype)
        for name, args in self.ops:
            if name == "H":
                (k,) = args; apply_H(st, k)
            elif name == "X":
                (k,) = args; apply_X(st, k)
            elif name == "CNOT":
                c,t = args; apply_CNOT(st, c, t)
            elif name == "RZ":
                k,theta = args; U = G.RZ(theta, dtype=st.dtype); apply_single_qubit(st, U, k)
            elif name == "RX":
                k,theta = args; U = G.RX(theta, dtype=st.dtype); apply_single_qubit(st, U, k)
            else:
                raise ValueError(f"Unknown gate {name}")
        if check_norm:
            st.check_normalized(tol=1e-6 if dtype==np.complex64 else 1e-12)
        return st
