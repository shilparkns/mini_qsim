# mini_qsim (Step 1 — Serial Baseline)

**Purpose:**  
A small real state-vector simulator for testing parallel performance.

**Key points**
- Little-endian basis; qubit *k* → bit `(1 << k)`
- Gates: H, X, CNOT (optional later: RZ(θ), RX(θ))
- Default dtype: complex64 (fast); use complex128 for validation
- Implements only the Serial (NumPy) backend in Step 1

## 📦 File layout
```
mini_qsim/
  python/
    gates.py
    state.py
    apply_serial.py
    circuit.py
    tests/
      test_correctness_small.py
  data/
  README.md
```

## 🚀 Quickstart
```bash
# Run tests
python -m pytest -q
```

## ✅ Example
```python
from python.circuit import Circuit
c = Circuit.empty(2).h(0).cnot(0,1)   # Bell |Φ+⟩
st = c.run()
print(abs(st.as_numpy())**2)
# → [0.5 0. 0. 0.5]
```
