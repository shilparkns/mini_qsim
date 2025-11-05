# python/bench.py
import argparse, csv, os, socket, subprocess, time, platform
from datetime import datetime
import numpy as np
from .circuit import Circuit

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def backend_dir(backend):
    path = os.path.join(DATA_DIR, backend)
    os.makedirs(path, exist_ok=True)
    return path

def warmup(circ, backend, threads=None):
    # one dummy run to JIT-compile & warm caches; no norm check
    _ = circ.run(backend=backend, dtype=np.complex64, num_threads=threads, check_norm=False)

# ---------------------------------------------------------------------

def meta_row():
    commit = ""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    return {
        "hostname": socket.gethostname(),
        "commit": commit,
        "dtype": "complex64",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
    }

HEADER = ["qubits","depth","backend","threads","gates","wall_ms","hostname","commit","dtype","timestamp"]

def new_csv(path):
    """Create/overwrite CSV with header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADER).writeheader()

def write_row(path, row):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADER).writerow(row)

# ---------------------------------------------------------------------

def random_circuit(n, depth, seed=0):
    rng = np.random.default_rng(seed)
    c = Circuit.empty(n)
    for layer in range(depth):
        if layer % 2 == 0:
            for k in range(n):
                if rng.integers(0, 2) == 0:
                    c.h(k)
                else:
                    c.x(k)
        else:
            for k in range(0, n-1, 2):
                if rng.integers(0, 2) == 0:
                    c.cnot(k, k+1)
                else:
                    c.cnot(k+1, k)
    return c

def time_run(circ, backend, threads=None):
    t0 = time.perf_counter()
    _ = circ.run(backend=backend, dtype=np.complex64, num_threads=threads, check_norm=False)
    return (time.perf_counter() - t0) * 1e3  # ms

def numba_max_threads():
    try:
        from .apply_numba import get_num_threads, set_threads
        set_threads(1_000_000)
        return get_num_threads()
    except Exception:
        return os.cpu_count() or 1

# ---------------------------------------------------------------------
# individual experiments

def bench_qubits(ns, depth, backend, out_path):
    print(f"[run] Qubits scaling → {out_path}")
    new_csv(out_path)
    did_warmup = False
    for n in ns:
        circ = random_circuit(n, depth, seed=42)
        if not did_warmup:
            warmup(circ, backend=backend)   # <-- warmup here
            did_warmup = True
        wall = time_run(circ, backend)
        m = meta_row()
        write_row(out_path, {
            "qubits": n, "depth": depth, "backend": backend, "threads": 0 if backend=="serial" else numba_max_threads(),
            "gates": len(circ.ops), "wall_ms": f"{wall:.3f}",
            "hostname": m["hostname"], "commit": m["commit"], "dtype": m["dtype"], "timestamp": m["timestamp"]
        })
        print(f"  n={n}  wall={wall:.2f} ms")
    print("✓ done.\n")

def bench_threads(n, depth, threads_list, out_path):
    print(f"[run] Thread scaling → {out_path}")
    new_csv(out_path)
    from .apply_numba import set_threads
    circ = random_circuit(n, depth, seed=123)
    set_threads(1)
    t1 = time_run(circ, "numba", threads=1)
    pool = numba_max_threads()
    print(f"  pool={pool}  T1={t1:.1f} ms")

    for t in threads_list:
        tt = min(int(t), pool)
        if tt != t:
            print(f"  requested t={t} > pool={pool}; using t={tt}")
        set_threads(tt)
        wall = time_run(circ, "numba", threads=tt)
        speedup = t1 / wall if wall > 0 else float("nan")
        m = meta_row()
        write_row(out_path, {
            "qubits": n, "depth": depth, "backend": "numba", "threads": tt,
            "gates": len(circ.ops), "wall_ms": f"{wall:.3f}",
            "hostname": m["hostname"], "commit": m["commit"], "dtype": m["dtype"], "timestamp": m["timestamp"]
        })
        print(f"  t={tt}  wall={wall:.2f} ms  speedup={speedup:.2f}×")
    print("✓ done.\n")

def bench_depth(n, depths, backend, out_path):
    print(f"[run] Depth scaling → {out_path}")
    new_csv(out_path)
    d0 = min(depths)
    circ0 = random_circuit(n, d0, seed=7)
    warmup(circ0, backend=backend)   # warmup once

    for d in depths:
        circ = random_circuit(n, d, seed=7)
        wall = time_run(circ, backend)
        m = meta_row()
        write_row(out_path, {
            "qubits": n, "depth": d, "backend": backend, "threads": 0 if backend=="serial" else numba_max_threads(),
            "gates": len(circ.ops), "wall_ms": f"{wall:.3f}",
            "hostname": m["hostname"], "commit": m["commit"], "dtype": m["dtype"], "timestamp": m["timestamp"]
        })
        print(f"  depth={d}  wall={wall:.2f} ms")
    print("✓ done.\n")

# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="mini_qsim benchmarks → data/<backend>/*.csv (auto)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_qubits = sub.add_parser("qubits")
    p_qubits.add_argument("--ns", type=str, required=True)
    p_qubits.add_argument("--depth", type=int, default=100)
    p_qubits.add_argument("--backend", type=str, default="numba", choices=["serial","numba"])

    p_threads = sub.add_parser("threads")
    p_threads.add_argument("--n", type=int, default=16)
    p_threads.add_argument("--depth", type=int, default=200)
    p_threads.add_argument("--threads", type=str, default="1,2,4,8,16")
    # threads always use numba backend
    p_threads.add_argument("--backend", type=str, default="numba", choices=["numba"])

    p_depth = sub.add_parser("depth")
    p_depth.add_argument("--n", type=int, default=12)
    p_depth.add_argument("--depths", type=str, default="10,50,100,300,600")
    p_depth.add_argument("--backend", type=str, default="numba", choices=["serial","numba"])

    args = p.parse_args()

    base = backend_dir(args.backend)

    if args.cmd == "qubits":
        ns = [int(x) for x in args.ns.split(",")]
        out_path = os.path.join(base, "qubits.csv")
        bench_qubits(ns, args.depth, args.backend, out_path)

    elif args.cmd == "threads":
        ts = [int(x) for x in args.threads.split(",")]
        out_path = os.path.join(base, "threads.csv")
        bench_threads(args.n, args.depth, ts, out_path)

    elif args.cmd == "depth":
        ds = [int(x) for x in args.depths.split(",")]
        out_path = os.path.join(base, "depth.csv")
        bench_depth(args.n, ds, args.backend, out_path)

if __name__ == "__main__":
    main()
