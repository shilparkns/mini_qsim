# python/bench.py
import argparse, csv, os, socket, subprocess, time, platform, json
from datetime import datetime
import numpy as np
from .circuit import Circuit

def numba_max_threads():
    try:
        from .apply_numba import get_threads, set_threads
        # get current pool size (Numba caps set_num_threads to this)
        cur = get_threads()
        # try to set back to cur (no-op) just to ensure import/init
        set_threads(cur)
        return cur
    except Exception:
        return 1

def warmup(circ):
    # one dry run to pay JIT cost before timing
    _ = circ.run(backend="numba", dtype=np.complex64, num_threads=numba_max_threads(), check_norm=False)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "results.csv")

def meta_row():
    commit = ""
    try:
        commit = subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
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

def ensure_header(path):
    header = ["qubits","depth","backend","threads","gates","wall_ms","hostname","commit","dtype","timestamp"]
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

def write_row(path, row):
    ensure_header(path)
    header = ["qubits","depth","backend","threads","gates","wall_ms","hostname","commit","dtype","timestamp"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow(row)

def random_circuit(n, depth, seed=0):
    rng = np.random.default_rng(seed)
    c = Circuit.empty(n)
    for _ in range(depth):
        # 1/2 single-qubit, 1/2 CNOTs
        if rng.random() < 0.5:
            k = int(rng.integers(0, n)); c.h(k) if rng.random()<0.5 else c.x(k)
        else:
            a = int(rng.integers(0, n))
            b = a
            while b == a:
                b = int(rng.integers(0, n))
            c.cnot(a, b)
    return c

def time_run(circ, backend, threads=None):
    t0 = time.perf_counter()
    _ = circ.run(backend=backend, dtype=np.complex64, num_threads=threads, check_norm=False)
    return (time.perf_counter() - t0) * 1e3  # ms

def bench_qubits(ns, depth, backend):
    for n in ns:
        circ = random_circuit(n, depth, seed=42)
        warmup(circ)
        wall = time_run(circ, backend)
        m = meta_row()
        write_row(RESULTS, {
            "qubits": n, "depth": depth, "backend": backend, "threads": (0 if backend=="serial" else os.cpu_count()),
            "gates": len(circ.ops), "wall_ms": f"{wall:.3f}",
            "hostname": m["hostname"], "commit": m["commit"], "dtype": m["dtype"], "timestamp": m["timestamp"]
        })
        print(f"[qubits] n={n} {backend} wall={wall:.1f} ms")

def bench_threads(n, depth, threads_list):
    from .apply_numba import set_threads
    circ = random_circuit(n, depth, seed=123)

    # Warm up JIT so t=1 isn’t polluted by compile time
    warmup(circ)

    max_t = numba_max_threads()
    for t in threads_list:
        t_eff = min(int(t), max_t)
        if t_eff != int(t):
            print(f"[threads] requested t={t} > pool={max_t}; using t={t_eff}")
        set_threads(t_eff)
        wall = time_run(circ, "numba", threads=t_eff)
        m = meta_row()
        write_row(RESULTS, {
            "qubits": n, "depth": depth, "backend": "numba", "threads": t_eff,
            "gates": len(circ.ops), "wall_ms": f"{wall:.3f}",
            "hostname": m["hostname"], "commit": m["commit"], "dtype": m["dtype"], "timestamp": m["timestamp"]
        })
        print(f"[threads] n={n} t={t_eff} wall={wall:.1f} ms")


def bench_depth(n, depths, backend="numba"):
    for d in depths:
        circ = random_circuit(n, d, seed=7)
        warmup(circ)
        wall = time_run(circ, backend)
        m = meta_row()
        write_row(RESULTS, {
            "qubits": n, "depth": d, "backend": backend, "threads": (0 if backend=="serial" else os.cpu_count()),
            "gates": len(circ.ops), "wall_ms": f"{wall:.3f}",
            "hostname": m["hostname"], "commit": m["commit"], "dtype": m["dtype"], "timestamp": m["timestamp"]
        })
        print(f"[depth] n={n} d={d} {backend} wall={wall:.1f} ms")

def main():
    p = argparse.ArgumentParser(description="mini_qsim benchmarks → data/results.csv")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_qubits = sub.add_parser("qubits")
    p_qubits.add_argument("--ns", type=str, required=True, help="comma list, e.g. 12,14,16")
    p_qubits.add_argument("--depth", type=int, default=100)
    p_qubits.add_argument("--backend", type=str, default="numba", choices=["serial","numba"])

    p_threads = sub.add_parser("threads")
    p_threads.add_argument("--n", type=int, default=16)
    p_threads.add_argument("--depth", type=int, default=200)
    p_threads.add_argument("--threads", type=str, default="1,2,4,8,16")

    p_depth = sub.add_parser("depth")
    p_depth.add_argument("--n", type=int, default=12)
    p_depth.add_argument("--depths", type=str, default="10,50,100,300,600")
    p_depth.add_argument("--backend", type=str, default="numba", choices=["serial","numba"])

    args = p.parse_args()
    if args.cmd == "qubits":
        ns = [int(x) for x in args.ns.split(",")]
        bench_qubits(ns, args.depth, args.backend)
    elif args.cmd == "threads":
        ts = [int(x) for x in args.threads.split(",")]
        bench_threads(args.n, args.depth, ts)
    elif args.cmd == "depth":
        ds = [int(x) for x in args.depths.split(",")]
        bench_depth(args.n, ds, args.backend)

if __name__ == "__main__":
    main()

