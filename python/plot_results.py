# python/plot_results.py
import csv, os, sys
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import median

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_rows(path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            row["qubits"]  = int(row["qubits"])
            row["depth"]   = int(row["depth"])
            row["threads"] = int(row["threads"])
            row["wall_ms"] = float(row["wall_ms"])
            rows.append(row)
    return rows

def median_by_key(rows, key_fields):
    buckets = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in key_fields)
        buckets[key].append(r["wall_ms"])
    agg = []
    for key, vals in buckets.items():
        out = dict(zip(key_fields, key))
        out["wall_ms"] = float(median(vals))
        agg.append(out)
    return agg

def plot_runtime_vs_qubits(rows, tag):
    pts = [r for r in rows if r["depth"] == 100]
    if not pts: return
    by_backend = defaultdict(list)
    for r in pts:
        by_backend[r["backend"]].append((r["qubits"], r["wall_ms"]))
    plt.figure()
    for be, p in by_backend.items():
        xs, ys = zip(*sorted(p))
        plt.plot(xs, ys, marker="o", label=be)
    plt.xlabel("Qubits (n)")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Runtime vs Qubits [{tag}]")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(DATA_DIR, f"runtime_vs_qubits_{tag}.png"), dpi=200)

def plot_speedup_vs_threads(rows, tag):
    pts = median_by_key(rows, ["threads"])
    pts = sorted(pts, key=lambda r: r["threads"])
    if not pts: return
    t1 = next((r["wall_ms"] for r in pts if r["threads"] == 1), None)
    if not t1: return
    xs = [r["threads"] for r in pts]
    ys = [t1 / r["wall_ms"] for r in pts]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Speedup (T1/Tt)")
    plt.title(f"Speedup vs Threads [{tag}]")
    plt.grid(True)
    plt.savefig(os.path.join(DATA_DIR, f"speedup_vs_threads_{tag}.png"), dpi=200)

def plot_runtime_vs_depth(rows, tag):
    from collections import defaultdict
    import matplotlib.pyplot as plt, os

    # detect which qubit counts exist
    qubits_list = sorted(set(r["qubits"] for r in rows))
    by_backend = defaultdict(list)

    # plot all qubit counts that exist
    for r in rows:
        by_backend[r["backend"]].append((r["depth"], r["wall_ms"]))

    plt.figure()
    for be, pts in by_backend.items():
        xs, ys = zip(*sorted(pts))
        plt.plot(xs, ys, marker="o", label=f"{be} (n={qubits_list[0]})")
    plt.xlabel("Depth")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Runtime vs Depth [{tag}]")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(DATA_DIR, f"runtime_vs_depth_{tag}.png"), dpi=200)

def plot_runtime_vs_threads(rows, tag):
    # expect rows from threads.csv (same n, depth; varying threads)
    from collections import defaultdict
    # group by threads and take median wall_ms (dedupe protection)
    buckets = defaultdict(list)
    for r in rows:
        buckets[int(r["threads"])].append(float(r["wall_ms"]))
    if not buckets: 
        return
    xs = sorted(buckets.keys())
    ys = [sorted(buckets[t])[len(buckets[t])//2] for t in xs]  # median

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Runtime vs Threads [{tag}]")
    plt.grid(True)
    plt.savefig(os.path.join(DATA_DIR, f"runtime_vs_threads_{tag}.png"), dpi=200)

def plot_depth_compare_all():
    import os, csv
    import matplotlib.pyplot as plt

    base = os.path.join(os.path.dirname(__file__), "..", "data")
    paths = {
        "serial": os.path.join(base, "serial", "depth.csv"),
        "numba":  os.path.join(base, "numba",  "depth.csv"),
        "cupy":   os.path.join(base, "cupy",   "depth.csv"),
    }

    series = {}
    for be, p in paths.items():
        if not os.path.exists(p):
            continue
        with open(p, "r") as f:
            r = csv.DictReader(f)
            pts = [(int(row["depth"]), float(row["wall_ms"])) for row in r]
        pts = sorted(pts)
        if pts:
            series[be] = pts

    if not series:
        print("plot_depth_compare_all: no depth.csv files found.")
        return

    plt.figure()
    for be, pts in series.items():
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", label=be)
    plt.xlabel("Depth")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Depth — serial vs numba vs cupy")
    plt.grid(True); plt.legend()
    out = os.path.join(base, "runtime_vs_depth_compare.png")
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


def plot_qubits_compare_all():
    import os, csv, matplotlib.pyplot as plt
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    paths = {
        "serial": os.path.join(base, "serial", "qubits.csv"),
        "numba":  os.path.join(base, "numba",  "qubits.csv"),
        "cupy":   os.path.join(base, "cupy",   "qubits.csv"),
    }
    series = {}
    for be, p in paths.items():
        if not os.path.exists(p): continue
        with open(p, "r") as f:
            r = csv.DictReader(f)
            pts = sorted([(int(x["qubits"]), float(x["wall_ms"])) for x in r])
        if pts: series[be] = pts
    if not series: return
    plt.figure()
    for be, pts in series.items():
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", label=be)
    plt.xlabel("Qubits (n)")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Qubits — serial vs numba vs cupy")
    plt.grid(True); plt.legend()
    out = os.path.join(base, "runtime_vs_qubits_compare.png")
    plt.savefig(out, dpi=200)

def plot_qubits_compare():
    import os, csv
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    paths = {
        "serial": os.path.join(base, "serial", "qubits.csv"),
        "numba":  os.path.join(base, "numba",  "qubits.csv"),
        "cupy":   os.path.join(base, "cupy",   "qubits.csv"),
    }

    rows_by = {}
    for be, p in paths.items():
        if not os.path.exists(p):
            continue
        xs, ys = [], []
        with open(p, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                xs.append(int(row["qubits"]))
                ys.append(float(row["wall_ms"]))
        rows_by[be] = (xs, ys)

    if not rows_by:
        return

    import matplotlib.pyplot as plt
    outdir = base
    plt.figure()
    for be, (xs, ys) in rows_by.items():
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        plt.plot(xs, ys, marker="o", label=be)

    plt.xlabel("Qubits (n)")
    plt.ylabel("Runtime (ms, log scale)")
    plt.title("Runtime vs Qubits (serial vs numba vs cupy)")
    plt.yscale("log")         # <-- add this line here
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "runtime_vs_qubits_compare.png"), dpi=200)

def plot_gpu_cpu_speedup_qubits(fixed_depth=None):
    import os, csv, math
    import matplotlib.pyplot as plt
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    numba_p = os.path.join(base, "numba", "qubits.csv")
    cupy_p  = os.path.join(base, "cupy",  "qubits.csv")
    if not (os.path.exists(numba_p) and os.path.exists(cupy_p)):
        print("plot_gpu_cpu_speedup_qubits: missing numba/cupy qubits.csv")
        return

    # load rows grouped by (n, depth)
    def load(path):
        rows = {}
        with open(path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                n = int(row["qubits"]); d = int(row["depth"])
                rows[(n, d)] = float(row["wall_ms"])
        return rows

    numba = load(numba_p)
    cupy  = load(cupy_p)

    # pick a depth: use `fixed_depth` if provided, else the most common intersection
    depths = sorted(set(d for (_, d) in numba.keys()) & set(d for (_, d) in cupy.keys()))
    if not depths:
        print("plot_gpu_cpu_speedup_qubits: no overlapping depths.")
        return
    if fixed_depth is None:
        from collections import Counter
        c = Counter([d for (_, d) in numba.keys() if ( _, d ) in numba])  # simple pick
        # choose the smallest overlapping depth to be safe
        d0 = min(depths)
    else:
        d0 = fixed_depth
        if d0 not in depths:
            print(f"plot_gpu_cpu_speedup_qubits: depth {d0} not found in overlap {depths}")
            return

    # collect matching n at depth d0
    ns = sorted(set(n for (n, d) in numba.keys() if d == d0) & set(n for (n, d) in cupy.keys() if d == d0))
    if not ns:
        print("plot_gpu_cpu_speedup_qubits: no matching ns at chosen depth.")
        return

    xs = [str(n) for n in ns]
    speedups = []
    print("\nGPU/CPU speedup table (Numba / CuPy) at depth =", d0)
    print("n\tCPU(ms)\tGPU(ms)\tSpeedup")
    for n in ns:
        cpu = numba[(n, d0)]
        gpu = cupy[(n, d0)]
        sp  = cpu / gpu if gpu > 0 else math.nan
        speedups.append(sp)
        print(f"{n}\t{cpu:.2f}\t{gpu:.2f}\t{sp:.2f}×")

    # bar chart
    plt.figure()
    plt.bar(xs, speedups)
    plt.xlabel("Qubits (n)")
    plt.ylabel("Speedup (Numba / CuPy)")
    plt.title(f"GPU/CPU Speedup vs Qubits (depth={d0})")
    plt.grid(True, axis="y")
    out = os.path.join(base, "gpu_cpu_speedup_qubits.png")
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

def plot_gpu_cpu_speedup_depth(fixed_n=None):
    import os, csv, math
    import matplotlib.pyplot as plt
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    numba_p = os.path.join(base, "numba", "depth.csv")
    cupy_p  = os.path.join(base, "cupy",  "depth.csv")
    if not (os.path.exists(numba_p) and os.path.exists(cupy_p)):
        print("plot_gpu_cpu_speedup_depth: missing numba/cupy depth.csv")
        return

    def load(path):
        rows = {}
        with open(path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                n = int(row["qubits"]); d = int(row["depth"])
                rows[(n, d)] = float(row["wall_ms"])
        return rows

    numba = load(numba_p)
    cupy  = load(cupy_p)

    # choose n
    ns = sorted(set(n for (n, _) in numba.keys()) & set(n for (n, _) in cupy.keys()))
    if not ns:
        print("plot_gpu_cpu_speedup_depth: no overlapping n.")
        return
    n0 = fixed_n or ns[0]  # pick first overlapping n if not provided

    depths = sorted(set(d for (n, d) in numba.keys() if n == n0) & set(d for (n, d) in cupy.keys() if n == n0))
    if not depths:
        print(f"plot_gpu_cpu_speedup_depth: no depths for n={n0}.")
        return

    xs = [str(d) for d in depths]
    speedups = []
    print(f"\nGPU/CPU speedup table (Numba / CuPy) at n = {n0}")
    print("depth\tCPU(ms)\tGPU(ms)\tSpeedup")
    for d in depths:
        cpu = numba[(n0, d)]
        gpu = cupy[(n0, d)]
        sp  = cpu / gpu if gpu > 0 else math.nan
        speedups.append(sp)
        print(f"{d}\t{cpu:.2f}\t{gpu:.2f}\t{sp:.2f}×")

    plt.figure()
    plt.bar(xs, speedups)
    plt.xlabel("Depth")
    plt.ylabel("Speedup (Numba / CuPy)")
    plt.title(f"GPU/CPU Speedup vs Depth (n={n0})")
    plt.grid(True, axis="y")
    out = os.path.join(base, "gpu_cpu_speedup_depth.png")
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


def main():
    global DATA_DIR
    # find all CSVs recursively under data/
    csvs = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".csv"):
                csvs.append(os.path.join(root, f))

    if not csvs:
        print("No CSV files found under data/")
        return

    for path in csvs:
        tag = os.path.splitext(os.path.basename(path))[0]
        backend = os.path.basename(os.path.dirname(path))  # 'serial' or 'numba'
        try:
            rows = load_rows(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        print(f"Plotting from {backend}/{tag}.csv ({len(rows)} rows)...")

        # send plots to the same backend folder
        local_out = os.path.dirname(path)
        old_dir = DATA_DIR
        DATA_DIR = local_out

        if tag.startswith("qubits"):
            plot_runtime_vs_qubits(rows, f"{backend}")
        elif tag.startswith("threads"):
            plot_speedup_vs_threads(rows, f"{backend}")
            plot_runtime_vs_threads(rows, f"{backend}")
        elif tag.startswith("depth"):
            plot_runtime_vs_depth(rows, f"{backend}")
        else:
            plot_runtime_vs_qubits(rows, f"{backend}")
            plot_runtime_vs_depth(rows, f"{backend}")
            plot_speedup_vs_threads(rows, f"{backend}")

        DATA_DIR = old_dir
        
    plot_qubits_compare_all()
    plot_depth_compare_all()
    plot_gpu_cpu_speedup_qubits(fixed_depth=200)
    plot_gpu_cpu_speedup_depth(fixed_n=18)
    print("\nSaved all plots under data/<backend>/*.png")




if __name__ == "__main__":
    main()
