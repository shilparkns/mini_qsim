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
    pts = [r for r in rows if r["qubits"] == 12]
    by_backend = defaultdict(list)
    for r in pts:
        by_backend[r["backend"]].append((r["depth"], r["wall_ms"]))
    plt.figure()
    for be, p in by_backend.items():
        xs, ys = zip(*sorted(p))
        plt.plot(xs, ys, marker="o", label=be)
    plt.xlabel("Depth")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Runtime vs Depth [{tag}]")
    plt.legend()
    plt.grid(True)
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
    plot_qubits_compare()
    print("\nSaved all plots under data/<backend>/*.png")



if __name__ == "__main__":
    main()
