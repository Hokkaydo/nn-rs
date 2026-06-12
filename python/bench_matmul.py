"""
cargo build --bin bench_runner --release   # once, or after Rust code changes
python python/bench_matmul.py
"""
import csv
import io
import os
import subprocess
import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch

SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
WARMUP = 5
REPS = 20

os.makedirs("bench_results", exist_ok=True)

# ---------------------------------------------------------------------------
# Rust timings via bench_runner binary
# ---------------------------------------------------------------------------

def run_rust(op: str) -> list[dict]:
    """Run the release bench_runner for a given op and return parsed rows."""
    binary = os.path.join("target", "release", "bench_runner")
    if not os.path.exists(binary):
        print(f"[error] {binary} not found — run: cargo build --bin bench_runner --release",
              file=sys.stderr)
        return []

    result = subprocess.run(
        [binary, op] + [str(s) for s in SIZES],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[error] bench_runner failed:\n{result.stderr}", file=sys.stderr)
        return []

    rows = []
    reader = csv.DictReader(io.StringIO(result.stdout))
    for row in reader:
        rows.append({
            "framework": "rust_cpu",
            "size": int(row["size"]),
            "mean_ns": int(row["mean_ns"]),
            "std_ns": int(row["std_ns"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Python timings
# ---------------------------------------------------------------------------

def _stats(times_sec):
    arr = np.array(times_sec)
    return int(arr.mean() * 1e9), int(arr.std() * 1e9)


def bench_numpy(size):
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    for _ in range(WARMUP):
        _ = a @ b
    times = [timeit.timeit(lambda: a @ b, number=1) for _ in range(REPS)]
    return _stats(times)


def bench_torch_cpu(size):
    a = torch.rand(size, size)
    b = torch.rand(size, size)
    for _ in range(WARMUP):
        _ = torch.mm(a, b)
    times = [timeit.timeit(lambda: torch.mm(a, b), number=1) for _ in range(REPS)]
    return _stats(times)


# ---------------------------------------------------------------------------
# Run everything and write CSV
# ---------------------------------------------------------------------------

print("=== Rust (bench_runner) ===")
rust_rows = run_rust("matmul")
for r in rust_rows:
    sz = r["size"]
    print(f"rust_cpu      {sz:4d}x{sz:<4d}  {r['mean_ns']/1e6:9.3f} ms  ±{r['std_ns']/1e6:.3f} ms")

all_rows = list(rust_rows)

print("\n=== Python (NumPy / PyTorch) ===")
for size in SIZES:
    for name, fn in [("numpy", bench_numpy), ("torch_cpu", bench_torch_cpu)]:
        mean_ns, std_ns = fn(size)
        all_rows.append({"framework": name, "size": size, "mean_ns": mean_ns, "std_ns": std_ns})
        print(f"{name:12s}  {size:4d}x{size:<4d}  {mean_ns/1e6:9.3f} ms  ±{std_ns/1e6:.3f} ms")

csv_path = "bench_results/matmul.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["framework", "size", "mean_ns", "std_ns"])
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nWrote {csv_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COLORS  = {"rust_cpu": "#d62728", "numpy": "#1f77b4", "torch_cpu": "#2ca02c"}
MARKERS = {"rust_cpu": "o",       "numpy": "s",        "torch_cpu": "^"}

fig, (ax_lat, ax_tput) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("matmul NxN — Rust CPU vs NumPy vs PyTorch CPU")

from collections import defaultdict
by_fw: dict[str, list] = defaultdict(list)
for r in all_rows:
    by_fw[r["framework"]].append(r)

for fw, rows in by_fw.items():
    rows.sort(key=lambda r: r["size"])
    sizes   = np.array([r["size"] for r in rows])
    mean_ns = np.array([r["mean_ns"] for r in rows], dtype=float)
    std_ns  = np.array([r["std_ns"]  for r in rows], dtype=float)
    mean_ms = mean_ns / 1e6

    ax_lat.plot(sizes, mean_ms, marker=MARKERS[fw], color=COLORS[fw], label=fw)
    ax_lat.fill_between(sizes, mean_ms - std_ns/1e6, mean_ms + std_ns/1e6,
                        alpha=0.15, color=COLORS[fw])

    gflops = 2.0 * sizes**3 / mean_ns   # GFLOP/s
    ax_tput.plot(sizes, gflops, marker=MARKERS[fw], color=COLORS[fw], label=fw)

for ax, ylabel, title in [
    (ax_lat,  "Latency (ms)",      "Latency"),
    (ax_tput, "Throughput (GFLOP/s)", "Throughput"),
]:
    ax.set_xlabel("Matrix size N (NxN)")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
# draw ref O(n) dashed black for latency
ref_sizes = np.array(SIZES)
ref_latency = 1e-6 * ref_sizes**3  # O(n^3) latency reference
ax_lat.plot(ref_sizes, ref_latency, linestyle="--", color="black", label="O(n³) ref")
ax_lat.legend()

plt.tight_layout()
out = "bench_results/matmul_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
