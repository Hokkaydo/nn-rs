"""
Benchmark forward and backward pass at increasing batch sizes:
Rust CPU (bench_runner) vs PyTorch CPU.

Run from the repo root:
    cargo build --bin bench_runner --release   # once, or after Rust code changes
    python python/bench_forward.py
"""

import csv
import io
import os
import subprocess
import sys
import timeit
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

BATCH_SIZES = [1, 32, 64, 128, 256, 512, 1024]
WARMUP = 5
REPS = 20

os.makedirs("bench_results", exist_ok=True)

# ---------------------------------------------------------------------------
# Rust timings via bench_runner binary
# ---------------------------------------------------------------------------

def run_rust(op: str) -> list[dict]:
    binary = os.path.join("target", "release", "bench_runner")
    if not os.path.exists(binary):
        print(f"[error] {binary} not found — run: cargo build --bin bench_runner --release",
              file=sys.stderr)
        return []

    result = subprocess.run([binary, op], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[error] bench_runner failed:\n{result.stderr}", file=sys.stderr)
        return []

    rows = []
    reader = csv.DictReader(io.StringIO(result.stdout))
    for row in reader:
        rows.append({
            "framework": "rust_cpu",
            "mode": op,          # "forward" or "backward"
            "batch_size": int(row["size"]),
            "mean_ns": int(row["mean_ns"]),
            "std_ns": int(row["std_ns"]),
        })
    return rows


# ---------------------------------------------------------------------------
# PyTorch timings
# ---------------------------------------------------------------------------

def _stats(times_sec):
    arr = np.array(times_sec)
    return int(arr.mean() * 1e9), int(arr.std() * 1e9)


def make_torch_net():
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.LogSoftmax(dim=1),
    )


def bench_torch_forward(net, batch_size):
    net.eval()
    x = torch.rand(batch_size, 784)
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = net(x)
        times = [timeit.timeit(lambda: net(x), number=1) for _ in range(REPS)]
    return _stats(times)


def bench_torch_backward(net, batch_size):
    net.train()
    x = torch.rand(batch_size, 784)
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_fn = nn.NLLLoss()
    labels = torch.zeros(batch_size, dtype=torch.long)  # all class 0

    def step():
        opt.zero_grad()
        out = net(x)
        loss = loss_fn(out, labels)
        loss.backward()
        opt.step()

    for _ in range(WARMUP):
        step()
    times = [timeit.timeit(step, number=1) for _ in range(REPS)]
    return _stats(times)


# ---------------------------------------------------------------------------
# Run and collect results
# ---------------------------------------------------------------------------

all_rows: list[dict] = []

for mode in ("forward", "backward"):
    print(f"\n=== Rust (bench_runner) — {mode} ===")
    rust_rows = run_rust(mode)
    for r in rust_rows:
        bs = r["batch_size"]
        print(f"rust_cpu  bs={bs:4d}  {r['mean_ns']/1e6:9.3f} ms  ±{r['std_ns']/1e6:.3f} ms")
    all_rows.extend(rust_rows)

torch_net = make_torch_net()

print("\n=== PyTorch CPU — forward ===")
for bs in BATCH_SIZES:
    mean_ns, std_ns = bench_torch_forward(torch_net, bs)
    all_rows.append({"framework": "torch_cpu", "mode": "forward",
                     "batch_size": bs, "mean_ns": mean_ns, "std_ns": std_ns})
    print(f"torch_cpu  bs={bs:4d}  {mean_ns/1e6:9.3f} ms  ±{std_ns/1e6:.3f} ms")

print("\n=== PyTorch CPU — backward ===")
for bs in BATCH_SIZES:
    mean_ns, std_ns = bench_torch_backward(torch_net, bs)
    all_rows.append({"framework": "torch_cpu", "mode": "backward",
                     "batch_size": bs, "mean_ns": mean_ns, "std_ns": std_ns})
    print(f"torch_cpu  bs={bs:4d}  {mean_ns/1e6:9.3f} ms  ±{std_ns/1e6:.3f} ms")

csv_path = "bench_results/forward.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["framework", "mode", "batch_size", "mean_ns", "std_ns"])
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nWrote {csv_path}")

# ---------------------------------------------------------------------------
# Plot: one subplot per mode (forward / backward)
# ---------------------------------------------------------------------------

COLORS  = {"rust_cpu": "#d62728", "torch_cpu": "#2ca02c"}
MARKERS = {"rust_cpu": "o",       "torch_cpu": "^"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MNIST net — Rust CPU vs PyTorch CPU")

for ax, mode in zip(axes, ("forward", "backward")):
    by_fw: dict[str, list] = defaultdict(list)
    for r in all_rows:
        if r["mode"] == mode:
            by_fw[r["framework"]].append(r)

    for fw, rows in by_fw.items():
        rows.sort(key=lambda r: r["batch_size"])
        sizes   = np.array([r["batch_size"] for r in rows])
        mean_ns = np.array([r["mean_ns"] for r in rows], dtype=float)
        std_ns  = np.array([r["std_ns"]  for r in rows], dtype=float)
        mean_ms = mean_ns / 1e6

        ax.plot(sizes, mean_ms, marker=MARKERS[fw], color=COLORS[fw], label=fw)
        ax.fill_between(sizes, mean_ms - std_ns/1e6, mean_ms + std_ns/1e6, alpha=0.15, color=COLORS[fw])

    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_xscale("log", base=2)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{mode} pass")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "bench_results/forward_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
