import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("bench_results")
COLORS  = {"rust_cpu": "#d62728", "numpy": "#1f77b4", "torch_cpu": "#2ca02c"}
MARKERS = {"rust_cpu": "o",       "numpy": "s",        "torch_cpu": "^"}


def _plot_latency(ax, rows_by_fw, x_col):
    for fw, rows in rows_by_fw.items():
        rows.sort(key=lambda r: r[x_col])
        x       = np.array([r[x_col]    for r in rows])
        mean_ms = np.array([r["mean_ns"] for r in rows], dtype=float) / 1e6
        std_ms  = np.array([r["std_ns"]  for r in rows], dtype=float) / 1e6
        ax.plot(x, mean_ms, marker=MARKERS.get(fw, "o"), color=COLORS.get(fw, "grey"), label=fw)
        ax.fill_between(x, mean_ms - std_ms, mean_ms + std_ms, alpha=0.15,
                        color=COLORS.get(fw, "grey"))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# matmul comparison
# ---------------------------------------------------------------------------

matmul_path = RESULTS / "matmul.csv"
if matmul_path.exists():
    df = pd.read_csv(matmul_path)
    by_fw: dict[str, list] = defaultdict(list)
    for _, row in df.iterrows():
        by_fw[row["framework"]].append(row.to_dict())

    fig, (ax_lat, ax_tput) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("matmul NxN - Rust CPU vs NumPy vs PyTorch CPU")

    _plot_latency(ax_lat, by_fw, "size")
    ax_lat.set_xlabel("Matrix size N (NxN)")
    ax_lat.set_title("Latency")

    for fw, rows in by_fw.items():
        rows.sort(key=lambda r: r["size"])
        sizes   = np.array([r["size"]    for r in rows])
        mean_ns = np.array([r["mean_ns"] for r in rows], dtype=float)
        gflops  = 2.0 * sizes**3 / mean_ns
        ax_tput.plot(sizes, gflops, marker=MARKERS.get(fw, "o"),
                     color=COLORS.get(fw, "grey"), label=fw)

    ax_tput.set_xlabel("Matrix size N (NxN)")
    ax_tput.set_xscale("log", base=2)
    ax_tput.set_yscale("log", base=10)
    ax_tput.set_ylabel("Throughput (GFLOP/s)")
    ax_tput.set_title("Throughput")
    ax_tput.legend()
    ax_tput.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS / "matmul_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()
else:
    print(f"[skip] {matmul_path} not found - run python/bench_matmul.py first", file=sys.stderr)

# ---------------------------------------------------------------------------
# forward / backward comparison
# ---------------------------------------------------------------------------

fwd_path = RESULTS / "forward.csv"
if fwd_path.exists():
    df = pd.read_csv(fwd_path)

    for mode, out_name in [("forward",  "forward_comparison.png"),
                           ("backward", "backward_comparison.png")]:
        sub = df[df["mode"] == mode]
        by_fw = defaultdict(list)
        for _, row in sub.iterrows():
            by_fw[row["framework"]].append(row.to_dict())

        if not any(by_fw.values()):
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"{mode} pass - Rust CPU vs PyTorch CPU (MNIST net)")

        _plot_latency(ax, by_fw, "batch_size")
        ax.set_xlabel("Batch size")
        ax.set_title(f"{mode} pass latency")

        plt.tight_layout()
        out = RESULTS / out_name
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
        plt.close()
else:
    print(f"[skip] {fwd_path} not found - run python/bench_forward.py first", file=sys.stderr)
