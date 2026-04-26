#!/usr/bin/env python3
"""
Plot vLLM scaling curves (TP=1 vs TP=2) on AMD R9700 workstation.

Usage:
    python plot_scaling.py [scaling_data.json] [output.png]

Reads benchmark results from JSON and generates a publication-ready chart
showing throughput vs concurrency for different tensor parallelism configurations.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot(data: dict, out_path: Path) -> None:
    hw = data["hardware"]
    sw = data["software"]
    wl = data["workload"]
    runs = data["runs"]
    kv = data["kv_cache"]

    # --- Figure setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("white")

    # ==========================================================
    # Panel 1 — Throughput scaling curve
    # ==========================================================
    color_tp1 = "#1f77b4"  # blue
    color_tp2 = "#d62728"  # red

    # TP=1 curve
    tp1_n = [r["n_prompts"] for r in runs["tp1"]]
    tp1_tput = [r["output_throughput_toks"] for r in runs["tp1"]]
    ax1.plot(tp1_n, tp1_tput, "o-", color=color_tp1, linewidth=2.5,
             markersize=9, label="TP=1 (single GPU)", zorder=3)

    # Annotate each TP=1 point with value
    for n, t in zip(tp1_n, tp1_tput):
        ax1.annotate(f"{t:.0f}", xy=(n, t), xytext=(0, 10),
                     textcoords="offset points", ha="center",
                     fontsize=9, color=color_tp1, fontweight="bold")

    # TP=2 points (may be sparse)
    tp2_n = [r["n_prompts"] for r in runs["tp2"]]
    tp2_tput = [r["output_throughput_toks"] for r in runs["tp2"]]
    if len(tp2_n) > 1:
        ax1.plot(tp2_n, tp2_tput, "s-", color=color_tp2, linewidth=2.5,
                 markersize=9, label="TP=2 (2 GPUs, tensor parallel)", zorder=3)
    else:
        ax1.plot(tp2_n, tp2_tput, "s", color=color_tp2, markersize=12,
                 label="TP=2 (single measurement — sweep pending)",
                 markerfacecolor="none", markeredgewidth=2.5, zorder=3)

    for n, t in zip(tp2_n, tp2_tput):
        ax1.annotate(f"{t:.0f}", xy=(n, t), xytext=(0, -18),
                     textcoords="offset points", ha="center",
                     fontsize=9, color=color_tp2, fontweight="bold")

    ax1.set_xscale("log")
    ax1.set_xticks([10, 25, 50, 100, 200, 500])
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlabel("Concurrent prompts (N)", fontsize=12)
    ax1.set_ylabel("Output throughput (tokens/s)", fontsize=12)
    ax1.set_title("vLLM Throughput Scaling — AMD Radeon AI PRO R9700",
                  fontsize=13, fontweight="bold", pad=14)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax1.set_ylim(0, max(tp1_tput + tp2_tput) * 1.20)

    # Add max-concurrency annotations (constraint lines)
    tp1_max_conc = kv["tp1"]["max_concurrency_4k"]
    tp2_max_conc = kv["tp2"]["max_concurrency_4k"]
    ax1.axvline(x=tp1_max_conc, color=color_tp1, linestyle=":", alpha=0.5, linewidth=1.5)
    ax1.text(tp1_max_conc, ax1.get_ylim()[1] * 0.05, f"TP=1 KV limit\n@4k ctx: {tp1_max_conc:.0f}×",
             fontsize=8, color=color_tp1, ha="center",
             bbox=dict(facecolor="white", edgecolor=color_tp1, alpha=0.85, boxstyle="round,pad=0.3"))
    ax1.axvline(x=tp2_max_conc, color=color_tp2, linestyle=":", alpha=0.5, linewidth=1.5)
    ax1.text(tp2_max_conc, ax1.get_ylim()[1] * 0.05, f"TP=2 KV limit\n@4k ctx: {tp2_max_conc:.0f}×",
             fontsize=8, color=color_tp2, ha="center",
             bbox=dict(facecolor="white", edgecolor=color_tp2, alpha=0.85, boxstyle="round,pad=0.3"))

    # ==========================================================
    # Panel 2 — Per-request throughput (reveals batching cost)
    # ==========================================================
    tp1_per_req = [t / n for n, t in zip(tp1_n, tp1_tput)]
    ax2.plot(tp1_n, tp1_per_req, "o-", color=color_tp1, linewidth=2,
             markersize=8, label="TP=1", zorder=3)

    if len(tp2_n) >= 1:
        tp2_per_req = [t / n for n, t in zip(tp2_n, tp2_tput)]
        ax2.plot(tp2_n, tp2_per_req, "s", color=color_tp2, markersize=10,
                 markerfacecolor="none", markeredgewidth=2, label="TP=2", zorder=3)

    ax2.set_xscale("log")
    ax2.set_xticks([10, 25, 50, 100, 200, 500])
    ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax2.set_xlabel("Concurrent prompts (N)", fontsize=11)
    ax2.set_ylabel("Per-request throughput (tok/s)", fontsize=11)
    ax2.set_title("Latency trade-off", fontsize=12, pad=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.axhline(y=15, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.text(10, 16, "~human reading speed", fontsize=8, color="gray", alpha=0.8)

    # ==========================================================
    # Footer — system info
    # ==========================================================
    footer = (
        f"Model: {wl['model']} (dtype={wl['dtype']}, ctx={wl['max_model_len']}, max_tokens={wl['max_tokens_per_request']}) | "
        f"Hardware: {hw['gpu_count']}× {hw['gpu_model']} [{hw['gpu_arch']}] on {hw['cpu']} | "
        f"Stack: ROCm {sw['rocm']}, vLLM {sw['vllm']}, {sw['os']} kernel {sw['kernel']} | "
        f"Run date: {data['date']}"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8,
             color="#555555", wrap=True)

    plt.tight_layout(rect=[0, 0.035, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")


def main():
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scaling_data.json")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("scaling_curve.png")
    data = load_data(json_path)
    plot(data, out_path)


if __name__ == "__main__":
    main()
