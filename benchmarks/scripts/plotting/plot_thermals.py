#!/usr/bin/env python3
"""
Plot thermal/utilization timeline from sample_system.py output.

Usage:
    python plot_thermals.py thermals.jsonl thermals.png
    python plot_thermals.py thermals.jsonl thermals.png --events events.json

events.json format:
    [
        {"t": 12.5, "label": "benchmark start", "color": "#2ca02c"},
        {"t": 45.3, "label": "benchmark end",   "color": "#d62728"}
    ]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_samples(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def load_events(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def by_gpu(samples: list[dict]) -> dict[int, dict]:
    """Transpose: {gpu_idx: {t: [...], temp: [...], use: [...], power: [...], name: str, is_igpu: bool}}"""
    result: dict[int, dict] = {}
    for s in samples:
        for g in s.get("gpus", []):
            idx = g["idx"]
            if idx not in result:
                result[idx] = {
                    "name": g.get("name") or f"GPU {idx}",
                    "is_igpu": g.get("is_igpu", False),
                    "t": [], "temp": [], "use": [], "power": [],
                }
            result[idx]["t"].append(s["t"])
            result[idx]["temp"].append(g.get("temp"))
            result[idx]["use"].append(g.get("use"))
            result[idx]["power"].append(g.get("power_w"))
    return result


def plot(samples: list[dict], events: list[dict], out: Path) -> None:
    if not samples:
        print("No samples to plot.")
        return

    gpus = by_gpu(samples)
    t = [s["t"] for s in samples]
    cpu_temp = [s.get("cpu_temp") for s in samples]
    cpu_use = [s.get("cpu_percent") for s in samples]

    fig, (ax_temp, ax_use) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                            gridspec_kw={"height_ratios": [1.2, 1]})
    fig.patch.set_facecolor("white")

    # Colors — discrete GPUs in warm palette, iGPU in gray, CPU in blue
    discrete_colors = ["#d62728", "#ff7f0e"]  # red, orange for 2 discrete GPUs
    igpu_color = "#7f7f7f"
    cpu_color = "#1f77b4"

    # ========================= Panel 1 — Temperatures =========================
    discrete_counter = 0
    for idx in sorted(gpus.keys()):
        g = gpus[idx]
        if g["is_igpu"]:
            color = igpu_color
            label = f"iGPU ({g['name']})" if g["name"] else "iGPU"
            ls = ":"
        else:
            color = discrete_colors[discrete_counter % len(discrete_colors)]
            discrete_counter += 1
            label = f"GPU {idx}: {g['name']}" if g["name"] else f"GPU {idx}"
            ls = "-"
        ax_temp.plot(g["t"], g["temp"], ls, color=color, linewidth=1.8,
                     label=label, alpha=0.9)

    if any(c is not None for c in cpu_temp):
        ax_temp.plot(t, cpu_temp, "-", color=cpu_color, linewidth=1.8,
                     label="CPU (Tctl)", alpha=0.9)

    ax_temp.set_ylabel("Temperature (°C)", fontsize=12)
    ax_temp.set_title("System thermals during vLLM benchmark",
                      fontsize=13, fontweight="bold", pad=10)
    ax_temp.grid(True, alpha=0.3, linestyle="--")
    ax_temp.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Reference lines for AMD thermal thresholds
    ax_temp.axhline(y=95, color="red", linestyle="--", alpha=0.3, linewidth=1)
    ax_temp.text(t[-1] * 0.99, 95.5, "thermal throttle (typical)",
                 fontsize=8, color="red", alpha=0.7, ha="right")

    # ========================= Panel 2 — Utilization =========================
    discrete_counter = 0
    for idx in sorted(gpus.keys()):
        g = gpus[idx]
        if g["is_igpu"]:
            color = igpu_color
            ls = ":"
            label = f"iGPU use"
        else:
            color = discrete_colors[discrete_counter % len(discrete_colors)]
            discrete_counter += 1
            ls = "-"
            label = f"GPU {idx} use"
        uses = [u if u is not None else 0 for u in g["use"]]
        ax_use.plot(g["t"], uses, ls, color=color, linewidth=1.6,
                    label=label, alpha=0.85)

    if any(c is not None for c in cpu_use):
        ax_use.plot(t, cpu_use, "-", color=cpu_color, linewidth=1.6,
                    label="CPU use (aggregate)", alpha=0.85)

    ax_use.set_xlabel("Time (seconds since sampler start)", fontsize=12)
    ax_use.set_ylabel("Utilization (%)", fontsize=12)
    ax_use.set_ylim(-2, 105)
    ax_use.grid(True, alpha=0.3, linestyle="--")
    ax_use.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # ========================= Event markers =========================
    for ev in events:
        for ax in (ax_temp, ax_use):
            ax.axvline(x=ev["t"], color=ev.get("color", "#2ca02c"),
                       linestyle="--", alpha=0.7, linewidth=1.5)
        # Place label in top panel
        ax_temp.text(ev["t"], ax_temp.get_ylim()[1] * 0.98, " " + ev["label"],
                     rotation=90, fontsize=9, color=ev.get("color", "#2ca02c"),
                     va="top", ha="left")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL file from sample_system.py")
    ap.add_argument("output", help="Output PNG path")
    ap.add_argument("--events", help="Optional events.json with {t,label,color}")
    args = ap.parse_args()

    samples = load_samples(Path(args.input))
    events = load_events(Path(args.events) if args.events else None)
    plot(samples, events, Path(args.output))


if __name__ == "__main__":
    main()
