#!/usr/bin/env python3
"""
Build a 2x3 grid of thermal charts from Plan A runs.
Shows TP=1 vs TP=2 at increasing concurrency (N=500, 2000, 3000).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    thermal_dir = Path("thermal-runs")
    out_path = Path("thermal_gallery.png")

    # Selected representative runs: plateau + max N for both TP configs
    runs = [
        ("tp1-n500",  "TP=1, N=500 (plateau start)"),
        ("tp1-n2000", "TP=1, N=2000 (plateau mid)"),
        ("tp1-n3000", "TP=1, N=3000 (plateau max)"),
        ("tp2-n500",  "TP=2, N=500 (plateau start)"),
        ("tp2-n2000", "TP=2, N=2000 (plateau mid)"),
        ("tp2-n3000", "TP=2, N=3000 (plateau max, 131s run)"),
    ]

    # Load scaling data for throughput labels
    scaling = json.loads(Path("scaling_data.json").read_text())
    tput = {}
    for run in scaling["runs"]["tp1"]:
        tput[f"tp1-n{run['n_prompts']}"] = run["output_throughput_toks"]
    for run in scaling["runs"]["tp2"]:
        tput[f"tp2-n{run['n_prompts']}"] = run["output_throughput_toks"]

    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    fig.patch.set_facecolor("white")

    for idx, (run_name, label) in enumerate(runs):
        ax = axes[idx // 3][idx % 3]
        img_path = thermal_dir / f"{run_name}-thermals.png"

        if not img_path.exists():
            ax.text(0.5, 0.5, f"Missing:\n{img_path.name}",
                    ha="center", va="center", fontsize=11, color="red",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_axis_off()

        throughput = tput.get(run_name, "?")
        title = f"{label}  —  {throughput:.0f} tok/s" if isinstance(throughput, (int, float)) else label
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    fig.suptitle(
        "Thermal timelines: TP=1 (top row) vs TP=2 (bottom row) at increasing concurrency",
        fontsize=14, fontweight="bold", y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
