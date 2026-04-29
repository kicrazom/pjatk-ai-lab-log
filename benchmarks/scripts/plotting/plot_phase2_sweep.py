"""
Phase 2 plot generator for NaviMed-UMB benchmark suite.

Reads results_table.csv produced by analyze_phase2_sweep.py and generates
three plots per METHODOLOGY §7.2:
    1. scaling_curve.png   — tok/s out vs N (log-x), knee annotated
    2. thermal_curve.png   — temp peak + power mean vs N
    3. efficiency_curve.png — mWh/token vs N (energy efficiency)

Universal across all NaviMed-UMB models. Same visual layout enables
cross-model figure stacking for paper.

Usage:
    python3 plot_phase2_sweep.py \
        --csv benchmarks/results/qwen36-27b/results_table.csv \
        --output-dir benchmarks/results/qwen36-27b/
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(message)s"

# Color palette — kept consistent across all NaviMed-UMB figures
COLOR_THROUGHPUT: Final[str] = "#1f77b4"   # blue
COLOR_TEMP: Final[str] = "#d62728"          # red
COLOR_POWER: Final[str] = "#ff7f0e"         # orange
COLOR_EFFICIENCY: Final[str] = "#2ca02c"   # green
COLOR_KNEE: Final[str] = "#7f7f7f"          # grey


@dataclass
class SweepRow:
    """One row from results_table.csv."""

    n: int
    tok_s_out: float
    total_s: float
    vram_peak_gb: float
    t_peak_c: float
    w_mean: float
    w_peak: float
    mwh_per_tok: float

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "SweepRow":
        return cls(
            n=int(row["N"]),
            tok_s_out=float(row["tok_s_out"]),
            total_s=float(row["total_s"]),
            vram_peak_gb=float(row["VRAM_peak_GB"]),
            t_peak_c=float(row["T_peak_C"]),
            w_mean=float(row["W_mean"]),
            w_peak=float(row["W_peak"]),
            mwh_per_tok=float(row["W_per_tok_Wh"]) * 1000.0,
        )


def load_csv(path: Path) -> tuple[list[SweepRow], dict[str, str]]:
    """Load all rows + metadata (model, quant, config) from results_table.csv."""
    rows: list[SweepRow] = []
    meta: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(SweepRow.from_csv_row(row))
            if not meta:
                meta = {
                    "model": row["model"],
                    "quant": row["quant"],
                    "tp": row["TP"],
                    "max_len": row["max_len"],
                    "util": row["util"],
                    "kv_dtype": row["KV_dtype"],
                    "tuning": row["tuning"],
                }
    rows.sort(key=lambda r: r.n)
    return rows, meta


def title_for(meta: dict[str, str]) -> str:
    return (
        f"{meta['model']} — {meta['quant'].upper()} TP={meta['tp']} "
        f"max_len={meta['max_len']} util={meta['util']} "
        f"KV={meta['kv_dtype']}"
    )


# --- Plot helpers ------------------------------------------------------------


def _setup_log_x_axis(ax) -> None:
    """Standard log-x axis with integer ticks at powers of N values."""
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)


def _annotate_knee(
    ax, rows: list[SweepRow], y_attr: str, label_fmt: str = "knee N={n}"
) -> None:
    """Mark the row with maximum y_attr value as the knee."""
    knee = max(rows, key=lambda r: getattr(r, y_attr))
    ax.axvline(knee.n, color=COLOR_KNEE, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(
        knee.n, ax.get_ylim()[1] * 0.95, label_fmt.format(n=knee.n),
        rotation=90, va="top", ha="right", fontsize=9,
        color=COLOR_KNEE,
    )


def plot_scaling_curve(
    rows: list[SweepRow], meta: dict[str, str], output_path: Path
) -> None:
    """Plot 1: tok/s output vs N (log-x), with knee annotated."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    n_values = [r.n for r in rows]
    tput_values = [r.tok_s_out for r in rows]

    ax.plot(
        n_values, tput_values, "o-",
        color=COLOR_THROUGHPUT, linewidth=2.5, markersize=10, zorder=3,
    )
    for r in rows:
        ax.annotate(
            f"{r.tok_s_out:.1f}",
            xy=(r.n, r.tok_s_out), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=9,
        )

    _setup_log_x_axis(ax)
    _annotate_knee(ax, rows, "tok_s_out")

    ax.set_xlabel("Concurrent requests (N)", fontsize=11)
    ax.set_ylabel("Output throughput [tok/s]", fontsize=11)
    ax.set_title(title_for(meta), fontsize=12, pad=14)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def plot_thermal_curve(
    rows: list[SweepRow], meta: dict[str, str], output_path: Path
) -> None:
    """Plot 2: temp peak + power mean vs N (twin axes)."""
    fig, ax_temp = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax_pow = ax_temp.twinx()

    n_values = [r.n for r in rows]
    temps = [r.t_peak_c for r in rows]
    powers = [r.w_mean for r in rows]

    line_t, = ax_temp.plot(
        n_values, temps, "s-",
        color=COLOR_TEMP, linewidth=2.0, markersize=8,
        label="Temp peak [°C]", zorder=3,
    )
    line_p, = ax_pow.plot(
        n_values, powers, "^-",
        color=COLOR_POWER, linewidth=2.0, markersize=8,
        label="Power mean [W]", zorder=3,
    )

    _setup_log_x_axis(ax_temp)
    ax_temp.set_xlabel("Concurrent requests (N)", fontsize=11)
    ax_temp.set_ylabel("Temperature peak [°C]", color=COLOR_TEMP, fontsize=11)
    ax_pow.set_ylabel("Power mean [W, both GPUs]", color=COLOR_POWER, fontsize=11)
    ax_temp.tick_params(axis="y", labelcolor=COLOR_TEMP)
    ax_pow.tick_params(axis="y", labelcolor=COLOR_POWER)

    ax_temp.set_title(title_for(meta), fontsize=12, pad=14)
    ax_temp.legend(handles=[line_t, line_p], loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def plot_efficiency_curve(
    rows: list[SweepRow], meta: dict[str, str], output_path: Path
) -> None:
    """Plot 3: mWh/tok vs N — energy efficiency. Lower is better."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    n_values = [r.n for r in rows]
    eff_values = [r.mwh_per_tok for r in rows]

    ax.plot(
        n_values, eff_values, "D-",
        color=COLOR_EFFICIENCY, linewidth=2.5, markersize=10, zorder=3,
    )
    for r in rows:
        ax.annotate(
            f"{r.mwh_per_tok:.3f}",
            xy=(r.n, r.mwh_per_tok), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=9,
        )

    # Annotate optimum (lowest mWh/tok)
    optimum = min(rows, key=lambda r: r.mwh_per_tok)
    ax.axvline(optimum.n, color=COLOR_KNEE, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(
        optimum.n, ax.get_ylim()[1] * 0.95,
        f"optimum N={optimum.n}",
        rotation=90, va="top", ha="right", fontsize=9, color=COLOR_KNEE,
    )

    _setup_log_x_axis(ax)
    ax.set_xlabel("Concurrent requests (N)", fontsize=11)
    ax.set_ylabel("Energy per output token [mWh/tok]", fontsize=11)
    ax.set_title(
        f"Energy efficiency — {title_for(meta)}\n(lower is better)",
        fontsize=12, pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, required=True,
        help="results_table.csv from analyze_phase2_sweep.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write the three PNG plots",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT,
    )

    if not args.csv.exists():
        logging.error("CSV not found: %s", args.csv)
        return 1

    rows, meta = load_csv(args.csv)
    if not rows:
        logging.error("No rows in CSV")
        return 1
    logging.info("Loaded %d rows for %s", len(rows), meta["model"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_scaling_curve(rows, meta, args.output_dir / "scaling_curve.png")
    plot_thermal_curve(rows, meta, args.output_dir / "thermal_curve.png")
    plot_efficiency_curve(rows, meta, args.output_dir / "efficiency_curve.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
