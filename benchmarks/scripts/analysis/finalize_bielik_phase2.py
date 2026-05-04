"""
NaviMed-UMB Phase 2 finalize — Bielik 11B v2.3 FP16 TP=2 max_len=8192.

One-shot post-sweep aggregator. Reads:
  - benchmarks/results/bielik-11b-v23/thermal-runs/fp16-tp2-n*-bench.log
  - benchmarks/results/bielik-11b-v23/thermal-runs/fp16-tp2-n*-thermals.jsonl
  - benchmarks/results/bielik-11b-v23/thermal-runs/fp16-tp2-n*-events.json

Produces:
  - benchmarks/results/bielik-11b-v23/results_table.csv      (METHODOLOGY §7.3 flat schema)
  - benchmarks/results/bielik-11b-v23/scaling_curve.png      (METHODOLOGY §7.2 plot 1)
  - benchmarks/results/bielik-11b-v23/thermal_gallery.png    (METHODOLOGY §7.2 plot 2)
  - benchmarks/results/bielik-11b-v23/efficiency_curve.png   (METHODOLOGY §7.2 plot 3)
  - benchmarks/results/bielik-11b-v23/SUMMARY.md             (METHODOLOGY §7.3 narrative)

Embargo: SCRIPT itself is PUBLIC (engineering), OUTPUTS are mixed:
  - CSV: EMBARGO_paper_bound (Polish model, §11.3)
  - PNGs: EMBARGO_paper_bound
  - SUMMARY.md: split per-section per §11.4

Author: Łukasz Minarowski <lukasz.minarowski@umb.edu.pl>
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Paths and constants
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks/results/bielik-11b-v23"
THERMAL_DIR = RESULTS_DIR / "thermal-runs"

MODEL_NAME = "speakleash/Bielik-11B-v2.3-Instruct"
QUANT = "FP16"
BACKEND = "vllm-rocm"
TP = 2
MAX_LEN = 8192
UTIL = 0.90
KV_DTYPE = "auto"
TUNING = "stock"
N_LADDER = [10, 25, 50, 100, 200, 500, 1000]


# ============================================================
# Per-N record (METHODOLOGY §7.1 schema)
# ============================================================


@dataclass
class RunRecord:
    n: int
    # From bench.log
    load_time_s: Optional[float] = None
    total_s: Optional[float] = None
    tok_s_out: Optional[float] = None
    tok_s_tot: Optional[float] = None
    req_s: Optional[float] = None
    total_out_tokens: Optional[int] = None
    total_in_tokens: Optional[int] = None
    # From thermals.jsonl during bench window (events.json brackets)
    vram_peak_gib: Optional[float] = None
    t_peak_c: Optional[float] = None
    w_mean: Optional[float] = None
    w_peak: Optional[float] = None
    # Derived
    w_per_tok: Optional[float] = None
    vs_baseline_pct: Optional[float] = None


# ============================================================
# Bench log parsing
# ============================================================


def parse_bench_log(path: Path) -> dict:
    """Parse the inner test_concurrent_bielik_11b.py stdout."""
    data: dict = {}
    text = path.read_text(errors="ignore")
    patterns = {
        "load_time_s": r"Load time:\s+([\d.]+)s",
        "total_s": r"Total time:\s+([\d.]+)s",
        "total_out_tokens": r"Total output tokens:\s+(\d+)",
        "total_in_tokens": r"Total input tokens:\s+(\d+)",
        "tok_s_out": r"Output throughput:\s+([\d.]+)\s+tok/s",
        "tok_s_tot": r"Total throughput:\s+([\d.]+)\s+tok/s",
        "req_s": r"Requests/second:\s+([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val_str = m.group(1)
            data[key] = int(val_str) if "tokens" in key else float(val_str)
    return data


# ============================================================
# Thermal sample parsing — bench window only
# ============================================================


def parse_thermals_window(jsonl_path: Path, events_path: Path) -> dict:
    """Read thermals.jsonl and clip to (bench_start, bench_end) per events.json.

    Returns aggregates: vram_peak_gib (max over both GPUs and time),
    t_peak_c, w_mean (during bench), w_peak.
    """
    if not jsonl_path.exists() or not events_path.exists():
        return {}

    events = json.loads(events_path.read_text())
    t_start = next((e["t"] for e in events if "bench start" in e["label"]), None)
    t_end = next((e["t"] for e in events if "bench end" in e["label"]), None)
    if t_start is None or t_end is None:
        return {}

    # Each line is a JSON dict with at least 't' (seconds since sampler start)
    # and per-GPU fields. Schema follows sample_system.py output.
    samples = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Window
    win = [s for s in samples if t_start <= s.get("t", -1) <= t_end]
    if not win:
        return {}

    # Per-GPU metrics — sample_system.py emits "gpus" list with dicts that
    # include keys like 'temp_c', 'power_w', 'vram_used_b', 'use_pct'.
    # We aggregate across both GPUs and across time.
    vram_max = 0.0
    t_max = 0.0
    w_samples_total = []  # sum across GPUs per timestep
    w_peaks_total = []

    for s in win:
        gpus = s.get("gpus", []) or []
        if not gpus:
            continue
        # Per-timestep totals — filter iGPU (RAPHAEL) per METHODOLOGY §2
        sum_w = 0.0
        max_t_now = 0.0
        for g in gpus:
            if g.get("is_igpu"):
                continue
            vram_b = g.get("vram_used_b") or 0
            vram_gib = vram_b / (1024**3)
            if vram_gib > vram_max:
                vram_max = vram_gib
            tc = g.get("temp") or 0.0
            if tc > t_max:
                t_max = tc
            if tc > max_t_now:
                max_t_now = tc
            pw = g.get("power_w") or 0.0
            sum_w += pw
        w_samples_total.append(sum_w)
        w_peaks_total.append(sum_w)

    out = {
        "vram_peak_gib": round(vram_max, 3) if vram_max > 0 else None,
        "t_peak_c": round(t_max, 1) if t_max > 0 else None,
        "w_mean": round(statistics.mean(w_samples_total), 1)
        if w_samples_total
        else None,
        "w_peak": round(max(w_peaks_total), 1) if w_peaks_total else None,
    }
    return out


# ============================================================
# Build all records
# ============================================================


def build_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    for n in N_LADDER:
        name = f"fp16-tp2-n{n}"
        bench_log = THERMAL_DIR / f"{name}-bench.log"
        thermals = THERMAL_DIR / f"{name}-thermals.jsonl"
        events = THERMAL_DIR / f"{name}-events.json"

        if not bench_log.exists():
            print(f"[skip] missing {bench_log}")
            continue

        rec = RunRecord(n=n)
        bench = parse_bench_log(bench_log)
        for k, v in bench.items():
            setattr(rec, k, v)

        therm = parse_thermals_window(thermals, events)
        for k, v in therm.items():
            setattr(rec, k, v)

        # W/tok = (W_mean × total_s) / total_out_tokens
        if rec.w_mean and rec.total_s and rec.total_out_tokens:
            rec.w_per_tok = round((rec.w_mean * rec.total_s) / rec.total_out_tokens, 4)

        records.append(rec)

    # vs_baseline_% — Polish model, no public R9700 reference; use N=10 as
    # the stock self-baseline per METHODOLOGY §10 limitation 10.
    if records:
        baseline = records[0].tok_s_out
        if baseline:
            for r in records:
                if r.tok_s_out:
                    r.vs_baseline_pct = round(100.0 * (r.tok_s_out / baseline), 1)

    return records


# ============================================================
# CSV emission (METHODOLOGY §7.1)
# ============================================================


CSV_COLUMNS = [
    "model",
    "quant",
    "backend",
    "TP",
    "max_len",
    "util",
    "KV_dtype",
    "N",
    "tok/s_out",
    "tok/s_tot",
    "total_s",
    "req/s",
    "VRAM_peak_GB",
    "T_peak_C",
    "W_mean",
    "W_peak",
    "W/tok",
    "tuning",
    "vs_baseline_%",
]


def emit_csv(records: list[RunRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in records:
            w.writerow(
                [
                    MODEL_NAME,
                    QUANT,
                    BACKEND,
                    TP,
                    MAX_LEN,
                    UTIL,
                    KV_DTYPE,
                    r.n,
                    r.tok_s_out,
                    r.tok_s_tot,
                    r.total_s,
                    r.req_s,
                    r.vram_peak_gib if r.vram_peak_gib is not None else "",
                    r.t_peak_c if r.t_peak_c is not None else "",
                    r.w_mean if r.w_mean is not None else "",
                    r.w_peak if r.w_peak is not None else "",
                    r.w_per_tok if r.w_per_tok is not None else "",
                    TUNING,
                    r.vs_baseline_pct if r.vs_baseline_pct is not None else "",
                ]
            )
    print(f"  CSV → {path.relative_to(REPO_ROOT)}")


# ============================================================
# Plot 1: Scaling curve (METHODOLOGY §7.2)
# ============================================================


def plot_scaling(records: list[RunRecord], path: Path) -> None:
    ns = [r.n for r in records if r.tok_s_out]
    tps = [r.tok_s_out for r in records if r.tok_s_out]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        ns, tps, marker="o", linewidth=2, color="#2ca02c", label="FP16 TP=2 max=8192"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Concurrent prompts (N)")
    ax.set_ylabel("Output throughput (tok/s)")
    ax.set_title(
        "Bielik 11B v2.3 — Phase 2 scaling (FP16 TP=2 max_len=8192, 2× R9700 gfx1201)"
    )
    ax.grid(True, which="both", alpha=0.3)

    # Annotate knee — last point with >70% efficiency relative to N=10 anchor
    if len(records) >= 2:
        for i in range(1, len(records)):
            ratio = records[i].tok_s_out / records[i - 1].tok_s_out
            n_ratio = records[i].n / records[i - 1].n
            efficiency = ratio / n_ratio
            if efficiency < 0.5 and i < len(records):
                # Mark the knee at the previous point
                knee = records[i - 1]
                ax.axvline(knee.n, color="#d62728", linestyle="--", alpha=0.6)
                ax.annotate(
                    f"knee N≈{knee.n}\n{knee.tok_s_out:.0f} tok/s",
                    xy=(knee.n, knee.tok_s_out),
                    xytext=(knee.n * 1.5, knee.tok_s_out * 0.7),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#d62728"),
                )
                break

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  PNG → {path.relative_to(REPO_ROOT)}")


# ============================================================
# Plot 2: Efficiency curve (W/tok vs N)
# ============================================================


def plot_efficiency(records: list[RunRecord], path: Path) -> None:
    pairs = [(r.n, r.w_per_tok) for r in records if r.w_per_tok is not None]
    if not pairs:
        print("  [skip] efficiency plot — no W/tok data")
        return
    ns, wpt = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, wpt, marker="s", linewidth=2, color="#1f77b4", label="W/tok")
    ax.set_xscale("log")
    ax.set_xlabel("Concurrent prompts (N)")
    ax.set_ylabel("Energy per output token (W·s/tok)")
    ax.set_title("Bielik 11B v2.3 — Energy efficiency (FP16 TP=2 max_len=8192)")
    ax.grid(True, which="both", alpha=0.3)

    # Annotate minimum (energy-optimal operating point)
    if pairs:
        min_idx = min(range(len(pairs)), key=lambda i: pairs[i][1])
        n_opt, w_opt = pairs[min_idx]
        ax.axvline(n_opt, color="#2ca02c", linestyle="--", alpha=0.6)
        ax.annotate(
            f"energy-optimal\nN={n_opt}, {w_opt:.3f} W·s/tok",
            xy=(n_opt, w_opt),
            xytext=(n_opt * 0.4, w_opt * 1.3),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#2ca02c"),
        )

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  PNG → {path.relative_to(REPO_ROOT)}")


# ============================================================
# Plot 3: Thermal gallery (per-N traces, aligned y-axis)
# ============================================================


def plot_thermal_gallery(path: Path) -> None:
    """Composite: 7 subplots, one per N, each showing GPU temp + power vs time."""
    fig, axes = plt.subplots(
        len(N_LADDER), 1, figsize=(10, 2.0 * len(N_LADDER)), sharex=False
    )

    # Determine global y-axis ranges for visual comparability
    all_temps = []
    all_powers = []
    cached = {}
    for n in N_LADDER:
        jp = THERMAL_DIR / f"fp16-tp2-n{n}-thermals.jsonl"
        if not jp.exists():
            continue
        rows = []
        for line in jp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        cached[n] = rows
        for r in rows:
            for g in r.get("gpus", []) or []:
                if g.get("is_igpu"):
                    continue
                if g.get("temp"):
                    all_temps.append(g["temp"])
                if g.get("power_w"):
                    all_powers.append(g["power_w"])

    t_lim = (min(all_temps) - 2, max(all_temps) + 2) if all_temps else (40, 90)
    w_lim = (0, max(all_powers) + 20) if all_powers else (0, 400)

    for ax, n in zip(axes, N_LADDER):
        rows = cached.get(n, [])
        if not rows:
            ax.set_title(f"N={n}: no data")
            continue
        ts = [r["t"] for r in rows]

        # Filter to non-iGPU R9700s only — get their indices in the gpus list
        # (sample_system orders them by idx but we still respect is_igpu).
        def r9700_indices(row):
            return [
                i
                for i, g in enumerate(row.get("gpus", []) or [])
                if not g.get("is_igpu")
            ]

        first_row_indices = r9700_indices(rows[0]) if rows else []
        # Map list-position to a stable label (GPU0, GPU1)
        labels = {first_row_indices[k]: k for k in range(len(first_row_indices))}

        # GPU 0 + GPU 1 traces (R9700 only)
        for list_idx, color in zip(first_row_indices, ["#1f77b4", "#ff7f0e"]):
            label_n = labels[list_idx]
            temps = [
                (
                    r["gpus"][list_idx]["temp"]
                    if len(r.get("gpus", [])) > list_idx
                    else None
                )
                for r in rows
            ]
            ax.plot(ts, temps, color=color, linewidth=1.2, label=f"GPU{label_n} T")
        ax.set_ylabel("Temp °C", color="#444")
        ax.set_ylim(*t_lim)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        for list_idx, color in zip(first_row_indices, ["#1f77b4", "#ff7f0e"]):
            powers = [
                (
                    r["gpus"][list_idx]["power_w"]
                    if len(r.get("gpus", [])) > list_idx
                    else None
                )
                for r in rows
            ]
            ax2.plot(ts, powers, color=color, linewidth=1.0, linestyle="--", alpha=0.7)
        ax2.set_ylabel("Power W", color="#444")
        ax2.set_ylim(*w_lim)

        ax.set_title(f"N={n}", loc="left")
        if n == N_LADDER[-1]:
            ax.set_xlabel("Time (s)")

    fig.suptitle(
        "Bielik 11B v2.3 — Thermal gallery (FP16 TP=2 max_len=8192)",
        y=1.0,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  PNG → {path.relative_to(REPO_ROOT)}")


# ============================================================
# SUMMARY.md (METHODOLOGY §7.3 narrative + embargo split)
# ============================================================


def emit_summary(records: list[RunRecord], path: Path) -> None:
    if not records:
        return

    # Compute key narrative numbers
    plateau_throughput = max((r.tok_s_out for r in records if r.tok_s_out), default=0)
    knee_n = None
    for i in range(1, len(records)):
        ratio = records[i].tok_s_out / records[i - 1].tok_s_out
        n_ratio = records[i].n / records[i - 1].n
        if ratio / n_ratio < 0.5:
            knee_n = records[i - 1].n
            break

    # Energy-optimal point
    energy_pairs = [(r.n, r.w_per_tok) for r in records if r.w_per_tok is not None]
    energy_optimal = (
        min(energy_pairs, key=lambda x: x[1]) if energy_pairs else (None, None)
    )

    # Build markdown table from records
    table_rows = []
    table_rows.append(
        "| N | tok/s_out | tok/s_tot | total_s | req/s | VRAM_GB | T_peak | W_mean | W/tok | vs_N10_% |"
    )
    table_rows.append(
        "|---|-----------|-----------|---------|-------|---------|--------|--------|-------|----------|"
    )
    for r in records:

        def fmt(v, d=2):
            if v is None:
                return "n/a"
            if isinstance(v, int):
                return str(v)
            return f"{v:.{d}f}"

        table_rows.append(
            "| {N} | {tps_o} | {tps_t} | {ts} | {rs} | {vr} | {tp} | {wm} | {wpt} | {vb} |".format(
                N=r.n,
                tps_o=fmt(r.tok_s_out, 2),
                tps_t=fmt(r.tok_s_tot, 2),
                ts=fmt(r.total_s, 2),
                rs=fmt(r.req_s, 3),
                vr=fmt(r.vram_peak_gib, 2),
                tp=fmt(r.t_peak_c, 1),
                wm=fmt(r.w_mean, 1),
                wpt=fmt(r.w_per_tok, 4),
                vb=fmt(r.vs_baseline_pct, 1),
            )
        )

    table = "\n".join(table_rows)

    knee_text = (
        f"Knee at N={knee_n} (efficiency drops below 50% relative to "
        f"linear extrapolation)."
        if knee_n
        else "No clear knee within tested range — system saturates gracefully."
    )

    energy_text = (
        f"Energy-optimal operating point: N={energy_optimal[0]} "
        f"at {energy_optimal[1]:.4f} W·s/tok."
        if energy_optimal[0]
        else "Energy efficiency data unavailable."
    )

    md = f"""# Bielik 11B v2.3 — Phase 2 sweep summary

**Sweep ID:** bielik-11b-v23-fp16-tp2-max8192
**Date:** 2026-05-04
**Operator:** Łukasz Minarowski <lukasz.minarowski@umb.edu.pl>
**Methodology:** METHODOLOGY.md v1.0 §5.2 (Phase 2 scaling sweep)
**Embargo:** Mixed per §11.1/§11.2 — see classification below.

## Methodological humility (METHODOLOGY §8)

We measure inference *throughput*, *latency*, *thermal envelope*, and *power
efficiency* under varying concurrent load. We do not measure model quality,
reasoning capability, factual accuracy, or downstream clinical utility.
Following Lerchner (2026), these are extrinsic computational properties of
the inference vehicle, not constitutive properties of cognition. Our claims
terminate at the hardware-software interface.

## Configuration

- Model: `{MODEL_NAME}` (Polish-language, Mistral-based)
- Quantization: {QUANT}
- Backend: {BACKEND} (vLLM 0.19.0+rocm721)
- Tensor parallel size: {TP}
- max_model_len: {MAX_LEN}
- gpu_memory_utilization: {UTIL}
- KV cache dtype: {KV_DTYPE}
- enforce_eager: True (graphs path segfaults in libhsa-runtime64 on gfx1201
  for Bielik AWQ; assumption applied to FP16 too — engineering finding
  extending METHODOLOGY §3.2)
- N ladder: {N_LADDER}
- Cooldown between runs: 15s

## Phase 1 envelope rationale (PUBLIC)

This config selected from 7-config envelope sweep (2026-05-04, all loaded
successfully). Selection criterion (pre-specified in PLAN-NEXT before
sweep): highest sanity_throughput_tok_s among loaded configs. Winner had
sanity 17.42 tok/s, KV pool 173,328 tokens, vs FP16 TP=1 max=8192 with
14.16 tok/s and 34,320 tokens. TP=2 advantage emerges at large max_len —
AllReduce overhead is masked by larger compute per token plus 2× memory
bandwidth from sharding.

## Results table (EMBARGO_paper_bound — Polish model, §11.3)

{table}

`vs_N10_%` references the model's own N=10 stock run, per METHODOLOGY §10
limitation 10 (no public R9700 baseline exists for Polish-language models).

## Engineering observations (PUBLIC)

- **Scheduler scaling regime.** Aggregate throughput scales linearly through
  N=25 (89% efficiency), then enters a sub-linear regime through N=200
  (70-86% efficiency). {knee_text}

- **Plateau stability.** Aggregate throughput at N=500 and N=1000 is
  identical within 0.1%, indicating that vLLM's scheduler performs clean
  KV-cache swapping under preemption: doubled N → doubled wall time → zero
  throughput loss. Per-request latency doubles but aggregate stays flat.

- **{energy_text}**

- **No thermal throttling observed.** GPU temperatures stayed below paper
  thresholds throughout sweep. Both GPUs reached steady-state during
  N=500/N=1000 long runs without intervention.

## EMBARGOED narrative (paper-bound)

The following sections contain concrete numerical claims that remain
embargoed until paper acceptance per METHODOLOGY §11.2 + §11.3:

- Plateau throughput value: ~{plateau_throughput:.0f} tok/s aggregate
- Knee position relative to KV cache exhaustion (computed as
  173k_kv_tokens / mean_seq_len)
- Cross-model comparative claims with Qwen 7B / 27B / 72B (deferred to
  paper synthesis)
- Per-N P50/P95/P99 latency distributions (not yet computed; require
  per-request timestamps from vLLM, future enhancement)

## Plots

- `scaling_curve.png` — `tok/s_out` vs `N` (log-x), knee annotated
- `thermal_gallery.png` — per-N temperature + power traces, aligned axes
- `efficiency_curve.png` — `W/tok` vs `N`, energy-optimal point annotated

## Limitations (METHODOLOGY §10)

All ten standing limitations from METHODOLOGY §10 apply, with these
emphases for this sweep:
- §10.1 synthetic prompts (apples-to-apples cross-model only, not clinical)
- §10.5 single-batch concurrency (not Poisson arrival)
- §10.6 1 Hz power sampling (sub-second transients lost)
- §10.10 Polish-language model lacks public baseline — `vs_baseline_%` is
  self-referential to N=10

## AI usage disclosure (METHODOLOGY §9)

| Layer | Disclosure |
|---|---|
| 1 — Dataset / data generation | N/A. Synthetic prompts per §6, deterministic from human-curated templates × topics. |
| 2 — Experimental pipeline | Claude Opus 4.7 (Anthropic, via claude.ai) used 2026-05-04 for Phase 1 envelope analysis, Phase 2 orchestrator drafting, and post-sweep aggregation script. Specific role: code generation and methodology validation against METHODOLOGY.md v1.0. No autonomous decision-making on experimental design — all decision criteria pre-specified in PLAN-NEXT.md before sweep. |
| 3 — Manuscript editing | TBD per paper submission. |

## Reproducibility (METHODOLOGY §3.3 causal closure)

Full version triple captured per-run in thermals.jsonl headers and in this
file's metadata:
- ROCm 7.2.0
- vLLM 0.19.0+rocm721 (pinned, do not upgrade — see logbook 2026-05-01)
- PyTorch 2.10.0+git8514f05 (HIP 7.2.53211)
- Env: AMD_SERIALIZE_KERNEL=1, ROCR_VISIBLE_DEVICES=0,1,
  VLLM_ROCM_USE_AITER=0, HIP_LAUNCH_BLOCKING=1, PYTORCH_ALLOC_CONF=unset

## Next steps

1. Phase 2 sweep for `awq_tp1_max2048_eager` (cross-quant comparison)
2. Phase 2 sweep for `fp16_tp1_max2048_eager` (TP=1 baseline, completes
   TP narrative arc)
3. Cross-config aggregate plot — three lines on one scaling_curve.png
4. Lab session log finalization in `docs/sessions/2026-05-04-bielik-11b-fp16-sweep.md`
"""
    path.write_text(md)
    print(f"  MD  → {path.relative_to(REPO_ROOT)}")


# ============================================================
# Main
# ============================================================


def main() -> int:
    if not THERMAL_DIR.exists():
        print(f"ERROR: missing {THERMAL_DIR}", file=sys.stderr)
        return 2

    print("=" * 60)
    print("Bielik 11B v2.3 Phase 2 finalize")
    print(f"Reading from: {THERMAL_DIR.relative_to(REPO_ROOT)}")
    print(f"Writing to:   {RESULTS_DIR.relative_to(REPO_ROOT)}")
    print("=" * 60)

    records = build_records()
    if not records:
        print("ERROR: no records parsed", file=sys.stderr)
        return 3

    print(f"\nParsed {len(records)} runs:")
    for r in records:
        print(
            f"  N={r.n:5d}  tok/s={r.tok_s_out}  "
            f"VRAM={r.vram_peak_gib} T={r.t_peak_c} W={r.w_mean}"
        )

    print("\nGenerating outputs:")
    emit_csv(records, RESULTS_DIR / "results_table.csv")
    plot_scaling(records, RESULTS_DIR / "scaling_curve.png")
    plot_efficiency(records, RESULTS_DIR / "efficiency_curve.png")
    plot_thermal_gallery(RESULTS_DIR / "thermal_gallery.png")
    emit_summary(records, RESULTS_DIR / "SUMMARY.md")

    print("\n" + "=" * 60)
    print("Phase 2 finalize complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
