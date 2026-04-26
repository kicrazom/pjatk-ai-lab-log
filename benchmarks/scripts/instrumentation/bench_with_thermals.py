#!/usr/bin/env python3
"""
Run a vLLM benchmark with background thermal/utilization sampling.
Produces timestamped JSONL + events file + runs the plotter.

Usage:
    python bench_with_thermals.py 1 100              # TP=1, N=100
    python bench_with_thermals.py 2 200 --name tp2-200
    python bench_with_thermals.py 1 100 --out-dir ~/benchmarks/thermal-runs/

What it does:
  1. Starts sample_system.py in background
  2. Waits 5s (baseline idle)
  3. Runs ~/benchmarks/test_concurrent.py with given TP and N
  4. Waits 5s (cooldown capture)
  5. Kills sampler
  6. Writes events.json with benchmark start/end markers
  7. Calls plot_thermals.py
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SAMPLER = HERE / "sample_system.py"
PLOTTER = HERE.parent / "plotting" / "plot_thermals.py"
BENCHMARK = Path.home() / "benchmarks" / "test_concurrent.py"
IDLE_BEFORE_S = 5.0
IDLE_AFTER_S  = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tp", type=int, help="Tensor parallel size (1 or 2)")
    ap.add_argument("n", type=int, help="Number of concurrent prompts")
    ap.add_argument("--name", help="Run name (default: tpX-nY)")
    ap.add_argument("--out-dir", default=".", help="Output directory")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampler interval")
    args = ap.parse_args()

    name = args.name or f"tp{args.tp}-n{args.n}"
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{name}-thermals.jsonl"
    events_json = out_dir / f"{name}-events.json"
    png = out_dir / f"{name}-thermals.png"
    bench_log = out_dir / f"{name}-bench.log"

    # Sanity checks
    for p in (SAMPLER, PLOTTER, BENCHMARK):
        if not p.exists():
            print(f"ERROR: missing {p}")
            sys.exit(1)

    print(f"Run name: {name}")
    print(f"Output dir: {out_dir}")

    # 1. Start sampler
    print("Starting sampler...")
    sampler_proc = subprocess.Popen(
        [sys.executable, str(SAMPLER), str(jsonl), "--interval", str(args.interval)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    t_wall_start = time.time()
    events = []

    try:
        # 2. Baseline idle
        print(f"Collecting {IDLE_BEFORE_S}s baseline...")
        time.sleep(IDLE_BEFORE_S)

        # 3. Benchmark
        t_bench_start = time.time() - t_wall_start
        events.append({"t": round(t_bench_start, 2),
                       "label": f"bench start (TP={args.tp}, N={args.n})",
                       "color": "#2ca02c"})
        print(f"Running benchmark (TP={args.tp}, N={args.n})...")

        env = os.environ.copy()
        if args.tp == 1:
            env["ROCR_VISIBLE_DEVICES"] = "0"
        else:
            env.pop("ROCR_VISIBLE_DEVICES", None)

        with open(bench_log, "w") as f:
            bench_result = subprocess.run(
                [sys.executable, "-u", str(BENCHMARK), str(args.tp), str(args.n)],
                stdout=f, stderr=subprocess.STDOUT, env=env, timeout=900,
            )

        t_bench_end = time.time() - t_wall_start
        events.append({"t": round(t_bench_end, 2),
                       "label": "bench end",
                       "color": "#d62728"})
        print(f"Benchmark done (rc={bench_result.returncode})")

        # 4. Cooldown capture
        print(f"Collecting {IDLE_AFTER_S}s cooldown...")
        time.sleep(IDLE_AFTER_S)

    finally:
        # 5. Stop sampler
        print("Stopping sampler...")
        sampler_proc.send_signal(signal.SIGINT)
        try:
            sampler_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sampler_proc.kill()

    # 6. Write events
    with open(events_json, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Events: {events_json}")

    # 7. Plot
    print("Generating thermal plot...")
    subprocess.run([sys.executable, str(PLOTTER), str(jsonl), str(png),
                    "--events", str(events_json)], check=True)

    # Also print throughput from bench log if available
    try:
        bench_text = bench_log.read_text()
        for line in bench_text.splitlines():
            if any(k in line for k in ("Output throughput", "Requests/second", "Total time")):
                print(f"  {line.strip()}")
    except Exception:
        pass

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
