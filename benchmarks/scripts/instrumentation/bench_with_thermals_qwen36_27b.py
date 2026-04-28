"""
Run a Qwen 3.6 27B TP=2 vLLM benchmark with background thermal/utilization
sampling.

Identical to bench_with_thermals_qwen72b.py except BENCHMARK points to the
27B-specific script and accepts --quant. All output conventions (JSONL,
events.json, PNG, bench.log) match the 7B/72B Plan A artifacts so
plot_scaling.py and plot_thermals.py work unchanged.

Usage:
   python bench_with_thermals_qwen36_27b.py 2 100 --quant fp8 \\
       --name fp8-tp2-n100 --out-dir thermal-runs/qwen36-27b/

   python bench_with_thermals_qwen36_27b.py 2 25 --quant bf16 \\
       --max-len 1024 --util 0.95 \\
       --name bf16-tp2-n25 --out-dir thermal-runs/qwen36-27b/
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
BENCHMARK = HERE.parent / "runners" / "test_concurrent_qwen36_27b.py"


IDLE_BEFORE_S = 5.0
IDLE_AFTER_S = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tp", type=int, help="Tensor parallel size (must be 2 for 27B)")
    ap.add_argument("n", type=int, help="Number of concurrent prompts")
    ap.add_argument("--quant", choices=["fp8", "bf16"], default="bf16",
                    help="Quantization variant (default: bf16)")
    ap.add_argument("--max-len", type=int, default=None,
                    help="Override max_model_len for inner benchmark")
    ap.add_argument("--util", type=float, default=None,
                    help="Override gpu_memory_utilization for inner benchmark")
    ap.add_argument("--kv-dtype", default=None,
                    help="Override kv_cache_dtype for inner benchmark")
    ap.add_argument("--name", help="Run name (default: <quant>-tpX-nY)")
    ap.add_argument("--out-dir", default=".", help="Output directory")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampler interval")
    args = ap.parse_args()

    name = args.name or f"{args.quant}-tp{args.tp}-n{args.n}"
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl = out_dir / f"{name}-thermals.jsonl"
    events_json = out_dir / f"{name}-events.json"
    png = out_dir / f"{name}-thermals.png"
    bench_log = out_dir / f"{name}-bench.log"

    for p in (SAMPLER, PLOTTER, BENCHMARK):
        if not p.exists():
            print(f"ERROR: missing {p}")
            sys.exit(1)

    print(f"Run name: {name}")
    print(f"Output dir: {out_dir}")
    print(f"Quantization: {args.quant.upper()}")

    print("Starting sampler...")
    sampler_proc = subprocess.Popen(
        [sys.executable, str(SAMPLER), str(jsonl), "--interval", str(args.interval)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    t_wall_start = time.time()
    events = []

    try:
        print(f"Collecting {IDLE_BEFORE_S}s baseline...")
        time.sleep(IDLE_BEFORE_S)

        t_bench_start = time.time() - t_wall_start
        events.append({"t": round(t_bench_start, 2),
                       "label": f"bench start ({args.quant.upper()}, "
                                f"TP={args.tp}, N={args.n})",
                       "color": "#2ca02c"})

        print(f"Running benchmark ({args.quant.upper()}, "
              f"TP={args.tp}, N={args.n})...")
        env = os.environ.copy()
        # For 27B both quantizations TP=2 mandatory — do not mask GPUs
        env.pop("ROCR_VISIBLE_DEVICES", None)

        # Build inner benchmark command
        cmd = [sys.executable, "-u", str(BENCHMARK),
               str(args.tp), str(args.n),
               "--quant", args.quant]
        if args.max_len is not None:
            cmd += ["--max-len", str(args.max_len)]
        if args.util is not None:
            cmd += ["--util", str(args.util)]
        if args.kv_dtype is not None:
            cmd += ["--kv-dtype", args.kv_dtype]

        with open(bench_log, "w") as f:
            bench_result = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=1800,
            )

        t_bench_end = time.time() - t_wall_start
        events.append({"t": round(t_bench_end, 2),
                       "label": "bench end",
                       "color": "#d62728"})
        print(f"Benchmark done (rc={bench_result.returncode})")

        print(f"Collecting {IDLE_AFTER_S}s cooldown...")
        time.sleep(IDLE_AFTER_S)

    finally:
        print("Stopping sampler...")
        sampler_proc.send_signal(signal.SIGINT)
        try:
            sampler_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sampler_proc.kill()

    with open(events_json, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Events: {events_json}")

    print("Generating thermal plot...")
    subprocess.run([sys.executable, str(PLOTTER), str(jsonl), str(png),
                    "--events", str(events_json)], check=True)

    try:
        bench_text = bench_log.read_text()
        for line in bench_text.splitlines():
            if any(k in line for k in ("Output throughput", "Requests/second",
                                        "Total time", "Load time")):
                print(f"  {line.strip()}")
    except Exception:
        pass

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
