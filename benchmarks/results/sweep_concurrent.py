#!/usr/bin/env python3
"""
Sweep vLLM concurrent benchmark across TP={1,2} and N={10,25,50,100,200,...}.
Dumps results to scaling_data.json for consumption by plot_scaling.py.

Usage:
    python sweep_concurrent.py                    # default sweep
    python sweep_concurrent.py 10,50,200          # custom N values
    python sweep_concurrent.py 10,50,200 1,2      # custom N and TP
"""
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

DEFAULTS_N = [10, 25, 50, 100, 200]
DEFAULTS_TP = [1, 2]
SCRIPT = Path.home() / "benchmarks" / "test_concurrent.py"
OUTPUT_JSON = Path(__file__).parent / "scaling_data.json"


def parse_throughput(stdout: str) -> float | None:
    """Extract 'Output throughput: X tok/s' from benchmark stdout."""
    for line in stdout.splitlines():
        if "Output throughput" in line and "tok/s" in line:
            # e.g. "Output throughput:   1731.3 tok/s"
            tokens = line.split(":")[-1].strip().split()[0]
            try:
                return float(tokens)
            except ValueError:
                return None
    return None


def run_single(tp: int, n: int) -> dict:
    """Run one benchmark invocation, return result dict."""
    print(f"  → TP={tp}, N={n} ...", end=" ", flush=True)
    env = os.environ.copy()
    if tp == 1:
        env["ROCR_VISIBLE_DEVICES"] = "0"
    else:
        env.pop("ROCR_VISIBLE_DEVICES", None)  # let both GPUs be visible

    t0 = time.time()
    result = subprocess.run(
        ["python", "-u", str(SCRIPT), str(tp), str(n)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    duration = time.time() - t0

    tput = parse_throughput(result.stdout)
    if tput is None:
        print(f"FAILED ({duration:.0f}s)")
        if result.returncode != 0:
            print(result.stderr[-500:])
        return {"n_prompts": n, "output_throughput_toks": None, "error": "parse_failed"}

    print(f"{tput:.1f} tok/s ({duration:.0f}s)")
    return {"n_prompts": n, "output_throughput_toks": tput}


def main():
    ns = DEFAULTS_N
    tps = DEFAULTS_TP
    if len(sys.argv) > 1:
        ns = [int(x) for x in sys.argv[1].split(",")]
    if len(sys.argv) > 2:
        tps = [int(x) for x in sys.argv[2].split(",")]

    print(f"Sweep: TP={tps}, N={ns}")
    print(f"Script: {SCRIPT}")
    print(f"Output: {OUTPUT_JSON}\n")

    if not SCRIPT.exists():
        print(f"ERROR: benchmark script not found at {SCRIPT}")
        sys.exit(1)

    # Load existing data if present (we'll merge)
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON) as f:
            data = json.load(f)
    else:
        data = {
            "hardware": {
                "cpu": "AMD Ryzen 9 9950X3D",
                "gpu_model": "AMD Radeon AI PRO R9700",
                "gpu_arch": "gfx1201 (RDNA 4)",
                "gpu_vram_gb": 32,
                "gpu_count": 2,
                "ram_gb": 96,
                "ram_speed": "DDR5-6000",
            },
            "software": {
                "os": "Kubuntu 24.04",
                "kernel": "6.17",
                "rocm": "7.2.1",
                "vllm": "0.19.0+rocm721",
                "pytorch": "2.10.0+git8514f05",
                "nccl": "2.27.7",
            },
            "workload": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "dtype": "float16",
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.70,
                "enforce_eager": True,
                "max_tokens_per_request": 128,
                "input_tokens_mean": 13.4,
                "output_tokens_mean": 126.9,
                "note": "Prompts share common template prefix; prefix caching active",
            },
            "kv_cache": {
                "tp1": {"size_tokens": 120944, "max_concurrency_4k": 29.53},
                "tp2": {"size_tokens": 487296, "max_concurrency_4k": 118.97},
            },
            "runs": {"tp1": [], "tp2": []},
        }

    # Run sweep
    for tp in tps:
        key = f"tp{tp}"
        print(f"\n=== TP={tp} ===")
        existing_ns = {r["n_prompts"] for r in data["runs"].get(key, [])}
        for n in ns:
            result = run_single(tp, n)
            if "error" not in result:
                # Remove any previous result for this N (re-run override)
                data["runs"][key] = [r for r in data["runs"].get(key, []) if r["n_prompts"] != n]
                data["runs"][key].append(result)

        # Sort by N for readable JSON
        data["runs"][key].sort(key=lambda r: r["n_prompts"])

    data["date"] = date.today().isoformat()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Regenerate plot: python plot_scaling.py")


if __name__ == "__main__":
    main()
