#!/usr/bin/env python3
"""
System thermal/utilization sampler for vLLM benchmarks.
Writes line-delimited JSON (one sample per line) so you can `tail -f` it live.

Usage:
    python sample_system.py thermals.jsonl          # run until Ctrl+C
    python sample_system.py thermals.jsonl --duration 300
    python sample_system.py thermals.jsonl --interval 0.5

Sampled:
    - CPU: aggregate utilization %, Tctl temperature (via psutil)
    - Each GPU (rocm-smi): utilization %, edge temp, VRAM used, power
"""
import argparse
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def cpu_temp() -> float | None:
    try:
        temps = psutil.sensors_temperatures()
        for chip in ("k10temp", "zenpower", "coretemp"):
            if chip in temps:
                for r in temps[chip]:
                    if r.label in ("Tctl", "Tdie", ""):
                        return r.current
                if temps[chip]:
                    return temps[chip][0].current
    except Exception:
        pass
    return None


def rocm_smi_json() -> dict:
    """Run rocm-smi --json, return parsed dict or {}."""
    try:
        r = subprocess.run(
            ["rocm-smi", "--json", "--showuse", "--showtemp",
             "--showmeminfo", "vram", "--showproductname", "--showpower"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return json.loads(r.stdout.strip() or "{}")
    except Exception:
        return {}


def parse_gpus(raw: dict) -> list[dict]:
    """Turn rocm-smi JSON into [{idx, name, use, temp, vram_used_b, power_w, is_igpu}, ...]"""
    gpus = []
    for key, val in raw.items():
        m = re.search(r"card(\d+)", key, re.IGNORECASE)
        if not m or not isinstance(val, dict):
            continue
        idx = int(m.group(1))
        g = {"idx": idx, "name": None, "use": None, "temp": None,
             "vram_used_b": None, "power_w": None, "is_igpu": False}

        for fn in ("Card Series", "Card series", "Card Model", "Product Name"):
            if fn in val:
                g["name"] = str(val[fn]).strip()
                break
        for fn in ("GPU use (%)", "GPU Activity"):
            if fn in val:
                try: g["use"] = int(float(val[fn]))
                except: pass
                break
        for fn in ("Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)"):
            if fn in val:
                try: g["temp"] = float(val[fn])
                except: pass
                break
        try: g["vram_used_b"] = int(val.get("VRAM Total Used Memory (B)", 0)) or None
        except: pass
        for fn in ("Average Graphics Package Power (W)", "GPU Power (W)", "Socket Power (W)"):
            if fn in val:
                try: g["power_w"] = float(val[fn])
                except: pass
                break

        # Heuristic: iGPU has "Graphics" in name (e.g. "AMD Radeon Graphics") or small VRAM
        if g["name"] and "AI PRO" not in g["name"] and "Graphics" in g["name"]:
            g["is_igpu"] = True

        gpus.append(g)
    gpus.sort(key=lambda x: x["idx"])
    return gpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="Output JSONL file")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between samples")
    ap.add_argument("--duration", type=float, default=None, help="Stop after N seconds")
    args = ap.parse_args()

    stop = False
    def handle_sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    out = Path(args.output)
    print(f"Sampling → {out} (interval={args.interval}s)", file=sys.stderr)
    if args.duration:
        print(f"Will stop after {args.duration}s", file=sys.stderr)
    print("Send SIGINT/Ctrl+C to stop.", file=sys.stderr)

    # Prime psutil (first cpu_percent call returns 0)
    psutil.cpu_percent(interval=None)
    time.sleep(0.1)

    t_start = time.time()
    n = 0
    with open(out, "w") as f:
        while not stop:
            t_elapsed = time.time() - t_start
            if args.duration and t_elapsed >= args.duration:
                break

            raw = rocm_smi_json()
            gpus = parse_gpus(raw)
            sample = {
                "t": round(t_elapsed, 3),
                "iso": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
                "cpu_temp": cpu_temp(),
                "gpus": gpus,
            }
            f.write(json.dumps(sample) + "\n")
            f.flush()
            n += 1

            # Sleep accounting for sample collection time
            next_t = t_start + n * args.interval
            sleep_for = max(0, next_t - time.time())
            time.sleep(sleep_for)

    print(f"\nCollected {n} samples over {time.time() - t_start:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
