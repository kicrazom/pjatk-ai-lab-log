"""
Phase 1 envelope exploration for Qwen 3.6 27B on 2x R9700 (gfx1201), TP=2.

Maps the (max_model_len, gpu_memory_utilization) configuration space for
both FP8 and BF16 quantizations, reporting KV cache size and theoretical
max concurrency for each working configuration. Failed configurations
(OOM, KV underflow) are recorded as such.

The output drives Phase 2 sweep design: BF16 with the current known
working config (max_len=1024, util=0.95) gives only 7x max concurrency,
which is too restrictive for a meaningful throughput sweep. This script
searches for BF16 configurations that yield >=20x max concurrency.

Based on sanity_qwen72b_awq.py with adjustments for Qwen 3.6 27B:
 - Hybrid Mamba+Transformer attention requires enforce_eager=True
 - AMD_SERIALIZE_KERNEL=1 (newer PyTorch rejects =3)
 - Both FP8 and BF16 variants tested
 - Each config attempted in fresh process (subprocess isolation) to
   avoid VRAM accumulation across attempts

Output: JSON file at benchmarks/results/qwen36-27b/phase1_envelope.json
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ConfigResult:
    """Result of one envelope probe."""
    quant: str                          # "fp8" or "bf16"
    max_model_len: int
    gpu_memory_utilization: float
    kv_cache_dtype: Optional[str]       # None or "fp8_e4m3" / "fp8_e5m2"
    status: str                         # "ok" / "oom" / "kv_underflow" / "crash"
    weights_gib_per_gpu: Optional[float] = None
    kv_cache_gib_total: Optional[float] = None
    kv_cache_tokens: Optional[int] = None
    max_concurrency: Optional[float] = None
    load_time_s: Optional[float] = None
    cold_inference_tok_s: Optional[float] = None
    error_message: Optional[str] = None


# Configuration space to probe
# Format: (quant, max_len, util, kv_cache_dtype)
PHASE1_CONFIGS = [
    # FP8 — already known good baseline (sanity check)
    ("fp8",  2048, 0.85, None),                # known: 7.53 GiB KV, 61K tokens, 52x
    ("fp8",  4096, 0.90, None),                # explore: more context

    # BF16 — current baseline + exploration
    ("bf16", 1024, 0.95, None),                # known: 2.68 GiB KV, 7K tokens, 7x
    ("bf16",  512, 0.95, None),                # halve context
    ("bf16",  512, 0.97, None),                # halve context + push util
    ("bf16", 1024, 0.97, None),                # same context + push util
    ("bf16", 2048, 0.97, None),                # known to fail at util=0.85, try 0.97
    ("bf16",  768, 0.96, None),                # intermediate context

    # BF16 with FP8 KV cache — may halve KV footprint
    ("bf16", 1024, 0.95, "fp8_e4m3"),
    ("bf16", 2048, 0.95, "fp8_e4m3"),
]


# Models on disk (paths use ~ for portability and privacy)
MODEL_PATHS = {
    "fp8":  os.path.expanduser("~/models/qwen36-27b-fp8"),
    "bf16": os.path.expanduser("~/models/qwen36-27b"),
}


# Output paths (use Path.home() to avoid hardcoded usernames)
RESULTS_DIR = Path.home() / "pjatk-ai-lab-log/benchmarks/results/qwen36-27b"
RESULTS_JSON = RESULTS_DIR / "phase1_envelope.json"


def make_probe_script(
    quant: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_cache_dtype: Optional[str],
) -> str:
    """Generate a self-contained Python script that probes one config.

    The script is run in a subprocess to ensure VRAM is fully released
    between probes (LLM destructor doesn't always free memory cleanly).

    The model_path is already expanded via os.path.expanduser at MODEL_PATHS
    level, so the subprocess receives an absolute path without exposing
    the original tilde-relative form to the embedded script.
    """
    model_path = MODEL_PATHS[quant]
    kv_arg = f', kv_cache_dtype="{kv_cache_dtype}"' if kv_cache_dtype else ""

    return f'''
import json
import sys
import time
from vllm import LLM, SamplingParams

result = {{
    "quant": "{quant}",
    "max_model_len": {max_model_len},
    "gpu_memory_utilization": {gpu_memory_utilization},
    "kv_cache_dtype": {repr(kv_cache_dtype)},
    "status": "crash",
    "error_message": None,
}}

try:
    t0 = time.time()
    llm = LLM(
        model="{model_path}",
        dtype="auto",
        max_model_len={max_model_len},
        gpu_memory_utilization={gpu_memory_utilization},
        enforce_eager=True,
        tensor_parallel_size=2{kv_arg},
    )
    result["load_time_s"] = time.time() - t0

    # Run a single inference to confirm working
    t0 = time.time()
    out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=20, temperature=0.3))
    result["cold_inference_tok_s"] = 20 / (time.time() - t0)

    result["status"] = "ok"

    # Read engine stats from vLLM internals if available
    # (best-effort — vLLM internal API varies between versions)
    try:
        engine = llm.llm_engine
        if hasattr(engine, "scheduler") and hasattr(engine.scheduler, "block_manager"):
            bm = engine.scheduler.block_manager
            block_size = bm.block_size
            num_blocks = bm.num_total_gpu_blocks
            kv_tokens = block_size * num_blocks
            result["kv_cache_tokens"] = kv_tokens
    except Exception as e:
        result["error_message"] = f"stats_read_failed: {{e}}"

except Exception as e:
    err_msg = str(e)
    if "out of memory" in err_msg.lower() or "OutOfMemoryError" in err_msg:
        result["status"] = "oom"
    elif "No available memory for the cache blocks" in err_msg:
        result["status"] = "kv_underflow"
    else:
        result["status"] = "crash"
    result["error_message"] = err_msg[:500]

print("===PROBE_RESULT===")
print(json.dumps(result))
'''


def run_probe(config: tuple) -> ConfigResult:
    """Run one probe in subprocess and parse result.

    Subprocess isolation ensures VRAM is fully released between probes.
    """
    quant, max_len, util, kv_dtype = config

    print(f"\n{'='*60}")
    print(f"Probing: quant={quant} max_len={max_len} util={util} "
          f"kv_dtype={kv_dtype or 'default'}")
    print('='*60)

    script = make_probe_script(quant, max_len, util, kv_dtype)

    # Environment for gfx1201 (mandatory)
    env = os.environ.copy()
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["VLLM_ROCM_USE_AITER"] = "0"
    env["AMD_SERIALIZE_KERNEL"] = "1"
    env["HIP_LAUNCH_BLOCKING"] = "1"
    # Use the activated venv python
    env["PYTHONPATH"] = ""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per probe
        )
    except subprocess.TimeoutExpired:
        return ConfigResult(
            quant=quant,
            max_model_len=max_len,
            gpu_memory_utilization=util,
            kv_cache_dtype=kv_dtype,
            status="crash",
            error_message="timeout (10 min)",
        )

    # Parse result from stdout
    stdout = proc.stdout
    if "===PROBE_RESULT===" not in stdout:
        return ConfigResult(
            quant=quant,
            max_model_len=max_len,
            gpu_memory_utilization=util,
            kv_cache_dtype=kv_dtype,
            status="crash",
            error_message=f"no probe result in stdout. stderr: {proc.stderr[:300]}",
        )

    json_line = stdout.split("===PROBE_RESULT===")[1].strip().split("\n")[0]
    raw = json.loads(json_line)

    # Compute max_concurrency if KV tokens known
    if raw.get("kv_cache_tokens"):
        raw["max_concurrency"] = raw["kv_cache_tokens"] / raw["max_model_len"]

    return ConfigResult(**raw)


def main():
    """Run all phase 1 probes and save results."""
    print(f"=== Phase 1: BF16/FP8 envelope exploration on 2x R9700 ===")
    print(f"Configurations to probe: {len(PHASE1_CONFIGS)}")
    print(f"Estimated time: {len(PHASE1_CONFIGS) * 60} seconds (1 min per probe)")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    t_start = time.time()

    for i, config in enumerate(PHASE1_CONFIGS, 1):
        print(f"\n[{i}/{len(PHASE1_CONFIGS)}] elapsed: {(time.time()-t_start):.0f}s")
        result = run_probe(config)
        results.append(result)

        # Print one-line summary
        if result.status == "ok":
            print(f"  ✓ OK — KV: {result.kv_cache_tokens} tokens, "
                  f"max_conc: {result.max_concurrency:.1f}x, "
                  f"cold tok/s: {result.cold_inference_tok_s:.2f}")
        else:
            print(f"  ✗ {result.status.upper()}: "
                  f"{(result.error_message or '')[:100]}")

        # Save incrementally (in case we crash later)
        with open(RESULTS_JSON, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    # Summary table
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY")
    print("=" * 80)
    print(f"{'quant':6} {'max_len':>8} {'util':>5} {'kv_dtype':>10} "
          f"{'status':>14} {'KV_tok':>8} {'max_conc':>9}")
    print("-" * 80)
    for r in results:
        kv_dtype = r.kv_cache_dtype or "-"
        kv_tok = str(r.kv_cache_tokens) if r.kv_cache_tokens else "-"
        max_conc = f"{r.max_concurrency:.1f}x" if r.max_concurrency else "-"
        print(f"{r.quant:6} {r.max_model_len:>8} {r.gpu_memory_utilization:>5.2f} "
              f"{kv_dtype:>10} {r.status:>14} {kv_tok:>8} {max_conc:>9}")

    print(f"\nResults saved to: {RESULTS_JSON}")
    print(f"Total elapsed: {(time.time()-t_start):.0f}s")

    # Recommend best configs for Phase 2
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR PHASE 2 SWEEP")
    print("=" * 80)
    ok_results = [r for r in results if r.status == "ok" and r.max_concurrency]
    fp8_best = max(
        (r for r in ok_results if r.quant == "fp8"),
        key=lambda r: r.max_concurrency,
        default=None,
    )
    bf16_best = max(
        (r for r in ok_results if r.quant == "bf16"),
        key=lambda r: r.max_concurrency,
        default=None,
    )
    if fp8_best:
        print(f"FP8 best:  max_len={fp8_best.max_model_len}, "
              f"util={fp8_best.gpu_memory_utilization}, "
              f"kv_dtype={fp8_best.kv_cache_dtype or 'default'}, "
              f"max_conc={fp8_best.max_concurrency:.1f}x")
    if bf16_best:
        print(f"BF16 best: max_len={bf16_best.max_model_len}, "
              f"util={bf16_best.gpu_memory_utilization}, "
              f"kv_dtype={bf16_best.kv_cache_dtype or 'default'}, "
              f"max_conc={bf16_best.max_concurrency:.1f}x")


if __name__ == "__main__":
    main()
