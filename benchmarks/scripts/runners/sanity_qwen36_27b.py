"""
Phase 1 envelope exploration for Qwen 3.6 27B on 2x R9700 (gfx1201), TP=2.

Maps the (max_model_len, gpu_memory_utilization) configuration space for
both FP8 and BF16 quantizations, reporting KV cache size and theoretical
max concurrency for each working configuration. Failed configurations
(OOM, KV underflow) are recorded as such.

Architecture notes:
 - Hybrid Mamba+Transformer attention requires enforce_eager=True
 - AMD_SERIALIZE_KERNEL=1 (newer PyTorch rejects =3)
 - Both FP8 and BF16 variants tested
 - Each config attempted in fresh process (subprocess isolation) to
   avoid VRAM accumulation across attempts
 - Host has 3 GPUs visible to ROCm (2x R9700 + iGPU RAPHAEL on Ryzen
   9950X3D); rocm-smi probe filters to R9700 only via Card Series.
   ROCR_VISIBLE_DEVICES=0,1 already scopes vLLM/torch.
 - vLLM 0.19 KV cache stats: engine.vllm_config.cache_config (NOT
   engine.scheduler.block_manager — that path was removed in 0.19).

Output: one JSON file per config under
benchmarks/results/hardware_envelope/, named {config_id}.json,
conforming to benchmarks/results/hardware_envelope/SCHEMA.md (v1).
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Schema version — bumped on breaking changes to JSON output shape
SCHEMA_VERSION = 1

# Model HuggingFace ids (separate from local model_path for provenance)
MODEL_HF_IDS = {
    "fp8":  "Qwen/Qwen3.6-27B-Instruct-FP8",
    "bf16": "Qwen/Qwen3.6-27B-Instruct",
}


@dataclass
class ConfigResult:
    """Result of one envelope probe.

    Field order and naming follow SCHEMA.md v1. Failed probes (loaded=
    False) get all schema fields populated, with measurement fields
    set to None — downstream consumers can iterate over partial records.
    """
    # Identity
    config_id: str
    model: str
    model_path: str
    # Config (vLLM args)
    quantization: str
    kv_cache_dtype: Optional[str]
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    enforce_eager: bool
    # Environment snapshot
    env: dict
    # Outcome
    loaded: bool
    error: Optional[str]
    error_class: Optional[str]
    # Measurements (None when loaded=False)
    load_time_s: Optional[float]
    peak_vram_gb_per_gpu: Optional[list]
    peak_vram_source: str                       # "rocm_smi" only in v1.5
    weights_gib_per_gpu: Optional[list]         # always None in v1.5; v2 work
    kv_cache_gb: Optional[float]                # always None in v1.5; v2 work
    kv_cache_tokens: Optional[int]
    max_concurrency: Optional[float]
    sanity_throughput_tok_s: Optional[float]
    # Provenance
    timestamp_utc: str
    rocm_version: str
    vllm_version: str
    torch_version: str
    torch_hip_version: Optional[str]
    gpu_model: str
    gpu_gcn_arch_name: str
    gpu_count: int
    schema_version: int = SCHEMA_VERSION


# Configuration space to probe
# Format: (quant, max_len, util, kv_cache_dtype)
PHASE1_CONFIGS = [
    # FP8 — already known good baseline (sanity check)
    ("fp8",  2048, 0.85, None),
    ("fp8",  4096, 0.90, None),

    # BF16 — current baseline + exploration
    ("bf16", 1024, 0.95, None),
    ("bf16",  512, 0.95, None),
    ("bf16",  512, 0.97, None),
    ("bf16", 1024, 0.97, None),
    ("bf16", 2048, 0.97, None),
    ("bf16",  768, 0.96, None),

    # BF16 with FP8 KV cache — may halve KV footprint
    ("bf16", 1024, 0.95, "fp8_e4m3"),
    ("bf16", 2048, 0.95, "fp8_e4m3"),
]

# Local model paths (Path.home() avoids hardcoded usernames)
MODEL_PATHS = {
    "fp8":  str(Path.home() / "models/qwen36-27b-fp8"),
    "bf16": str(Path.home() / "models/qwen36-27b"),
}

# Output directory — aligns with SCHEMA.md and embargo policy
RESULTS_DIR = Path.home() / "navimed-umb/benchmarks/results/hardware_envelope"

# Architecture constants — currently fixed for Qwen 3.6 27B on 2x R9700
TENSOR_PARALLEL_SIZE = 2
ENFORCE_EAGER = True


# ============================================================
# rocm-smi VRAM measurement (R9700-only filtering)
# ============================================================

def _rocm_smi_card_series() -> dict:
    """Map card index → Card Series name via `rocm-smi --showproductname --json`.

    Returns {0: "AMD Radeon AI PRO R9700", 1: "AMD Radeon AI PRO R9700",
    2: "AMD Radeon Graphics", ...}. Empty dict on rocm-smi failure.

    The host has 3 GPUs visible to ROCm (2x R9700 + iGPU). VRAM probes
    must filter to R9700 only — both for correctness (iGPU "VRAM" is
    system RAM) and reproducibility (PCIe enumeration order may differ
    across reboots).
    """
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(proc.stdout)
    except Exception:
        return {}

    series_map = {}
    for card_key, card in data.items():
        if not card_key.startswith("card"):
            continue
        try:
            idx = int(card_key.removeprefix("card"))
        except ValueError:
            continue
        # Tolerate key variations across rocm-smi versions
        series_key = next(
            (k for k in card if "Card Series" in k or "Card series" in k),
            None,
        )
        if series_key:
            series_map[idx] = card[series_key]
    return series_map


def _r9700_card_indices() -> list:
    """Return sorted list of card indices that are R9700.

    Filters by 'R9700' substring in Card Series. Returns empty list
    on rocm-smi failure (callers must handle gracefully).
    """
    series = _rocm_smi_card_series()
    if not series:
        return []
    return sorted(idx for idx, name in series.items() if "R9700" in name)


def _rocm_smi_vram_used_gib(card_indices: list) -> Optional[list]:
    """Read VRAM used (GiB) for the given card indices.

    Returns list aligned with card_indices order, or None on failure.
    """
    if not card_indices:
        return None
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(proc.stdout)
    except Exception:
        return None

    out = []
    for idx in card_indices:
        card_key = f"card{idx}"
        if card_key not in data:
            return None
        card = data[card_key]
        used_key = next(
            (k for k in card if "Used" in k and "VRAM" in k),
            None,
        )
        if used_key is None:
            return None
        try:
            used_bytes = int(card[used_key])
        except (ValueError, TypeError):
            return None
        out.append(round(used_bytes / 1024**3, 3))
    return out


# ============================================================
# Subprocess probe script generation
# ============================================================

def make_config_id(
    quant: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_cache_dtype: Optional[str],
) -> str:
    """Build canonical config_id matching SCHEMA.md grammar.

    Format: qwen36_27b_{quant}_max{max_len}_util{util_x100}[_kv{kv_short}]

    util is multiplied by 100 and zero-padded to 3 digits so that
    lexicographic sort matches numeric sort (085 < 095 < 097).
    """
    util_x100 = f"{int(round(gpu_memory_utilization * 100)):03d}"
    base = f"qwen36_27b_{quant}_max{max_model_len}_util{util_x100}"
    if kv_cache_dtype:
        kv_short = kv_cache_dtype.replace("_", "")
        return f"{base}_kv{kv_short}"
    return base


def make_probe_script(
    quant: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_cache_dtype: Optional[str],
) -> str:
    """Generate a self-contained Python script that probes one config.

    Reports back a single JSON line via stdout marked with PROBE_RESULT
    sentinel. Includes vLLM 0.19 internals reading via the correct
    `engine.vllm_config.cache_config` path discovered on 2026-04-29.
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
    "error": None,
    "load_time_s": None,
    "sanity_throughput_tok_s": None,
    "kv_cache_tokens": None,
    "peak_vram_per_card_gib": {{}},
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

    # Sanity inference (single prompt, 20 tokens) — proves model lives
    t0 = time.time()
    out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=20, temperature=0.3))
    result["sanity_throughput_tok_s"] = 20 / (time.time() - t0)

    result["status"] = "ok"

    # vLLM 0.19 KV cache stats via vllm_config.cache_config
    # (engine.scheduler does not exist in 0.19)
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        num_blocks = cc.num_gpu_blocks
        block_size = cc.block_size
        if num_blocks and block_size:
            result["kv_cache_tokens"] = num_blocks * block_size
    except Exception as e:
        result["error"] = f"cache_config_read_failed: {{e}}"

except Exception as e:
    err_msg = str(e)
    if "out of memory" in err_msg.lower() or "OutOfMemoryError" in err_msg:
        result["status"] = "oom"
    elif "No available memory for the cache blocks" in err_msg:
        result["status"] = "kv_underflow"
    elif "HSA_STATUS_ERROR" in err_msg or "hipError" in err_msg or "HIP error" in err_msg:
        result["status"] = "rocm_error"
    else:
        result["status"] = "crash"
    result["error"] = err_msg[:500]

# Probe rocm-smi from inside subprocess — VRAM is still allocated here.
# Done after the try/except so it runs whether load succeeded or not
# (failed loads may still have partial allocations worth reporting).
import subprocess as _smi_sub
try:
    _smi_proc = _smi_sub.run(
        ["rocm-smi", "--showmeminfo", "vram", "--json"],
        capture_output=True, text=True, timeout=10,
    )
    _smi_data = json.loads(_smi_proc.stdout)
    _vram_per_card = {{}}
    for _card_key, _card in _smi_data.items():
        if not _card_key.startswith("card"):
            continue
        _used_key = next(
            (k for k in _card if "Used" in k and "VRAM" in k),
            None,
        )
        if _used_key:
            try:
                _vram_per_card[_card_key] = round(int(_card[_used_key]) / 1024**3, 3)
            except (ValueError, TypeError):
                pass
    result["peak_vram_per_card_gib"] = _vram_per_card
except Exception as _e:
    result["peak_vram_per_card_gib"] = {{"error": str(_e)}}

print("===PROBE_RESULT===")
print(json.dumps(result))
'''


# ============================================================
# Provenance capture (versions, GPU info)
# ============================================================

def capture_versions_and_gpu() -> dict:
    """Capture ROCm/vLLM/torch versions and GPU identification once.

    Called once in main() — values are constant per host.
    """
    import torch
    import vllm

    props = torch.cuda.get_device_properties(0)
    return {
        "rocm_version": _read_rocm_version(),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "gpu_model": props.name,
        "gpu_gcn_arch_name": props.gcnArchName,
        "gpu_count": torch.cuda.device_count(),
    }


def _read_rocm_version() -> str:
    """Read ROCm release version from /opt/rocm/.info/version.

    Uses filesystem rather than `rocm-smi --version` because rocm-smi
    has its own independent versioning. Returns 'unknown' on missing file.
    """
    version_file = Path("/opt/rocm/.info/version")
    if version_file.is_file():
        try:
            return version_file.read_text().strip()
        except Exception:
            return "unknown"
    return "unknown"


def _utc_now_z() -> str:
    """RFC 3339 UTC timestamp with Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def classify_status(legacy_status: str) -> tuple:
    """Map legacy 4+1-way status to schema's (loaded, error_class)."""
    if legacy_status == "ok":
        return True, None
    return False, legacy_status


# ============================================================
# Probe orchestration
# ============================================================

def run_probe(config: tuple, host_info: dict, r9700_indices: list) -> ConfigResult:
    """Run one probe in subprocess, measure VRAM via rocm-smi, parse result."""
    quant, max_len, util, kv_dtype = config
    config_id = make_config_id(quant, max_len, util, kv_dtype)

    print(f"\n{'='*60}")
    print(f"Probing: {config_id}")
    print('='*60)

    # Pre-load VRAM baseline (R9700 only)
    vram_pre = _rocm_smi_vram_used_gib(r9700_indices) if r9700_indices else None

    script = make_probe_script(quant, max_len, util, kv_dtype)

    # Subprocess environment — gfx1201 mandatory env vars
    env = os.environ.copy()
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["VLLM_ROCM_USE_AITER"] = "0"
    env["AMD_SERIALIZE_KERNEL"] = "1"
    env["HIP_LAUNCH_BLOCKING"] = "1"
    env["PYTHONPATH"] = ""

    env_snapshot = {
        k: env.get(k) for k in (
            "VLLM_ROCM_USE_AITER",
            "AMD_SERIALIZE_KERNEL",
            "HIP_LAUNCH_BLOCKING",
            "PYTORCH_ALLOC_CONF",
        )
    }

    base_fields = {
        "config_id": config_id,
        "model": MODEL_HF_IDS[quant],
        "model_path": MODEL_PATHS[quant],
        "quantization": quant,
        "kv_cache_dtype": kv_dtype,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "max_model_len": max_len,
        "gpu_memory_utilization": util,
        "enforce_eager": ENFORCE_EAGER,
        "env": env_snapshot,
        "peak_vram_source": "rocm_smi",
        "timestamp_utc": _utc_now_z(),
        **host_info,
    }

    # Run probe in subprocess
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return _build_failed_result(
            base_fields,
            error_class="timeout",
            error="probe subprocess exceeded 600s wall time",
        )

    # Parse subprocess result (peak VRAM is now captured INSIDE subprocess
    # via rocm-smi probe before teardown — see make_probe_script). Parent
    # only needs vram_pre as baseline; vram_post comes from child JSON.
    stdout = proc.stdout
    if "===PROBE_RESULT===" not in stdout:
        stderr_tail = proc.stderr[-500:] if proc.stderr else ""
        return _build_failed_result(
            base_fields,
            error_class="crash",
            error=f"no probe result marker in stdout. stderr tail: {stderr_tail}",
        )

    json_line = stdout.split("===PROBE_RESULT===")[1].strip().split("\n")[0]
    raw = json.loads(json_line)

    legacy_status = raw["status"]
    loaded, error_class = classify_status(legacy_status)

    # Compute peak VRAM as delta (post - pre), R9700 only.
    # post values come from child subprocess (captured before teardown)
    # via raw["peak_vram_per_card_gib"]; filter to R9700 indices only.
    peak_vram = None
    child_vram = raw.get("peak_vram_per_card_gib")
    if vram_pre is not None and isinstance(child_vram, dict) and "error" not in child_vram:
        try:
            post_filtered = [child_vram[f"card{idx}"] for idx in r9700_indices]
            peak_vram = [round(post - pre, 3)
                         for pre, post in zip(vram_pre, post_filtered)]
        except (KeyError, TypeError):
            peak_vram = None

    # Compute max_concurrency if KV tokens known
    max_concurrency = None
    kv_tokens = raw.get("kv_cache_tokens")
    if kv_tokens:
        max_concurrency = round(kv_tokens / max_len, 2)

    return ConfigResult(
        **base_fields,
        loaded=loaded,
        error=raw.get("error"),
        error_class=error_class,
        load_time_s=raw.get("load_time_s"),
        peak_vram_gb_per_gpu=peak_vram,
        weights_gib_per_gpu=None,
        kv_cache_gb=None,
        kv_cache_tokens=kv_tokens,
        max_concurrency=max_concurrency,
        sanity_throughput_tok_s=raw.get("sanity_throughput_tok_s"),
    )


def _build_failed_result(base_fields: dict, error_class: str, error: str) -> ConfigResult:
    """Construct ConfigResult for cases where subprocess never produced
    a parseable result (timeout or no PROBE_RESULT marker)."""
    return ConfigResult(
        **base_fields,
        loaded=False,
        error=error[:500],
        error_class=error_class,
        load_time_s=None,
        peak_vram_gb_per_gpu=None,
        weights_gib_per_gpu=None,
        kv_cache_gb=None,
        kv_cache_tokens=None,
        max_concurrency=None,
        sanity_throughput_tok_s=None,
    )


def write_per_config_json(result: ConfigResult) -> Path:
    """Write one config's result to its own JSON file (crash-resilient)."""
    out_path = RESULTS_DIR / f"{result.config_id}.json"
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    return out_path


# ============================================================
# Main entry point
# ============================================================

def main():
    """Run all phase 1 probes and save results."""
    print("=" * 70)
    print("Phase 1: BF16/FP8 envelope exploration on 2x R9700 (gfx1201)")
    print("=" * 70)
    print(f"Configurations to probe: {len(PHASE1_CONFIGS)}")
    print(f"Estimated time: {len(PHASE1_CONFIGS) * 60} seconds (~1 min per probe)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover R9700 cards via rocm-smi (filters out iGPU on Ryzen X3D)
    r9700_indices = _r9700_card_indices()
    if r9700_indices:
        print(f"R9700 cards detected at indices: {r9700_indices}")
    else:
        print("WARNING: rocm-smi probe failed — peak_vram measurements unavailable")

    host_info = capture_versions_and_gpu()
    print(f"Host: {host_info['gpu_model']} ({host_info['gpu_gcn_arch_name']}) "
          f"x{host_info['gpu_count']}")
    print(f"ROCm: {host_info['rocm_version']} | vLLM: {host_info['vllm_version']} "
          f"| torch: {host_info['torch_version']}")
    print(f"Output dir: {RESULTS_DIR}")
    print()

    results = []
    t_start = time.time()

    for i, config in enumerate(PHASE1_CONFIGS, 1):
        print(f"\n[{i}/{len(PHASE1_CONFIGS)}] elapsed: {(time.time()-t_start):.0f}s")
        result = run_probe(config, host_info, r9700_indices)
        results.append(result)

        # Persist immediately (crash-resilient)
        out_path = write_per_config_json(result)

        # One-line summary
        if result.loaded:
            kv_tok = result.kv_cache_tokens or 0
            max_conc = result.max_concurrency or 0.0
            tok_s = result.sanity_throughput_tok_s or 0.0
            vram = result.peak_vram_gb_per_gpu or [0, 0]
            print(f"  ✓ OK — VRAM: {vram} GiB, KV: {kv_tok} tokens, "
                  f"max_conc: {max_conc:.1f}x, cold tok/s: {tok_s:.2f}")
        else:
            err_short = (result.error or '')[:100]
            print(f"  ✗ {(result.error_class or 'unknown').upper()}: {err_short}")
        print(f"  → {out_path.name}")

    # Summary table
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY")
    print("=" * 80)
    print(f"{'config_id':<48} {'loaded':>10} {'KV_tok':>8} {'max_conc':>9}")
    print("-" * 80)
    for r in results:
        kv_tok = str(r.kv_cache_tokens) if r.kv_cache_tokens else "-"
        max_conc = f"{r.max_concurrency:.1f}x" if r.max_concurrency else "-"
        loaded_str = "yes" if r.loaded else f"no/{r.error_class}"
        print(f"{r.config_id:<48} {loaded_str:>10} {kv_tok:>8} {max_conc:>9}")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Total elapsed: {(time.time()-t_start):.0f}s")

    # Recommendations for Phase 2
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR PHASE 2 SWEEP")
    print("=" * 80)
    ok_results = [r for r in results if r.loaded and r.max_concurrency]
    fp8_best = max(
        (r for r in ok_results if r.quantization == "fp8"),
        key=lambda r: r.max_concurrency,
        default=None,
    )
    bf16_best = max(
        (r for r in ok_results if r.quantization == "bf16"),
        key=lambda r: r.max_concurrency,
        default=None,
    )
    if fp8_best:
        print(f"FP8 best:  {fp8_best.config_id} (max_conc={fp8_best.max_concurrency:.1f}x)")
    if bf16_best:
        print(f"BF16 best: {bf16_best.config_id} (max_conc={bf16_best.max_concurrency:.1f}x)")


if __name__ == "__main__":
    main()
