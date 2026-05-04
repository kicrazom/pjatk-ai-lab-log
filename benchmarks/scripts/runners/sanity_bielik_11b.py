"""
NaviMed-UMB Phase 1 envelope probe — Bielik 11B v2.3.

Adapted from sanity_qwen36_27b.py per METHODOLOGY.md v1.0 §5.1.

Differences from Qwen 3.6 27B:
- Bielik 11B is NOT hybrid attention (Mistral-based architecture) —
  enforce_eager is an EXPLORATION AXIS, not a constraint per §3.2.
- Per METHODOLOGY §4, Bielik FP16 supports both TP=1 and TP=2;
  Bielik AWQ is TP=1 only — TP is therefore an EXPLORATION AXIS.
- Two model variants share one runner: fp16 and awq_marlin.

Embargo: PUBLIC (engineering envelope, methodology only).
Output: benchmarks/results/hardware_envelope/bielik_11b_*.json

Author: Łukasz Minarowski <lukasz.minarowski@umb.edu.pl>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ============================================================
# Constants — Bielik 11B v2.3
# ============================================================

# Model HuggingFace ids (separate from local model_path for provenance)
MODEL_HF_IDS = {
    "fp16": "speakleash/Bielik-11B-v2.3-Instruct",
    "awq": "speakleash/Bielik-11B-v2.3-Instruct-AWQ",
}

# Local model paths (Path.home() avoids hardcoded usernames)
# Verified on disk 2026-05-04: directory names use lowercase
MODEL_PATHS = {
    "fp16": str(Path.home() / "models/bielik-11b-v23"),
    "awq": str(Path.home() / "models/bielik-11b-v23-awq"),
}

# Output directory — aligns with METHODOLOGY §5.1 and embargo §11.1
RESULTS_DIR = Path.home() / "navimed-umb/benchmarks/results/hardware_envelope"

# ============================================================
# Phase 1 envelope grid
# ============================================================
#
# Tuple structure: (quant, tp, max_model_len, gpu_memory_utilization,
#                   enforce_eager, kv_cache_dtype)
#
# Bielik is Mistral-based (not hybrid attention). Per METHODOLOGY §3.2,
# enforce_eager is mandatory ONLY for hybrid attention models. For
# Bielik, eager=False (CUDA graphs) is a legitimate exploration axis.
#
# AWQ is TP=1 per METHODOLOGY §4 (model 6).
# FP16 explores both TP=1 and TP=2.
#
# Total: 9 configs (within METHODOLOGY §5.1 8-12 envelope budget).
# ============================================================

PHASE1_CONFIGS = [
    # AWQ-4bit, TP=1 (METHODOLOGY §4 model 6)
    # Note: enforce_eager=False (CUDA graphs) DROPPED after empirical
    # finding 2026-05-04: graphs path segfaults in libhsa-runtime64 on
    # gfx1201 for Bielik AWQ (config bielik_11b_awq_tp1_max2048_util090_graphs).
    # This extends METHODOLOGY §3.2: graphs unsupported on gfx1201 for
    # Mistral-based models too, not just hybrid attention.
    ("awq", 1, 2048, 0.90, True, None),
    ("awq", 1, 8192, 0.90, True, None),
    # FP16, TP=1 — fits on single 32 GiB R9700 with headroom for KV
    ("fp16", 1, 2048, 0.90, True, None),
    ("fp16", 1, 4096, 0.90, True, None),
    ("fp16", 1, 8192, 0.90, True, None),
    # FP16, TP=2 — needed for max_len=8192 if KV does not fit on TP=1
    ("fp16", 2, 2048, 0.90, True, None),
    ("fp16", 2, 8192, 0.90, True, None),
]

# ============================================================
# Result schema (matches sanity_qwen36_27b.py SCHEMA v1)
# ============================================================


@dataclass
class ConfigResult:
    """Result of one envelope probe.

    Field order and naming follow SCHEMA.md v1. Failed probes (loaded=
    False) get all schema fields populated, with measurement fields
    set to None — downstream consumers can iterate over partial records.

    Bielik-specific addition: enforce_eager is part of identity (not
    constant True like Qwen 3.6 27B), so it appears alongside other
    config dimensions.
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
    peak_vram_source: str  # "rocm_smi" only in v1.5

    kv_cache_tokens: Optional[int]
    kv_cache_gb: Optional[float]
    max_concurrency: Optional[int]
    sanity_throughput_tok_s: Optional[float]


# ============================================================
# Environment snapshot — for causal closure (METHODOLOGY §3.3)
# ============================================================


def env_snapshot() -> dict:
    """Capture mandatory env vars per METHODOLOGY §3.1."""
    keys = [
        "AMD_SERIALIZE_KERNEL",
        "HIP_LAUNCH_BLOCKING",
        "ROCR_VISIBLE_DEVICES",
        "VLLM_ROCM_USE_AITER",
        "PYTORCH_ALLOC_CONF",
        "HIP_VISIBLE_DEVICES",
        "NCCL_P2P_DISABLE",
    ]
    return {k: os.environ.get(k, "<unset>") for k in keys}


def stack_versions() -> dict:
    """Capture full version triple per METHODOLOGY §3.3."""
    out = {}
    try:
        import vllm

        out["vllm_version"] = vllm.__version__
    except Exception as e:
        out["vllm_version"] = f"<error: {e}>"
    try:
        import torch

        out["torch_version"] = torch.__version__
        out["torch_hip_version"] = getattr(torch.version, "hip", "<unset>")
    except Exception as e:
        out["torch_version"] = f"<error: {e}>"
    try:
        proc = subprocess.run(
            ["rocm-smi", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out["rocm_smi_version"] = (
            proc.stdout.strip().splitlines()[0] if proc.stdout else "<unknown>"
        )
    except Exception as e:
        out["rocm_smi_version"] = f"<error: {e}>"
    return out


# ============================================================
# rocm-smi VRAM read (per-card) — METHODOLOGY §2 GPU filtering
# ============================================================


def _rocm_smi_vram_used_gib(card_indices: list) -> Optional[list]:
    """Read VRAM used (GiB) for the given card indices.

    Returns list aligned with card_indices order, or None on failure.
    """
    if not card_indices:
        return None
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
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
# Config ID — canonical, sortable
# ============================================================


def make_config_id(
    quant: str,
    tp: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    kv_cache_dtype: Optional[str],
) -> str:
    """Build canonical config_id matching SCHEMA.md grammar.

    Format: bielik_11b_{quant}_tp{TP}_max{max_len}_util{util_x100}_{eager|graphs}[_kv{kv_short}]

    util is multiplied by 100 and zero-padded to 3 digits so that
    lexicographic sort matches numeric sort (085 < 095 < 097).

    Eager flag is part of ID because it is an exploration axis for
    Bielik (unlike Qwen 3.6 27B where it is hardcoded True).
    """
    util_x100 = f"{int(round(gpu_memory_utilization * 100)):03d}"
    eager_tag = "eager" if enforce_eager else "graphs"
    base = f"bielik_11b_{quant}_tp{tp}_max{max_model_len}_util{util_x100}_{eager_tag}"
    if kv_cache_dtype:
        kv_short = kv_cache_dtype.replace("_", "")
        return f"{base}_kv{kv_short}"
    return base


# ============================================================
# Subprocess probe script generation
# ============================================================


def make_probe_script(
    quant: str,
    tp: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    kv_cache_dtype: Optional[str],
) -> str:
    """Generate a self-contained Python script that probes one config.

    Reports back a single JSON line via stdout marked with PROBE_RESULT
    sentinel. Includes vLLM 0.19 internals reading via the correct
    `engine.vllm_config.cache_config` path (validated 2026-04-29 on
    Qwen 3.6 27B sweep).

    For AWQ variants, sets `quantization="awq_marlin"` (vLLM auto-detects
    AWQ-Marlin kernel for gfx1201 — confirmed working on Bielik AWQ in
    PLAN-NEXT cache; falls back to "awq" if Marlin path unavailable).
    """
    model_path = MODEL_PATHS[quant]
    kv_arg = f', kv_cache_dtype="{kv_cache_dtype}"' if kv_cache_dtype else ""
    quant_arg = ', quantization="awq_marlin"' if quant == "awq" else ""
    eager_arg = "True" if enforce_eager else "False"

    return f"""
import json
import sys
import time
from vllm import LLM, SamplingParams

result = {{
    "quant": "{quant}",
    "tp": {tp},
    "max_model_len": {max_model_len},
    "gpu_memory_utilization": {gpu_memory_utilization},
    "enforce_eager": {eager_arg},
    "kv_cache_dtype": {repr(kv_cache_dtype)},
    "status": "crash",
    "error": None,
    "load_time_s": None,
    "sanity_throughput_tok_s": None,
    "kv_cache_tokens": None,
    "kv_cache_gb": None,
    "max_concurrency": None,
}}

try:
    t0 = time.time()
    llm = LLM(
        model="{model_path}",
        dtype="auto",
        max_model_len={max_model_len},
        gpu_memory_utilization={gpu_memory_utilization},
        enforce_eager={eager_arg},
        tensor_parallel_size={tp}{quant_arg}{kv_arg},
    )
    result["load_time_s"] = time.time() - t0

    # Sanity inference (single prompt, 20 tokens) — proves model lives
    t0 = time.time()
    out = llm.generate(
        ["What is 2+2?"],
        SamplingParams(max_tokens=20, temperature=0.3),
    )
    result["sanity_throughput_tok_s"] = round(20 / (time.time() - t0), 3)
    result["status"] = "ok"

    # vLLM 0.19 KV cache stats via vllm_config.cache_config
    try:
        engine = llm.llm_engine
        cache_config = engine.vllm_config.cache_config
        if hasattr(cache_config, "num_gpu_blocks") and cache_config.num_gpu_blocks:
            block_size = cache_config.block_size
            result["kv_cache_tokens"] = int(cache_config.num_gpu_blocks * block_size)
        if hasattr(cache_config, "gpu_cache_size_bytes"):
            result["kv_cache_gb"] = round(
                cache_config.gpu_cache_size_bytes / 1024**3, 3
            )
        # max_concurrency estimate (vLLM scheduler hint)
        sched = engine.vllm_config.scheduler_config
        if hasattr(sched, "max_num_seqs"):
            result["max_concurrency"] = int(sched.max_num_seqs)
    except Exception as e:
        result["kv_cache_introspection_error"] = str(e)

except Exception as e:
    result["error"] = str(e)
    result["error_class"] = type(e).__name__
    result["status"] = "fail"

print("PROBE_RESULT:" + json.dumps(result))
"""


# ============================================================
# Probe runner — one config per subprocess
# ============================================================


def run_one_probe(
    quant: str,
    tp: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    kv_cache_dtype: Optional[str],
) -> ConfigResult:
    """Run a single probe config in a fresh subprocess.

    Subprocess isolation is mandatory: vLLM cannot reliably tear down
    its CUDA context for a TP-changing reload within one process.
    """
    config_id = make_config_id(
        quant,
        tp,
        max_model_len,
        gpu_memory_utilization,
        enforce_eager,
        kv_cache_dtype,
    )
    model_path = MODEL_PATHS[quant]
    model_hf = MODEL_HF_IDS[quant]
    env = env_snapshot()

    print(f"\n[{config_id}]", flush=True)
    print(f"  model={model_hf}", flush=True)
    print(
        f"  TP={tp} max_len={max_model_len} util={gpu_memory_utilization}"
        f" eager={enforce_eager} kv={kv_cache_dtype}",
        flush=True,
    )

    script = make_probe_script(
        quant,
        tp,
        max_model_len,
        gpu_memory_utilization,
        enforce_eager,
        kv_cache_dtype,
    )

    # Pre-probe VRAM read (baseline)
    vram_before = _rocm_smi_vram_used_gib([0, 1])
    if vram_before:
        print(f"  VRAM before: {vram_before} GiB", flush=True)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,  # 10 min — ample for 11B load + sanity
    )

    # Post-probe VRAM read (peak proxy)
    vram_after = _rocm_smi_vram_used_gib([0, 1])

    # Parse PROBE_RESULT line
    probe_data = None
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE_RESULT:"):
            try:
                probe_data = json.loads(line[len("PROBE_RESULT:") :])
                break
            except json.JSONDecodeError:
                pass

    if probe_data is None:
        # Subprocess died before reporting — capture stderr tail
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
        print(f"  STATUS: subprocess_died returncode={proc.returncode}", flush=True)
        return ConfigResult(
            config_id=config_id,
            model=model_hf,
            model_path=model_path,
            quantization=quant,
            kv_cache_dtype=kv_cache_dtype,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            env=env,
            loaded=False,
            error=f"subprocess died: {stderr_tail}",
            error_class="SubprocessDied",
            load_time_s=None,
            peak_vram_gb_per_gpu=None,
            peak_vram_source="rocm_smi",
            kv_cache_tokens=None,
            kv_cache_gb=None,
            max_concurrency=None,
            sanity_throughput_tok_s=None,
        )

    loaded = probe_data["status"] == "ok"
    print(f"  STATUS: {probe_data['status']}", flush=True)
    if loaded:
        print(f"  load_time_s={probe_data['load_time_s']:.2f}", flush=True)
        print(f"  sanity_tok/s={probe_data['sanity_throughput_tok_s']}", flush=True)
        print(
            f"  kv_tokens={probe_data.get('kv_cache_tokens')}"
            f" kv_gb={probe_data.get('kv_cache_gb')}"
            f" max_concurrency={probe_data.get('max_concurrency')}",
            flush=True,
        )
        print(f"  VRAM after: {vram_after} GiB", flush=True)
    else:
        err_short = (probe_data.get("error") or "")[:200]
        print(f"  error_class={probe_data.get('error_class')}", flush=True)
        print(f"  error={err_short}", flush=True)

    return ConfigResult(
        config_id=config_id,
        model=model_hf,
        model_path=model_path,
        quantization=quant,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        env=env,
        loaded=loaded,
        error=probe_data.get("error"),
        error_class=probe_data.get("error_class"),
        load_time_s=probe_data.get("load_time_s"),
        peak_vram_gb_per_gpu=vram_after,
        peak_vram_source="rocm_smi",
        kv_cache_tokens=probe_data.get("kv_cache_tokens"),
        kv_cache_gb=probe_data.get("kv_cache_gb"),
        max_concurrency=probe_data.get("max_concurrency"),
        sanity_throughput_tok_s=probe_data.get("sanity_throughput_tok_s"),
    )


# ============================================================
# Main
# ============================================================


def main() -> int:
    """Run all Phase 1 configs sequentially, write per-config JSON.

    Output filename per METHODOLOGY §5.1:
      benchmarks/results/hardware_envelope/{config_id}.json

    Each file is fully self-contained — no cross-file references.
    Failed configs get the same schema with measurement fields None.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    versions = stack_versions()
    print("=" * 60)
    print("Bielik 11B v2.3 — Phase 1 envelope")
    print("METHODOLOGY.md v1.0 §5.1")
    print(f"vLLM:  {versions.get('vllm_version')}")
    print(
        f"Torch: {versions.get('torch_version')} (HIP {versions.get('torch_hip_version')})"
    )
    print(f"ROCm:  {versions.get('rocm_smi_version')}")
    print(f"Configs: {len(PHASE1_CONFIGS)}")
    print(f"Output:  {RESULTS_DIR}")
    print("=" * 60)

    # Validate model dirs exist before launching any probe
    for quant, path in MODEL_PATHS.items():
        if not Path(path).is_dir():
            print(
                f"ERROR: model dir missing for quant={quant}: {path}", file=sys.stderr
            )
            return 2

    n_loaded = 0
    n_failed = 0
    for cfg in PHASE1_CONFIGS:
        result = run_one_probe(*cfg)
        out_path = RESULTS_DIR / f"{result.config_id}.json"
        # Inject stack versions per METHODOLOGY §3.3 causal closure
        record = asdict(result)
        record["stack_versions"] = versions
        out_path.write_text(json.dumps(record, indent=2) + "\n")
        if result.loaded:
            n_loaded += 1
        else:
            n_failed += 1

    print()
    print("=" * 60)
    print(f"Phase 1 complete: {n_loaded} loaded, {n_failed} failed")
    print(f"Records: {RESULTS_DIR}/bielik_11b_*.json")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
