"""
Phase 1 envelope analyzer for NaviMed-UMB benchmark suite.

Reads all hardware-envelope JSON results for Qwen 3.6 27B on 2x R9700,
validates schema, and produces:
    - Console summary table (logged at INFO level).
    - SUMMARY.md (markdown report, PUBLIC — envelope only).
    - phase1_envelope.csv (machine-readable for plotting).
    - Phase 2 sweep recommendation: best FP8 and best BF16 configs.

EMBARGO classification:
    All envelope outputs (load status, VRAM, KV cache tokens, max_concurrency,
    single-prompt sanity throughput) are PUBLIC. They describe the engineering
    envelope, not the scaling law. Phase 2 sweep results (throughput@N, P50/P95
    latency vs concurrent users) remain EMBARGOED until paper submission.

Usage:
    python3 analyze_phase1_envelope.py \
        --results-dir benchmarks/results/hardware_envelope \
        --output-dir benchmarks/results/hardware_envelope
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# --- Constants ---------------------------------------------------------------

EXPECTED_SCHEMA_VERSION: Final[int] = 1
GPU_VRAM_TOTAL_GB: Final[float] = 32.0
NEAR_OOM_THRESHOLD_GB: Final[float] = 29.0  # >30 GB out of 32 GB = warning
LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(message)s"

# --- Data structures ---------------------------------------------------------


@dataclass(frozen=True)
class EnvelopeRecord:
    """Single hardware-envelope measurement (one config × one model variant)."""

    config_id: str
    quantization: str
    kv_cache_dtype: str | None
    max_model_len: int
    gpu_memory_utilization: float
    loaded: bool
    error_class: str | None
    load_time_s: float
    peak_vram_gb_per_gpu: list[float]
    kv_cache_tokens: int | None
    max_concurrency: float | None
    sanity_throughput_tok_s: float | None

    @property
    def peak_vram_max(self) -> float:
        """Highest per-GPU VRAM (relevant for OOM proximity)."""
        return max(self.peak_vram_gb_per_gpu) if self.peak_vram_gb_per_gpu else 0.0

    @property
    def is_near_oom(self) -> bool:
        return self.peak_vram_max >= NEAR_OOM_THRESHOLD_GB


# --- I/O ---------------------------------------------------------------------


def load_json_safe(path: Path) -> dict | None:
    """Load JSON file with explicit error handling. Returns None on failure."""
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Failed to read %s: %s", path.name, exc)
        return None


def parse_record(payload: dict, source_file: str) -> EnvelopeRecord | None:
    """Convert raw JSON payload to typed EnvelopeRecord, validating schema."""
    if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        logging.warning(
            "%s has schema_version=%s (expected %d); skipping",
            source_file, payload.get("schema_version"), EXPECTED_SCHEMA_VERSION,
        )
        return None
    try:
        return EnvelopeRecord(
            config_id=payload["config_id"],
            quantization=payload["quantization"],
            kv_cache_dtype=payload.get("kv_cache_dtype"),
            max_model_len=payload["max_model_len"],
            gpu_memory_utilization=payload["gpu_memory_utilization"],
            loaded=payload["loaded"],
            error_class=payload.get("error_class"),
            load_time_s=payload.get("load_time_s") or 0.0,
            peak_vram_gb_per_gpu=payload.get("peak_vram_gb_per_gpu") or [],
            kv_cache_tokens=payload.get("kv_cache_tokens"),
            max_concurrency=payload.get("max_concurrency"),
            sanity_throughput_tok_s=payload.get("sanity_throughput_tok_s"),
        )
    except KeyError as exc:
        logging.error("%s missing required field: %s", source_file, exc)
        return None


def load_all_envelope_records(results_dir: Path) -> list[EnvelopeRecord]:
    """Load every qwen36_27b_*.json from hardware_envelope/ as EnvelopeRecord."""
    json_paths = sorted(results_dir.glob("qwen36_27b_*.json"))
    logging.info("Found %d envelope JSON files in %s", len(json_paths), results_dir)
    records = []
    for path in json_paths:
        payload = load_json_safe(path)
        if payload is None:
            continue
        record = parse_record(payload, path.name)
        if record is not None:
            records.append(record)
    return records


# --- Analysis ----------------------------------------------------------------


def select_best_by_throughput(
    records: list[EnvelopeRecord], quantization: str
) -> EnvelopeRecord | None:
    """
    Pick the loaded config with highest sanity throughput for given quantization.

    Why throughput as criterion: Phase 2 sweep measures throughput@N — picking
    the envelope config with best single-prompt throughput is a defensible
    starting point. Trade-off (larger max_model_len → smaller max_concurrency)
    is left to manual review of the summary table.
    """
    candidates = [
        r for r in records
        if r.quantization == quantization
        and r.loaded
        and r.sanity_throughput_tok_s is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.sanity_throughput_tok_s or 0.0)


def select_largest_context(
    records: list[EnvelopeRecord], quantization: str
) -> EnvelopeRecord | None:
    """Pick the loaded config with largest max_model_len for given quantization."""
    candidates = [r for r in records if r.quantization == quantization and r.loaded]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.max_model_len)


# --- Reporting ---------------------------------------------------------------


def format_record_row(record: EnvelopeRecord) -> str:
    """Format single record as one markdown table row."""
    status = "OK" if record.loaded else "FAIL"
    kv_dtype = record.kv_cache_dtype or "default"
    vram = f"{record.peak_vram_max:.2f}"
    if record.is_near_oom:
        vram = f"**{vram}**"  # bold marker for near-OOM
    kv_tok = record.kv_cache_tokens if record.kv_cache_tokens is not None else "—"
    max_conc = (
        f"{record.max_concurrency:.2f}" if record.max_concurrency is not None else "—"
    )
    tok_s = (
        f"{record.sanity_throughput_tok_s:.2f}"
        if record.sanity_throughput_tok_s is not None else "—"
    )
    return (
        f"| `{record.config_id}` | {record.quantization} | {kv_dtype} | "
        f"{record.max_model_len} | {record.gpu_memory_utilization:.2f} | "
        f"{status} | {record.load_time_s:.1f} | {vram} | {kv_tok} | "
        f"{max_conc} | {tok_s} |"
    )


def build_summary_markdown(records: list[EnvelopeRecord]) -> str:
    """Assemble full SUMMARY.md content (PUBLIC — envelope data only)."""
    header = (
        "# Phase 1 — Hardware Envelope Summary (Qwen 3.6 27B, 2× R9700)\n\n"
        "**EMBARGO:** PUBLIC. Envelope data only — single-config measurements, "
        "no scaling sweep.\n\n"
        "## Per-config results\n\n"
        "Bold VRAM = within 2 GB of OOM (peak ≥ 30/32 GB). KV dtype `default` "
        "= same as model weights.\n\n"
        "| Config | Quant | KV dtype | max_len | util | Status | Load [s] | "
        "Peak VRAM/GPU [GB] | KV tokens | Max conc. | Sanity tok/s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = "\n".join(format_record_row(r) for r in records)

    best_fp8 = select_best_by_throughput(records, "fp8")
    best_bf16 = select_best_by_throughput(records, "bf16")
    largest_bf16 = select_largest_context(records, "bf16")

    recommendation = (
        "\n\n## Phase 2 sweep recommendation\n\n"
        f"- **Best BF16 (max throughput):** "
        f"`{best_bf16.config_id if best_bf16 else 'none'}`\n"
        f"- **Largest-context BF16 loaded:** "
        f"`{largest_bf16.config_id if largest_bf16 else 'none'}`\n"
        f"- **Best FP8 (max throughput):** "
        f"`{best_fp8.config_id if best_fp8 else 'none'}`\n\n"
        "Trade-off note: large `max_model_len` reduces `max_concurrency`, "
        "which caps Phase 2 throughput at high N. Pick the BF16 envelope "
        "matching the longest prompt+response budget actually expected in "
        "the sweep, not unconditionally the highest tok/s.\n"
    )

    near_oom = [r for r in records if r.loaded and r.is_near_oom]
    if near_oom:
        warnings = "\n## ⚠ Near-OOM configurations\n\n" + "\n".join(
            f"- `{r.config_id}`: peak {r.peak_vram_max:.2f} GB/GPU"
            for r in near_oom
        ) + (
            "\n\nThese configs loaded but leave little headroom. Avoid for "
            "Phase 2 sweep at high N — KV cache pressure may trigger "
            "preemption or runtime OOM.\n"
        )
    else:
        warnings = ""

    return header + rows + recommendation + warnings


def write_csv(records: list[EnvelopeRecord], output_path: Path) -> None:
    """Write flat CSV for downstream plotting (matplotlib / pandas)."""
    columns = [
        "config_id", "quantization", "kv_cache_dtype", "max_model_len",
        "gpu_memory_utilization", "loaded", "error_class", "load_time_s",
        "peak_vram_gb_max", "kv_cache_tokens", "max_concurrency",
        "sanity_throughput_tok_s",
    ]
    lines = [",".join(columns)]
    for r in records:
        lines.append(",".join([
            r.config_id,
            r.quantization,
            r.kv_cache_dtype or "",
            str(r.max_model_len),
            f"{r.gpu_memory_utilization:.2f}",
            str(r.loaded),
            r.error_class or "",
            f"{r.load_time_s:.2f}",
            f"{r.peak_vram_max:.3f}",
            str(r.kv_cache_tokens) if r.kv_cache_tokens is not None else "",
            f"{r.max_concurrency:.2f}" if r.max_concurrency is not None else "",
            f"{r.sanity_throughput_tok_s:.2f}"
            if r.sanity_throughput_tok_s is not None else "",
        ]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote CSV: %s", output_path)


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI args. Defaults match navimed-umb repo layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("benchmarks/results/hardware_envelope"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("benchmarks/results/hardware_envelope"),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )
    if not args.results_dir.exists():
        logging.error("Results directory not found: %s", args.results_dir)
        return 1

    records = load_all_envelope_records(args.results_dir)
    if not records:
        logging.error("No valid envelope records found.")
        return 1

    loaded_count = sum(1 for r in records if r.loaded)
    logging.info("Loaded successfully: %d/%d", loaded_count, len(records))
    for record in records:
        if not record.loaded:
            logging.warning(
                "FAILED: %s (%s)", record.config_id, record.error_class or "unknown"
            )
        elif record.is_near_oom:
            logging.warning(
                "Near-OOM: %s peak %.2f GB/GPU",
                record.config_id, record.peak_vram_max,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "SUMMARY.md"
    csv_path = args.output_dir / "phase1_envelope.csv"

    summary_path.write_text(build_summary_markdown(records), encoding="utf-8")
    logging.info("Wrote summary: %s", summary_path)
    write_csv(records, csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
