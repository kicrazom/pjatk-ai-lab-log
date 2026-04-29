"""
Phase 2 sweep analyzer for NaviMed-UMB benchmark suite (v2 — universal).

Reads thermal-runs output from any vLLM benchmark wrapper in the suite,
auto-detecting the filename convention and parsing schema. Supports:
    - Legacy format (Qwen 7B v0.1.0):  tp{N}-n{N}-bench.log
                                       (no quant prefix; quant from dtype log)
    - Current format (Qwen 27B+):      {quant}-tp{N}-n{N}-bench.log
                                       (explicit quant; full run header)

Produces per METHODOLOGY §7:
    - results_table.csv  (flat, all N rows, kyuz0+ schema)
    - SUMMARY.md         (markdown report, embargo-aware)

Universal across all NaviMed-UMB models. Cross-config sweeps (e.g. 7B has
both TP=1 and TP=2 in one folder) produce one row per run, sorted by (TP, N).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Final

LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(message)s"

RE_RUN_NAME = re.compile(
    r"^(?:(?P<quant>\w+)-)?tp(?P<tp>\d+)-n(?P<n>\d+)-bench\.log$"
)

RE_PROMPTS = re.compile(r"^Prompts:\s+(\d+)", re.MULTILINE)
RE_INPUT_TOTAL = re.compile(r"^Input tokens total:\s+(\d+)", re.MULTILINE)
RE_OUTPUT_TOTAL = re.compile(r"^Output tokens total:\s+(\d+)", re.MULTILINE)
RE_OUTPUT_MEAN = re.compile(r"^Output tokens mean:\s+([\d.]+)", re.MULTILINE)
RE_OUTPUT_MIN = re.compile(r"^Output tokens min:\s+(\d+)", re.MULTILINE)
RE_OUTPUT_MAX = re.compile(r"^Output tokens max:\s+(\d+)", re.MULTILINE)
RE_TOTAL_TIME = re.compile(r"^Total time:\s+([\d.]+)s", re.MULTILINE)
RE_OUT_TPUT = re.compile(r"^Output throughput:\s+([\d.]+)\s+tok/s", re.MULTILINE)
RE_TOT_TPUT = re.compile(r"^Total throughput:\s+([\d.]+)\s+tok/s", re.MULTILINE)
RE_REQ_S = re.compile(r"^Requests/second:\s+([\d.]+)", re.MULTILINE)
RE_LOAD_TIME = re.compile(r"^Load time:\s+([\d.]+)s", re.MULTILINE)

# Flexible: "max_model_len:        2048" OR "'max_model_len': 4096"
RE_MAX_LEN = re.compile(r"max_model_len['\":\s]+(\d+)")
RE_UTIL = re.compile(r"gpu_memory_util(?:ization)?['\":\s]+([\d.]+)")
RE_KV = re.compile(r"kv_cache_dtype['\":=\s]+(\S+?)[\s,'\")]")

RE_DTYPE_VLLM = re.compile(r"dtype['\":\s]+'?(\w+)'?")
RE_QUANTIZATION_VLLM = re.compile(r"quantization['\":=\s]+'?(\w+)'?")

DTYPE_TO_QUANT: Final[dict[str, str]] = {
    "float16": "fp16",
    "fp16": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float8_e4m3fn": "fp8",
    "fp8": "fp8",
    "auto": "fp16",
}


@dataclass
class BenchMetrics:
    quant: str
    tp: int
    n_prompts: int
    input_tokens_total: int
    output_tokens_total: int
    output_tokens_mean: float
    output_tokens_min: int
    output_tokens_max: int
    total_time_s: float
    output_throughput_tok_s: float
    total_throughput_tok_s: float
    req_per_s: float
    load_time_s: float
    max_model_len: int
    gpu_memory_util: float
    kv_cache_dtype: str


@dataclass
class ThermalSummary:
    bench_start_s: float
    bench_end_s: float
    n_samples_in_window: int
    vram_peak_gb_per_gpu: list[float] = field(default_factory=list)
    temp_peak_c_per_gpu: list[float] = field(default_factory=list)
    power_mean_w_per_gpu: list[float] = field(default_factory=list)
    power_peak_w_per_gpu: list[float] = field(default_factory=list)

    @property
    def vram_peak_gb_max(self) -> float:
        return max(self.vram_peak_gb_per_gpu) if self.vram_peak_gb_per_gpu else 0.0

    @property
    def temp_peak_c_max(self) -> float:
        return max(self.temp_peak_c_per_gpu) if self.temp_peak_c_per_gpu else 0.0

    @property
    def power_mean_w_total(self) -> float:
        return sum(self.power_mean_w_per_gpu)

    @property
    def power_peak_w_total(self) -> float:
        return sum(self.power_peak_w_per_gpu)


@dataclass
class RunRecord:
    bench: BenchMetrics
    thermal: ThermalSummary

    @property
    def w_per_tok(self) -> float:
        if self.bench.output_tokens_total == 0:
            return 0.0
        energy_wh = (
            self.thermal.power_mean_w_total * self.bench.total_time_s / 3600.0
        )
        return energy_wh / self.bench.output_tokens_total


def _grep_one(
    pattern: re.Pattern, text: str, default: str | None = None
) -> str:
    m = pattern.search(text)
    if m:
        return m.group(1)
    if default is not None:
        return default
    raise ValueError(f"Pattern {pattern.pattern!r} not found")


def detect_quant(text: str, filename_hint: str | None) -> str:
    """Return canonical quant label from filename hint, vLLM init log, or fallback."""
    if filename_hint:
        return filename_hint.lower()
    quant_match = RE_QUANTIZATION_VLLM.search(text)
    if quant_match and quant_match.group(1).lower() not in {"none", "null"}:
        return quant_match.group(1).lower()
    dtype_match = RE_DTYPE_VLLM.search(text)
    if dtype_match:
        return DTYPE_TO_QUANT.get(dtype_match.group(1).lower(), "unknown")
    return "unknown"


def parse_bench_log(path: Path) -> BenchMetrics:
    text = path.read_text(encoding="utf-8", errors="replace")
    name_match = RE_RUN_NAME.match(path.name)
    if not name_match:
        raise ValueError(f"Filename {path.name} does not match run pattern")

    quant = detect_quant(text, name_match.group("quant"))

    return BenchMetrics(
        quant=quant,
        tp=int(name_match.group("tp")),
        n_prompts=int(_grep_one(RE_PROMPTS, text)),
        input_tokens_total=int(_grep_one(RE_INPUT_TOTAL, text)),
        output_tokens_total=int(_grep_one(RE_OUTPUT_TOTAL, text)),
        output_tokens_mean=float(_grep_one(RE_OUTPUT_MEAN, text)),
        output_tokens_min=int(_grep_one(RE_OUTPUT_MIN, text)),
        output_tokens_max=int(_grep_one(RE_OUTPUT_MAX, text)),
        total_time_s=float(_grep_one(RE_TOTAL_TIME, text)),
        output_throughput_tok_s=float(_grep_one(RE_OUT_TPUT, text)),
        total_throughput_tok_s=float(_grep_one(RE_TOT_TPUT, text)),
        req_per_s=float(_grep_one(RE_REQ_S, text)),
        load_time_s=float(_grep_one(RE_LOAD_TIME, text, default="0.0")),
        max_model_len=int(_grep_one(RE_MAX_LEN, text, default="0")),
        gpu_memory_util=float(_grep_one(RE_UTIL, text, default="0.0")),
        kv_cache_dtype=_grep_one(RE_KV, text, default="default"),
    )


def parse_events_json(path: Path) -> tuple[float, float]:
    events = json.loads(path.read_text(encoding="utf-8"))
    starts = [e["t"] for e in events if "start" in e["label"].lower()]
    ends = [e["t"] for e in events if "end" in e["label"].lower()]
    if not starts or not ends:
        raise ValueError(f"events.json {path.name} missing bench start/end")
    return starts[0], ends[0]


def parse_thermals_jsonl(
    path: Path, bench_start_s: float, bench_end_s: float
) -> ThermalSummary:
    samples_in_window = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            sample = json.loads(line)
            if bench_start_s <= sample["t"] <= bench_end_s:
                samples_in_window.append(sample)

    if not samples_in_window:
        raise ValueError(f"thermals.jsonl {path.name} has no in-window samples")

    r9700_indices = [
        g["idx"] for g in samples_in_window[0]["gpus"]
        if not g.get("is_igpu", False)
    ]

    summary = ThermalSummary(
        bench_start_s=bench_start_s,
        bench_end_s=bench_end_s,
        n_samples_in_window=len(samples_in_window),
    )
    for gpu_idx in r9700_indices:
        vram_series = [
            _bytes_to_gb(s["gpus"][gpu_idx]["vram_used_b"])
            for s in samples_in_window
        ]
        temp_series = [s["gpus"][gpu_idx]["temp"] for s in samples_in_window]
        power_series = [
            s["gpus"][gpu_idx]["power_w"] for s in samples_in_window
            if s["gpus"][gpu_idx]["power_w"] is not None
        ]
        summary.vram_peak_gb_per_gpu.append(max(vram_series) if vram_series else 0.0)
        summary.temp_peak_c_per_gpu.append(max(temp_series) if temp_series else 0.0)
        summary.power_mean_w_per_gpu.append(mean(power_series) if power_series else 0.0)
        summary.power_peak_w_per_gpu.append(max(power_series) if power_series else 0.0)
    return summary


def _bytes_to_gb(b: int) -> float:
    return b / (1024 ** 3)


def discover_runs(thermal_runs_dir: Path) -> list[Path]:
    paths = []
    for p in thermal_runs_dir.glob("*-bench.log"):
        if RE_RUN_NAME.match(p.name):
            paths.append(p)
    paths.sort(
        key=lambda p: (
            int(RE_RUN_NAME.match(p.name).group("tp")),
            int(RE_RUN_NAME.match(p.name).group("n")),
        )
    )
    return paths


def load_run(bench_log_path: Path) -> RunRecord | None:
    base = bench_log_path.with_suffix("").name.removesuffix("-bench")
    events_path = bench_log_path.parent / f"{base}-events.json"
    thermals_path = bench_log_path.parent / f"{base}-thermals.jsonl"
    if not events_path.exists() or not thermals_path.exists():
        logging.warning("Missing companion files for %s; skipping", bench_log_path.name)
        return None
    try:
        bench = parse_bench_log(bench_log_path)
        bench_start, bench_end = parse_events_json(events_path)
        thermal = parse_thermals_jsonl(thermals_path, bench_start, bench_end)
        return RunRecord(bench=bench, thermal=thermal)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        logging.error("Failed to parse %s: %s", bench_log_path.name, exc)
        return None


CSV_COLUMNS: Final[list[str]] = [
    "model", "quant", "backend", "TP", "max_len", "util", "KV_dtype", "N",
    "n_runs",
    "tok_s_out", "tok_s_tot", "total_s", "req_s",
    "VRAM_peak_GB", "T_peak_C", "W_mean", "W_peak", "W_per_tok_Wh",
    "load_time_s", "input_tok_total", "output_tok_total", "output_tok_mean",
    "tuning",
]


def format_csv_row(
    rec: RunRecord, model: str, backend: str, tuning: str
) -> str:
    return ",".join([
        model,
        rec.bench.quant,
        backend,
        str(rec.bench.tp),
        str(rec.bench.max_model_len),
        f"{rec.bench.gpu_memory_util:.2f}",
        rec.bench.kv_cache_dtype,
        str(rec.bench.n_prompts),
        "1",
        f"{rec.bench.output_throughput_tok_s:.2f}",
        f"{rec.bench.total_throughput_tok_s:.2f}",
        f"{rec.bench.total_time_s:.2f}",
        f"{rec.bench.req_per_s:.3f}",
        f"{rec.thermal.vram_peak_gb_max:.2f}",
        f"{rec.thermal.temp_peak_c_max:.0f}",
        f"{rec.thermal.power_mean_w_total:.0f}",
        f"{rec.thermal.power_peak_w_total:.0f}",
        f"{rec.w_per_tok:.6f}",
        f"{rec.bench.load_time_s:.1f}",
        str(rec.bench.input_tokens_total),
        str(rec.bench.output_tokens_total),
        f"{rec.bench.output_tokens_mean:.1f}",
        tuning,
    ])


def write_csv(
    records: list[RunRecord], output_path: Path,
    model: str, backend: str, tuning: str,
) -> None:
    lines = [",".join(CSV_COLUMNS)]
    lines.extend(format_csv_row(r, model, backend, tuning) for r in records)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote CSV: %s", output_path)


def format_summary_row(rec: RunRecord) -> str:
    return (
        f"| {rec.bench.tp} | {rec.bench.n_prompts} | "
        f"{rec.bench.output_throughput_tok_s:.1f} | "
        f"{rec.bench.total_time_s:.1f} | "
        f"{rec.bench.req_per_s:.2f} | "
        f"{rec.thermal.vram_peak_gb_max:.1f} | "
        f"{rec.thermal.temp_peak_c_max:.0f} | "
        f"{rec.thermal.power_mean_w_total:.0f} | "
        f"{rec.thermal.power_peak_w_total:.0f} | "
        f"{rec.w_per_tok * 1000:.3f} |"
    )


def build_summary_markdown(
    records: list[RunRecord], model: str, backend: str, tuning: str,
) -> str:
    if not records:
        return "# No runs to report\n"
    first = records[0]

    tp_values = sorted({r.bench.tp for r in records})
    config_label = (
        f"{first.bench.quant.upper()} TP={'/'.join(map(str, tp_values))} "
        f"max_len={first.bench.max_model_len} "
        f"util={first.bench.gpu_memory_util:.2f} "
        f"KV={first.bench.kv_cache_dtype}"
    )

    header = (
        f"# Phase 2 Scaling Sweep — {model}\n\n"
        f"**Configuration:** `{config_label}`  \n"
        f"**Backend:** `{backend}`  \n"
        f"**Tuning:** `{tuning}`  \n"
        f"**N reruns per cell (n_runs):** 1 (v0.2.0 exploratory; "
        f"Tier A n=10 deferred per METHODOLOGY §7.4)\n\n"
        "## ⚠ Embargo classification\n\n"
        "**EMBARGO — paper figures.** All concrete throughput, latency, and "
        "power numbers below are paper-bound until publication acceptance. "
        "Engineering observations (knee shape, vLLM scheduler robustness, "
        "preemption regime onset, TP=1 vs TP=2 trade-off) are PUBLIC. See "
        "[METHODOLOGY.md §11](../../../METHODOLOGY.md#11-embargo-policy) for "
        "the per-artifact split.\n\n"
        "## Results table\n\n"
        "Energy per token reported in mWh (= 0.001 Wh). Power columns are "
        "totals across all R9700 GPUs in use (1× for TP=1, 2× for TP=2).\n\n"
        "| TP | N | tok/s out | total [s] | req/s | VRAM peak [GB] | "
        "T peak [°C] | W mean | W peak | mWh/tok |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = "\n".join(format_summary_row(r) for r in records)

    by_tp = {tp: [r for r in records if r.bench.tp == tp] for tp in tp_values}
    obs_lines: list[str] = []

    for tp, tp_records in by_tp.items():
        by_n = {r.bench.n_prompts: r for r in tp_records}
        sorted_n = sorted(by_n.keys())
        peak_n = max(sorted_n, key=lambda n: by_n[n].bench.output_throughput_tok_s)
        peak_tput = by_n[peak_n].bench.output_throughput_tok_s
        max_n_tput = by_n[sorted_n[-1]].bench.output_throughput_tok_s
        obs_lines.append(
            f"- **TP={tp} throughput knee:** peak at N={peak_n} "
            f"({peak_tput:.1f} tok/s output). At N={sorted_n[-1]} "
            f"({sorted_n[-1] // peak_n}× over knee) throughput is "
            f"{(max_n_tput / peak_tput - 1) * 100:+.1f}% vs peak."
        )

    if len(tp_values) >= 2:
        sets_per_tp = [{r.bench.n_prompts for r in by_tp[tp]} for tp in tp_values]
        common_n = sorted(set.intersection(*sets_per_tp))
        if common_n:
            obs_lines.append(
                f"- **TP=1 vs TP=2 crossover:** common N values "
                f"{common_n} — see results table for per-N comparison."
            )

    obs_lines.append(
        "- **Methodological humility (METHODOLOGY §8):** these numbers "
        "characterize inference *throughput* and *thermal envelope*; they "
        "do not measure model quality, reasoning, or clinical utility."
    )

    observations = (
        "\n\n## Engineering observations (PUBLIC)\n\n"
        + "\n".join(obs_lines) + "\n"
    )
    return header + rows + observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermal-runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="vllm-0.19.0+rocm721")
    parser.add_argument("--tuning", default="stock", choices=["stock", "uv"])
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT,
    )

    if not args.thermal_runs_dir.exists():
        logging.error("Thermal runs dir not found: %s", args.thermal_runs_dir)
        return 1

    bench_logs = discover_runs(args.thermal_runs_dir)
    if not bench_logs:
        logging.error("No bench.log files in %s", args.thermal_runs_dir)
        return 1
    logging.info("Found %d run(s) to analyze", len(bench_logs))

    records: list[RunRecord] = []
    for log_path in bench_logs:
        rec = load_run(log_path)
        if rec is not None:
            records.append(rec)
            logging.info(
                "Parsed %s: quant=%s TP=%d N=%d → %.1f tok/s, VRAM %.1f GB, %d W mean",
                log_path.name, rec.bench.quant, rec.bench.tp, rec.bench.n_prompts,
                rec.bench.output_throughput_tok_s,
                rec.thermal.vram_peak_gb_max,
                rec.thermal.power_mean_w_total,
            )

    if not records:
        logging.error("All runs failed to parse")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results_table.csv"
    summary_path = args.output_dir / "SUMMARY.md"

    write_csv(records, csv_path, args.model, args.backend, args.tuning)
    summary_path.write_text(
        build_summary_markdown(records, args.model, args.backend, args.tuning),
        encoding="utf-8",
    )
    logging.info("Wrote summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
