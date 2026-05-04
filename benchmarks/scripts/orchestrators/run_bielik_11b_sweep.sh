#!/usr/bin/env bash
# NaviMed-UMB Phase 2 sweep — Bielik 11B v2.3 FP16 TP=2 max_len=8192
#
# Per METHODOLOGY.md v1.0 §5.2:
#   - N ladder: {10, 25, 50, 100, 200, 500, 1000}
#   - 15s cooldown between runs
#   - 1 Hz background thermal sampling
#   - Output: benchmarks/results/bielik-11b-v23/thermal-runs/<quant>-tp<TP>-n<N>-{bench.log,events.json,thermals.jsonl,thermals.png}
#
# Phase 1 selected config (2026-05-04):
#   fp16_tp2_max8192_util090_eager  → sanity 17.42 tok/s, 173k KV tokens
#
# Embargo: EMBARGO_paper_bound (Polish model, METHODOLOGY §11.3).
#
# Estimated runtime: ~1-1.5h
#   Per-N: load (~25s) + warmup (~5s) + bench (varies 10-300s) + cooldown 15s
#   N=10 fastest, N=1000 slowest.
#
# Usage from repo root:
#   bash benchmarks/scripts/orchestrators/run_bielik_11b_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ===========================================================================
# Pre-flight: env vars (METHODOLOGY §3.1)
# ===========================================================================
unset PYTORCH_ALLOC_CONF
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=1
export HIP_LAUNCH_BLOCKING=1
# Note: bench_with_thermals_bielik_11b.py pops ROCR_VISIBLE_DEVICES
# before launching inner benchmark, but we set it for the wrapper itself.
export ROCR_VISIBLE_DEVICES=0,1

# Activate venv if not already
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck source=/dev/null
    source "$HOME/venvs/vllm/bin/activate"
fi

# ===========================================================================
# Configuration
# ===========================================================================
QUANT="fp16"
TP=2
MAX_LEN=8192
UTIL=0.90
N_LADDER=(10 25 50 100 200 500 1000)
COOLDOWN_S=15

OUT_DIR="benchmarks/results/bielik-11b-v23/thermal-runs"
LOG="docs/sessions/2026-05-04-bielik-11b-fp16-sweep.md"

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$LOG")"

# ===========================================================================
# Sanity check
# ===========================================================================
python3 -c "import vllm; assert vllm.__version__.startswith('0.19.0'), f'WRONG vLLM: {vllm.__version__}'"
test -d "$HOME/models/bielik-11b-v23" || { echo "BRAK ~/models/bielik-11b-v23"; exit 1; }

# ===========================================================================
# Lab log header
# ===========================================================================
{
    echo "# Bielik 11B v2.3 FP16 TP=2 — Phase 2 sweep"
    echo ""
    echo "**Date:** $(date -Iseconds)"
    echo "**Operator:** Łukasz Minarowski <lukasz.minarowski@umb.edu.pl>"
    echo "**Methodology:** METHODOLOGY.md v1.0 §5.2"
    echo "**Embargo:** EMBARGO_paper_bound (Polish model, §11.3)"
    echo ""
    echo "## Configuration"
    echo ""
    echo "- Model: speakleash/Bielik-11B-v2.3-Instruct"
    echo "- Quantization: FP16"
    echo "- Tensor parallel size: 2"
    echo "- max_model_len: 8192"
    echo "- gpu_memory_utilization: 0.90"
    echo "- enforce_eager: True (graphs path segfaults on gfx1201)"
    echo "- N ladder: ${N_LADDER[*]}"
    echo "- Cooldown between runs: ${COOLDOWN_S}s"
    echo ""
    echo "## Phase 1 selection rationale"
    echo ""
    echo "Selected from envelope sweep (2026-05-04, all 7 configs loaded successfully):"
    echo "- Highest sanity tok/s_out: 17.42 (vs 14.16 for FP16 TP=1 max=8192)"
    echo "- Largest KV pool: 173,328 tokens (vs 34,320 for TP=1)"
    echo "- TP=2 advantage emerges at max_len=8192 (AllReduce overhead masked by"
    echo "  larger compute per token + 2x memory bandwidth from sharding)"
    echo ""
    echo "## Methodological humility (METHODOLOGY §8)"
    echo ""
    echo "We measure inference *throughput*, *latency*, *thermal envelope*, and"
    echo "*power efficiency* under varying concurrent load. We do not measure"
    echo "model quality, reasoning capability, factual accuracy, or downstream"
    echo "clinical utility. Following Lerchner (2026), these are extrinsic"
    echo "computational properties of the inference vehicle, not constitutive"
    echo "properties of cognition. Our claims terminate at the hardware-software"
    echo "interface."
    echo ""
    echo "## Run log"
    echo ""
} > "$LOG"

# ===========================================================================
# Main sweep loop
# ===========================================================================
T_START=$(date +%s)

for N in "${N_LADDER[@]}"; do
    NAME="${QUANT}-tp${TP}-n${N}"
    echo ""
    echo "============================================================"
    echo "Run: $NAME ($(date -Iseconds))"
    echo "============================================================"

    {
        echo "### N=$N"
        echo ""
        echo "- Start: $(date -Iseconds)"
    } >> "$LOG"

    # Capture per-run start/end in lab log
    RUN_START=$(date +%s)

    python3 benchmarks/scripts/instrumentation/bench_with_thermals_bielik_11b.py \
        "$TP" "$N" \
        --quant "$QUANT" \
        --max-len "$MAX_LEN" \
        --util "$UTIL" \
        --name "$NAME" \
        --out-dir "$OUT_DIR" \
        --interval 1.0 || {
            echo "FAIL: N=$N — see ${OUT_DIR}/${NAME}-bench.log" | tee -a "$LOG"
            # Continue to next N rather than abort entire sweep
            continue
        }

    RUN_END=$(date +%s)
    RUN_S=$((RUN_END - RUN_START))

    {
        echo "- End: $(date -Iseconds) (${RUN_S}s wall)"
        echo "- Artifacts: ${OUT_DIR}/${NAME}-{bench.log,events.json,thermals.jsonl,thermals.png}"
        echo ""
    } >> "$LOG"

    echo "Cooldown ${COOLDOWN_S}s..."
    sleep "$COOLDOWN_S"
done

T_END=$(date +%s)
T_TOTAL=$((T_END - T_START))

# ===========================================================================
# Summary footer
# ===========================================================================
{
    echo ""
    echo "## Sweep summary"
    echo ""
    echo "- Total wall time: ${T_TOTAL}s ($((T_TOTAL / 60)) min)"
    echo "- Configs completed: $(find "$OUT_DIR" -maxdepth 1 -name "${QUANT}-tp${TP}-n*-bench.log" 2>/dev/null | wc -l) / ${#N_LADDER[@]}"
    echo "- End: $(date -Iseconds)"
    echo ""
    echo "## Next steps"
    echo ""
    echo "1. Aggregate results into results_table.csv per METHODOLOGY §7.3"
    echo "2. Generate scaling_curve.png, thermal_gallery.png, efficiency_curve.png"
    echo "3. Write SUMMARY.md with embargo split (engineering PUBLIC, scaling EMBARGOED)"
    echo "4. Atomic commits: results dir + lab log + (later) plots"
    echo ""
    echo "## AI usage disclosure (METHODOLOGY §9)"
    echo ""
    echo "- Layer 1 (data): N/A — synthetic prompts per §6, deterministic from"
    echo "  human-curated templates and topics."
    echo "- Layer 2 (pipeline): Claude Opus 4.7 via claude.ai used 2026-05-04"
    echo "  for Phase 1 envelope analysis and Phase 2 sweep orchestrator drafting."
    echo "- Layer 3 (manuscript): TBD per paper submission."
} >> "$LOG"

echo ""
echo "============================================================"
echo "Sweep complete. Total: ${T_TOTAL}s ($((T_TOTAL / 60)) min)"
echo "Lab log: $LOG"
echo "Results: $OUT_DIR"
echo "============================================================"
