#!/usr/bin/env bash
# End-to-end Qwen 3.6 27B sweep orchestrator on 2x R9700 (gfx1201).
#
# Runs both FP8 and BF16 variants with thermal monitoring per N value.
# Configurations are Phase 1 envelope-validated: pass best (max_len, util)
# from Phase 1 results as FP8_MAX_LEN, FP8_UTIL, BF16_MAX_LEN, BF16_UTIL
# environment variables (or use defaults).
#
# Outputs land in:
#   benchmarks/results/qwen36-27b-fp8/thermal-runs/
#   benchmarks/results/qwen36-27b/thermal-runs/
#
# Each run produces 5 files:
#   {name}-thermals.jsonl  — per-second sampler output
#   {name}-events.json     — bench start/end timestamps
#   {name}-thermals.png    — thermal/util plot with events
#   {name}-bench.log       — full benchmark stdout (load time, throughput, etc.)
#
# Usage:
#   bash run_qwen36_27b_sweep.sh                    # use Phase 1 defaults
#   FP8_MAX_LEN=4096 bash run_qwen36_27b_sweep.sh   # override via env
#   ONLY_QUANT=fp8 bash run_qwen36_27b_sweep.sh     # skip BF16

set -euo pipefail

REPO_ROOT="$HOME/navimed-umb"
SCRIPTS_DIR="$REPO_ROOT/benchmarks/scripts"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"

WRAPPER="$SCRIPTS_DIR/instrumentation/bench_with_thermals_qwen36_27b.py"

# Validate scripts exist
if [[ ! -f "$WRAPPER" ]]; then
    echo "ERROR: missing $WRAPPER"
    echo "Did you copy bench_with_thermals_qwen36_27b.py to instrumentation/?"
    exit 1
fi

# Activate venv if not already
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "$HOME/venvs/vllm/bin/activate"
fi

# Mandatory env for gfx1201
unset PYTORCH_ALLOC_CONF
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=1
export HIP_LAUNCH_BLOCKING=1

# Phase 1-validated configurations (override via env vars)
FP8_MAX_LEN="${FP8_MAX_LEN:-2048}"
FP8_UTIL="${FP8_UTIL:-0.85}"
FP8_KV_DTYPE="${FP8_KV_DTYPE:-}"

BF16_MAX_LEN="${BF16_MAX_LEN:-1024}"
BF16_UTIL="${BF16_UTIL:-0.95}"
BF16_KV_DTYPE="${BF16_KV_DTYPE:-}"

# Sweep N values per quantization
# FP8 has ~52x max concurrency at default config — sweep up to 50
# BF16 has ~7x max concurrency at default config — sweep up to 7
# Phase 1 may discover better BF16 config (e.g. 30x); update arrays below
FP8_N_VALUES=(1 5 10 25 50)
BF16_N_VALUES=(1 2 5 7)

# Quantization filter
ONLY_QUANT="${ONLY_QUANT:-both}"

run_quant_sweep() {
    local quant="$1"
    local max_len="$2"
    local util="$3"
    local kv_dtype="$4"
    shift 4
    local n_values=("$@")

    local out_dir="$RESULTS_DIR/qwen36-27b"
    if [[ "$quant" == "fp8" ]]; then
        out_dir="$RESULTS_DIR/qwen36-27b-fp8"
    fi
    out_dir="$out_dir/thermal-runs"
    mkdir -p "$out_dir"

    echo ""
    echo "=========================================="
    echo "  Sweep: ${quant^^} (TP=2)"
    echo "=========================================="
    echo "  max_model_len:        $max_len"
    echo "  gpu_memory_util:      $util"
    echo "  kv_cache_dtype:       ${kv_dtype:-default}"
    echo "  N values:             ${n_values[*]}"
    echo "  Output dir:           $out_dir"
    echo ""

    local kv_arg=""
    if [[ -n "$kv_dtype" ]]; then
        kv_arg="--kv-dtype $kv_dtype"
    fi

    for n in "${n_values[@]}"; do
        local run_name="${quant}-tp2-n${n}"
        echo ""
        echo "--- Running: $run_name ---"

        python3 "$WRAPPER" 2 "$n" \
            --quant "$quant" \
            --max-len "$max_len" \
            --util "$util" \
            $kv_arg \
            --name "$run_name" \
            --out-dir "$out_dir"

        # Brief cooldown between runs (let GPU temps drop)
        echo "Cooldown 15s..."
        sleep 15
    done

    echo ""
    echo "${quant^^} sweep complete. Artifacts in: $out_dir"
}

t_start=$(date +%s)

if [[ "$ONLY_QUANT" == "both" || "$ONLY_QUANT" == "fp8" ]]; then
    run_quant_sweep "fp8" "$FP8_MAX_LEN" "$FP8_UTIL" "$FP8_KV_DTYPE" "${FP8_N_VALUES[@]}"
fi

if [[ "$ONLY_QUANT" == "both" || "$ONLY_QUANT" == "bf16" ]]; then
    run_quant_sweep "bf16" "$BF16_MAX_LEN" "$BF16_UTIL" "$BF16_KV_DTYPE" "${BF16_N_VALUES[@]}"
fi

t_end=$(date +%s)
elapsed=$((t_end - t_start))

echo ""
echo "=========================================="
echo "  SWEEP COMPLETE"
echo "=========================================="
echo "  Total elapsed: $((elapsed / 60))m $((elapsed % 60))s"
echo ""
echo "Next steps:"
echo "  1. Aggregate scaling data:"
echo "     python3 $SCRIPTS_DIR/plotting/plot_scaling.py \\"
echo "       $RESULTS_DIR/qwen36-27b-fp8/thermal-runs/ \\"
echo "       $RESULTS_DIR/qwen36-27b-fp8/scaling_curve.png"
echo ""
echo "  2. Generate thermal gallery:"
echo "     python3 $SCRIPTS_DIR/plotting/plot_thermal_gallery.py \\"
echo "       $RESULTS_DIR/qwen36-27b-fp8/thermal-runs/"
echo ""
echo "  3. Write session report at:"
echo "     docs/sessions/$(date +%Y-%m-%d)-qwen36-27b-throughput-sweep.md"
