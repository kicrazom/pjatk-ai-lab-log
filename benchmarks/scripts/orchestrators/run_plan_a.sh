#!/bin/bash
# Plan A — full sweep with thermal instrumentation
# 7 concurrency levels × 2 TP configs = 14 runs

set -u  # error on unset vars

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
OUT_DIR="$REPO_ROOT/benchmarks/results/qwen2.5-7b-fp16/thermal-runs"
SUMMARY="$OUT_DIR/plan-a-summary.log"

# Plan A: logarithmic spacing; 3000 (not 5000) as last point per user decision
N_VALUES=(50 100 200 500 1000 2000 3000)
TP_VALUES=(1 2)
COOLDOWN_S=45

mkdir -p "$OUT_DIR"

# Sanity checks
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: venv not active. Run: source ~/venvs/vllm/bin/activate"
    exit 1
fi

if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm not importable. Wrong venv?"
    exit 1
fi

if [ ! -f ~/benchmarks/test_concurrent.py ]; then
    echo "ERROR: ~/benchmarks/test_concurrent.py not found."
    echo "Create symlink: ln -sf $REPO_ROOT/benchmarks/scripts/runners/test_concurrent.py ~/benchmarks/test_concurrent.py"
    exit 1
fi

export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1

echo "============================================" | tee "$SUMMARY"
echo "Plan A sweep started: $(date)"                 | tee -a "$SUMMARY"
echo "N values: ${N_VALUES[*]}"                      | tee -a "$SUMMARY"
echo "TP values: ${TP_VALUES[*]}"                    | tee -a "$SUMMARY"
echo "Cooldown between runs: ${COOLDOWN_S}s"         | tee -a "$SUMMARY"
echo "Total runs: $((${#N_VALUES[@]} * ${#TP_VALUES[@]}))" | tee -a "$SUMMARY"
echo "Output: $OUT_DIR"                              | tee -a "$SUMMARY"
echo "============================================" | tee -a "$SUMMARY"

RUN_IDX=0
TOTAL_RUNS=$((${#N_VALUES[@]} * ${#TP_VALUES[@]}))
GLOBAL_START=$(date +%s)

for tp in "${TP_VALUES[@]}"; do
    for n in "${N_VALUES[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        NAME="tp${tp}-n${n}"
        echo "" | tee -a "$SUMMARY"
        echo "--- Run $RUN_IDX/$TOTAL_RUNS: $NAME ---" | tee -a "$SUMMARY"
        echo "Time: $(date +%H:%M:%S) | Elapsed: $(( ($(date +%s) - GLOBAL_START) / 60 ))min" | tee -a "$SUMMARY"

        # Kill any lingering vLLM processes from a previous failed run
        pkill -f EngineCore 2>/dev/null || true
        pkill -f test_concurrent 2>/dev/null || true
        pkill -f sample_system 2>/dev/null || true
        sleep 2

        python "$REPO_ROOT/benchmarks/scripts/instrumentation/bench_with_thermals.py" "$tp" "$n" \
            --name "$NAME" --out-dir "$OUT_DIR" 2>&1 | \
            tee -a "$OUT_DIR/${NAME}-wrapper.log" | \
            grep -E "Output throughput|Requests/second|Total time|ERROR|Load time" | \
            tee -a "$SUMMARY"

        if [ $RUN_IDX -lt $TOTAL_RUNS ]; then
            echo "Cooldown ${COOLDOWN_S}s..." | tee -a "$SUMMARY"
            sleep "$COOLDOWN_S"
        fi
    done
done

ELAPSED=$(( ($(date +%s) - GLOBAL_START) / 60 ))
echo "" | tee -a "$SUMMARY"
echo "============================================" | tee -a "$SUMMARY"
echo "Plan A complete: $(date), total ${ELAPSED}min" | tee -a "$SUMMARY"
echo "============================================" | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "=== All results ===" | tee -a "$SUMMARY"
for tp in "${TP_VALUES[@]}"; do
    for n in "${N_VALUES[@]}"; do
        bench_log="$OUT_DIR/tp${tp}-n${n}-bench.log"
        if [ -f "$bench_log" ]; then
            tput=$(grep "Output throughput" "$bench_log" | tail -1 | awk '{print $(NF-1)}')
            printf "  TP=%d N=%-5d throughput=%s tok/s\n" "$tp" "$n" "${tput:-FAILED}" | tee -a "$SUMMARY"
        else
            printf "  TP=%d N=%-5d MISSING BENCH LOG\n" "$tp" "$n" | tee -a "$SUMMARY"
        fi
    done
done

echo ""
echo "Summary: $SUMMARY"
echo "Next: regenerate chart with updated data"
