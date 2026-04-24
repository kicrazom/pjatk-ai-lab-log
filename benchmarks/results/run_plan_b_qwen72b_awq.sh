#!/bin/bash
### Plan B - Qwen 72B AWQ TP=2 concurrency sweep with thermal/power instrumentation
### 7 concurrency levels x 1 TP config = 7 runs
### TP=1 excluded: 72B AWQ (39 GB) does not fit on single 32 GB R9700
###
### Based on run_plan_a.sh (Qwen 7B) with adjustments for 72B AWQ:
### - N values scaled for smaller KV cache (32k tokens vs 7B's 120k)
### - TP=1 removed (architectural constraint)
### - Env vars per H3 finding (2026-04-24): enforce_eager=True is sufficient,
###   AMD_SERIALIZE_KERNEL / HIP_LAUNCH_BLOCKING removed (36%+ overhead)
### - Cooldown raised to 60s (72B warmup takes ~90s, cards stay warm longer)

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$HERE/thermal-runs/qwen72b-awq"
SUMMARY="$OUT_DIR/plan-b-summary.log"

### N sweep for 72B - logarithmic spacing, ceiling at KV cache preemption point
### 72B has 0.175 MB/token KV vs 7B's 0.055 MB/token, so preemption enters earlier
N_VALUES=(10 25 50 100 200 500 1000)
TP_VALUES=(2)
COOLDOWN_S=60

mkdir -p "$OUT_DIR"

### Sanity checks
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: venv not active. Run: source ~/venvs/vllm/bin/activate"
    exit 1
fi

if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm not importable. Wrong venv?"
    exit 1
fi

if [ ! -f "$HERE/bench_with_thermals_qwen72b.py" ]; then
    echo "ERROR: bench_with_thermals_qwen72b.py not found at $HERE"
    exit 1
fi

if [ ! -d /home/mozarcik/models/qwen25-72b-awq ]; then
    echo "ERROR: Model directory missing"
    exit 1
fi

### H3 config env vars (from logbook/2026-04-24.md hypothesis testing)
### enforce_eager=True is still required (H2 crashed with HSA_STATUS_ERROR)
### AMD_SERIALIZE_KERNEL and HIP_LAUNCH_BLOCKING REMOVED (H3: 36% speedup)
export VLLM_ROCM_USE_AITER=0
unset AMD_SERIALIZE_KERNEL
unset HIP_LAUNCH_BLOCKING

echo "============================================" | tee "$SUMMARY"
echo "Plan B sweep started: $(date)"                 | tee -a "$SUMMARY"
echo "Model: Qwen 2.5 72B AWQ (local)"               | tee -a "$SUMMARY"
echo "N values: ${N_VALUES[*]}"                      | tee -a "$SUMMARY"
echo "TP values: ${TP_VALUES[*]}"                    | tee -a "$SUMMARY"
echo "Cooldown between runs: ${COOLDOWN_S}s"         | tee -a "$SUMMARY"
echo "Config: H3 (enforce_eager=True, no debug flags)" | tee -a "$SUMMARY"
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

        ### Kill any lingering vLLM processes from a previous failed run
        pkill -f EngineCore 2>/dev/null || true
        pkill -f test_concurrent_qwen72b 2>/dev/null || true
        pkill -f sample_system 2>/dev/null || true
        sleep 2

        python "$HERE/bench_with_thermals_qwen72b.py" "$tp" "$n" \
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
echo "Plan B complete: $(date), total ${ELAPSED}min" | tee -a "$SUMMARY"
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
