#!/usr/bin/env bash
### Sekwencyjne testowanie 4 hipotez dla Qwen 72B AWQ TP=2
### Kolejnosc: H4 (cold compile) -> H3 (debug flags) -> H2 (eager) -> H1 (marlin)
### Kazdy test: osobny Python proces, cooldown 30s, log do osobnego pliku

set -u  ### unset vars = error, ale NIE set -e (chcemy kontynuowac przy padach)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
HERE="$REPO_ROOT/benchmarks/scripts/runners"
LOG_DIR="$REPO_ROOT/logs/qwen72b_hypothesis_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

SCRIPT="$HERE/sanity_qwen72b_awq_hypothesis.py"
COOLDOWN_S=30

echo "=========================================="
echo "Qwen 72B AWQ TP=2 Hypothesis Testing"
echo "Log dir: $LOG_DIR"
echo "=========================================="

### Sanity check — vLLM venv active?
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: venv not active. Run: source ~/venvs/vllm/bin/activate"
    exit 1
fi

### H4 — drugi run, TE SAME flagi co baseline (cache warm?)
echo ""
echo "--- Test 1/4: H4 (cache warm, identyczne flagi jak baseline) ---"
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
HYPOTHESIS=H4 python "$SCRIPT" 2>&1 | tee "$LOG_DIR/h4_cache_warm.log"

echo "Cooldown ${COOLDOWN_S}s..."
sleep "$COOLDOWN_S"

### H3 — bez debug flags (ZMIANA: async kernels enabled)
echo ""
echo "--- Test 2/4: H3 (bez AMD_SERIALIZE_KERNEL, bez HIP_LAUNCH_BLOCKING) ---"
export VLLM_ROCM_USE_AITER=0
unset AMD_SERIALIZE_KERNEL
unset HIP_LAUNCH_BLOCKING
HYPOTHESIS=H3 python "$SCRIPT" 2>&1 | tee "$LOG_DIR/h3_no_debug.log"

echo "Cooldown ${COOLDOWN_S}s..."
sleep "$COOLDOWN_S"

### H2 — bez enforce_eager (RYZYKO: moze wrocic HSA_ERROR)
echo ""
echo "--- Test 3/4: H2 (enforce_eager=False, async kernels enabled) ---"
export VLLM_ROCM_USE_AITER=0
unset AMD_SERIALIZE_KERNEL
unset HIP_LAUNCH_BLOCKING
HYPOTHESIS=H2 python "$SCRIPT" 2>&1 | tee "$LOG_DIR/h2_no_eager.log"

echo "Cooldown ${COOLDOWN_S}s..."
sleep "$COOLDOWN_S"

### H1 — awq_marlin zamiast awq_triton (kernel change)
echo ""
echo "--- Test 4/4: H1 (quantization=awq_marlin, z powrotem debug flags dla pewnosci) ---"
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
HYPOTHESIS=H1 python "$SCRIPT" 2>&1 | tee "$LOG_DIR/h1_awq_marlin.log"

### Podsumowanie
echo ""
echo "=========================================="
echo "Wszystkie 4 testy zakonczone"
echo "Throughput comparison:"
echo "=========================================="
for hyp in h4_cache_warm h3_no_debug h2_no_eager h1_awq_marlin; do
    log_file="$LOG_DIR/${hyp}.log"
    if [ -f "$log_file" ]; then
        tput=$(grep "Throughput:" "$log_file" | tail -1 | awk '{print $2, $3}')
        load=$(grep "Load time:" "$log_file" | tail -1 | awk '{print $3}')
        printf "  %-20s throughput=%s  load=%s\n" "$hyp" "${tput:-FAILED}" "${load:-?}"
    else
        printf "  %-20s MISSING LOG\n" "$hyp"
    fi
done

echo ""
echo "Logs: $LOG_DIR"
