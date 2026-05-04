# PLAN-NEXT — Bielik 11B AWQ + FP16 TP=1 (2026-05-05)

**Last session:** 2026-05-04 — Phase 1 envelope (7/7 OK) + Phase 2 sweep
fp16-tp2-max8192 (7/7 OK, plateau 1794 tok/s @ N≈500). See lab log
`docs/sessions/2026-05-04-bielik-11b-fp16-sweep.md` and
`benchmarks/results/bielik-11b-v23/SUMMARY.md`.

**Current state (post-Phase-2 first config):**

* v0.1.0 (DOI 10.5281/zenodo.19851347) and v0.2.0 released — DOIs intact.
* Phase 1 envelope JSONs in `benchmarks/results/hardware_envelope/bielik_11b_*.json`
* Phase 2 fp16-tp2-max8192 thermal-runs + plots + SUMMARY.md committed.
* `main` head: 911493a (pre-commit infra).
* Worktree dirty — pending triage decision tomorrow morning.

---

## Tomorrow 2026-05-05 — Phase 2 AWQ + FP16 TP=1 (~1.5-2h)

### Pre-flight (~10 min)

```
cd ~/navimed-umb
source ~/venvs/vllm/bin/activate
unset PYTORCH_ALLOC_CONF
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=1
export HIP_LAUNCH_BLOCKING=1
export ROCR_VISIBLE_DEVICES=0,1
export HF_HUB_DISABLE_XET=1

git status                          # decide on dirty triage
ls benchmarks/results/bielik-11b-v23/
```

**Dirty repo decision** — two options:
  - (A) Triage 45 min upfront (atomic commits per scope, see SESSION_RESUME)
  - (B) Stash everything, sweep first, commit single batch at end of day

Recommend (A) for scope hygiene; recommend (B) only if mental bandwidth low.

### Sweep 1 — Bielik AWQ TP=1 (~15-20 min)

Adapt `benchmarks/scripts/orchestrators/run_bielik_11b_sweep.sh` — change
3 lines:

```
QUANT="awq"
TP=1
MAX_LEN=2048      # AWQ Phase 1 best
```

Output dir: `benchmarks/results/bielik-11b-v23/thermal-runs/` (same dir,
filenames will be `awq-tp1-n*-...` not `fp16-tp2-n*-...` so no collision).

Lab log: `docs/sessions/2026-05-05-bielik-11b-awq-sweep.md`.

Run:
```
systemd-inhibit --what=sleep:idle --who="bielik-awq-sweep" \
    bash benchmarks/scripts/orchestrators/run_bielik_11b_awq_sweep.sh 2>&1 \
    | tee /tmp/bielik-awq-$(date +%Y%m%d-%H%M).log
```

**Pre-specified expectation (AVOID HARKing):**

AWQ ma 3.13× więcej KV pool niż FP16 TP=1 (107k vs 34k). Spodziewamy się
że AWQ plateau będzie WYŻSZE niż fp16-tp1 plateau, ale NIŻSZE niż
fp16-tp2-max8192 (z racji single-GPU memory bandwidth limit). Knee
prawdopodobnie później niż dla fp16 (więcej KV → więcej concurrent before
preemption).

Jeśli AWQ tok/s_out @ N=10 jest >50 tok/s, plateau będzie wysokie. Jeśli
<30 tok/s, AWQ kernel na gfx1201 jest niewydajny (consistent z Phase 1
sanity 3.9 tok/s vs FP16 14.2 tok/s — AWQ kernel slowdown na single-prompt).

### Sweep 2 — Bielik FP16 TP=1 max_len=2048 (~10-15 min)

Adapt orchestrator — zmiany:

```
QUANT="fp16"
TP=1
MAX_LEN=2048
```

Lab log: `docs/sessions/2026-05-05-bielik-11b-fp16-tp1-sweep.md`.

**Pre-specified expectation:** Plateau wyraźnie niższe niż fp16-tp2-max8192
(34k vs 173k KV tokens = 5× mniej concurrent capacity). Knee przy N≈100-150.
Ten sweep zamyka narrację TP=2 vs TP=1: TP=2 max=8192 plateau 1794 tok/s,
TP=1 max=2048 plateau ~600-800 tok/s spodziewane.

### Cross-config aggregation (~15 min)

Adapt `finalize_bielik_phase2.py` na 3-config mode:

```python
CONFIGS = {
    "fp16-tp2-max8192": {color: "#2ca02c", label: "FP16 TP=2 max=8192"},
    "awq-tp1-max2048":  {color: "#1f77b4", label: "AWQ TP=1 max=2048"},
    "fp16-tp1-max2048": {color: "#ff7f0e", label: "FP16 TP=1 max=2048"},
}
```

Output:
- `benchmarks/results/bielik-11b-v23/scaling_curve_3config.png` (3 lines)
- `benchmarks/results/bielik-11b-v23/efficiency_curve_3config.png`
- Update `SUMMARY.md` z cross-config narrative

### Commit + push (~15 min)

Atomic commits:
1. `feat(bielik): Phase 2 sweep fp16-tp2-max8192 results + plots + SUMMARY`
2. `feat(bielik): Phase 2 sweep awq-tp1-max2048 results`
3. `feat(bielik): Phase 2 sweep fp16-tp1-max2048 results`
4. `feat(bielik): cross-config aggregate plots + SUMMARY update`
5. `docs(sessions): Bielik 11B v2.3 lab log entries 2026-05-04/05`

Plus dirty repo cleanup (jeśli (A) we wcześniejszym kroku):
- `style: apply pre-commit hooks across repo`
- `fix(scripts): F841 + SC1128 + SC1090 lint cleanup`

---

## What NOT to do tomorrow

- NIE skracaj N-laddra (wczoraj cały N={10..1000} działał w 9 min — szybko)
- NIE wprowadzaj nowych quantization/TP combinacji bez Phase 1 envelope
- NIE pomijaj ROCR_VISIBLE_DEVICES=0,1 — bez tego iGPU może się wmieszać
- NIE upgrade vLLM (pinned 0.19.0+rocm721)
- NIE rewrite commitów które są ancestors v0.1.0 / v0.2.0 (DOI immutable)

---

## Decision criterion for paper-bound config choice (PRE-SPECIFIED)

If asked which config to feature in the paper as "headline Bielik result":

> Highest tok/s_out at the energy-optimal N (lowest W·s/tok). Tie-breaker:
> highest plateau throughput. Tie-breaker 2: lowest VRAM peak.

This avoids HARKing — criterion fixed before seeing AWQ + FP16 TP=1 results.

---

## After 2026-05-05 (if time permits or next session)

- Bielik 11B v2.3 paper-bound figures finalize
- Move to next model in METHODOLOGY §4: Llama-PLLuM-8B-instruct
- Or: cross-model comparison (Bielik 11B vs Qwen 7B vs Qwen 27B) — needs
  fresh Qwen 7B re-run on current vLLM 0.19.0+rocm721 stack
