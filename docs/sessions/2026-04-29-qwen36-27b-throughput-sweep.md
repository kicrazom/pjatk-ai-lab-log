# 2026-04-29 — Qwen 3.6 27B Phase 2 throughput sweep + universal reporting

**Session goal:** complete v0.2.0 milestone — Phase 1 envelope analysis,
Phase 2 BF16 scaling sweep, universal reporting infrastructure, and
methodology consolidation.

**Branch:** `phase1-qwen36-27b` (8 atomic commits, ready to merge to main).

---

## Methodological humility (METHODOLOGY §8)

This session measures inference *throughput*, *latency*, *thermal envelope*,
and *power efficiency* for Qwen 3.6 27B on dual R9700 (gfx1201) under varying
concurrent request count. We do not measure model quality, reasoning
capability, factual accuracy, or downstream clinical utility. Following
Lerchner (2026), these are extrinsic computational properties of the
inference vehicle, not constitutive properties of cognition. All claims
terminate at the hardware-software interface.

---

## What was done

### Phase 1 — Hardware envelope (10 configurations)

Sweep over `(quantization, max_model_len, gpu_memory_utilization, kv_cache_dtype)`
combinations, single sanity prompt per config. **All 10 configurations
loaded successfully** on 2× R9700 with vLLM 0.19.0+rocm721.

Selected for Phase 2: **BF16 max_len=2048 util=0.97 KV=default** — highest
sanity throughput (7.89 tok/s), peak VRAM 29.32 GB/GPU (~9% headroom).

**Negative result captured:** KV FP8 e4m3 quantization on R9700 produces
**5× slowdown** vs default KV (1.48-1.53 vs 7-8 tok/s). vLLM 0.19.0 lacks
native FP8 KV kernels for gfx1201 — kernels do on-the-fly conversion. FP8
weights similarly slower than BF16 (4.77-5.34 vs 7-8 tok/s) due to absent
FP8 compute kernels. Both are publishable engineering findings for the
paper's "What doesn't work" section.

### Phase 2 — Scaling sweep (N=10..1000)

Single configuration (BF16 max2048 util097), seven N values, n_runs=1.
Total compute: 54m33s sweep + ~50s/run model load.

**Three engineering findings:**

1. **Throughput knee at N=200**, NOT at vLLM scheduler `max_concurrency`
   estimate of 24. Optimum batch size is ~8× the scheduler estimate for
   this model+config. Practitioners should sweep, not trust scheduler
   defaults.

2. **vLLM scheduler robustness under preemption.** At N=1000 (5× over
   throughput knee, 40× over scheduler max_concurrency), output throughput
   is -0.9% vs peak. No starvation pathology observed even at extreme
   over-saturation. PagedAttention + continuous batching scheduler are
   well-tuned for graceful degradation.

3. **Energy efficiency optimum (lowest mWh/token) is at N=25**, not at
   peak throughput N=200. Trade-off: 16% energy savings per token at the
   cost of ~14% aggregate throughput. Operators should choose operating
   point per priority — interactive (low N), throughput-max (peak N), or
   energy-optimal (energy-min N). This trade-off is invisible in marketing
   benchmarks that report only peak aggregate tok/s.

### Universal reporting infrastructure

Three new tools applied to both Qwen 3.6 27B (today's data) and Qwen 2.5
7B (retrofit of v0.1.0 data):

- `analyze_phase1_envelope.py` — Phase 1 JSON aggregator → SUMMARY.md + CSV.
- `analyze_phase2_sweep.py` — Phase 2 sweep aggregator (legacy + current
  filename support, quant auto-detection) → SUMMARY.md + results_table.csv
  in kyuz0+ schema.
- `plot_phase2_sweep.py` — three plots per METHODOLOGY §7.2 (scaling curve
  with knee, thermal twin-axes, efficiency curve with optimum). v1
  single-config; multi-config split lines deferred to v0.3.0.

### Methodology consolidation

`METHODOLOGY.md` — single source of truth, version 1.0. Synthesizes
Capitelli 2025 schema, Kim et al. 2026 disclosure framework, Lerchner 2026
humility position, ICMJE/COPE compliance, internal embargo policy. Defines
13 sections (scope, hardware, software, model suite, two-phase design,
workload, reporting schema, humility, AI disclosure, limitations, embargo,
references, versioning). Authoritative for v0.2.0+; future per-model session
logs reference by section number rather than duplicating text.

---

## Cross-model engineering insight (PUBLIC)

Today's data permits one cross-model comparison appropriate for public
release:

**Energy efficiency scales dramatically with model size.** Qwen 2.5 7B
operates at 0.010-0.035 mWh/token; Qwen 3.6 27B at 0.9-1.2 mWh/token —
roughly **35× higher energy cost per token at ~4× parameter count**.
Comparison only meaningful per-watt-per-token; raw throughput numbers
without energy normalization mislead operational planning.

**TP=2 is harmful for 7B-class models** that fit a single 32 GB GPU.
At N≥200, TP=1 outperforms TP=2 by ~24% throughput plus ~75% lower energy
per token. Tensor-parallel network overhead dominates compute savings when
the model is not memory-bound. Practitioners should default to TP=1 for
7B-class deployment unless KV cache pressure forces TP=2 (e.g. very long
contexts).

---

## AI disclosure (per METHODOLOGY §9)

| Layer | Practice this session |
|---|---|
| **1. Dataset** | Not applicable. Synthetic prompts deterministic from human-curated 8-template × 20-topic grid. |
| **2. Pipeline** | Claude Opus 4.7 (Anthropic, accessed via claude.ai) used during 2026-04-29 for: Phase 1 result interpretation, statistical methodology design (Tier A n=10 + Holm-Bonferroni decision), analyzer/plotter Python scaffolding, embargo-classification review per artifact, METHODOLOGY.md drafting. No autonomous agent operation; all code reviewed before execution. |
| **3. Manuscript** | Not applicable in this session — paper writing deferred to v0.3+. |

---

## Embargo classification of session artefacts

| Artefact | Status |
|---|---|
| Engineering findings narrative (above) | PUBLIC |
| `benchmarks/results/hardware_envelope/SUMMARY.md` + CSV | PUBLIC |
| `benchmarks/results/qwen36-27b/SUMMARY.md` + CSV + 3 plots | PUBLIC |
| `benchmarks/results/qwen2.5-7b-fp16/SUMMARY.md` + CSV (retrofit) | PUBLIC |
| `benchmarks/results/qwen36-27b/thermal-runs/` (raw per-run) | EMBARGO (gitignored) |
| Tier A n=10 reruns | DEFERRED to v0.2.1 |
| Cross-model figures with concrete numbers (paper-bound) | EMBARGO (not produced this session) |

---

## Lessons learned

1. **vLLM `max_concurrency` is conservative — sweep, don't trust.** The
   scheduler estimate underestimated optimum batch by 8×. This is a key
   recommendation for any practitioner deploying vLLM on consumer GPUs.

2. **Energy-optimal operating point ≠ throughput-optimal.** Marketing
   benchmarks report only peak throughput; honest deployment planning
   requires the W/tok curve. The METHODOLOGY §7.2 efficiency curve plot
   is the right artifact for this question.

3. **Filename conventions matter for tool reuse.** Legacy 7B
   (`tp{N}-n{N}-bench.log`) and current 27B (`{quant}-tp{N}-n{N}-bench.log`)
   formats both work with v2 analyzer thanks to optional regex group +
   quant auto-detection from vLLM dtype log. New models in suite should
   use the current convention; legacy data is forward-compatible.

4. **Embargo policy granularity per artifact, not per directory.**
   Aggregated SUMMARY.md and CSV go public (engineering findings); raw
   `thermal-runs/` stay local (paper-bound figure regeneration). The
   gitignore pattern `benchmarks/results/*/thermal-runs/` enforces this
   automatically for new sweeps without affecting v0.1.0 already-tracked
   files.

5. **Statistical methodology decision: Tier A only n=10.** Phase 2 v0.2.0
   is single-shot exploratory; v0.2.1 will rerun key configs (1-2 per
   model) at n=10 with Holm-Bonferroni correction for confirmatory
   hypothesis tests. Estimated compute: ~6-7h per config × 13 models ×
   2 configs avg ~ 170h total — split across multiple weekend sessions.

---

## Hardware/software environment (causal closure per METHODOLOGY §3.3)

```
GPU:           2x GIGABYTE Radeon AI PRO R9700 (gfx1201, 32 GB GDDR6 each)
CPU:           AMD Ryzen 9 9950X3D (16 cores / 32 threads)
RAM:           96 GB DDR5-6000
OS:            Kubuntu 24.04, kernel 6.17
ROCm:          7.2.0
vLLM:          0.19.0+rocm721
PyTorch:       2.10.0+git8514f05
flash_attn:    2.8.3
triton:        3.6.0
Env (mandatory for gfx1201):
  unset PYTORCH_ALLOC_CONF
  VLLM_ROCM_USE_AITER=0
  AMD_SERIALIZE_KERNEL=1
  HIP_LAUNCH_BLOCKING=1
  ROCR_VISIBLE_DEVICES=0,1
Model construction: enforce_eager=True (mandatory for Qwen 3.5/3.6 hybrid attention)
```

---

## Commit log this session

```
913ce35 chore(gitignore): raw terminal logs are local-only
d80e86f docs(methodology): single source of truth for benchmark methodology
55c3819 data(phase2): Qwen 2.5 7B FP16 retrofit to kyuz0+ schema (PUBLIC)
37ca804 data(phase2): Qwen 3.6 27B BF16 sweep N=10..1000 (aggregated artefacts)
a51f1bc data(phase1): Qwen 3.6 27B hardware envelope SUMMARY (PUBLIC)
4c61aed feat(analysis): universal Phase 1/Phase 2 analyzer + plotter (kyuz0+ schema)
570dc9d feat(sweep): align Phase 2 N values to research plan (10..1000)
4e83c37 data(phase1): Qwen 3.6 27B envelope on 2x R9700 (10 configs)
```

---

## Next session

See `docs/sessions/PLAN-NEXT.md`.
