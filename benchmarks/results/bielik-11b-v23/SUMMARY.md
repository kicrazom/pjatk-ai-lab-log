# Bielik 11B v2.3 — Phase 2 sweep summary

**Sweep ID:** bielik-11b-v23-fp16-tp2-max8192
**Date:** 2026-05-04
**Operator:** Łukasz Minarowski <lukasz.minarowski@umb.edu.pl>
**Methodology:** METHODOLOGY.md v1.0 §5.2 (Phase 2 scaling sweep)
**Embargo:** Mixed per §11.1/§11.2 — see classification below.

## Methodological humility (METHODOLOGY §8)

We measure inference *throughput*, *latency*, *thermal envelope*, and *power
efficiency* under varying concurrent load. We do not measure model quality,
reasoning capability, factual accuracy, or downstream clinical utility.
Following Lerchner (2026), these are extrinsic computational properties of
the inference vehicle, not constitutive properties of cognition. Our claims
terminate at the hardware-software interface.

## Configuration

- Model: `speakleash/Bielik-11B-v2.3-Instruct` (Polish-language, Mistral-based)
- Quantization: FP16
- Backend: vllm-rocm (vLLM 0.19.0+rocm721)
- Tensor parallel size: 2
- max_model_len: 8192
- gpu_memory_utilization: 0.9
- KV cache dtype: auto
- enforce_eager: True (graphs path segfaults in libhsa-runtime64 on gfx1201
  for Bielik AWQ; assumption applied to FP16 too — engineering finding
  extending METHODOLOGY §3.2)
- N ladder: [10, 25, 50, 100, 200, 500, 1000]
- Cooldown between runs: 15s

## Phase 1 envelope rationale (PUBLIC)

This config selected from 7-config envelope sweep (2026-05-04, all loaded
successfully). Selection criterion (pre-specified in PLAN-NEXT before
sweep): highest sanity_throughput_tok_s among loaded configs. Winner had
sanity 17.42 tok/s, KV pool 173,328 tokens, vs FP16 TP=1 max=8192 with
14.16 tok/s and 34,320 tokens. TP=2 advantage emerges at large max_len —
AllReduce overhead is masked by larger compute per token plus 2× memory
bandwidth from sharding.

## Results table (EMBARGO_paper_bound — Polish model, §11.3)

| N | tok/s_out | tok/s_tot | total_s | req/s | VRAM_GB | T_peak | W_mean | W/tok | vs_N10_% |
|---|-----------|-----------|---------|-------|---------|--------|--------|-------|----------|
| 10 | 183.14 | 204.03 | 6.99 | 1.431 | 28.13 | 45.0 | 188.0 | 1.0267 | 100.0 |
| 25 | 407.16 | 453.85 | 7.75 | 3.224 | 28.13 | 48.0 | 185.5 | 0.4554 | 222.3 |
| 50 | 699.47 | 783.28 | 8.69 | 5.756 | 28.13 | 51.0 | 196.2 | 0.2806 | 381.9 |
| 100 | 1182.64 | 1321.03 | 10.55 | 9.479 | 28.16 | 53.0 | 217.2 | 0.1837 | 645.8 |
| 200 | 1657.78 | 1855.00 | 14.75 | 13.555 | 28.25 | 57.0 | 229.3 | 0.1383 | 905.2 |
| 500 | 1793.96 | 2009.53 | 33.77 | 14.806 | 28.35 | 62.0 | 308.5 | 0.1720 | 979.6 |
| 1000 | 1793.39 | 2004.92 | 68.78 | 14.538 | 28.36 | 67.0 | 352.1 | 0.1963 | 979.2 |

`vs_N10_%` references the model's own N=10 stock run, per METHODOLOGY §10
limitation 10 (no public R9700 baseline exists for Polish-language models).

## Engineering observations (PUBLIC)

- **Scheduler scaling regime.** Aggregate throughput scales linearly through
  N=25 (89% efficiency), then enters a sub-linear regime through N=200
  (70-86% efficiency). Knee at N=200 (efficiency drops below 50% relative to linear extrapolation).

- **Plateau stability.** Aggregate throughput at N=500 and N=1000 is
  identical within 0.1%, indicating that vLLM's scheduler performs clean
  KV-cache swapping under preemption: doubled N → doubled wall time → zero
  throughput loss. Per-request latency doubles but aggregate stays flat.

- **Energy-optimal operating point: N=200 at 0.1383 W·s/tok.**

- **No thermal throttling observed.** GPU temperatures stayed below paper
  thresholds throughout sweep. Both GPUs reached steady-state during
  N=500/N=1000 long runs without intervention.

## EMBARGOED narrative (paper-bound)

The following sections contain concrete numerical claims that remain
embargoed until paper acceptance per METHODOLOGY §11.2 + §11.3:

- Plateau throughput value: ~1794 tok/s aggregate
- Knee position relative to KV cache exhaustion (computed as
  173k_kv_tokens / mean_seq_len)
- Cross-model comparative claims with Qwen 7B / 27B / 72B (deferred to
  paper synthesis)
- Per-N P50/P95/P99 latency distributions (not yet computed; require
  per-request timestamps from vLLM, future enhancement)

## Plots

- `scaling_curve.png` — `tok/s_out` vs `N` (log-x), knee annotated
- `thermal_gallery.png` — per-N temperature + power traces, aligned axes
- `efficiency_curve.png` — `W/tok` vs `N`, energy-optimal point annotated

## Limitations (METHODOLOGY §10)

All ten standing limitations from METHODOLOGY §10 apply, with these
emphases for this sweep:
- §10.1 synthetic prompts (apples-to-apples cross-model only, not clinical)
- §10.5 single-batch concurrency (not Poisson arrival)
- §10.6 1 Hz power sampling (sub-second transients lost)
- §10.10 Polish-language model lacks public baseline — `vs_baseline_%` is
  self-referential to N=10

## AI usage disclosure (METHODOLOGY §9)

| Layer | Disclosure |
|---|---|
| 1 — Dataset / data generation | N/A. Synthetic prompts per §6, deterministic from human-curated templates × topics. |
| 2 — Experimental pipeline | Claude Opus 4.7 (Anthropic, via claude.ai) used 2026-05-04 for Phase 1 envelope analysis, Phase 2 orchestrator drafting, and post-sweep aggregation script. Specific role: code generation and methodology validation against METHODOLOGY.md v1.0. No autonomous decision-making on experimental design — all decision criteria pre-specified in PLAN-NEXT.md before sweep. |
| 3 — Manuscript editing | TBD per paper submission. |

## Reproducibility (METHODOLOGY §3.3 causal closure)

Full version triple captured per-run in thermals.jsonl headers and in this
file's metadata:
- ROCm 7.2.0
- vLLM 0.19.0+rocm721 (pinned, do not upgrade — see logbook 2026-05-01)
- PyTorch 2.10.0+git8514f05 (HIP 7.2.53211)
- Env: AMD_SERIALIZE_KERNEL=1, ROCR_VISIBLE_DEVICES=0,1,
  VLLM_ROCM_USE_AITER=0, HIP_LAUNCH_BLOCKING=1, PYTORCH_ALLOC_CONF=unset

## Next steps

1. Phase 2 sweep for `awq_tp1_max2048_eager` (cross-quant comparison)
2. Phase 2 sweep for `fp16_tp1_max2048_eager` (TP=1 baseline, completes
   TP narrative arc)
3. Cross-config aggregate plot — three lines on one scaling_curve.png
4. Lab session log finalization in `docs/sessions/2026-05-04-bielik-11b-fp16-sweep.md`
