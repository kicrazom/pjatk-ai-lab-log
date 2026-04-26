# Qwen 2.5 72B AWQ — Plan B pilot

vLLM 0.19.0+rocm721 serving `Qwen/Qwen2.5-72B-Instruct-AWQ` on 2× R9700
(gfx1201) with mandatory TP=2. The 39 GB AWQ Int4 weights do not fit on
one 32 GB R9700, so the question was the plateau, not the choice. This
is the **TP=2-mandatory** corner of the
[three-point arc](../../README.md#the-tp2-arc-harmful-optional-necessary).

Pilot executed 2026-04-24, plus two negative-result follow-ups. Full
narrative: [`logbook/2026-04-24.md`](../../../logbook/2026-04-24.md).

## Pilot — N sweep, TP=2

| N    | Total time | Output throughput | Requests/s |
|-----:|-----------:|------------------:|-----------:|
|   10 |      92.8s |       13.8 tok/s  |       0.11 |
|   25 |      97.6s |       32.8 tok/s  |       0.26 |
|   50 |     105.0s |       61.0 tok/s  |       0.48 |
|  100 |     116.2s |      110.1 tok/s  |       0.86 |
|  200 |     217.6s |      117.6 tok/s  |       0.92 |
|  500 |     526.2s |      121.5 tok/s  |       0.95 |
| 1000 |    1051.2s |      121.7 tok/s  |       0.95 |

**Plateau ~121 tok/s for N ≥ 500** (two points within 0.2%); total time
scales linearly beyond, meaning additional requests queue. KV cache
stayed below capacity (peak 26.3 / 31.86 GiB per card) — no preemption.

**Linear up to N=100, knee at N=100→200.** N=10→100 grows ~8×; the
next doubling adds only 7%. KV is much smaller than Plan A's (32k
tokens vs ~120k for 7B FP16), so concurrency saturates earlier.

### Comparison to the TP=2-harmful regime (Qwen 7B Plan A)

| N    | Qwen 7B FP16 TP=2 | Qwen 72B AWQ TP=2 | Ratio |
|-----:|------------------:|------------------:|------:|
|   50 |              1154 |              61.0 | 18.9× |
|  100 |              1772 |             110.1 | 16.1× |
|  500 |              2938 |             121.5 | 24.2× |
| 1000 |              2946 |             121.7 | 24.2× |

The 24× plateau gap decomposes as 10× model size, ~2-3× AWQ kernel
overhead (`AWQMarlinLinearMethod` falls back to `ConchLinearKernel` —
a generic Triton path on ROCm 7.2.1, *not* a fused dequant+matmul like
NVIDIA Marlin), and higher per-token TP=2 sync cost (80 layers vs 28).
Only the first is intrinsic; AWQ kernel quality is gated on AITER.

## Configuration

| Parameter | Value | Why |
|---|---|---|
| `dtype` | `"auto"` | vLLM picks awq_marlin |
| `gpu_memory_utilization` | 0.92 | 72B AWQ needs more headroom than 7B's 0.70 |
| `max_model_len` | 4096 | KV-cache capacity vs prompt length tradeoff |
| `enforce_eager` | `True` | gfx1201 graph-capture workaround |
| `tensor_parallel_size` | 2 | Mandatory — weights don't fit one card |
| `max_tokens` | 128 | Matches Plan A for cross-comparison |
| Cooldown | 60s | 72B keeps cards warm longer than 7B |
| `AMD_SERIALIZE_KERNEL` / `HIP_LAUNCH_BLOCKING` | unset | H3: redundant when eager, costs ~36% |

Orchestrator:
[`../../scripts/orchestrators/run_plan_b_qwen72b_awq.sh`](../../scripts/orchestrators/run_plan_b_qwen72b_awq.sh)
— full sweep in ~65 minutes with thermal sampling on every run.

## Negative-result follow-ups

**H7 — kyuz0 attention flags.** `VLLM_ROCM_USE_AITER=1`,
`VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1`, `VLLM_USE_TRITON_FLASH_ATTN=0`
per the [kyuz0 toolbox](https://github.com/kyuz0/amd-r9700-vllm-toolboxes).
vLLM emitted `Unknown vLLM environment variable` for the
PREFILL_DECODE flag and resolved attention to `ROCM_ATTN` — identical
to Plan B. Throughput unchanged at **110.1 tok/s** at N=100. The flag
exists only in the TheRock-patched git build kyuz0 ships
(vLLM `0.14.0rc1.dev27+g1501a4070.d20251220`, Dec 2025 HEAD); not in
stable PyPI 0.19.0+rocm721. Data: [`h7-attn-negative/`](h7-attn-negative/).
TheRock migration queued as Phase D.

**CCD1 pinning.** `taskset -c 8-15,24-31` to force both workers onto
the 96 MB 3D V-Cache CCD. Affinity confirmed via `taskset -cp`. Result:
**three identical 110.1 tok/s measurements** across baseline, H7, and
CCD1. Workload is GPU-bound (100% util both cards; CPU ~14%); V-Cache
cannot accelerate a non-bottleneck. Data:
[`ccd1-negative/`](ccd1-negative/).

## What stays open

- **Graph capture.** `enforce_eager=True` costs ~20–30%; recovery
  needs a working gfx1201 graph-capture path. Periodic retest queued.
- **Unique-prompt throughput.** 24× ratio is on prefix-cached prompts;
  unique would compress the gap. Queued for Polish-language phase.
- **AWQ kernel quality on ROCm.** AITER AWQ work is the upstream;
  current stable falls back to generic Triton.

## Files

| Path | Content |
|---|---|
| [`pilot/`](pilot/) | 7 N points × TP=2 + 2 smoke runs |
| [`h7-attn-negative/`](h7-attn-negative/) | One smoke run with kyuz0 flags |
| [`ccd1-negative/`](ccd1-negative/) | One smoke run with `taskset -c 8-15,24-31` |
