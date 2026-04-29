# Phase 2 Scaling Sweep — Qwen/Qwen3.6-27B

**Configuration:** `BF16 TP=2 max_len=2048 util=0.97 KV=default`  
**Backend:** `vllm-0.19.0+rocm721`  
**Tuning:** `stock`  
**N reruns per cell (n_runs):** 1 (v0.2.0 exploratory; Tier A n=10 deferred per METHODOLOGY §7.4)

## ⚠ Embargo classification

**EMBARGO — paper figures.** All concrete throughput, latency, and power numbers below are paper-bound until publication acceptance. Engineering observations (knee shape, vLLM scheduler robustness, preemption regime onset) are PUBLIC. See repo [METHODOLOGY.md §11](../../../METHODOLOGY.md#11-embargo-policy) for the per-artifact split.

## Results table

Energy per token reported in mWh (= 0.001 Wh). Power columns are totals across both R9700 GPUs.

| N | tok/s out | total [s] | req/s | VRAM peak [GB] | T peak [°C] | W mean | W peak | mWh/tok |
|---|---|---|---|---|---|---|---|---|
| 10 | 68.7 | 18.6 | 0.54 | 29.3 | 50 | 255 | 679 | 1.032 |
| 25 | 84.0 | 38.1 | 0.66 | 29.4 | 57 | 275 | 716 | 0.909 |
| 50 | 83.6 | 76.6 | 0.65 | 29.3 | 63 | 320 | 889 | 1.065 |
| 100 | 86.5 | 147.4 | 0.68 | 29.3 | 67 | 346 | 657 | 1.113 |
| 200 | 98.2 | 256.6 | 0.78 | 29.4 | 72 | 376 | 889 | 1.063 |
| 500 | 91.7 | 696.4 | 0.72 | 29.3 | 72 | 404 | 825 | 1.224 |
| 1000 | 97.3 | 1314.4 | 0.76 | 29.4 | 72 | 413 | 685 | 1.179 |

## Engineering observations (PUBLIC)

- **Throughput knee:** peak at N=200 (98.2 tok/s output). Below knee: linear scaling; above knee: graceful degradation under KV-cache preemption.
- **Range:** N=10 → 68.7 tok/s; N=1000 → 97.3 tok/s.
- **vLLM scheduler robustness:** at N=1000 (=5× over knee), throughput is -0.9% vs peak — no starvation pathology.
- **Methodological humility (METHODOLOGY §8):** these numbers characterize inference *throughput* and *thermal envelope*; they do not measure model quality, reasoning, or clinical utility.
