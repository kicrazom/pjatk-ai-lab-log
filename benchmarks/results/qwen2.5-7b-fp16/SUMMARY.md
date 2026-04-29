# Phase 2 Scaling Sweep — Qwen/Qwen2.5-7B-Instruct

**Configuration:** `FP16 TP=1/2 max_len=4096 util=0.70 KV=auto`  
**Backend:** `vllm-0.19.0+rocm721`  
**Tuning:** `stock`  
**N reruns per cell (n_runs):** 1 (v0.2.0 exploratory; Tier A n=10 deferred per METHODOLOGY §7.4)

## ⚠ Embargo classification

**EMBARGO — paper figures.** All concrete throughput, latency, and power numbers below are paper-bound until publication acceptance. Engineering observations (knee shape, vLLM scheduler robustness, preemption regime onset, TP=1 vs TP=2 trade-off) are PUBLIC. See [METHODOLOGY.md §11](../../../METHODOLOGY.md#11-embargo-policy) for the per-artifact split.

## Results table

Energy per token reported in mWh (= 0.001 Wh). Power columns are totals across all R9700 GPUs in use (1× for TP=1, 2× for TP=2).

| TP | N | tok/s out | total [s] | req/s | VRAM peak [GB] | T peak [°C] | W mean | W peak | mWh/tok |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 1057.0 | 6.0 | 8.30 | 22.1 | 42 | 119 | 334 | 0.031 |
| 1 | 100 | 1787.3 | 7.1 | 14.09 | 22.1 | 45 | 110 | 334 | 0.017 |
| 1 | 200 | 3327.3 | 7.7 | 26.06 | 22.3 | 48 | 124 | 371 | 0.010 |
| 1 | 500 | 3866.4 | 16.5 | 30.28 | 22.3 | 54 | 174 | 477 | 0.012 |
| 1 | 1000 | 3862.5 | 33.0 | 30.30 | 22.3 | 60 | 204 | 424 | 0.015 |
| 1 | 2000 | 3880.4 | 65.7 | 30.43 | 22.3 | 65 | 234 | 523 | 0.017 |
| 1 | 3000 | 3847.7 | 99.5 | 30.16 | 22.3 | 66 | 257 | 491 | 0.019 |
| 2 | 50 | 1153.9 | 5.5 | 9.02 | 21.9 | 50 | 140 | 430 | 0.034 |
| 2 | 100 | 1772.3 | 7.2 | 13.92 | 21.9 | 48 | 155 | 525 | 0.024 |
| 2 | 200 | 2627.9 | 9.7 | 20.60 | 22.0 | 48 | 185 | 1022 | 0.020 |
| 2 | 500 | 2937.5 | 21.7 | 23.05 | 22.0 | 51 | 217 | 691 | 0.021 |
| 2 | 1000 | 2945.7 | 43.3 | 23.10 | 22.0 | 56 | 294 | 1112 | 0.028 |
| 2 | 2000 | 2942.1 | 86.8 | 23.05 | 22.0 | 62 | 320 | 647 | 0.030 |
| 2 | 3000 | 2917.0 | 131.4 | 22.83 | 22.0 | 66 | 363 | 963 | 0.035 |

## Engineering observations (PUBLIC)

- **TP=1 throughput knee:** peak at N=2000 (3880.4 tok/s output). At N=3000 (1× over knee) throughput is -0.8% vs peak.
- **TP=2 throughput knee:** peak at N=1000 (2945.7 tok/s output). At N=3000 (3× over knee) throughput is -1.0% vs peak.
- **TP=1 vs TP=2 crossover:** common N values [50, 100, 200, 500, 1000, 2000, 3000] — see results table for per-N comparison.
- **Methodological humility (METHODOLOGY §8):** these numbers characterize inference *throughput* and *thermal envelope*; they do not measure model quality, reasoning, or clinical utility.
