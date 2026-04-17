# vLLM Concurrent Throughput - Scaling Curves

Benchmarking tensor-parallelism throughput of vLLM 0.19 on 2x AMD Radeon AI PRO
R9700 (RDNA 4 / gfx1201). Measures output throughput (tokens/s) against
increasing concurrency (number of simultaneous prompts), with system thermal
instrumentation.

## Results

![Scaling curve](scaling_curve.png)

### Headline findings (Plan A sweep, 2026-04-17)

**Configuration:** Qwen 2.5 7B Instruct, FP16, max_tokens=128, enforce_eager=True,
gpu_memory_utilization=0.70. Full sweep: N in {50, 100, 200, 500, 1000, 2000, 3000}
x TP in {1, 2} = 14 runs with thermal instrumentation.

**Two distinct performance regimes, both remarkably flat:**

- **TP=1 plateau: 3870 +/- 12 tok/s** across N=500-3000 (std dev 0.3%). Single
  R9700 saturates around N=500 concurrent requests. Higher N only increases
  total time proportionally; throughput ceiling is fixed.
- **TP=2 plateau: 2940 +/- 16 tok/s** across the same range (std dev 0.5%).
  24% lower than TP=1, and this gap is the PCIe 5.0 all_reduce tax.

**The only point where TP=2 beats TP=1: N=50** (1154 vs 1057 tok/s, +9%).
At very low concurrency, TP=2's parallel prefill gives a small edge before
all_reduce overhead starts dominating.

### Full measurement table

| N | TP=1 tok/s | TP=2 tok/s | TP=2 / TP=1 |
|---|---|---|---|
| 50 | 1057 | **1154** | 1.09x (TP=2 wins) |
| 100 | 1787 | 1772 | 0.99x |
| 200 | 3327 | 2628 | 0.79x |
| 500 | 3866 | 2938 | 0.76x |
| 1000 | 3863 | 2946 | 0.76x |
| 2000 | 3880 | 2942 | 0.76x |
| 3000 | 3848 | 2917 | 0.76x |

### Why TP=2 underperforms for this workload

The 7B model fits comfortably on a single 32 GB R9700 with 6.5 GiB headroom
for KV cache (120k tokens available, never pressured in this sweep). TP=2
splits each layer's weights 50/50 across the two GPUs, requiring an
`all_reduce` synchronization after every attention and MLP block. On PCIe 5.0
x16 without NVLink/xGMI, this synchronization cost exceeds the benefit of
halved per-GPU weight reads.

**When TP=2 would win** (not exercised in this sweep):

- Concurrency exceeding TP=1's KV cache (requires either unique prompts with
  no prefix caching, or max_tokens much greater than 128). With prefix caching
  active and only 8 template prefixes, even N=3000 never pressures the 120k
  KV budget.
- Models that don't fit on a single GPU (e.g., 32B+ dense or 70B quantized).
  TP=2 becomes required, not optional.
- Long-form decode (max_tokens >= 1024) where per-request compute amortizes
  all_reduce cost across more token steps.

### Thermal observations

From `tp2-n3000-thermals.png` (longest single run, 131s of active compute):

- **GPU temperature asymmetry:** GPU 0 ran consistently 5C hotter than GPU 1
  (peak 66C vs 61C). Same workload (tensor parallel splits evenly), so the
  delta is airflow or case position, not load.
- **Both GPUs well below throttle:** Peak temps 30C below the 95C typical
  throttle threshold. The 45s inter-run cooldown is sufficient.
- **CPU thermally dominant:** AMD 9950X3D Tctl reached 75C during compute,
  hotter than either GPU. Tokenization, HTTP scheduling, and NCCL bookkeeping
  run on CPU at ~16-20% utilization, generating significant heat.
- **iGPU visible in dashboard interference:** Periodic 80-99% iGPU spikes at
  ~15s intervals correspond to AI workstation dashboard (WebSocket pushes)
  and browser rendering. Adds small thermal load but does not impact benchmark
  throughput (iGPU is not in compute path).

## Files

| File | Purpose |
|---|---|
| `scaling_data.json` | Complete Plan A measurements (14 points) + hw/sw context |
| `scaling_curve.png` | 2-panel chart (throughput + latency trade-off) |
| `plot_scaling.py` | Regenerator from JSON |
| `sweep_concurrent.py` | Additional sweep runner with JSON merge |
| `bench_with_thermals.py` | Wrapper orchestrating sampler + benchmark + plot |
| `sample_system.py` | Background CPU/GPU/iGPU sampler to JSONL |
| `plot_thermals.py` | Timeline plotter for thermal JSONL |
| `run_plan_a.sh` | Orchestrator for full Plan A sweep |
| `logs/` | Raw vLLM stdout from earlier exploratory benchmarks |
| `scripts/` | Benchmark source scripts |
| `thermal-runs/` | Plan A per-run artifacts (JSONL + PNG + logs + events) |

## Reproducing

Prerequisites: vLLM 0.19+rocm721 in `~/venvs/vllm/`, Qwen 2.5 7B cached,
matplotlib + psutil installed.

```bash
source ~/venvs/vllm/bin/activate
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1

# Single run with thermal instrumentation
python bench_with_thermals.py 1 100 --name tp1-n100 --out-dir thermal-runs/

# Full Plan A sweep (~30 min)
./run_plan_a.sh

# Regenerate chart after new data
python plot_scaling.py scaling_data.json scaling_curve.png
```

## Configuration notes

- `enforce_eager=True` is **required** on gfx1201 with vLLM 0.19+rocm721 as
  a workaround for `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` during CUDA-graph
  capture. Non-eager mode may work after upstream fixes in future vLLM
  releases, worth retesting periodically.
- Prompts share a small set of template prefixes, so **prefix caching is
  active** and amplifies measured throughput. Throughput for truly unique
  prompts would be lower, but the **TP=1:TP=2 ratio should hold**.

## Next experiments

- [x] Plan A high-concurrency sweep (N=50-3000, both TP configs) - DONE
- [ ] **Qwen 2.5 72B AWQ** (~40 GB) - TP=2 becomes mandatory (doesn't fit on
      single 32 GiB R9700). Expected throughput 150-400 tok/s. Completes the
      narrative arc: 7B (TP=2 harmful) -> 72B (TP=2 required).
- [ ] **Polish-language models** - Bielik-11B-v2.3-Instruct (SpeakLeash),
      PLLuM-12B (CYFRAGOVPL). Cross-track value for Scientific Writing 3.0
      course case study.
- [ ] **Long-form decode** (max_tokens=1024) - TP=2 scaling in decode-bound
      regime, where all_reduce amortizes better.
- [ ] **Unique prompts without prefix caching** - force KV cache pressure,
      expose TP=1 preempting point and TP=2's 4x cache advantage.
- [ ] **NUMA pinning A/B** on 9950X3D (2 CCDs) - CCD-local GPU pinning may
      reduce CPU-side thermal spikes and improve TP=2 all_reduce latency.
