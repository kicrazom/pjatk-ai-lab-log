# vLLM Concurrent Throughput — Scaling Curves

Benchmarking tensor-parallelism throughput of vLLM 0.19 on 2× AMD Radeon AI PRO
R9700 (RDNA 4 / gfx1201). Measures output throughput (tokens/s) against
increasing concurrency (number of simultaneous prompts).

## Results

![Scaling curve](scaling_curve.png)

### Headline findings (2026-04-17 baseline)

**Configuration:** Qwen 2.5 7B Instruct, FP16, max_tokens=128, enforce_eager=True

1. **TP=2 is slower than TP=1 at every tested concurrency level (N=100 to N=500).**
   Difference ranges from 1% (N=100) to 23% (N=500). On PCIe 5.0 without NVLink,
   `all_reduce` overhead exceeds the benefit of halved per-GPU weight reads for
   a model that already fits comfortably in one 32 GiB R9700.

2. **Both configurations show a reproducible ~20% throughput dip at N=300.**
   TP=1: 3351 → 2675 → 3853 tok/s across N=200/300/500.
   TP=2: 2615 → 2327 → 2969 tok/s across same points.
   Likely a chunked-prefill scheduler artifact at that batch size. Worth
   replicating with a thermals-instrumented run to rule out thermal transient.

3. **TP=1 has not yet hit its KV cache ceiling.** At N=500 with ~141 tok/req,
   peak KV demand is ~70k tokens, well under the 120k budget. TP=2's 487k-token
   KV cache (4× larger) remains unexploited at these concurrency levels.
   TP=2 would only begin to win for N ≫ 850 (TP=1 cache exhaustion) or with
   significantly longer per-request output (max_tokens ≥ 1024).

4. **Per-request throughput stays above human reading speed (~15 tok/s) for
   TP=1 up to N≈100.** Above that, individual requests slow noticeably; past
   N=200 per-request rate drops below 17 tok/s. For user-facing interactive
   serving on this single R9700, sweet spot is N ≈ 50-100.

### When would TP=2 actually help?

Based on this dataset, TP=2 is a pessimization for this workload. It should
win in three situations not yet tested:

- **Concurrency beyond TP=1 KV budget** (N > 800 with 4k context). At that
  point TP=1 must preempt requests; TP=2's 4× cache keeps them resident.
- **Long-form decode** (max_tokens ≥ 1024). Longer decode amortizes `all_reduce`
  overhead across more token steps.
- **Models that don't fit in 32 GiB** (e.g., 32B+ in FP16, or 70B AWQ). TP=2
  becomes mandatory, not optional.

## Files

| File | Purpose |
|---|---|
| `scaling_data.json` | Raw measurements + hardware/software context |
| `plot_scaling.py` | Regenerate `scaling_curve.png` from JSON |
| `sweep_concurrent.py` | Run full sweep and update JSON |
| `scaling_curve.png` | Published chart |
| `logs/` | Raw vLLM stdout from each benchmark run |
| `scripts/` | Benchmark source scripts |
| `thermal-runs/` | (Pending) Time-series thermal/utilization data |

## Reproducing

Prerequisites: vLLM 0.19+rocm721 in `~/venvs/vllm/`, Qwen 2.5 7B cached.

```bash
source ~/venvs/vllm/bin/activate

# Full sweep (overwrites JSON entries for specified N values)
python sweep_concurrent.py

# Custom sweep
python sweep_concurrent.py 50,100,200,300,500 1,2

# Regenerate chart after sweep
python plot_scaling.py scaling_data.json scaling_curve.png
```

## Configuration notes

- `enforce_eager=True` is **required** on gfx1201 with vLLM 0.19+rocm721 as a
  workaround for `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` during CUDA-graph
  capture. See parent repo for diagnosis. Non-eager mode may work after
  upstream fixes (file bug if still broken by next vLLM release).
- Benchmark prompts share a template prefix, so **prefix caching is active**
  and amplifies throughput for both configs. Real-world throughput with unique
  prompts would be proportionally lower but the *ratio* TP=1:TP=2 should hold.

## Pending experiments

- [ ] **Plan A high-concurrency sweep** (N ∈ {50, 100, 200, 500, 1000, 2000, 5000}
      × TP ∈ {1, 2}) — reveals TP=1 KV exhaustion point and TP=2 break-even
- [ ] **Long-form decode** (max_tokens=1024) — test TP=2 scaling in decode-bound regime
- [ ] **Thermal instrumentation** — timeline of CPU/GPU0/GPU1/iGPU temps + utilization
- [ ] **Larger model** (Qwen 2.5 32B AWQ) — TP=2 as necessity not choice
- [ ] **NUMA pinning A/B** on 9950X3D (2 CCDs)
