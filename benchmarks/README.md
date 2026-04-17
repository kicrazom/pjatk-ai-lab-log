# Benchmarks

Performance characterization of the workstation's ML compute stack.
Benchmarks fall into two categories.

## Low-level compute

Raw hardware throughput — matrix multiply, memory bandwidth, ROCm primitives.
These establish a theoretical ceiling and validate the GPU / ROCm / PyTorch
stack is correctly installed.

- [`run_single_r9700_benchmark.sh`](run_single_r9700_benchmark.sh) —
  GEMM benchmark (FP32/FP16/BF16 TFLOPS) on single R9700, using the
  `~/venvs/rocm72/` environment.

## Inference serving

Real workload throughput — vLLM serving LLM inference with multiple
concurrent requests. These measure what the workstation actually delivers
when used for local AI.

- [`results/`](results/) — **vLLM throughput scaling study** (April 2026)
  - Qwen 2.5 7B Instruct, FP16
  - N in {50, 100, 200, 500, 1000, 2000, 3000} x TP in {1, 2} = 14 runs
  - Per-run thermal and utilization timelines (CPU + 2x GPU + iGPU)
  - Key finding: TP=1 saturates at 3870 tok/s, TP=2 at 2940 tok/s
    (PCIe all_reduce overhead makes TP=2 a pessimization for this
    workload on a model that fits in a single 32 GB R9700)

![Scaling curve](results/scaling_curve.png)

See [`results/README.md`](results/README.md) for the full writeup.

## Planned

- Qwen 2.5 72B AWQ — where TP=2 becomes mandatory (model does not fit
  on single GPU). Validates the "other half" of the tensor parallel story.
- Bielik-11B-v2.3-Instruct, PLLuM-12B — Polish-language models, cross-track
  relevance for Scientific Writing 3.0 course material.
- Mixtral-8x7B-AWQ — mixture-of-experts architecture as a third data point
  alongside dense 7B and dense 72B.
- Long-form decode (max_tokens >= 1024) — regime where TP=2 may amortize
  its all_reduce cost.

## Reproducing

All inference benchmarks require the vLLM virtual environment:

```bash
source ~/venvs/vllm/bin/activate
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
```

The environment variables are a workaround for
`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` during CUDA-graph capture on
gfx1201 with vLLM 0.19+rocm721. See [`results/README.md`](results/README.md)
for details and suggested retest after upstream fixes.
