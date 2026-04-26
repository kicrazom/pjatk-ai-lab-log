# Methodology

Cross-cutting protocol notes for every study under [`results/`](../results/).
See [`hardware_context.md`](hardware_context.md) for the dGPU/iGPU mask
and CCD topology.

## Sweep design

`run_plan_*.sh` orchestrators sweep **N** (concurrency) on a logarithmic
spacing and pivot **TP** ∈ {1, 2} where both are valid. Plateau =
two consecutive N points within ±0.5%. **N spacing is per-model**:
Plan A used {50, 100, 200, 500, 1000, 2000, 3000} for Qwen 7B (~120k-token
KV absorbs all); Plan B used {10, 25, 50, 100, 200, 500, 1000} for
Qwen 72B AWQ (KV shrinks to ~32k, `max_model_len=4096`).

Per-run thermal sampling produces `<run-name>-{bench.log, wrapper.log,
thermals.jsonl, events.json, thermals.png}`. Together they regenerate
every chart from `scripts/plotting/`.

## Env-var floor (gfx1201 / RDNA 4)

```bash
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=1     # NOT 3 — current PyTorch rejects 3
export HIP_LAUNCH_BLOCKING=1      # Qwen 3.5 / 3.6 hybrid attention needs it
unset PYTORCH_ALLOC_CONF          # expandable_segments unsupported on ROCm
# enforce_eager=True passed via the LLM(...) constructor
```

`enforce_eager=True` is mandatory: vLLM 0.19.0+rocm721 hits
`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` during CUDA-graph capture
on gfx1201. Reproduced on Qwen 72B AWQ (H2, 2026-04-24) and again
on Qwen 3.6 27B FP8 TP=2 (2026-04-26). Recovery from re-enabling
graphs is expected at 20–30% once upstream patches the path.

**Protocol evolution.** Plan A used `SERIALIZE_KERNEL=3`; current
PyTorch rejects values > 1. Plan B's H3 (2026-04-24) recovered ~36%
throughput on Qwen 72B AWQ by dropping `HIP_LAUNCH_BLOCKING=1` — but
Qwen 3.5 / 3.6 hybrid-attention models need it back (verified
2026-04-26). The block above is the conservative floor; the H3 drop is
a per-model speedup, not a universal default.

**Canonical Qwen 3.6 27B config** (FP8 and BF16 both verified
2026-04-26): `tensor_parallel_size=2` + `enforce_eager=True` + the
env-var floor above. FP8: `max_model_len=2048`, `gpu_memory_utilization=0.85`.
BF16: `max_model_len=1024`, `gpu_memory_utilization=0.95`. FP8 + TP=1
OOMs (29 GiB runtime weights vs 32 GiB VRAM); both variants under
CUDA-graph capture crash with the gfx1201 HSA error.

## Negative results that rule out optimization paths

- **H7** — `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1` (from
  [`kyuz0/amd-r9700-vllm-toolboxes`](https://github.com/kyuz0/amd-r9700-vllm-toolboxes))
  is silently a no-op on stable vLLM 0.19.0+rocm721; implemented only
  in the TheRock-patched git build kyuz0 ships. Phase D.
- **CCD1 pinning** — `taskset -c 8-15,24-31` (96 MB 3D V-Cache CCD)
  produced **three identical 110.1 tok/s measurements** at N=100.
  GPU-bound at 100% util; CPU at ~14%. V-Cache can't accelerate a
  non-bottleneck. Documented as a calibration point.

## Prompt set and prefix caching

Plan A/B use 8 prefixes × 20 topics = 160 unique prompts, recycled when
N > 160. vLLM prefix caching is on by default — **amplifies** absolute
throughput vs unique prompts, but the **TP=1 : TP=2 ratio** is invariant.
A unique-prompt sweep is queued for the Polish-language phase.
