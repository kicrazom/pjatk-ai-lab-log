# Benchmarks

![The battle of the models in AMD kingdom on vLLM-powered horses, commencing at dawn...](assets/battle_of_LLM_models_gemini.png)

Performance characterization of vLLM 0.19 + ROCm 7.2.1 on 2× R9700
(gfx1201 / Navi 48 / RDNA 4), plus a low-level GEMM ceiling. The driving
question is **when TP=2 helps, hurts, or is mandatory** as model size and
quantization vary. PCIe 5.0 x16 (no xGMI) makes the per-layer `all_reduce`
the cost; halved per-GPU weight reads are the benefit.

## The TP=2 arc — harmful, envelope-defining, mandatory

| Regime | Representative model | TP=2 verdict |
|---|---|---|
| Harmful | Qwen 2.5 7B FP16 (15 GB) | TP=2 plateau is 24% *below* TP=1 — sync cost > parallelism gain; single-GPU optimal |
| Envelope-defining | **Qwen 3.6 27B** FP8 / BF16 | FP8 OOMs on TP=1 (~29 GiB runtime weights leave no KV headroom); both variants need TP=2 |
| Mandatory | Qwen 2.5 72B AWQ (39 GB) | Even quantized doesn't fit one 32 GB card |

Qwen 3.6 27B (released 2026-04-22, dense, Apache 2.0) defines the
practical envelope at this parameter count. Both variants run on
TP=2 with `enforce_eager=True` (smoke 2026-04-26): **FP8** at
`max_model_len=2048`, `gpu_memory_utilization=0.85` — 15.35 GiB weights
+ 7.53 GiB KV per card; **BF16** at `max_model_len=1024`, `util=0.95` —
25.76 GiB + 2.68 GiB KV. FP8 + TP=1 OOMs (29 GiB runtime weights leave
no headroom). CUDA-graph capture on either TP setting hits the same
`HSA_STATUS_ERROR` familiar from the 72B AWQ work. The "phase
transition" is the boundary between *TP=1 optimal* (≤7B FP16) and
*TP=2 required* (≥27B at any quantization), not a TP=1 / TP=2 choice
within one model.

**Unexpected on R9700: BF16 outpaces FP8 by ~75%** in cold first
inference (7.23 vs 4.15 tok/s) — vLLM lacks FP8 kernel configs for
gfx1201 and falls back to a generic block-FP8 path, inverting the
FP8 > BF16 ordering typical on H100/MI300X. The 7.23 tok/s BF16
cold figure suggests substantial warm-state headroom: equivalent
CUDA setups (Q4_K_M on RTX 4090, llama.cpp) reach ~46 tok/s
token-gen, an order of magnitude above our cold-start measurement.
Detailed throughput sweep is next.

## The model set (13 models, ~770 GB)

| # | Directory | Repo ID | Size | Variant |
|--:|---|---|---:|---|
|  1 | `bielik-11b-v23` | `speakleash/Bielik-11B-v2.3-Instruct` | 21 GB | FP16 |
|  2 | `bielik-11b-v23-awq` | `speakleash/Bielik-11B-v2.3-Instruct-AWQ` | 5.8 GB | AWQ Int4 |
|  3 | `llama-pllum-8b-instruct` | `CYFRAGOVPL/Llama-PLLuM-8B-instruct` | 15 GB | BF16 |
|  4 | `pllum-12b-chat` | `CYFRAGOVPL/PLLuM-12B-chat` | 23 GB | BF16 |
|  5 | `mistral-nemo-instruct-2407` | `mistralai/Mistral-Nemo-Instruct-2407` | 46 GB | BF16 |
|  6 | `qwen25-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | 15 GB | FP16 |
|  7 | `qwen25-72b-awq` | `Qwen/Qwen2.5-72B-Instruct-AWQ` | 39 GB | AWQ Int4 |
|  8 | `qwen36-27b-fp8` | `Qwen/Qwen3.6-27B-FP8` | 29 GB | FP8 |
|  9 | `qwen36-27b` | `Qwen/Qwen3.6-27B` | 56 GB | BF16 |
| 10 | `mixtral-8x7b-awq` | `TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ` | 23 GB | AWQ Int4 (MoE) |
| 11 | `llama-pllum-70b-base` | `CYFRAGOVPL/Llama-PLLuM-70B-base` | 132 GB | BF16 (base) |
| 12 | `llama-pllum-70b-instruct` | `CYFRAGOVPL/Llama-PLLuM-70B-instruct` | 132 GB | BF16 (SFT) |
| 13 | `llama-pllum-70b-chat` | `CYFRAGOVPL/Llama-PLLuM-70B-chat` | 132 GB | BF16 (SFT+RLHF) |

Axes: 12 dense + 1 MoE; 5× FP16/BF16 instruct, 4× AWQ Int4, 1× FP8, 3×
pretrain BF16; Polish-native (Bielik ×2, PLLuM ×5) + multilingual general
purpose (Qwen 2.5 ×2, Qwen 3.6 ×2, Mistral-Nemo); the three Llama-PLLuM
70B variants isolate SFT + RLHF cost on identical base weights.

## Layout

```
assets/         Cover image and static media
methodology/    Sweep design, gfx1201 env vars, hardware/iGPU separation
scripts/
  low_level/    GEMM benchmark (single R9700 ceiling)
  plotting/     Scaling and thermal chart regenerators
  instrumentation/  Sampler + thermal-instrumented benchmark wrappers
  runners/      vLLM benchmark entry points (one per model family)
  orchestrators/  Plan A / Plan B / hypothesis-test sweep drivers
results/<model-config>/  One subdir per study, each with its own README
```

## Status

- [x] Plan A — Qwen 2.5 7B FP16, TP=1 vs TP=2, 14 runs (2026-04-17).
      [`results/qwen2.5-7b-fp16/`](results/qwen2.5-7b-fp16/README.md)
- [x] Plan B pilot — Qwen 2.5 72B AWQ TP=2, plateau ~121 tok/s, plus
      H7 / CCD1 negative-result follow-ups (2026-04-24).
      [`results/qwen2.5-72b-awq/`](results/qwen2.5-72b-awq/README.md)
- [ ] Phase-transition sweep — Qwen 3.6 27B FP8 + BF16, both TP=2 (FP8
      smoke test passed 2026-04-26; full N sweep pending).
- [ ] Polish-language sweep (Bielik ×2, PLLuM ×5).
- [ ] Mixtral-8x7B-AWQ (MoE data point), Mistral-Nemo (multilingual ref).

For sweep design, env-var rationale, and the dGPU/iGPU separation that
keeps the integrated `gfx1036` out of compute, see
[`methodology/`](methodology/README.md).
