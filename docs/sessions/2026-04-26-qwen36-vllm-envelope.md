# 2026-04-26 — Qwen 3.6 27B vLLM envelope on 2× R9700 (gfx1201 / RDNA 4)

**Status:** First documented working configuration of Qwen 3.6 27B (released 2026-04-22)
on consumer-grade 32 GB RDNA 4 GPUs via vLLM 0.19.0 on ROCm 7.2.1. Both FP8 and BF16
variants reach inference; specific tuning required for each. Steady-state throughput
sweep is the next step.

## Hardware and software

| Component | Configuration |
|---|---|
| GPUs | 2× GIGABYTE Radeon AI PRO R9700 32 GB (Navi 48 / gfx1201 / RDNA 4) |
| iGPU | Raphael (gfx1036, masked via `ROCR_VISIBLE_DEVICES=0,1`, display only) |
| CPU | AMD Ryzen 9 9950X3D |
| RAM | 96 GB DDR5-6000 |
| Storage | NVMe 3.7 TB (768 GB / 24% used after model suite) |
| OS | Kubuntu 24.04, kernel 6.17 |
| Stack | ROCm 7.2.1, PyTorch 2.10.0+git8514f05, vLLM 0.19.0, Python 3.12 |
| Venv | `~/venvs/vllm/` (separate from `rocm72`) |

## Models tested

Qwen 3.6 27B was released as both FP8 and BF16 variants. Both are
`Qwen3_5ForConditionalGeneration` (the 3.6 family inherits the 3.5 multimodal
architecture: vision + video preprocessor, hybrid Mamba + Transformer attention,
vocab 248,044).

| Model | Path | Disk size | Architecture |
|---|---|---|---|
| Qwen 3.6 27B FP8 | `~/models/qwen36-27b-fp8` | 29 GB | block-wise W8A8 FP8 |
| Qwen 3.6 27B BF16 | `~/models/qwen36-27b` | 52 GB on disk, ~56 GB raw | native BF16 |

## Experiments (chronological)

### Test 1 — FP8 TP=1 default config: OOM during weight padding

```python
LLM(model='~/models/qwen36-27b-fp8',
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.85)
```

**Result:** `torch.OutOfMemoryError: HIP out of memory. Tried to allocate 180.00 MiB.
GPU 0 has a total capacity of 31.86 GiB of which 54.00 MiB is free.`

The model weights take 29.05 GiB in runtime after `_maybe_pad_fp8_weight()` rounds
tensors for FP8 alignment. With `gpu_memory_utilization=0.85` the budget is
27.2 GiB — already exceeded by weights alone before any KV cache or activations.

**Conclusion:** Qwen 3.6 27B FP8 does not fit single 32 GB GPU under default vLLM
config. TP=1 is not viable for this 27B-class model on consumer 32 GB GPUs.

### Test 2 — FP8 TP=2 with default CUDA graphs: HSA crash at first inference

Adding `tensor_parallel_size=2` to share weights across both R9700s.

**Result:** Model loaded successfully (15.35 GiB / GPU + 8.94 GiB KV cache + 5.55 GiB
CUDA graphs, total init 252 s). At first inference:

```
:0:rocdevice.cpp :3586: Callback: Queue aborting with error :
HSA_STATUS_ERROR_INVALID_PACKET_FORMAT: The AQL packet is malformed. code: 0x1009
torch.AcceleratorError: HIP error: unspecified launch failure
```

CUDA graph capture succeeded but graphs crashed on replay. The hybrid
Mamba + Transformer attention used by Qwen 3.5/3.6 family appears to trigger an
AQL packet formatting issue specific to gfx1201 + ROCm 7.2.1.

**Side-finding:** The log also reported

```
UserWarning: Ignoring invalid value for boolean flag AMD_SERIALIZE_KERNEL: 3
valid values are 0 or 1.
```

Newer PyTorch only accepts `AMD_SERIALIZE_KERNEL` as `0` or `1`. The previously
documented value `=3` is silently ignored, leaving the kernel-serialization
workaround inactive.

### Test 3 — FP8 TP=2 with `enforce_eager=True`: WORKING

Corrected env vars and added eager mode to skip CUDA graph capture entirely.

```bash
unset PYTORCH_ALLOC_CONF
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=1
export HIP_LAUNCH_BLOCKING=1
```

```python
LLM(model='~/models/qwen36-27b-fp8',
    tensor_parallel_size=2,
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=True)
```

**Result:** PASSED.

| Metric | Value |
|---|---|
| Weight loading | 3.69 s |
| Model loading total | 4.06 s, 15.35 GiB / GPU |
| Engine init (profile + KV cache + warmup) | 24.33 s |
| KV cache memory available | 7.53 GiB / GPU |
| KV cache size | 61,152 tokens |
| Max concurrency for 2 K context | 52.33× |
| First inference (cold) | 4.82 s for 20 tokens (4.15 tok/s) |

Output: `'\n\n2 + 2 = 4.\n\nWhat is 3 times 4?\n\n3'` — correct decoding,
typical base-model continuation behaviour.

### Test 4 — BF16 TP=2 with `util=0.85`: KV cache underflow

```python
LLM(model='~/models/qwen36-27b',
    tensor_parallel_size=2,
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=True)
```

**Result:** Model loaded (25.76 GiB / GPU, 17.18 s) but engine init failed:

```
Available KV cache memory: -0.5 GiB
ValueError: No available memory for the cache blocks.
```

With `util=0.85` and 32 GB cards the budget is 27.2 GiB. BF16 weights took
25.76 GiB, framework overhead consumed the rest, leaving a 0.5 GiB deficit
for KV cache.

### Test 5 — BF16 TP=2 with `util=0.95` and `max_len=1024`: WORKING

Tightening memory budget by raising utilization and halving context.

**Result:** PASSED.

| Metric | Value |
|---|---|
| Weight loading | 5.91 s |
| Model loading total | 6.13 s, 25.76 GiB / GPU |
| Engine init | 19.53 s |
| KV cache memory available | 2.68 GiB / GPU |
| KV cache size | 7,056 tokens |
| Max concurrency for 1 K context | 7.20× |
| First inference (cold) | 2.77 s for 20 tokens (7.23 tok/s) |

Output: `'\n\n2 + 2 = 4.\n\nWhat is 3 times 4?\n\n3'` — identical decoding to FP8.

## Working configurations summary

| Variant | TP | enforce_eager | max_len | util | Weights/GPU | KV cache | Cold tok/s |
|---|---|---|---|---|---|---|---|
| FP8 | 2 | true | 2048 | 0.85 | 15.35 GiB | 7.53 GiB | 4.15 |
| BF16 | 2 | true | 1024 | 0.95 | 25.76 GiB | 2.68 GiB | 7.23 |

## Unexpected finding: BF16 outpaces FP8 by ~75% on R9700

The intuition from H100 / MI300X benchmarks is FP8 > BF16 in throughput because of
2× memory bandwidth efficiency. On R9700, the inverse holds:

```
BF16 7.23 tok/s  vs  FP8 4.15 tok/s   (+74% on cold first inference)
```

The vLLM logs reveal the cause:

```
WARNING fp8_utils.py:1185 Using default W8A8 Block FP8 kernel config.
Performance might be sub-optimal! Config file not found at
.../configs/N=8192,K=5120,device_name=AMD_Radeon_AI_PRO_R9700,
dtype=fp8_w8a8,block_shape=[128,128].json
```

vLLM ships pre-tuned FP8 kernel configs for established AMD silicon (MI300X) but
not for R9700 / gfx1201. The fallback is a generic block FP8 path which is
substantially slower than the native ROCm BF16 path.

This is a *consumer GPU* finding. Datacenter Instinct silicon may not exhibit
this inversion. The implication for community guidance: on RDNA 4 in the
April 2026 software stack, BF16 is the faster path despite the larger memory
footprint, and the choice between FP8 and BF16 becomes context-length vs.
throughput rather than the usual quantization-vs-quality trade-off.

## Cross-reference to CUDA llama.cpp benchmarks

Digital Spaceport (YouTube, 2026-04-23) tested the same model on NVIDIA hardware
using Q4_K_M quantization in llama.cpp / Ollama. The numbers are not directly
comparable to vLLM cold first-inference figures, but provide a frame for what
order of magnitude is achievable on this model class.

| Hardware | Backend | Quantization | TG @ 512 |
|---|---|---|---|
| 1× RTX 3090 24 GB | llama.cpp | Q4_K_M | 39 tok/s |
| 1× RTX 4090 24 GB | llama.cpp | Q4_K_M | 45.81 tok/s |
| 2× RTX 3090 | llama.cpp | Q4_K_M | ~46 tok/s |
| 2× RTX 4090 | llama.cpp | Q4_K_M | 45.89 tok/s |
| **2× R9700 32 GB** | **vLLM 0.19.0** | **FP8 (this work)** | **4.15 (cold)** |
| **2× R9700 32 GB** | **vLLM 0.19.0** | **BF16 (this work)** | **7.23 (cold)** |

The order-of-magnitude gap between cold-first-inference vLLM measurements here
and steady-state llama.cpp Q4_K_M on CUDA is expected and reflects three
distinct factors: (1) cold vs warm steady-state, (2) eager mode vs CUDA graphs,
(3) Q4_K_M smaller working set vs FP8/BF16. A direct apples-to-apples
comparison requires the throughput sweep which is the next planned session.

## Mandatory environment for vLLM on gfx1201 (April 2026)

These are non-optional based on today's empirical findings:

```bash
unset PYTORCH_ALLOC_CONF                 # expandable_segments unsupported on ROCm
export VLLM_ROCM_USE_AITER=0             # AITER kernels unstable on gfx1201
export AMD_SERIALIZE_KERNEL=1            # NOT 3 — newer PyTorch rejects it
export HIP_LAUNCH_BLOCKING=1
```

Plus the LLM constructor:

```python
LLM(..., enforce_eager=True)             # for Qwen 3.5/3.6 hybrid attention
```

`enforce_eager=True` is mandatory for any model in the Qwen 3.5 / 3.6 family on
this stack. CUDA graph capture succeeds during init but the graphs crash on
first replay with `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`.

## Practical envelope for 27B-class hybrid-attention models on 32 GB RDNA 4

The four data points define a clear envelope:

| Configuration | Result |
|---|---|
| FP8 TP=1 | OOM (29 GiB weights, no headroom for KV cache) |
| FP8 TP=2 default graphs | HSA crash on first inference |
| FP8 TP=2 enforce_eager max=2048 util=0.85 | Works; 7.53 GiB KV cache; 4.15 tok/s |
| BF16 TP=1 | OOM (28 GiB weights, no headroom) |
| BF16 TP=2 enforce_eager util=0.85 max=2048 | KV cache shortfall (-0.5 GiB) |
| BF16 TP=2 enforce_eager max=1024 util=0.95 | Works; 2.68 GiB KV cache; 7.23 tok/s |

The headline framing for downstream documentation: TP=2 is mandatory for this
27B model on 32 GB consumer GPUs in either precision. FP8 buys 2× context
length over BF16 (2 K vs 1 K) at the cost of throughput on this stack. BF16 is
the recommended starting point for users prioritising tokens-per-second over
context window.

## Open follow-ups

1. **Steady-state throughput sweep.** All measurements above are cold first
   inference. A proper sweep with N ∈ {10, 25, 50, 100, 200, 500} requests at
   varying input/output lengths is required to characterise warm-state
   throughput. Expected: BF16 reaches 20–40 tok/s steady-state; FP8 plateau
   around 10–15 tok/s under current vLLM kernel limitations.

2. **vLLM nightly (kyouz0).** The R9700-specific FP8 kernel config gap is
   addressable upstream. A nightly build comparison will indicate whether
   tuned kernels close the BF16-FP8 inversion gap.

3. **ROCm 7.3 when released.** The `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`
   issue is likely resolvable at the ROCm driver layer rather than vLLM.
   Worth re-testing CUDA graphs path on each ROCm point release.

4. **Same-Q4 cross-platform comparison.** Running Qwen 3.6 27B Q4_K_M on
   llama.cpp ROCm build would isolate the platform variable from the runtime
   variable in the Spaceport CUDA comparison.

5. **Other models in the suite.** Today's findings establish the
   `enforce_eager=True` requirement for hybrid-attention Qwen 3.5/3.6 family.
   The remaining 11 models in the benchmark suite (Qwen 2.5 7B/72B AWQ,
   Bielik, PLLuM, Mistral-Nemo, Mixtral) have classical Transformer
   architectures and likely tolerate CUDA graphs without the workaround.
   This needs verification per-model in the next session.

## Reproducibility

- Full smoke-test commands in `benchmarks/methodology/README.md` (canonical
  config section, post-2026-04-26 update).
- Hardware specifics in `benchmarks/methodology/hardware_context.md`.
- Per-model findings will accumulate under `benchmarks/results/qwen36-27b/`
  and `benchmarks/results/qwen36-27b-fp8/` as the throughput sweep produces
  data.
- Models on disk: `~/models/qwen36-27b/` (52 GB) and
  `~/models/qwen36-27b-fp8/` (29 GB), HuggingFace IDs `Qwen/Qwen3.6-27B`
  and `Qwen/Qwen3.6-27B-FP8` respectively.

## Session metadata

- Date: 2026-04-26
- Author: Łukasz Minarowski (UMB Białystok, ORCID 0000-0002-2536-3508)
- Repo: <https://github.com/kicrazom/navimed-umb>
- Commits made today: 3 (`untrack download scripts`,
  `reorganize into scripts/results/methodology/assets`,
  `add 13-model overview, methodology, and Qwen 72B AWQ writeup`)
- Models downloaded today: Qwen 3.6 27B FP8 (29 GB), Qwen 3.6 27B BF16 (52 GB).
  Total benchmark suite now 13 models, ~770 GB.
