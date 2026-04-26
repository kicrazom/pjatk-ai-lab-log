# Session 2026-04-26 — gpt-oss-20b on dual R9700: routing API skew blocks MXFP4 path

**Status:** Negative result — model deferred from current sweep.

**Reason:** Version skew between `triton_kernels==1.0.0` and `vllm 0.19.0+rocm721`. The routing API (`routing_from_bitmatrix`) that vLLM's `gpt_oss_triton_kernels_moe.py` imports unconditionally is absent from the installed `triton_kernels` package.

**Decision:** Skip `openai/gpt-oss-20b` in the current 9-model benchmark sweep. Re-evaluate when either (a) `triton_kernels` major version increments with the routing API exposed, or (b) vLLM publishes a ROCm wheel that bundles a compatible `triton_kernels` build.

---

## 1. Context and motivation

`openai/gpt-oss-20b` was considered as a complementary data point for the NaviMed Hardware Envelope characterization (paper v0.1.0, released 2026-04-26). Motivation:

- **Architectural diversity** — dense-expert MoE with attention sinks and sliding window. Different from the MoE path already represented in the suite (Mixtral 8x7B AWQ, sparse routing) and orthogonal to the dense-attention models (Qwen 2.5, Bielik, PLLuM, Mistral-Nemo).
- **Native MXFP4 quantization** — extends the suite's quantization coverage beyond AWQ-int4.
- **Single-GPU fit on R9700** — ~12–13 GB MXFP4 footprint, enabling a clean TP=1 vs TP=2 contrast complementary to the Qwen 2.5 72B AWQ "TP=2 mandatory" arc.
- **External recognition** — OpenAI gpt-oss is the most-recognized "frontier American open-weights" reference for international reviewers, contrasting cleanly with Polish-tuned models (Bielik, PLLuM) on identical hardware.

## 2. Sources reviewed prior to attempt

The following references were consulted before allocating sweep time:

| Source | Stack | Reported result |
|---|---|---|
| Reddit r/ROCm — Cyp9715 (AMD GPU loaner program) | vLLM 0.11.0 + ROCm 6.4.2, 2× R9700, Colfax bare-metal | Working: 1925 tok/s output throughput, 200 concurrent, ShareGPT V3 200 prompts |
| Reddit r/ROCm — `no_no_no_oh_yes` (commenter, identical hardware) | `rocm/vllm-dev:nightly` (vLLM 0.11.1rc2.dev161), 2× R9700 | Failed: `triton_kernels.tensor` module missing entirely; `_swizzle_mxfp4` import blocked |
| vLLM official recipes (GPT-OSS) | Listed as supported on Radeon AI PRO R9700 | Theoretical support claim |
| Hostkey benchmark | Ollama / GGUF + ROCm 7.1.1, single R9700 | Working but different inference engine (llama.cpp backend, not vLLM) |
| Phoronix R9700 dual-GPU review | ROCm 7.0.2 + assorted AI frameworks | Hardware / driver baseline confirmed for general AI workloads |
| vLLM blog — GPT-OSS optimizations on Blackwell | NVIDIA H100/B200 + vLLM | Reference for upstream tuning intent (stream-interval, attention sinks kernels) |

**Reading of the source landscape:** the Reddit OP's Colfax loaner was almost certainly running a custom AMD-supplied stack with `triton_kernels` built from source in lockstep with the vLLM commit being tested. This configuration is not reproducible from commodity ROCm wheel installs. The commenter's failure on `rocm/vllm-dev:nightly` is itself supporting evidence — same hardware class, more recent vLLM, no working configuration.

## 3. Hardware and stack at attempt

```
GPUs:           2× GIGABYTE Radeon AI PRO R9700 (RDNA 4 / gfx1201 / Navi 48), 32 GB GDDR6 each
CPU:            AMD Ryzen 9 9950X3D
RAM:            96 GB DDR5-6000
OS:             Kubuntu 24.04, kernel 6.17
ROCm:           7.2
PyTorch:        2.10.0+git8514f05 (ROCm 7.2.1)
vLLM:           0.19.0+rocm721 (gfx1201 native build)
Triton:         3.6.0
triton_kernels: 1.0.0
venv:           ~/venvs/vllm/
```

## 4. Diagnostic chain

### 4.1 Initial smoke test — MoE routing API import

```bash
source ~/venvs/vllm/bin/activate

python -c "
from triton_kernels.routing import routing_from_bitmatrix
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
print('MXFP4 dependencies OK')
"
```

Output:

```
Traceback (most recent call last):
  File "<string>", line 2, in <module>
ModuleNotFoundError: No module named 'triton_kernels.routing'
```

The same error class is reported in the vLLM startup path (`gpt_oss_triton_kernels_moe.py:27`) when attempting `vllm serve openai/gpt-oss-20b`.

### 4.2 Package introspection

```bash
python -c "
import triton_kernels
print(triton_kernels.__file__)
print(dir(triton_kernels))
"
```

Output:

```
/home/mozarcik/venvs/vllm/lib/python3.12/site-packages/triton_kernels/__init__.py
['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__']
```

Empty `__init__.py` (0 bytes), no namespace re-exports. Submodules are still importable directly — this is an intentional design choice in the package, not a broken install.

### 4.3 Submodule inventory

```bash
ls -la ~/venvs/vllm/lib/python3.12/site-packages/triton_kernels/
```

Relevant entries:

| File / dir | Size | Status |
|---|---|---|
| `__init__.py` | 0 B | Empty (intentional, no namespace pollution) |
| `tensor.py` | 9 382 B | **Present** — provides `FP4`, `convert_layout`, `wrap_torch_tensor` |
| `matmul_ogs.py` | 40 135 B | Present — MoE matmul core |
| `topk.py` | 7 542 B | Present — top-k gating |
| `swiglu.py` | 3 193 B | Present |
| `distributed.py` | 16 792 B | Present |
| `reduce.py` | 13 465 B | Present |
| `compaction.py` | 2 538 B | Present |
| `numerics.py` | 1 145 B | Present |
| `specialize.py` | 7 251 B | Present |
| `roofline.py` | 11 743 B | Present |
| `routing.py` | — | **Absent** |

### 4.4 Targeted verification

```bash
# (a) MXFP4 layout path — does the dequant infrastructure work?
python -c "from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor; print('MXFP4 layout OK')"
# → MXFP4 layout OK

# (b) Search for routing_from_bitmatrix anywhere in the package tree
grep -rn "routing_from_bitmatrix" ~/venvs/vllm/lib/python3.12/site-packages/triton_kernels/
# → (no matches; empty output)

# (c) Package version metadata
python -c "import importlib.metadata as md; print(md.version('triton_kernels'))"
# → 1.0.0
```

## 5. Root cause

`triton_kernels==1.0.0` does **not** export the routing API that vLLM's gpt-oss MoE integration imports unconditionally:

```python
# vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py (line 27)
from triton_kernels.routing import routing_from_bitmatrix
```

`routing_from_bitmatrix` was introduced in upstream `openai/triton` (`python/triton_kernels/`) after the cut for `triton_kernels==1.0.0`. The ROCm wheel for `vllm 0.19.0+rocm721` resolves `triton_kernels` from its pinned dependency at version 1.0.0, but the embedded vLLM source already calls into a newer API.

This is a **version skew**, not an absent feature class:

- **MXFP4 layout / dequant path is fully functional** in the installed `triton_kernels==1.0.0` (verified by direct import in §4.4(a)).
- **MoE routing path that gpt-oss specifically requires** has not yet propagated to a published `triton_kernels` release that the ROCm wheel pulls.

## 6. Three observed configurations on 2× R9700

| Configuration | vLLM | `triton_kernels` state | Result |
|---|---|---|---|
| Colfax loaner (AMD-supplied, Cyp9715) | 0.11.0 | Custom build, lockstep with vLLM commit | gpt-oss-20b runs, 1925 tok/s output |
| `rocm/vllm-dev:nightly` (no_no_no_oh_yes) | 0.11.1rc2.dev | `tensor.py` itself missing | Fails on layout import |
| This setup (commodity ROCm wheel) | 0.19.0+rocm721 | `tensor.py` present, `routing.py` absent | Fails on routing import |

The gradient is itself the observation. "Supported on R9700" in vLLM's official recipes presupposes a stack alignment that none of the publicly distributed wheels currently provide.

## 7. Decision

**Skip `gpt-oss-20b` in the current sweep.** Three workarounds were considered and rejected:

1. **Monkey-patch the routing import.** Would require synthesizing `routing_from_bitmatrix` from `topk.py` primitives or stubbing it. Brittle; breaks reproducibility (any reader trying to replicate would need the same patch); correctness of model outputs would require independent validation. Rejected.
2. **Downgrade vLLM** to a release predating the routing API import (e.g. matching the OP's 0.11.0). Loses gfx1201 native support, which is the entire reason this stack was chosen. Non-starter.
3. **Build `triton_kernels` from upstream source.** Estimated 4–8 h of work; ABI compatibility with vLLM 0.19.0 build flags unclear; RDNA 4 / gfx1201 is a second-class target in upstream Triton testing (CUDA and MI300x are primary). Cost is disproportionate to the marginal value of one additional model in the suite. Rejected for now.

Re-evaluate when either:

- `triton_kernels >= X.Y` (whichever release exposes `routing_from_bitmatrix`) propagates as a ROCm wheel dependency, or
- vLLM ships a ROCm wheel with an embedded compatible `triton_kernels` (analogous to what AMD does internally for loaner-hardware demonstrations).

## 8. Implications for the Hardware Envelope characterization

This is a positive finding for the paper, not merely a deferred model:

- **Documents a reproducibility gap** in an officially-supported configuration. vLLM recipes list R9700 for the gpt-oss family. Useful as a *Limitations / Software stack* observation for v0.2.x of the Hardware Envelope paper, or as material for the planned NaviMed Arena methodology paper.
- **Strengthens the sovereignty narrative.** AWQ-quantized European models (Bielik, PLLuM) and the Qwen 2.5 family run on the same commodity stack without bleeding-edge MoE kernel dependencies. The frontier American MXFP4-native MoE requires a vendor-supplied stack alignment to run on the same hardware. This is operationally significant for institutions evaluating local-first infrastructure under sovereignty constraints (RODO/GDPR, EU AI Act).
- **Distinguishes "supported" from "reproducible".** A useful axis to add to the methodology section: a model entry in the suite is meaningful only when reproducibility is demonstrable on a commodity wheel install, not only on a vendor-provided lockstep build.

## 9. References

1. Reddit r/ROCm — *Benchmarking GPT-OSS-20B on AMD Radeon AI PRO R9700 × 2 (Loaner Hardware Results)*. <https://www.reddit.com/r/ROCm/comments/1omdbb2/>
2. vLLM Recipes — *GPT OSS*. <https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html>
3. Hostkey — *Benchmarking the Radeon AI Pro R9700* (ROCm 7.1.1 + Ollama). <https://hostkey.com/blog/142-benchmarking-the-radeon-ai-pro-r9700-amds-red-team-buys-into-on-device-ai/>
4. Phoronix — *AMD Radeon AI PRO R9700 Linux Performance For Single & Dual GPU Benchmarks Review*. <https://www.phoronix.com/review/amd-radeon-ai-pro-r9700>
5. vLLM Blog — *GPT-OSS Performance Optimizations on NVIDIA Blackwell: Pushing the Pareto Frontier* (2026-02-01). <https://blog.vllm.ai/2026/02/01/gpt-oss-optimizations.html>

---

*Session conducted on 2026-04-26 as part of pre-sweep model triage for the NaviMed Hardware Envelope characterization. Negative result, recorded for reproducibility.*
