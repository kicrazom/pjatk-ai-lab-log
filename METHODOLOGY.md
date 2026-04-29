# NaviMed-UMB Benchmark Methodology

**Document version:** 1.0 (2026-04-29)
**Maintainer:** Łukasz Minarowski, UMB Białystok (ORCID 0000-0002-2536-3508)
**Repository:** https://github.com/kicrazom/navimed-umb
**License:** CC-BY-4.0 (text), MIT (code)
**Citation:** Minarowski Ł. NaviMed-UMB Benchmark Methodology. Zenodo. DOI 10.5281/zenodo.19851347

This document is the single source of truth for how every model in the NaviMed-UMB benchmark suite is measured and reported. Every per-model session report, every paper figure, and every public release derives from the conventions defined here. When in doubt during a benchmarking session, this document overrides ad-hoc decisions.

---

## 1. Scope and goals

NaviMed-UMB is a universal AI hardware envelope and inference benchmark platform built on consumer-grade AMD Radeon AI PRO R9700 GPUs (RDNA 4, gfx1201). The platform exists to answer two related but distinct questions:

1. **Engineering envelope (PUBLIC):** What configurations of `(model, quantization, max_model_len, gpu_memory_utilization, kv_cache_dtype, tensor_parallel_size)` load successfully on this hardware, what is their VRAM footprint, KV cache capacity, and per-GPU thermal/power profile under load?

2. **Scaling sweep (EMBARGOED until publication):** At envelope-validated best configurations, how does aggregate throughput, latency, and power efficiency scale with concurrent request count `N`, where does the throughput knee occur relative to `max_concurrency`, and how do these properties compare across quantizations, model architectures, and inference backends?

The platform deliberately does **not** measure model quality, reasoning capability, or clinical utility. Those questions belong to the separate NaviMed Arena methodology paper and to domain-specific validation studies (e.g., Broncho-Nome). Mixing hardware engineering with semantic evaluation conflates two distinct epistemic regimes; this is the methodological humility position elaborated in §8.

---

## 2. Hardware platform

| Component | Specification |
|---|---|
| GPU (×2) | GIGABYTE Radeon AI PRO R9700, 32 GB GDDR6, RDNA 4 / Navi 48, gfx1201 |
| CPU | AMD Ryzen 9 9950X3D (16 cores / 32 threads, 2 CCDs) |
| RAM | 96 GB DDR5-6000 |
| Storage | NVMe (model weights cached locally) |
| OS | Kubuntu 24.04, kernel 6.17 |
| Display | iGPU RAPHAEL (integrated in 9950X3D), separate from compute path |

The system exposes three GPUs to ROCm: `card0` and `card1` are the R9700 pair, `card2` is the integrated RAPHAEL graphics. All compute work is restricted to the R9700 pair via `ROCR_VISIBLE_DEVICES=0,1`. Per-GPU sampling code MUST filter by `Card Series` string match rather than by index assumption, because index ordering is not stable across kernel boots.

---

## 3. Software stack

| Layer | Version | Notes |
|---|---|---|
| ROCm | 7.2.0 | Native install, not Docker |
| vLLM | 0.19.0+rocm721 | gfx1201 supported as of this build |
| PyTorch | 2.10.0+git8514f05 | Bundled with vLLM wheel |
| flash_attn | 2.8.3 | Bundled |
| triton | 3.6.0 | Bundled |
| Python venv | `~/venvs/vllm/` | Always activated before any benchmark command |

### 3.1 Mandatory environment variables for gfx1201

```bash
unset PYTORCH_ALLOC_CONF
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=1            # NOT 3 — rejected by current PyTorch
export HIP_LAUNCH_BLOCKING=1
export ROCR_VISIBLE_DEVICES=0,1          # exclude iGPU RAPHAEL
```

### 3.2 Mandatory model-construction flags

For every model with hybrid attention (Qwen 3.5 family, Qwen 3.6 family, including 27B variants), `enforce_eager=True` is mandatory. Default CUDA graph capture crashes with `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` on gfx1201. This is a known runtime envelope constraint, not a model defect.

### 3.3 Causal closure

Every benchmark JSON record captures the full software stack version triple (`rocm_version`, `vllm_version`, `torch_version`, `torch_hip_version`) plus the full env-var dictionary. Any reproduction attempt that does not reproduce this triple is not a reproduction; it is a different experiment. This requirement follows Lerchner (2026) — computation as description requires complete vehicle specification.

---

## 4. Model suite

The benchmark suite covers thirteen models, deliberately mixing international (Qwen, Mistral, Mixtral) and Polish-language (Bielik, PLLuM) families. Polish models are the platform's community-differentiating contribution; no other public benchmark targets RDNA 4 with these models.

| # | Model | Params | Quant | TP | Family | Notes |
|---|---|---|---|---|---|---|
| 1 | `Qwen/Qwen2.5-7B-Instruct` | 7B | BF16 | 1 or 2 | Qwen 2.5 | TP=2 harmful below high N |
| 2 | `Qwen/Qwen2.5-72B-Instruct-AWQ` | 72B | AWQ-4bit | 2 | Qwen 2.5 | TP=2 mandatory |
| 3 | `Qwen/Qwen3.6-27B` | 27B | BF16 | 2 | Qwen 3.6 | Hybrid attention, eager only |
| 4 | `Qwen/Qwen3.6-27B-FP8` | 27B | FP8 | 2 | Qwen 3.6 | No FP8 kernels on R9700 |
| 5 | `speakleash/Bielik-11B-v2.3-Instruct` | 11B | FP16 | 1 or 2 | Bielik (PL) | |
| 6 | `speakleash/Bielik-11B-v2.3-Instruct-AWQ` | 11B | AWQ-4bit | 1 | Bielik (PL) | |
| 7 | `CYFRAGOVPL/Llama-PLLuM-8B-instruct` | 8B | BF16 | 1 | PLLuM (PL) | Llama base |
| 8 | `CYFRAGOVPL/PLLuM-12B-chat` | 12B | BF16 | 1 or 2 | PLLuM (PL) | Mistral base |
| 9 | `CYFRAGOVPL/Llama-PLLuM-70B-chat-250801` | 70B | BF16 | 2 | PLLuM (PL) | Finale |
| 10 | `mistralai/Mistral-Nemo-Instruct-2407` | 12B | BF16 | 1 or 2 | Mistral | |
| 11 | `TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ` | 47B (MoE) | AWQ-4bit | 2 | Mixtral | |

Model HF identifiers for Qwen 3.6 family use no `-Instruct` suffix (i.e. `Qwen/Qwen3.6-27B` and `Qwen/Qwen3.6-27B-FP8`). PLLuM naming is family-dependent: `Llama-PLLuM-` for Llama-based variants (8B, 70B), bare `PLLuM-` for the Mistral-based 12B.

A future v0.4 phase will repeat the suite under Lemonade Server (AMD-sponsored, GGUF native, OpenAI-compatible API at `localhost:13305`) for stack-level comparison: vLLM as server-grade (high-N AWQ throughput) vs Lemonade as desktop-grade (low-N GGUF latency).

---

## 5. Two-phase experimental design

Each model is measured in two phases with disjoint outputs, separate filesystems, and distinct embargo classifications.

### 5.1 Phase 1 — Hardware envelope

**Goal:** Determine which configurations load and characterize each one along non-load-dependent axes.

**Procedure:** For every `(quantization, max_model_len, gpu_memory_utilization, kv_cache_dtype)` tuple in a curated grid (typically 8–12 configs per model), the runner attempts model construction, executes a single sanity prompt, and records:

- `loaded` (boolean), `error_class` if failed
- `load_time_s`
- `peak_vram_gb_per_gpu` (list, one entry per GPU)
- `kv_cache_tokens`, `kv_cache_gb` (when computable)
- `max_concurrency` (from vLLM scheduler estimate)
- `sanity_throughput_tok_s` (single-prompt, memory-bandwidth-bound baseline)

**Output location:** `benchmarks/results/hardware_envelope/<model>_<quant>_<config>.json`

**Embargo:** PUBLIC. Envelope data describes only the engineering vehicle, not the scaling law.

### 5.2 Phase 2 — Scaling sweep

**Goal:** Characterize aggregate throughput, latency, thermal, and power under increasing concurrent load.

**Procedure:** At Phase 1 envelope-validated best configurations (typically one per quantization, sometimes additional configs covering trade-off space), run the concurrent benchmark at `N ∈ {10, 25, 50, 100, 200, 500, 1000}` requests. The N values are chosen to span:

- Linear-scaling region below `max_concurrency`
- Knee at `N ≈ max_concurrency`
- Preemption regime above `max_concurrency` up to ~40× over

Each N value is one independent run with full model load and 15-second cooldown between runs. Background sampling at 1 Hz captures `cpu_percent`, `cpu_temp`, and per-GPU `use`, `temp`, `vram_used_b`, `power_w`.

**Output location:** `benchmarks/results/<model>/thermal-runs/<quant>-tp<TP>-n<N>-{bench.log,events.json,thermals.jsonl,thermals.png}`

**Embargo:** Mixed. Engineering observations (knee position, preemption onset, thermal envelope) are PUBLIC. Concrete scaling numbers (throughput@N, P50/P95/P99 latency, KV cache curves, cross-model comparisons) are EMBARGOED until paper acceptance. The split is enforced per artifact, not per file — see §11.

---

## 6. Workload

All Phase 2 sweeps use a synthetic prompt generator identical across models, ensuring apples-to-apples cross-model comparison.

- **Templates:** 8 instruction templates (explain, write story, list benefits, summarize history, compare, describe from first principles, address misconceptions, give practical example)
- **Topics:** 20 general-knowledge topics (quantum entanglement, photosynthesis, machine learning, TCP/IP, black holes, mRNA vaccines, …)
- **Construction:** N prompts via `templates[i % 8].format(topics[i % 20])` for `i ∈ [0, N)`
- **Sampling:** `temperature=0.7`, `max_tokens=128`
- **Warmup:** `min(5, N)` prompts before timed run, results discarded

Synthetic prompts are not equivalent to real medical workloads. This is a stated limitation (§10). Domain-specific evaluation belongs to the NaviMed Arena and to per-domain validation studies (Broncho-Nome).

---

## 7. Universal reporting schema

Every Phase 2 sweep produces the same flat table, regardless of model. Cross-model figures depend on this discipline.

### 7.1 Mandatory columns

```
model | quant | backend | TP | max_len | util | KV_dtype | N |
tok/s_out | tok/s_tot | total_s | req/s |
VRAM_peak_GB | T_peak_C | W_mean | W_peak | W/tok |
tuning(stock|uv) | vs_baseline_%
```

Where:
- `tok/s_out` = output tokens / total wall time (aggregate, all streams)
- `tok/s_tot` = (input + output) tokens / total wall time
- `req/s` = N / total wall time
- `VRAM_peak_GB` = max over both GPUs and over the run window
- `T_peak_C`, `W_peak` = same, from sampler
- `W_mean` = sampler mean during benchmark window (after baseline, before cooldown)
- `W/tok` = energy efficiency, computed as `(W_mean × total_s) / total_output_tokens`
- `tuning` = `stock` for default firmware, `uv` for undervolted (-75 mV / +15 W typical)
- `vs_baseline_%` = relative throughput vs reference; for international models the reference is kyuz0's published numbers, for Polish models (no public reference) the reference is the model's own stock run

This schema follows Capitelli (2025) for the engineering columns and extends with the energy efficiency column W/tok per Ziskind's methodology. Cross-stack comparison (§4 v0.4 phase) will add a `backend` column distinguishing vLLM from Lemonade.

### 7.2 Mandatory plots

Three plots per model, identical layout for cross-model comparison:

1. **Scaling curve.** `tok/s_out` vs `N`, log-x. Vertical dashed line at `max_concurrency` from Phase 1. Annotation at the knee.
2. **Thermal profile gallery.** Grid of per-N thermal traces (temp + power vs time), aligned y-axis across N values for visual comparability.
3. **Efficiency curve.** `W/tok` vs `N`. Lower is better; the minimum identifies the energy-optimal operating point.

### 7.3 Output structure (per model)

```
benchmarks/results/<model>/
├── thermal-runs/
│   ├── <quant>-tp<TP>-n<N>-bench.log         # full benchmark stdout
│   ├── <quant>-tp<TP>-n<N>-events.json       # benchmark start/end timestamps
│   ├── <quant>-tp<TP>-n<N>-thermals.jsonl    # 1 Hz sampler trace
│   └── <quant>-tp<TP>-n<N>-thermals.png      # per-N thermal plot
├── scaling_curve.png                         # plot 1
├── thermal_gallery.png                       # plot 2
├── efficiency_curve.png                      # plot 3
├── results_table.csv                         # flat schema, all N rows
└── SUMMARY.md                                # narrative + table + plots + embargo split
```

Plus a session lab log entry at `docs/sessions/YYYY-MM-DD-<model>-sweep.md` containing methodology recap, results table, methodological humility statement, AI disclosure block, lessons learned, and embargo classification.

---

## 8. Methodological humility (Lerchner 2026)

Every paper section, every session log entry, and every public summary opens with the following statement, adapted to context:

> We measure inference *throughput*, *latency*, *thermal envelope*, and *power efficiency* under varying concurrent load. We do not measure model quality, reasoning capability, factual accuracy, or downstream clinical utility. Following Lerchner (2026), these are extrinsic computational properties of the inference vehicle, not constitutive properties of cognition. Our claims terminate at the hardware-software interface.

Two corollaries:

- **Throughput is not capability.** A model that produces 100 tok/s of confidently-wrong medical advice is exactly as fast as one that produces 100 tok/s of carefully-cited medical advice. The benchmark cannot tell them apart, and does not try.
- **Reproducibility is causal closure.** Computation = description requires the full vehicle specification (hardware, ROCm wheel, env vars, model weights hash). Our benchmark JSON captures this; ad-hoc reproductions that omit the triple are different experiments.

---

## 9. AI disclosure framework (Kim et al. 2026; ICMJE; COPE)

NaviMed-UMB benchmarks involve AI assistance during analysis and reporting, but no AI involvement in data generation. Every paper, lab log entry, and release uses the three-layer disclosure:

| Layer | NaviMed-UMB practice |
|---|---|
| **1. Dataset / data generation** | Not applicable. Synthetic prompts generated deterministically from human-curated templates and topics (see §6). No LLM-generated data enters the experimental pipeline. |
| **2. Experimental pipeline** | Claude (Anthropic) used during 2026-04 sessions for: analysis script drafting, debugging vLLM env issues, designing Phase 1 envelope grids, and validating scaling-curve interpretation. Specific model: Claude Opus 4.7 via claude.ai. Each session log entry records the specific date and the role played. |
| **3. Manuscript editing** | To be disclosed per paper at submission time, with model name, version, date range, and editing scope. No LLM-authored claims; LLM used as editor and sounding-board only. |

The disclosure block is mandatory in every `docs/sessions/*.md` file and every `paper/*.md` file. Format follows Kim et al. (2026) trabular convention, referenced for stylistic consistency across the project.

---

## 10. Limitations

The following limitations apply to every result produced under this methodology. They are restated in every paper and every release.

1. **Synthetic prompts.** General-knowledge prompt templates do not reflect the token-length distributions, vocabulary, or structural complexity of real medical workloads (clinical notes, radiology reports, discharge summaries, multi-turn consultation transcripts). Phase 2 throughput numbers should be read as upper-bound engineering capacity, not as predictors of clinical-deployment latency.

2. **Single hardware platform.** All measurements come from one specific dual-R9700 system with one specific ROCm/vLLM/PyTorch triple. Generalization to other gfx1201 systems is plausible but unverified; generalization to CDNA (MI250/MI300), other RDNA generations, or NVIDIA hardware requires re-running Phase 1 envelope from scratch.

3. **Inference backend.** v0.1–v0.3 measure vLLM only. v0.4 will add Lemonade Server (AMD-sponsored, GGUF). llama.cpp, TGI, and SGLang are explicitly out of scope.

4. **No model-quality dimension.** As stated in §1 and §8, this benchmark does not evaluate semantic correctness. Pairing throughput with quality belongs to the separate NaviMed Arena methodology and to domain-specific studies.

5. **Concurrency model is embarrassingly parallel.** Real production traffic exhibits Poisson-arrival patterns, mixed prompt lengths, and streaming-with-cancellation; our sweep submits N requests as a single batch. The knee position and shape are representative but the absolute throughput at high N is best-case.

6. **Power sampling at 1 Hz.** Mean and peak watts are computed over discrete samples; sub-second power transients are not captured. Energy in Wh requires trapezoidal integration over the sample timeline, which the analysis pipeline performs but cannot recover information lost between samples.

7. **Thermal envelope assumes ambient ~22 °C.** Lab ambient temperature is not actively controlled. Hot-summer reproductions may show higher steady-state temperatures and thermal throttling behavior not present in our data.

8. **vLLM version drift.** vLLM 0.19.0+rocm721 is one specific build. Newer vLLM versions may add gfx1201 optimizations (FP8 kernels, AITER) that change the envelope substantially. Results are timestamped to allow longitudinal comparison.

9. **AWQ vs FP8 vs BF16 are different things.** Cross-quantization comparisons are valid only at fixed model identity. Comparing Qwen 72B AWQ to Qwen 3.6 27B BF16 on throughput is methodologically suspect; the suite includes both for envelope characterization, not for direct quantization-vs-quantization claims.

10. **Polish-language models lack public baselines.** For Bielik and PLLuM the `vs_baseline_%` column refers to the model's own stock run rather than an external reference, because no other public R9700 benchmark targets these models. This is a contribution of the platform, not a methodological compromise.

---

## 11. Embargo policy

The platform operates under a structured embargo policy distinguishing engineering content (PUBLIC, immediate release to repository) from research content (EMBARGOED until paper acceptance).

### 11.1 PUBLIC artifacts

- Hardware envelope tables (load y/n, peak VRAM, KV tokens, max_concurrency)
- Engineering workarounds (env vars, `enforce_eager`, ROCR_VISIBLE_DEVICES filter)
- Sanity throughput numbers (single-prompt, memory-bandwidth baseline)
- All scripts (runners, orchestrators, analysis, plotting)
- Methodology document (this file)
- AI disclosure structure
- Knee-position observations ("knee occurs at N ≈ max_concurrency from Phase 1")

### 11.2 EMBARGOED artifacts (until paper acceptance)

- Phase 2 scaling tables: throughput@N for N ∈ {10..1000}
- Latency distributions: P50/P95/P99 per N
- KV cache utilization curves
- Cross-model comparative claims with concrete numbers
- Quantization-vs-quantization scaling-law interpretations
- Paper-bound figures (scaling curves, thermal galleries, efficiency curves) with concrete y-axis values

### 11.3 Stricter embargo for Polish models

Bielik and PLLuM scaling numbers carry stricter embargo than Qwen / Mistral / Mixtral. The Polish AI community is small; scoop risk is materially higher. PLLuM 70B sweep results in particular remain local-only until co-author review and paper submission.

### 11.4 Per-step labeling

Every benchmark session, every result-generating step is explicitly labeled at the time of generation: `EMBARGO — paper figure` or `PUBLIC — engineering note for repo`. This avoids accidental commits of paper-bound numbers to the public repository.

---

## 12. References

- Capitelli, D. (2025). *AMD R9700 vLLM benchmarks.* GitHub. https://github.com/kyuz0/amd-r9700-vllm
- Capitelli, D. (2025). *AMD R9700 AI Toolboxes.* https://kyuz0.github.io/amd-r9700-ai-toolboxes/
- Kim et al. (2026). *Context-Aware Diversity (CAD) score and AI disclosure framework.* [Citation per submitted paper]
- Lerchner, A. (2026). *The Abstraction Fallacy: Why Computation Is Not Cognition.* arXiv preprint. [Citation pending]
- ICMJE (2024). *Recommendations for the Conduct, Reporting, Editing, and Publication of Scholarly Work in Medical Journals.* https://www.icmje.org/recommendations/
- COPE (n.d.). *Authorship and AI tools — position statement.* https://publicationethics.org/cope-position-statements/ai-author
- vLLM project. https://github.com/vllm-project/vllm
- ROCm 7.2.0 release notes. https://rocm.docs.amd.com/

---

## 13. Versioning of the benchmark series

| Version | Scope | Status |
|---|---|---|
| v0.1.0 | Hardware envelope paper, Qwen 3.6 27B | Released 2026-04-26, DOI 10.5281/zenodo.19851347 |
| v0.2.0 | Phase 2 scaling sweep, Qwen 3.6 27B (BF16 + FP8) | In progress 2026-04-29 |
| v0.3.0 | Remaining model suite, Phase 1 + Phase 2 | Planned |
| v0.4.0 | Lemonade Server cross-stack comparison | Planned post-v0.3 |
| v0.5.0 | Undervolted re-run (-75 mV / +15 W) of full suite | Planned |

This methodology document itself is versioned independently; methodological revisions bump its version (currently 1.0) and are recorded in the changelog at the bottom of this file.

---

## Changelog

- **1.0 (2026-04-29):** Initial consolidated methodology. Synthesizes engineering practice from v0.1.0 release, Capitelli-style reporting schema, Kim et al. (2026) AI disclosure framework, Lerchner (2026) methodological humility position, and embargo policy operational since 2026-04-28. Authoritative for v0.2.0 and onward.
