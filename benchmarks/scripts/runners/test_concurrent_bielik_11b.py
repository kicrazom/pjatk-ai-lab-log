"""
Concurrent request benchmark for Bielik 11B v2.3 on 2x R9700 (gfx1201).

Usage:
    python test_concurrent_bielik_11b.py 2 100 --quant fp16
    python test_concurrent_bielik_11b.py 1 50  --quant awq

Based on test_concurrent_qwen36_27b.py with adjustments for Bielik 11B v2.3:
  - Mistral-based architecture (NOT hybrid attention)
  - Two quantizations: FP16 (~22 GB weights) and AWQ-4bit (~6 GB weights)
  - Per METHODOLOGY §4 model 5: FP16 supports TP=1 and TP=2
  - Per METHODOLOGY §4 model 6: AWQ supports TP=1 only
  - enforce_eager=True confirmed mandatory empirically (2026-05-04 Phase 1):
    graphs=False segfaults in libhsa-runtime64 on gfx1201 even for
    Mistral-based models. Extends METHODOLOGY §3.2.
  - Default configs from Phase 1 envelope study (2026-05-04):
      FP16 TP=2: max_len=8192, util=0.90 (173k KV tokens, 17.4 sanity tok/s)
      FP16 TP=1: max_len=4096, util=0.90 (34k KV tokens, 14.7 sanity tok/s)
      AWQ  TP=1: max_len=2048, util=0.90 (107k KV tokens,  3.9 sanity tok/s)
    Override via --max-len / --util to use Phase 1-discovered optima.
  - All vLLM code wrapped in main() to avoid spawn-multiprocessing recursion.

Embargo: EMBARGO_paper_bound (Polish model, stricter §11.3).

Author: Łukasz Minarowski <lukasz.minarowski@umb.edu.pl>
"""

import argparse
import os
import sys
import time

# Default configurations — locked from Phase 1 envelope (2026-05-04).
# Phase 2 sweep typically uses one config; pass via --max-len / --util to
# override on a per-run basis if needed.
DEFAULT_CONFIGS = {
    "fp16": {
        "model_path": os.path.expanduser("~/models/bielik-11b-v23"),
        "max_model_len": 8192,  # Phase 1 best at TP=2; reduce for TP=1
        "gpu_memory_utilization": 0.90,
    },
    "awq": {
        "model_path": os.path.expanduser("~/models/bielik-11b-v23-awq"),
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.90,
    },
}


def make_prompts(n: int) -> list[str]:
    """Build n varied prompts from templates x topics.

    IDENTICAL to 7B / 27B / 72B benchmarks per METHODOLOGY §6 — same
    workload, different model. Cross-model comparability depends on this
    being byte-for-byte identical.
    """
    templates = [
        "Explain {} in simple terms, with an example:",
        "Write a short story (about 100 words) involving {}:",
        "What are the three key benefits of {}? Give specific reasons:",
        "Summarize the history of {} in 3-4 sentences:",
        "Compare {} with a related concept, highlighting differences:",
        "Describe how {} works from first principles:",
        "What are common misconceptions about {}? Address them:",
        "Give a practical example of using {} in everyday life:",
    ]
    topics = [
        "quantum entanglement",
        "photosynthesis",
        "machine learning",
        "the TCP/IP protocol",
        "black holes",
        "mRNA vaccines",
        "distributed systems",
        "neural plasticity",
        "supply chain logistics",
        "tensor parallelism",
        "climate feedback loops",
        "the Krebs cycle",
        "cryptographic hashing",
        "CRISPR gene editing",
        "monetary policy",
        "reinforcement learning",
        "ocean currents",
        "magnetic resonance imaging",
        "fermentation",
        "GPS triangulation",
    ]
    return [
        templates[i % len(templates)].format(topics[i % len(topics)]) for i in range(n)
    ]


def main() -> int:
    """Argument parsing and benchmark execution.

    Imports vLLM inside main() to avoid CUDA init at module load —
    vLLM with TP>=2 uses 'spawn' multiprocessing which re-imports this
    module per worker; module-level CUDA init causes recursion errors.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("tp", type=int, help="Tensor parallel size (1 or 2)")
    ap.add_argument("n", type=int, help="Number of concurrent prompts")
    ap.add_argument(
        "--quant",
        choices=["fp16", "awq"],
        default="fp16",
        help="Quantization variant (default: fp16)",
    )
    ap.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Override max_model_len (default: from DEFAULT_CONFIGS)",
    )
    ap.add_argument(
        "--util",
        type=float,
        default=None,
        help="Override gpu_memory_utilization (default: from DEFAULT_CONFIGS)",
    )
    ap.add_argument(
        "--kv-dtype",
        default=None,
        help="Override kv_cache_dtype (e.g. fp8_e4m3, default: vLLM default)",
    )
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    # Resolve configuration from defaults + CLI overrides
    config = DEFAULT_CONFIGS[args.quant].copy()
    if args.max_len is not None:
        config["max_model_len"] = args.max_len
    if args.util is not None:
        config["gpu_memory_utilization"] = args.util

    if args.quant == "awq" and args.tp != 1:
        print(
            "WARNING: Bielik AWQ per METHODOLOGY §4 model 6 supports TP=1 only. "
            f"Got TP={args.tp}; behavior undefined."
        )

    print(
        f"=== Concurrent benchmark: Bielik 11B v2.3 {args.quant.upper()} "
        f"TP={args.tp}, N={args.n} prompts ==="
    )
    print(f"    model:                {config['model_path']}")
    print(f"    max_model_len:        {config['max_model_len']}")
    print(f"    gpu_memory_util:      {config['gpu_memory_utilization']}")
    print(f"    kv_cache_dtype:       {args.kv_dtype or 'default'}")
    print("    enforce_eager:        True (graphs path segfaults on gfx1201)")

    prompts = make_prompts(args.n)

    # Build LLM kwargs — quantization arg only for AWQ; kv_cache_dtype only
    # included if explicitly set
    llm_kwargs = dict(
        model=config["model_path"],
        dtype="auto",
        max_model_len=config["max_model_len"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enforce_eager=True,
        tensor_parallel_size=args.tp,
    )
    if args.quant == "awq":
        llm_kwargs["quantization"] = "awq_marlin"
    if args.kv_dtype:
        llm_kwargs["kv_cache_dtype"] = args.kv_dtype

    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"Load time: {time.time()-t0:.1f}s")

    sampling = SamplingParams(temperature=0.7, max_tokens=128)

    # Warmup so scheduler and attention backend are hot
    warmup_n = min(5, args.n)
    print(f"\n=== Warmup ({warmup_n} prompts) ===")
    llm.generate(prompts[:warmup_n], sampling)

    print(f"\n=== Benchmark ({args.n} prompts concurrent) ===")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t_gen = time.time() - t0

    out_lens = [len(o.outputs[0].token_ids) for o in outputs]
    in_lens = [len(o.prompt_token_ids) for o in outputs]
    total_out = sum(out_lens)
    total_in = sum(in_lens)

    print()
    print(f"Total time:           {t_gen:.2f}s")
    print(f"Total output tokens:  {total_out}")
    print(f"Total input tokens:   {total_in}")
    print(f"Output throughput:    {total_out / t_gen:.2f} tok/s")
    print(f"Total throughput:     {(total_out + total_in) / t_gen:.2f} tok/s")
    print(f"Requests/second:      {args.n / t_gen:.3f}")
    print(f"Mean output len:      {total_out / args.n:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
