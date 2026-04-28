"""
Concurrent request benchmark for Qwen 3.6 27B on 2x R9700 (gfx1201), TP=2.

Usage:
    python test_concurrent_qwen36_27b.py 2 100 --quant fp8
    python test_concurrent_qwen36_27b.py 2 25  --quant bf16 --max-len 1024 --util 0.95

Based on test_concurrent_qwen72b_awq.py (72B AWQ) with adjustments for Qwen 3.6 27B:
  - Two quantizations: FP8 (29 GB) and BF16 (52 GB), both require TP=2
  - enforce_eager=True is mandatory (Qwen 3.5/3.6 hybrid Mamba+Transformer
    attention causes HSA_STATUS_ERROR_INVALID_PACKET_FORMAT with default
    CUDA graph capture on gfx1201)
  - Default configurations from Phase 1 envelope study:
      FP8:  max_len=2048, util=0.85 (61K KV tokens, 52x max concurrency)
      BF16: max_len=1024, util=0.95 (7K KV tokens, 7x max concurrency)
    Override via --max-len / --util to use Phase 1-discovered optima.
  - All vLLM code wrapped in main() to avoid spawn-multiprocessing recursion
"""

import argparse
import os
import sys
import time


# Default configurations — these are the v0.1.0 working configs.
# Phase 1 may discover better BF16 configs; pass via --max-len / --util.
# Paths use os.path.expanduser to avoid hardcoded usernames in public repo.
DEFAULT_CONFIGS = {
    "fp8": {
        "model_path": os.path.expanduser("~/models/qwen36-27b-fp8"),
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.85,
    },
    "bf16": {
        "model_path": os.path.expanduser("~/models/qwen36-27b"),
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.95,
    },
}


def make_prompts(n: int) -> list[str]:
    """Build n varied prompts from templates x topics.

    Identical to 7B and 72B benchmarks to keep cross-model comparison
    apples-to-apples (same workload, different model).
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
        "quantum entanglement", "photosynthesis", "machine learning",
        "the TCP/IP protocol", "black holes", "mRNA vaccines",
        "distributed systems", "neural plasticity", "supply chain logistics",
        "tensor parallelism", "climate feedback loops", "the Krebs cycle",
        "cryptographic hashing", "CRISPR gene editing", "monetary policy",
        "reinforcement learning", "ocean currents", "magnetic resonance imaging",
        "fermentation", "GPS triangulation",
    ]
    return [templates[i % len(templates)].format(topics[i % len(topics)])
            for i in range(n)]


def main() -> int:
    """Argument parsing and benchmark execution.

    Imports vLLM inside main() to avoid CUDA init at module load —
    vLLM with TP>=2 uses 'spawn' multiprocessing which re-imports this
    module per worker; module-level CUDA init causes recursion errors.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("tp", type=int, help="Tensor parallel size (must be 2 for 27B)")
    ap.add_argument("n", type=int, help="Number of concurrent prompts")
    ap.add_argument("--quant", choices=["fp8", "bf16"], default="bf16",
                    help="Quantization variant (default: bf16)")
    ap.add_argument("--max-len", type=int, default=None,
                    help="Override max_model_len (default: from DEFAULT_CONFIGS)")
    ap.add_argument("--util", type=float, default=None,
                    help="Override gpu_memory_utilization (default: from DEFAULT_CONFIGS)")
    ap.add_argument("--kv-dtype", default=None,
                    help="Override kv_cache_dtype (e.g. fp8_e4m3, default: vLLM default)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    # Resolve configuration from defaults + CLI overrides
    config = DEFAULT_CONFIGS[args.quant].copy()
    if args.max_len is not None:
        config["max_model_len"] = args.max_len
    if args.util is not None:
        config["gpu_memory_utilization"] = args.util

    if args.tp != 2:
        print(f"WARNING: Qwen 3.6 27B requires TP=2 on 32 GB GPUs (single-GPU "
              f"configurations OOM at weight padding stage)")

    print(f"=== Concurrent benchmark: Qwen 3.6 27B {args.quant.upper()} "
          f"TP={args.tp}, N={args.n} prompts ===")
    print(f"    model:                {config['model_path']}")
    print(f"    max_model_len:        {config['max_model_len']}")
    print(f"    gpu_memory_util:      {config['gpu_memory_utilization']}")
    print(f"    kv_cache_dtype:       {args.kv_dtype or 'default'}")
    print(f"    enforce_eager:        True (mandatory for Qwen 3.5/3.6)")

    prompts = make_prompts(args.n)

    # Build LLM kwargs — kv_cache_dtype only included if explicitly set
    llm_kwargs = dict(
        model=config["model_path"],
        dtype="auto",
        max_model_len=config["max_model_len"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enforce_eager=True,
        tensor_parallel_size=args.tp,
    )
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

    print(f"\n{'='*50}")
    print(f"RESULTS - Qwen 3.6 27B {args.quant.upper()} TP={args.tp}")
    print(f"{'='*50}")
    print(f"Prompts:             {args.n}")
    print(f"Input tokens total:  {total_in}")
    print(f"Input tokens mean:   {total_in/args.n:.1f}")
    print(f"Output tokens total: {total_out}")
    print(f"Output tokens mean:  {total_out/args.n:.1f}")
    print(f"Output tokens min:   {min(out_lens)}")
    print(f"Output tokens max:   {max(out_lens)}")
    print(f"Total time:          {t_gen:.2f}s")
    print(f"Output throughput:   {total_out/t_gen:.1f} tok/s")
    print(f"Total throughput:    {(total_in+total_out)/t_gen:.1f} tok/s")
    print(f"Requests/second:     {args.n/t_gen:.2f}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
