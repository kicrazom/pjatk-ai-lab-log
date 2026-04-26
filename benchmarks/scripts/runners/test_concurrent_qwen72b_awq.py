"""
Concurrent request benchmark for Qwen 2.5 72B AWQ on 2x R9700 (gfx1201), TP=2.

Usage:
    python test_concurrent_qwen72b_awq.py 2 100   # TP=2 is the only valid value
                                                   # (72B AWQ doesn't fit on single GPU)

Based on test_concurrent.py (Qwen 7B) with adjustments for 72B AWQ:
  - Local model path (~/models/qwen25-72b-awq per research protocol)
  - dtype='auto' (AWQ handles its own dtype, vLLM auto-selects awq_marlin)
  - gpu_memory_utilization=0.92 (72B AWQ needs more headroom than 7B's 0.70)
  - Selected H3 config: enforce_eager=True (mandatory on gfx1201)
  - All vLLM code wrapped in main() to avoid spawn-multiprocessing recursion
    (vLLM TP>=2 uses 'spawn' which re-imports this module per worker)
"""
import sys
import time


def make_prompts(n: int) -> list[str]:
    """Build n varied prompts from templates x topics. Identical to 7B benchmark
    to keep cross-model comparison apples-to-apples."""
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
    ### Import inside main() to avoid CUDA init at module load (spawn recursion)
    from vllm import LLM, SamplingParams

    tp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    if tp_size != 2:
        print(f"WARNING: Qwen 72B AWQ requires TP=2 (39 GB > 32 GB single-GPU)")

    print(f"=== Concurrent benchmark: Qwen 72B AWQ TP={tp_size}, N={n_prompts} prompts ===")

    prompts = make_prompts(n_prompts)

    t0 = time.time()
    llm = LLM(
        model="/home/mozarcik/models/qwen25-72b-awq",
        dtype="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
    )
    print(f"Load time: {time.time()-t0:.1f}s")

    sampling = SamplingParams(temperature=0.7, max_tokens=128)

    ### Warmup so scheduler and attention backend are hot
    print("\n=== Warmup (5 prompts) ===")
    llm.generate(prompts[:5], sampling)

    print(f"\n=== Benchmark ({n_prompts} prompts concurrent) ===")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t_gen = time.time() - t0

    out_lens = [len(o.outputs[0].token_ids) for o in outputs]
    in_lens = [len(o.prompt_token_ids) for o in outputs]
    total_out = sum(out_lens)
    total_in = sum(in_lens)

    print(f"\n{'='*50}")
    print(f"RESULTS - Qwen 72B AWQ TP={tp_size}")
    print(f"{'='*50}")
    print(f"Prompts:             {n_prompts}")
    print(f"Input tokens total:  {total_in}")
    print(f"Input tokens mean:   {total_in/n_prompts:.1f}")
    print(f"Output tokens total: {total_out}")
    print(f"Output tokens mean:  {total_out/n_prompts:.1f}")
    print(f"Output tokens min:   {min(out_lens)}")
    print(f"Output tokens max:   {max(out_lens)}")
    print(f"Total time:          {t_gen:.2f}s")
    print(f"Output throughput:   {total_out/t_gen:.1f} tok/s")
    print(f"Total throughput:    {(total_in+total_out)/t_gen:.1f} tok/s")
    print(f"Requests/second:     {n_prompts/t_gen:.2f}")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
