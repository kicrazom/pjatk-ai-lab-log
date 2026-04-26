"""
Concurrent request benchmark — single GPU vs TP=2.
Uruchamianie:
    python test_concurrent.py 1 100   # TP=1, 100 promptów
    python test_concurrent.py 2 100   # TP=2, 100 promptów
"""
import sys
import time
from vllm import LLM, SamplingParams


def make_prompts(n: int) -> list[str]:
    """Zbuduj n zróżnicowanych promptów z szablonów + tematów."""
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
    prompts = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        prompts.append(tmpl.format(topic))
    return prompts


def main():
    tp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print(f"=== Concurrent benchmark: TP={tp_size}, N={n_prompts} prompts ===")

    prompts = make_prompts(n_prompts)

    t0 = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
    )
    print(f"Load time: {time.time()-t0:.1f}s")

    sampling = SamplingParams(temperature=0.7, max_tokens=128)

    # Warmup (żeby scheduler i attention backend rozgrzały się)
    print("\n=== Warmup (5 prompts) ===")
    llm.generate(prompts[:5], sampling)

    # Prawdziwy benchmark
    print(f"\n=== Benchmark ({n_prompts} prompts concurrent) ===")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t_gen = time.time() - t0

    # Statystyki
    out_lens = [len(o.outputs[0].token_ids) for o in outputs]
    in_lens = [len(o.prompt_token_ids) for o in outputs]
    total_out = sum(out_lens)
    total_in = sum(in_lens)

    print(f"\n{'='*50}")
    print(f"RESULTS — TP={tp_size}")
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


if __name__ == "__main__":
    main()
