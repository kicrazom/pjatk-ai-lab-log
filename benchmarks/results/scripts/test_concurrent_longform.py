"""
Concurrent benchmark — LONG FORM output (max_tokens=1024).
Powinno pokazać realne skalowanie TP=2 (długi decode = amortyzacja komunikacji).
"""
import sys
import time
from vllm import LLM, SamplingParams


def make_prompts(n: int) -> list[str]:
    """Dłuższe prompty wymagające rozwiniętych odpowiedzi."""
    templates = [
        "Write a detailed technical guide on {} covering history, key concepts, practical applications, and future directions:",
        "Explain {} from basic principles to advanced implications. Structure your answer with clear sections:",
        "Compose a comprehensive analysis of {} addressing theory, practice, limitations, and current research frontiers:",
        "Walk through {} step-by-step, from foundational ideas to contemporary developments, with examples:",
    ]
    topics = [
        "transformer attention mechanisms", "RNA splicing regulation",
        "distributed consensus protocols", "neural coding in the visual cortex",
        "quantum error correction", "supply chain resilience",
        "CRISPR off-target effects", "reinforcement learning policy gradients",
        "plate tectonics feedback loops", "cryptographic zero-knowledge proofs",
        "mRNA vaccine thermostability", "tensor network methods in physics",
        "circadian rhythm molecular biology", "carbon capture technologies",
        "graph neural networks for chemistry", "climate sensitivity estimation",
        "prokaryotic vs eukaryotic gene regulation", "bandwidth-delay product in networking",
        "large language model alignment", "gene therapy delivery vectors",
    ]
    prompts = []
    for i in range(n):
        prompts.append(templates[i % len(templates)].format(topics[i % len(topics)]))
    return prompts


def main():
    tp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

    print(f"=== Long-form: TP={tp_size}, N={n_prompts}, max_tokens={max_tokens} ===")

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

    sampling = SamplingParams(temperature=0.7, max_tokens=max_tokens)

    print("\n=== Warmup (3 prompts) ===")
    llm.generate(prompts[:3], sampling)

    print(f"\n=== Benchmark ({n_prompts} prompts) ===")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t_gen = time.time() - t0

    out_lens = [len(o.outputs[0].token_ids) for o in outputs]
    in_lens = [len(o.prompt_token_ids) for o in outputs]
    total_out = sum(out_lens)
    total_in = sum(in_lens)

    print(f"\n{'='*50}")
    print(f"RESULTS — TP={tp_size}")
    print(f"{'='*50}")
    print(f"Prompts:             {n_prompts}")
    print(f"Input mean:          {total_in/n_prompts:.1f} tokens")
    print(f"Output mean:         {total_out/n_prompts:.1f} tokens")
    print(f"Output min/max:      {min(out_lens)} / {max(out_lens)}")
    print(f"Total time:          {t_gen:.2f}s")
    print(f"Output throughput:   {total_out/t_gen:.1f} tok/s")
    print(f"Total throughput:    {(total_in+total_out)/t_gen:.1f} tok/s")
    print(f"Requests/second:     {n_prompts/t_gen:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
