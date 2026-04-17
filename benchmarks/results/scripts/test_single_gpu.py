import time
from vllm import LLM, SamplingParams


def main():
    prompts = [
        "Explain quantum entanglement in two sentences:",
        "Write a haiku about GPU benchmarking:",
        "What is the capital of Poland?",
        "List three benefits of open science:",
    ]

    print("=== Loading model on GPU 0 (EAGER MODE, no CUDA graphs) ===")
    t0 = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
    )
    t_load = time.time() - t0
    print(f"Load time: {t_load:.1f}s")

    sampling = SamplingParams(temperature=0.7, max_tokens=128)

    print("\n=== Warmup ===")
    llm.generate(["warmup"], sampling)

    print("\n=== Benchmark ===")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t_gen = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\nTotal output tokens: {total_tokens}")
    print(f"Generation time:     {t_gen:.2f}s")
    print(f"Throughput:          {total_tokens/t_gen:.1f} tokens/s")

    print("\n=== Sample outputs ===")
    for i, out in enumerate(outputs[:2]):
        print(f"\n[{i}] {out.prompt}")
        print(f"    → {out.outputs[0].text[:200]}")


if __name__ == "__main__":
    main()
