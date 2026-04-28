"""
Sanity-load test for Qwen 2.5 72B AWQ on 2x R9700 (gfx1201), TP=2.

Validates that the model loads correctly with tensor_parallel_size=2,
runs a small set of prompts, and reports throughput. No thermal
instrumentation (that comes with the pilot benchmark).

Based on test_dual_gpu_tp.py (Qwen 7B sanity) with adjustments for:
  - Local model path (~/models/qwen25-72b-awq per research protocol)
  - AWQ dtype (auto-detect)
  - Higher gpu_memory_utilization (0.92) — 72B AWQ requires most VRAM
"""
import time
import os
from vllm import LLM, SamplingParams


def main():
    prompts = [
        "Explain quantum entanglement in two sentences:",
        "Write a haiku about GPU benchmarking:",
        "What is the capital of Poland?",
        "List three benefits of open science:",
    ]

    print("=== Loading Qwen 2.5 72B AWQ with tensor_parallel_size=2 (EAGER MODE) ===")
    t0 = time.time()
    llm = LLM(
        model=os.path.expanduser("~/models/qwen25-72b-awq"),
        dtype="auto",                       # AWQ handles its own dtype
        max_model_len=4096,
        gpu_memory_utilization=0.92,        # 72B AWQ ~39 GB / 2 GPU = ~20 GB each
        enforce_eager=True,                 # gfx1201 HSA_STATUS workaround
        tensor_parallel_size=2,
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
