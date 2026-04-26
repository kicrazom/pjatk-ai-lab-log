"""
Parametrized sanity-load for Qwen 2.5 72B AWQ on 2x R9700 (gfx1201).

Tests hypotheses about low throughput (4.6 tok/s baseline from H0).
Hypothesis selected via environment variable HYPOTHESIS:
  H0 - baseline (all original flags: debug + eager)
  H4 - second run (cache warm, no changes from baseline)
  H3 - no debug serialization (async kernels enabled)
  H2 - no enforce_eager (CUDA graph capture enabled, risky on gfx1201)
  H1 - awq_marlin kernel (explicit)

IMPORTANT: All code must be inside main() because vLLM TP>=2 uses 'spawn'
multiprocessing which re-imports this module. Module-level LLM creation
causes recursive spawn failures.
"""
import os
import sys
import time


def build_vllm_kwargs(hypothesis: str, model_path: str) -> dict:
    """Return vLLM LLM() kwargs modified per hypothesis under test."""
    kwargs = {
        "model": model_path,
        "dtype": "auto",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.92,
        "enforce_eager": True,
        "tensor_parallel_size": 2,
    }
    if hypothesis == "H2":
        ### H2: test whether enforce_eager is the bottleneck (CUDA graphs gain)
        kwargs["enforce_eager"] = False
    elif hypothesis == "H1":
        ### H1: force awq_marlin kernel (vLLM auto-selects it anyway per log)
        kwargs["quantization"] = "awq_marlin"
    return kwargs


def print_environment(hypothesis: str, kwargs: dict) -> None:
    """Log experiment parameters for reproducibility."""
    print(f"=== Hypothesis: {hypothesis} ===")
    print(f"vllm kwargs: {kwargs}")
    print(f"Env VLLM_ROCM_USE_AITER={os.environ.get('VLLM_ROCM_USE_AITER', 'unset')}")
    print(f"Env AMD_SERIALIZE_KERNEL={os.environ.get('AMD_SERIALIZE_KERNEL', 'unset')}")
    print(f"Env HIP_LAUNCH_BLOCKING={os.environ.get('HIP_LAUNCH_BLOCKING', 'unset')}")


def run_sanity(hypothesis: str, model_path: str) -> int:
    """
    Execute one sanity-load run. Returns exit code.

    Side effects: prints load time, warmup time, throughput, sample output.
    """
    ### Import vllm INSIDE function - avoids triggering CUDA init at module load
    from vllm import LLM, SamplingParams

    kwargs = build_vllm_kwargs(hypothesis, model_path)
    print_environment(hypothesis, kwargs)

    print("\n=== Loading model ===")
    t0 = time.time()
    try:
        llm = LLM(**kwargs)
    except Exception as e:
        print(f"\nModel load FAILED: {e}")
        return 1
    t_load = time.time() - t0
    print(f"Load time: {t_load:.1f}s")

    sampling = SamplingParams(temperature=0.7, max_tokens=128)

    print("\n=== Warmup (1 prompt) ===")
    t0 = time.time()
    llm.generate(["warmup"], sampling)
    t_warmup = time.time() - t0
    print(f"Warmup time: {t_warmup:.1f}s")

    prompts = [
        "Explain quantum entanglement in two sentences:",
        "Write a haiku about GPU benchmarking:",
        "What is the capital of Poland?",
        "List three benefits of open science:",
    ]

    print("\n=== Benchmark (4 prompts) ===")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t_gen = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / t_gen

    print(f"\n=== RESULT (Hypothesis {hypothesis}) ===")
    print(f"Total output tokens: {total_tokens}")
    print(f"Generation time:     {t_gen:.2f}s")
    print(f"Throughput:          {throughput:.1f} tokens/s")
    print(f"Load time:           {t_load:.1f}s")
    print(f"Warmup time:         {t_warmup:.1f}s")

    print("\n=== Sample output [0] ===")
    print(outputs[0].outputs[0].text[:200])
    return 0


def main() -> int:
    hypothesis = os.environ.get("HYPOTHESIS", "H0")
    model_path = "/home/mozarcik/models/qwen25-72b-awq"
    return run_sanity(hypothesis, model_path)


if __name__ == "__main__":
    sys.exit(main())
