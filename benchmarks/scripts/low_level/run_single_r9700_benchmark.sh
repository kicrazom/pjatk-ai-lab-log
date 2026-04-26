
#!/usr/bin/env bash

# ROCm + PyTorch benchmark script for Radeon AI PRO R9700
# Reproduces the GEMM benchmark used in this repository.

set -e

echo "Activating ROCm Python environment..."
source ~/venvs/rocm72/bin/activate

echo "Creating benchmark script..."

cat << 'EOF' > matmul_tflops_v2.py
import time
import torch

assert torch.cuda.is_available(), "GPU ROCm not available"

device = "cuda"
n = 8192
warmup = 20
iters = 50

print(f"torch: {torch.__version__}")
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"matrix: {n} x {n}")

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    try:
        print(f"\nTesting dtype: {dtype}")
        x = torch.randn((n, n), device=device, dtype=dtype)
        y = torch.randn((n, n), device=device, dtype=dtype)

        with torch.no_grad():
            for _ in range(warmup):
                z = x @ y
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(iters):
                z = x @ y
            torch.cuda.synchronize()
            end = time.perf_counter()

        avg_time = (end - start) / iters
        flops = 2 * (n ** 3)
        tflops = flops / avg_time / 1e12

        print(f"avg_time_per_matmul_s: {avg_time:.6f}")
        print(f"throughput_tflops: {tflops:.2f}")
        print(f"checksum: {float(z.float().sum()):.6f}")
    except Exception as e:
        print(f"FAILED for {dtype}: {e}")
EOF

echo "Running benchmark with autotuning..."

PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_VERBOSE=1 HIP_VISIBLE_DEVICES=0 python matmul_tflops_v2.py

echo "Benchmark finished."
