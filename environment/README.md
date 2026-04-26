# Environment manifests

Snapshots of the software and system environment used for benchmarks
in this repository. Each snapshot is dated and corresponds to a
specific session or release.

## Files (2026-04-26 snapshot — release v0.1.0)

- `critical_versions_2026-04-26.txt` — short list of the most
  important runtime versions (Python, vLLM, PyTorch, HIP, etc.).
- `pip_freeze_2026-04-26.txt` — full `pip freeze` from the vLLM
  virtual environment.
- `system_info_2026-04-26.txt` — kernel, distribution, ROCm runtime,
  CPU info from `uname`, `lsb_release`, `rocm-smi --version`, `lscpu`.

## Cited from

- `paper/v0.1-hardware-envelope.md` (top 8 critical versions)
- `benchmarks/methodology/hardware_context.md`
- `docs/sessions/2026-04-26-qwen36-vllm-envelope.md`

## Versioning

New manifests added on stack changes (new model, ROCm/PyTorch/vLLM
upgrade). Older manifests remain in git history.
