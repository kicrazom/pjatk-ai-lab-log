# 2026-05-02 — TP=2 sanity test

Pre-flight check before Bielik 11B benchmark sessions. Verifies that
the stack used for the v0.2.0 Phase 2 sweep still produces a working
tensor-parallel-2 deployment.

## Stack

- vLLM 0.19.0+rocm721 (bundled wheel in ~/venvs/vllm)
- PyTorch 2.10.0+git8514f05 (bundled with vLLM)
- ROCm 7.2.0 (system, /opt/rocm-7.2.0)
- RCCL 2.27.7 (system, /opt/rocm-7.2.0/lib/librccl.so)
- Kubuntu 24.04 LTS, Python 3.12

## Test

Local model: ~/models/qwen25-7b-instruct
Config: TP=2, enforce_eager, max_model_len=4096, util=0.85,
attention=TRITON_ATTN, served on 127.0.0.1:8765.
Env: HIP_VISIBLE_DEVICES=0,1, NCCL_P2P_DISABLE=1, AMD_SERIALIZE_KERNEL=1,
VLLM_ROCM_USE_AITER=0, HIP_LAUNCH_BLOCKING=1.

## Result

Server READY in 35 seconds. /v1/chat/completions returns valid
response. Clean shutdown of both workers. Status: WORKING.

## Upstream context

vLLM issue #40980 (filed 2026-04-27) reports TP=2 deadlock on dual
R9700 with vLLM 0.19.1 / 0.19.2rc1 and PyTorch 2.11.0 built from
source. The configuration here (vLLM 0.19.0 bundled wheel + bundled
PyTorch 2.10.0) does not reproduce the deadlock. No investigation of
the functional difference is included here — this is an empirical
working-snapshot record only.

## Reproducibility note

vLLM in ~/venvs/vllm is pinned at 0.19.0+rocm721. Do not pip-upgrade
vLLM without a deliberate retest.

## Files

- 2026-05-02-tp2-qwen25-7b.log — full server log
