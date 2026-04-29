# Phase 1 — Hardware Envelope Summary (Qwen 3.6 27B, 2× R9700)

**EMBARGO:** PUBLIC. Envelope data only — single-config measurements, no scaling sweep.

## Per-config results

Bold VRAM = within 2 GB of OOM (peak ≥ 30/32 GB). KV dtype `default` = same as model weights.

| Config | Quant | KV dtype | max_len | util | Status | Load [s] | Peak VRAM/GPU [GB] | KV tokens | Max conc. | Sanity tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen36_27b_bf16_max1024_util095` | bf16 | default | 1024 | 0.95 | OK | 60.4 | 28.66 | 28224 | 27.56 | 7.82 |
| `qwen36_27b_bf16_max1024_util095_kvfp8e4m3` | bf16 | fp8_e4m3 | 1024 | 0.95 | OK | 52.5 | 28.82 | 61152 | 59.72 | 1.48 |
| `qwen36_27b_bf16_max1024_util097` | bf16 | default | 1024 | 0.97 | OK | 50.4 | 29.85 | 67424 | 65.84 | 7.46 |
| `qwen36_27b_bf16_max2048_util095_kvfp8e4m3` | bf16 | fp8_e4m3 | 2048 | 0.95 | OK | 51.9 | 28.82 | 61152 | 29.86 | 1.53 |
| `qwen36_27b_bf16_max2048_util097` | bf16 | default | 2048 | 0.97 | OK | 51.4 | 29.32 | 49392 | 24.12 | 7.89 |
| `qwen36_27b_bf16_max512_util095` | bf16 | default | 512 | 0.95 | OK | 51.3 | 28.66 | 28224 | 55.12 | 4.61 |
| `qwen36_27b_bf16_max512_util097` | bf16 | default | 512 | 0.97 | OK | 50.9 | 29.32 | 49392 | 96.47 | 7.40 |
| `qwen36_27b_bf16_max768_util096` | bf16 | default | 768 | 0.96 | OK | 51.4 | 28.98 | 38416 | 50.02 | 7.29 |
| `qwen36_27b_fp8_max2048_util085` | fp8 | default | 2048 | 0.85 | OK | 58.6 | 24.83 | 246960 | 120.59 | 4.77 |
| `qwen36_27b_fp8_max4096_util090` | fp8 | default | 4096 | 0.90 | OK | 48.8 | 26.39 | 298704 | 72.93 | 5.34 |

## Phase 2 sweep recommendation

- **Best BF16 (max throughput):** `qwen36_27b_bf16_max2048_util097`
- **Largest-context BF16 loaded:** `qwen36_27b_bf16_max2048_util095_kvfp8e4m3`
- **Best FP8 (max throughput):** `qwen36_27b_fp8_max4096_util090`

Trade-off note: large `max_model_len` reduces `max_concurrency`, which caps Phase 2 throughput at high N. Pick the BF16 envelope matching the longest prompt+response budget actually expected in the sweep, not unconditionally the highest tok/s.
