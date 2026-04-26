# Hardware context

Specifics of the workstation that surface in benchmark configuration
or thermal panels. Full BOM lives in [`bom/`](../../bom/).

## Three GPUs, two used for compute

| Device | Role | Arch | gcnArchName | VRAM | Notes |
|---|---|---|---|---|---|
| GPU 0 | Compute | RDNA 4 / Navi 48 | `gfx1201` | 32 GB | Gigabyte R9700 AI TOP |
| GPU 1 | Compute | RDNA 4 / Navi 48 | `gfx1201` | 32 GB | Gigabyte R9700 AI TOP |
| iGPU | **Display only** | Raphael | `gfx1036` | shared | Ryzen 9 9950X3D internal graphics |

The integrated `gfx1036` is masked from compute via `~/.bashrc`:

```bash
export ROCR_VISIBLE_DEVICES=0,1
```

Without it, ROCm enumerates all three devices and frameworks scanning
`device 0..N` (vLLM included) can land tensors on the iGPU and crash
or silently produce garbage. The mask hides the iGPU from compute
discovery while leaving display untouched. Thermal panels still sample
the iGPU via `rocm-smi` — periodic 80–99% spikes are the dashboard
browser tab and WebSocket pushes, not benchmark traffic. **Recurring
point of confusion**; keep `ROCR_VISIBLE_DEVICES=0,1` set.

## Architecture-name pitfall

R9700 reports `gcnArchName = gfx1201` — **not** `gfx1101` (RDNA 3 /
Navi 31, a different chip). Confirmed 2026-04-17 via
`torch.cuda.get_device_properties(0).gcnArchName`. ROCm tutorials older
than late 2025 frequently target `gfx1101` and do not apply without
verification. R9700 support landed only in ROCm 7.2.x.

## CPU — dual CCD with asymmetric L3

Ryzen 9 9950X3D, 16C/32T, two CCDs on one package, **single NUMA
node**. CCD 0: 32 MB L3 (CPUs 0–7 + SMT 16–23). CCD 1: **96 MB L3 with
3D V-Cache** (CPUs 8–15 + SMT 24–31). Cross-CCD access traverses
Infinity Fabric within the single node, not a NUMA hop. GPUs report
`NUMA: -1`, meaning "no preference" on single-node systems. CCD pinning
produced no measurable throughput change on GPU-bound AWQ workloads —
calibration data in [`README.md`](README.md).
