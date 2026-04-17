# pjatk-ai-lab-log

Engineering log of a local AI / LLM workstation build — hardware, power
infrastructure, ROCm environment, and benchmarks.

> **Security note:** This repository documents a workstation build and
> experiments. Operational details such as hostnames, domains, network
> addresses, and secrets are intentionally omitted.

## Hardware

| Component   | Model                                               |
|-------------|-----------------------------------------------------|
| CPU         | AMD Ryzen 9 9950X3D                                 |
| GPU         | 2x GIGABYTE Radeon AI PRO R9700 AI TOP 32G (gfx1201)|
| Motherboard | ASUS ProArt X870E                                   |
| RAM         | Corsair 96 GB DDR5-6000 CL30 (2x48 GB)              |
| Storage     | GOODRAM PX700 4 TB + 1 TB NVMe                      |
| PSU         | FSP PTM PRO 1650 W 80+ Platinum (ATX 3.1)           |
| Cooling     | Noctua NH-D15 G2 + 2x NF-A14                        |
| Case        | ASUS ProArt PA602                                   |
| UPS         | ARMAC 2000 Online (NUT-monitored)                   |
| OS          | Kubuntu 24.04, kernel 6.17, ROCm 7.2.1              |

See [`bom/readme.md`](bom/readme.md) for the full bill of materials,
power topology, and PCIe layout.

## Repository layout

| Directory                         | Contents                                               |
|-----------------------------------|--------------------------------------------------------|
| [`bom/`](bom/)                    | Hardware BOM, power infrastructure, PCIe topology, UPS |
| [`benchmarks/`](benchmarks/)      | vLLM / ROCm / PyTorch benchmarks and findings          |
| [`ai-workstation-dashboard/`](ai-workstation-dashboard/) | Real-time CPU/GPU monitoring dashboard (FastAPI + psutil + rocm-smi) |
| [`logbook/`](logbook/)            | Dated engineering notes from the build process         |

## Highlights

### vLLM throughput scaling (April 2026)

Full concurrency sweep of Qwen 2.5 7B Instruct on vLLM 0.19 with both
tensor-parallel configurations, including per-run thermal instrumentation.

![Scaling curve](benchmarks/results/scaling_curve.png)

**Key findings:**

- **TP=1 plateau: 3870 tok/s** (saturates from N=500, std dev 0.3%)
- **TP=2 plateau: 2940 tok/s** (24% PCIe all_reduce tax vs TP=1)
- **TP=2 only wins at N=50** (+9%), loses at every higher concurrency
- Thermal asymmetry between GPU 0 and GPU 1 (5C delta) — same workload,
  airflow-dependent

See [`benchmarks/results/README.md`](benchmarks/results/README.md) for the
full writeup, all 14 measurement points, thermal observations, and planned
next experiments (Qwen 72B AWQ, Polish-language models, long-form decode).

## Focus areas

- **Local LLM inference** — vLLM, llama.cpp on AMD ROCm
- **GPU benchmarking** — throughput, latency, thermal characterization
- **Workstation infrastructure** — power, UPS integration, PCIe topology
- **System monitoring** — custom dashboard for multi-GPU health

## Status

- [x] Hardware assembled and validated (2026-02-28 / 2026-03-04)
- [x] ROCm 7.2.1 + PyTorch validation benchmark (2026-03-08)
- [x] Second NVMe scratch disk integrated (2026-03-09)
- [x] UPS monitoring operational (NUT + nutdrv_qx, 2026-03-30)
- [x] AI workstation dashboard deployed (systemd autostart)
- [x] vLLM 0.19 + ROCm 7.2.1 installation validated (2026-04-17)
- [x] Plan A: TP=1 vs TP=2 scaling sweep with thermal instrumentation
- [ ] Plan B: Qwen 2.5 72B AWQ — TP=2 mandatory test
- [ ] Polish-language model benchmarks (Bielik, PLLuM)
- [ ] Long-form decode sweep (max_tokens=1024)
- [ ] NUMA pinning A/B on 9950X3D dual-CCD

## License

[GPL-3.0](LICENSE) — see file for full terms.
