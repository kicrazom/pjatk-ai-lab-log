# AI Workstation – BOM & Infrastructure

This repository is a lab logbook documenting the build, configuration, and ongoing development of an AMD-based AI workstation used for local ML inference, research tooling, and system automation.

The `bom/` directory contains the hardware bill of materials, power infrastructure, and low-level system topology.

---

## Hardware BOM

| Component   | Model                                              |
|-------------|----------------------------------------------------|
| CPU         | AMD Ryzen 9 9950X3D                                |
| GPU         | 2× GIGABYTE Radeon AI PRO R9700 AI TOP 32G        |
| Motherboard | ASUS ProArt X870E                                  |
| RAM         | Corsair 96 GB (2×48 GB) DDR5 6000 CL30            |
| Storage     | GOODRAM PX700 4 TB NVMe (primary) + 1 TB (secondary) |
| PSU         | FSP PTM PRO 1650 W 80+ Platinum ATX 3.1 (IEC C20) |
| Cooling     | Noctua NH-D15 G2 (dual-tower, 2× 140 mm)          |
| Case fans   | 2× Noctua NF-A14 140 mm (stacked)                 |
| Case        | ASUS ProArt PA602                                  |
| OS          | Kubuntu 24.04, kernel 6.17                         |

---

## Files in this directory

| File                             | Contents                                      |
|----------------------------------|-----------------------------------------------|
| `power_ups_cables.md`            | Power topology, UPS specs, cable map          |
| `nut-ups_armac-integration.md`   | NUT driver config for ARMAC PF1 (serial/CH340)|
| `pci-topology.md`                | Full PCIe device map from `lspci`             |

---

## Status

- [x] Hardware assembled and validated
- [x] UPS monitoring operational (NUT + `nutdrv_qx`)
- [x] Power failure detection tested
- [x] PCIe link width verification (LnkSta)
- [x] IOMMU group confirmation for both GPUs
