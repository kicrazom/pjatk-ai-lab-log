# AI Workstation – Hardware & Infrastructure (BOM)

## Overview
This directory contains the full hardware specification, power infrastructure, and low-level system integrations of the AI workstation.

It documents real-world configuration, troubleshooting, and validated solutions.

---

## Hardware
Full system specification (CPU, GPU, RAM, storage).

## CPU
AMD Ryzen 9 9950X3D

## GPU
2× GIGABYTE Radeon AI PRO R9700 AI TOP 32G

## Motherboard
ASUS ProArt X870E

## RAM
Corsair 96GB (2x48GB) DDR5 6000 CL30

## Storage
GOODRAM 4TB M.2 PCIe Gen4 NVMe PX700 - primary disk
GOODRAM 1TB M.2 PCIe Gen4 NVMe PX700 - secondary disk

## PSU
FSP PTM PRO 1650W 80 Plus Platinum ATX 3.1
Input: IEC C20

## Case
ASUS ProArt PA602 Wood Metal PWM Black

---

## Power & UPS
Power delivery, UPS setup, and Linux integration.

power_ups_cables.md  
nut-ups_armac-integration.md  

---

## PCIe / System topology
GPU layout and PCIe bandwidth analysis.

pci-topology.md

---

## Logbook / Real-world debugging
Integration issues, fixes, and validated configurations.

---

## Key notes

- UPS (ARMAC PF1) works via **serial-over-USB (CH340)**  
- Not compatible with `usbhid-ups`  
- Fully operational with `nutdrv_qx`  
- No kernel downgrade required (tested on 6.17)  

---

## Status

✔ Production-ready hardware stack  
✔ Working UPS monitoring (NUT)  
✔ Power failure detection validated  





