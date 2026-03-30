# PCIe Topology

Source: `lspci` on Kubuntu 24.04

## Discrete GPUs

| BDF     | Device          | Path (root port → switch → endpoint)     |
|---------|-----------------|-------------------------------------------|
| 03:00.0 | AMD/ATI 7551    | 00:01.1 → 01:00.0 (upstream) → 02:00.0 (downstream) → **03:00.0** |
| 07:00.0 | AMD/ATI 7551    | 00:01.3 → 05:00.0 (upstream) → 06:00.0 (downstream) → **07:00.0** |

Both GPUs sit behind separate root ports (00:01.1 and 00:01.3) with individual Navi 10 XL PCIe switches.

## NVMe Storage

| BDF     | Controller                              | Root port |
|---------|-----------------------------------------|-----------|
| 04:00.0 | Phison PS5027-E27T (GOODRAM PX700)      | 00:01.2   |
| 08:00.0 | Phison PS5027-E27T (GOODRAM PX700)      | 00:01.4   |

## Networking

| BDF     | Device                                  | Path                          |
|---------|-----------------------------------------|-------------------------------|
| 0e:00.0 | MediaTek MT7922 WiFi 6E                 | chipset switch → 0d:05.0      |
| 0f:00.0 | Intel I226-V 2.5GbE                     | chipset switch → 0d:06.0      |
| 10:00.0 | Aquantia AQC113CS 10GbE                 | chipset switch → 0d:08.0      |

## Chipset & Peripherals

| BDF     | Device                                  | Notes                         |
|---------|-----------------------------------------|-------------------------------|
| 12:00.0 | AMD USB controller                      | chipset USB                   |
| 13:00.0 | AMD 600 Series SATA                     | chipset SATA                  |
| 14:00.0 | AMD USB controller                      | chipset USB                   |
| 15:00.0 | AMD 600 Series SATA                     | chipset SATA                  |
| 16:00.0 | ASMedia 2421 → 2423 hub                 | front-panel USB hub           |
| 7c:00.0 | ASMedia 2426 USB                        | ASMedia USB (rear?)           |
| 7d:00.0 | ASMedia 2425 USB                        | ASMedia USB                   |

## Integrated GPU & SoC

| BDF     | Device                                  | Notes                         |
|---------|-----------------------------------------|-------------------------------|
| 7e:00.0 | AMD/ATI 13c0 (Raphael iGPU)            | integrated RDNA 2             |
| 7e:00.1 | Rembrandt HD Audio                      | HDMI/DP audio (iGPU)         |
| 7e:00.2 | AMD PSP/CCP                             | platform security processor   |
| 7e:00.3 | AMD USB                                 | SoC USB                      |
| 7e:00.4 | AMD USB                                 | SoC USB                      |
| 7e:00.6 | AMD HD Audio                            | onboard audio codec           |
| 7f:00.0 | AMD USB                                 | SoC USB                      |

## Verification checklist

- [x] Both discrete GPUs on separate root ports (00:01.1, 00:01.3)
- [ ] Both GPUs at PCIe 4.0 x16 — run: `lspci -vv -s 03:00.0 | grep LnkSta` and `lspci -vv -s 07:00.0 | grep LnkSta`
- [ ] IOMMU groups confirmed separate — run: `for d in /sys/kernel/iommu_groups/*/devices/*; do echo "$(basename $(dirname $(dirname $d))): $(lspci -nns $(basename $d))"; done | grep 7551`
