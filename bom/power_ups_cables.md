# Power & UPS Infrastructure

## Topology

```
Wall outlet 230V → UPS → workstation PSU
                 → UPS → power strip (monitor, router, peripherals)
```

## UPS

ARMAC OFFICE ON-LINE PF1 2000VA DUST-FREE

- 2000VA / 2000W, PF = 1.0
- Online (VFI – double conversion), 0 ms transfer
- 72V battery pack (6×12V), 8× IEC C13 outputs
- Default mode: INV (inverter)

## Cables

| Cable                       | Role                          |
|-----------------------------|-------------------------------|
| IEC C13 → C20, 16A          | UPS output → workstation PSU  |
| InLine 16659F (C14 ↔ C19)   | panel interconnect            |
| Techly 6-socket strip        | monitor & peripherals         |

Workstation PSU connected **directly to UPS** — no intermediate adapters.

## Load

- Idle: ~25% UPS capacity
- AI workload target: 55–65%

## Validation

Tested under load. Voltage regulation confirmed (240V in → 230V out). Status: **PASS**
