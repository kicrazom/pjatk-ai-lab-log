# Power & UPS Infrastructure

## Power cables
- IEC C13 → C20 (16A) for PSU 1650W
- InLine 16659F (C14 panel ↔ C19 panel)
- Techly 6-socket power strip (monitor & peripherals)

## Topology
Wall outlet 230V → (optional energy meter / Shelly Plug) → UPS → workstation PSU → workstation

## UPS
ARMAC OFFICE ON-LINE PF1 2000VA DUST-FREE

Key parameters:
- 2000VA / 2000W
- Online topology (VFI – double conversion)
- PF = 1.0
- 72V battery pack (6×12V)
- 8× IEC C13 outputs
- 0 ms transfer time

## Output wiring to workstation
- UPS output: IEC C13
- Workstation PSU input: IEC C20 (FSP/Fortron PTM PRO 1650W)
- Power cable: IEC C13 → IEC C20, 16A, heavy-duty

The workstation PSU is connected **directly to the UPS output** without intermediate adapters.

## Peripheral wiring
Low-power devices are powered through a secondary strip:

- Monitor
- Router/networking devices
- Small peripherals

High-load workstation PSU is connected **directly to the UPS**.

## UPS operating mode
Default operation mode:

- **INV (Inverter / Double Conversion)**

This ensures:

- stable 230V output
- isolation from grid fluctuations
- protection against voltage spikes

Bypass mode is used only for configuration.

## Expected display behaviour

### BYPASS
Output voltage follows grid input:
Input ~239–240V
Output ~239–240V

### INV (normal operation)
UPS regulates output voltage:
Input ~239–240V
Output ~230V stable


A faint high-frequency inverter noise may be audible in INV mode.  
This is expected behaviour for double-conversion UPS systems.


## Validation test

The workstation was powered from UPS and tested under load.

Results:

- System boots and runs normally on UPS power.
- Voltage regulation confirmed (240V input → 230V output).
- No instability observed.

Status: **PASS**

## Monitoring (planned / optional)

Energy monitoring via:

- Shelly Plug S Gen3
- or external wattmeter

Measurement point:
Wall outlet → Energy meter → UPS

This allows measurement of:

- total system power draw
- UPS conversion losses
- AI workload energy consumption

## Notes

UPS idle load indicator: **~25%**
Target UPS load during AI workloads: **55–65%**

This range provides optimal efficiency and thermal behaviour for online UPS operation.


