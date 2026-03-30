# ARMAC PF1 UPS -- NUT Integration Guide

## Model

ARMAC OFFICE ON-LINE PF1 2000VA / 2000W

## Architecture

Online UPS (double conversion)

------------------------------------------------------------------------

## Communication layer

USB → CH340 → Serial (RS232 emulation)

NOT: - USB HID - NOT compatible with usbhid-ups

------------------------------------------------------------------------

## Linux detection

``` bash
lsusb
```

    1a86:7523 QinHeng CH340 serial converter

``` bash
ls /dev/ttyUSB*
```

    /dev/ttyUSB0

------------------------------------------------------------------------

## Required driver

    nutdrv_qx

Protocol auto-detected:

    Megatec 0.07

------------------------------------------------------------------------

## Working configuration

``` ini
[armac]
    driver = nutdrv_qx
    port = /dev/ttyUSB0
```

------------------------------------------------------------------------

## Required permissions

``` bash
sudo usermod -aG dialout nut
```

------------------------------------------------------------------------

## What NOT to do

❌ do NOT use: - usbhid-ups - subdriver = armac (breaks config) - vendor
software (kernel limited)

------------------------------------------------------------------------

## Known limitations

-   battery.charge unreliable
-   battery.voltage unreliable
-   runtime estimation unavailable

✔ reliable: - ups.status - voltage - load

------------------------------------------------------------------------

## Tested environment

-   Kubuntu 24.04
-   Kernel 6.17
-   NUT 2.8.1

------------------------------------------------------------------------

## Status

✔ Fully working\
✔ Real-time monitoring\
✔ Power loss detection\
✔ Ready for shutdown automation
