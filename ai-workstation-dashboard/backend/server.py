#!/usr/bin/env python3
"""
AI Workstation Dashboard — Backend
───────────────────────────────────
FastAPI server collecting real system metrics via psutil + rocm-smi,
streamed to React frontend over WebSocket.

Usage:
    python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import re
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

TICK_INTERVAL = 1.0


@dataclass
class GpuInfo:
    index: int
    name: str = "AMD GPU"
    use_percent: Optional[int] = None
    temp_c: Optional[float] = None
    vram_used_b: Optional[int] = None
    vram_total_b: Optional[int] = None
    gpu_type: str = "discrete"


def run_cmd(cmd: list[str], timeout: int = 3) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def find_rocm_smi() -> str:
    for p in ["rocm-smi", "/opt/rocm/bin/rocm-smi"]:
        if run_cmd([p, "--showuse"]):
            return p
    return "rocm-smi"


ROCM_SMI = find_rocm_smi()


# ── Static system info (cached) ─────────────────────────────────────────────

def _get_static():
    os_name = ""
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                os_name = line.split("=", 1)[1].strip('"')
    except Exception:
        pass

    kernel = run_cmd(["uname", "-r"])

    ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    cpu_name = "Unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    freq = psutil.cpu_freq()
    cpu_freq = round(freq.max / 1000, 2) if freq and freq.max else 0.0

    disk_fs = ""
    for part in psutil.disk_partitions(all=False):
        if part.mountpoint == "/":
            disk_fs = part.fstype
            break

    # RAM modules — try multiple methods
    ram_modules = []
    dmi_out = ""
    # Try without sudo first, then with sudo
    for cmd in [["dmidecode", "-t", "memory"], ["sudo", "-n", "dmidecode", "-t", "memory"]]:
        dmi_out = run_cmd(cmd)
        if dmi_out and "Size:" in dmi_out:
            break

    if dmi_out:
        current = {}
        for line in dmi_out.splitlines():
            line = line.strip()
            if line.startswith("Size:") and "No Module" not in line:
                current["size"] = line.split(":", 1)[1].strip()
            elif line.startswith("Type:") and "Unknown" not in line and "Correction" not in line:
                current["type"] = line.split(":", 1)[1].strip()
            elif line.startswith("Configured Memory Speed:") and "Unknown" not in line:
                current["speed"] = line.split(":", 1)[1].strip()
            elif line.startswith("Manufacturer:") and "Not Specified" not in line:
                current["manufacturer"] = line.split(":", 1)[1].strip()
            elif line.startswith("Part Number:") and "Not Specified" not in line:
                current["part_number"] = line.split(":", 1)[1].strip()
            elif line == "" and current.get("size"):
                ram_modules.append(current)
                current = {}
        if current.get("size"):
            ram_modules.append(current)

    # NVMe / SATA disks via lsblk
    disks = []
    lsblk_out = run_cmd(["lsblk", "-d", "-o", "NAME,SIZE,MODEL,TRAN", "--noheadings"])
    for line in lsblk_out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            name = parts[0]
            if name.startswith("loop") or name.startswith("ram"):
                continue
            size = parts[1]
            # Model can have spaces, TRAN is last token (nvme/sata/usb)
            tran = parts[-1] if parts[-1] in ("nvme", "sata", "usb", "ata") else ""
            if tran:
                model = " ".join(parts[2:-1])
            else:
                model = " ".join(parts[2:])
            disks.append({"name": name, "size": size, "model": model, "tran": tran})

    return {
        "hostname": socket.gethostname(),
        "os_name": os_name,
        "kernel": kernel,
        "ip": ip,
        "cpu_name": cpu_name,
        "cpu_cores": psutil.cpu_count(logical=True),
        "cpu_freq_ghz": cpu_freq,
        "disk_fs": disk_fs,
        "ram_modules": ram_modules,
        "disks": disks,
    }


STATIC = _get_static()


# ── GPU parsing ──────────────────────────────────────────────────────────────

def parse_gpus_json() -> list[GpuInfo]:
    raw = run_cmd([ROCM_SMI, "--json", "--showuse", "--showtemp",
                   "--showmeminfo", "vram", "--showproductname"])
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    gpus = []
    for key, val in data.items():
        m = re.search(r"card(\d+)", key, re.IGNORECASE)
        if not m or not isinstance(val, dict):
            continue
        gpu = GpuInfo(index=int(m.group(1)))

        for fn in ("Card Series", "Card series", "Card Model", "Card model", "Product Name"):
            if fn in val:
                gpu.name = str(val[fn]).strip()
                break
        for fn in ("GPU use (%)", "GPU Activity"):
            if fn in val:
                try: gpu.use_percent = int(float(val[fn]))
                except: pass
                break
        for fn in ("Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)", "Temperature"):
            if fn in val:
                try: gpu.temp_c = float(val[fn])
                except: pass
                break
        try: gpu.vram_total_b = int(val.get("VRAM Total Memory (B)", 0)) or None
        except: pass
        try: gpu.vram_used_b = int(val.get("VRAM Total Used Memory (B)", 0)) or None
        except: pass

        if "Graphics" in gpu.name or (gpu.vram_total_b and gpu.vram_total_b < 4 * 1024**3):
            gpu.gpu_type = "integrated"
        gpus.append(gpu)

    gpus.sort(key=lambda g: g.index)
    return gpus


def parse_gpus_text() -> list[GpuInfo]:
    gd: dict[int, GpuInfo] = {}
    out = {
        "use": run_cmd([ROCM_SMI, "--showuse"]),
        "vram": run_cmd([ROCM_SMI, "--showmeminfo", "vram"]),
        "temp": run_cmd([ROCM_SMI, "--showtemp"]),
        "name": run_cmd([ROCM_SMI, "--showproductname"]),
    }
    for line in out["use"].splitlines():
        m = re.search(r"GPU\[(\d+)\].*?(\d+)\s*%", line)
        if m:
            idx = int(m.group(1))
            gd.setdefault(idx, GpuInfo(index=idx)).use_percent = int(m.group(2))
    for line in out["temp"].splitlines():
        if "emp" not in line: continue
        m = re.search(r"GPU\[(\d+)\].*?(\d+(?:\.\d+)?)", line)
        if m:
            idx = int(m.group(1))
            gd.setdefault(idx, GpuInfo(index=idx)).temp_c = float(m.group(2))
    for line in out["name"].splitlines():
        m = re.search(r"GPU\[(\d+)\].*?:\s*(.+)$", line)
        if m:
            idx = int(m.group(1))
            g = gd.setdefault(idx, GpuInfo(index=idx))
            g.name = m.group(2).strip()
            if "Graphics" in g.name: g.gpu_type = "integrated"
    for line in out["vram"].splitlines():
        t = re.search(r"GPU\[(\d+)\].*Total(?! Used).*?:\s*(\d+)", line)
        u = re.search(r"GPU\[(\d+)\].*Used.*?:\s*(\d+)", line)
        if t:
            idx = int(t.group(1))
            gd.setdefault(idx, GpuInfo(index=idx)).vram_total_b = int(t.group(2))
        if u:
            idx = int(u.group(1))
            gd.setdefault(idx, GpuInfo(index=idx)).vram_used_b = int(u.group(2))
    return [gd[k] for k in sorted(gd)]


def get_gpus() -> list[GpuInfo]:
    return parse_gpus_json() or parse_gpus_text()


# ── Metrics snapshot ─────────────────────────────────────────────────────────

def get_cpu_temp() -> Optional[float]:
    try:
        temps = psutil.sensors_temperatures()
        for chip in ("k10temp", "zenpower", "coretemp"):
            if chip in temps:
                for r in temps[chip]:
                    if r.label in ("Tctl", "Tdie", ""):
                        return r.current
                if temps[chip]:
                    return temps[chip][0].current
    except Exception:
        pass
    return None


def get_top_procs(limit: int = 12) -> list[dict]:
    rows = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            cpu = p.cpu_percent(interval=None)
            mem = p.memory_info().rss
            rows.append({"pid": p.pid, "name": p.info.get("name") or "?",
                         "cpu": round(cpu, 1), "mem_mib": round(mem / (1024**2), 1)})
        except Exception:
            continue
    rows.sort(key=lambda x: (x["cpu"], x["mem_mib"]), reverse=True)
    return rows[:limit]


def build_snapshot() -> dict:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    l1, l5, l15 = psutil.getloadavg()
    boot = psutil.boot_time()
    up_sec = int(time.time() - boot)
    d, rem = divmod(up_sec, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    up_str = f"{d}d {h}h {m}m" if d else f"{h}h {m}m"
    dk = psutil.disk_usage("/")

    # Per-partition usage for real disks (skip snap/loop/tmpfs)
    disk_partitions = []
    seen_devs = set()
    SKIP_FS = {"squashfs", "tmpfs", "devtmpfs", "overlay", "efivarfs"}
    for part in psutil.disk_partitions(all=True):
        if part.device in seen_devs:
            continue
        if part.fstype in SKIP_FS:
            continue
        if part.device.startswith("/dev/loop"):
            continue
        if part.mountpoint.startswith("/snap"):
            continue
        if part.mountpoint == "/boot/efi":
            continue
        seen_devs.add(part.device)
        try:
            usage = psutil.disk_usage(part.mountpoint)
            if usage.total < 100 * 1024**2:  # skip tiny partitions (<100MB)
                continue
            disk_partitions.append({
                "device": part.device,
                "mount": part.mountpoint,
                "fs": part.fstype,
                "used_gib": round(usage.used / (1024**3), 2),
                "total_gib": round(usage.total / (1024**3), 2),
                "percent": usage.percent,
            })
        except PermissionError:
            continue

    return {
        **STATIC,
        "timestamp": datetime.now().isoformat(),
        "uptime_sec": up_sec,
        "uptime_str": up_str,
        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "cpu_temp": get_cpu_temp(),
        "load": [round(l1, 2), round(l5, 2), round(l15, 2)],
        "ram_used_gib": round(vm.used / (1024**3), 2),
        "ram_total_gib": round(vm.total / (1024**3), 2),
        "ram_percent": vm.percent,
        "swap_used_gib": round(sw.used / (1024**3), 2),
        "swap_total_gib": round(sw.total / (1024**3), 2),
        "disk_mount": "/",
        "disk_used_gib": round(dk.used / (1024**3), 2),
        "disk_total_gib": round(dk.total / (1024**3), 2),
        "disk_percent": dk.percent,
        "disk_partitions": disk_partitions,
        "gpus": [asdict(g) for g in get_gpus()],
        "processes": get_top_procs(),
    }


# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Workstation Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND = Path(__file__).parent / "frontend"

if (FRONTEND / "index.html").exists():
    @app.get("/")
    async def root():
        return FileResponse(FRONTEND / "index.html")
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/api/snapshot")
async def snapshot():
    return build_snapshot()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(build_snapshot())
            await asyncio.sleep(TICK_INTERVAL)
    except (WebSocketDisconnect, Exception):
        pass


# Warm up
psutil.cpu_percent(interval=None)
for p in psutil.process_iter():
    try: p.cpu_percent(interval=None)
    except: pass


if __name__ == "__main__":
    import uvicorn
    print("┌──────────────────────────────────────────────┐")
    print("│  AI Workstation Dashboard                    │")
    print("│  http://0.0.0.0:8000       ws://…:8000/ws    │")
    print("└──────────────────────────────────────────────┘")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
