#!/usr/bin/env python3
"""
download_progress.py
Live monitor pobierania HuggingFace do ~/models/.

Zrodlo prawdy: HF API (endpoint /api/models/<repo>/tree/main).
Repo_id czytany z download_models.sh (mapping katalog -> repo).

Pokazuje: aktualny model, aktywny plik, % postepu, predkosc, ETA.
Refresh co 2s. Wymaga tylko biblioteki standardowej.
"""

from __future__ import annotations

import json
import re
from collections import deque
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REFRESH_INTERVAL_SEC: float = 2.0
MODELS_DIR: Path = Path.home() / "models"
SCRIPT_PATH: Path = Path.home() / "pjatk-ai-lab-log/scripts/download_models.sh"
BAR_WIDTH: int = 40
API_TIMEOUT_SEC: float = 10.0


# --- Data model -------------------------------------------------------------

@dataclass
class Snapshot:
    """Stan pobierania modelu w jednym momencie."""
    model_dir: Path
    downloaded: int
    expected: int
    active_file: str
    at_time: float


# --- Mapping: nazwa katalogu -> repo_id (ze skryptu) ------------------------

def load_subdir_to_repo(script_path: Path) -> dict[str, str]:
    """
    Parsuje wpisy MODELS z download_models.sh.
    Format: "repo_id|subdir|rozmiar|komentarz"
    Zwraca dict w kolejnosci wystepowania (Python 3.7+ zachowuje insertion order).
    """
    pattern = re.compile(r'"([^"|]+)\|([^"|]+)\|[^"|]+\|[^"]+"')
    mapping: dict[str, str] = {}
    try:
        for match in pattern.finditer(script_path.read_text(encoding="utf-8")):
            repo_id, subdir = match.group(1), match.group(2)
            # Komentarze w skrypcie pasuja do tego samego wzorca co wpisy
            # (np. "repo_id|local_subdir|rozmiar|komentarz")
            # Rzeczywiste repo_id zawiera ukosnik "org/nazwa" — tym filtrujemy
            if "/" not in repo_id:
                continue
            mapping[subdir] = repo_id
    except OSError:
        return {}
    return mapping


def get_download_order(subdir_to_repo: dict[str, str]) -> list[str]:
    """Lista katalogow docelowych w kolejnosci pobierania (z dict keys)."""
    return list(subdir_to_repo.keys())


# --- HF API: lista plikow z rozmiarami --------------------------------------

def fetch_repo_files(repo_id: str) -> dict[str, int]:
    """
    Pobiera {nazwa_pliku: rozmiar} z HF API tree endpoint.
    Zwraca pusty dict przy bledzie sieci.
    """
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        with urllib.request.urlopen(url, timeout=API_TIMEOUT_SEC) as response:
            data = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return {}
    return {
        entry["path"]: entry["size"]
        for entry in data
        if isinstance(entry, dict) and entry.get("type") == "file"
    }


# --- Discovery: co sie aktualnie pobiera ------------------------------------

def find_active_model(models_dir: Path) -> Optional[Path]:
    """Katalog modelu z najnowszym .incomplete = aktywnie pobierany."""
    incompletes = list(models_dir.glob("*/.cache/huggingface/download/*.incomplete"))
    if not incompletes:
        return None
    latest = max(incompletes, key=lambda p: p.stat().st_mtime)
    return latest.parents[3]


# --- Rozmiary: ile pobrano ---------------------------------------------------

def sum_sizes(paths) -> int:
    """Suma rozmiarow plikow z iterowalnego zrodla."""
    total = 0
    for path in paths:
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def count_downloaded(model_dir: Path) -> int:
    """Bajty: pliki finalne w katalogu modelu + aktywne .incomplete."""
    download_dir = model_dir / ".cache" / "huggingface" / "download"
    return sum_sizes(model_dir.iterdir()) + sum_sizes(download_dir.glob("*.incomplete"))


# --- Identyfikacja aktywnego pliku ------------------------------------------

def find_active_filename(model_dir: Path, repo_files: dict[str, int]) -> str:
    """
    Nazwa pliku aktualnie pobieranego.
    Aktywny = istnieje w repo_files (z API), ale nie w katalogu modelu.
    Sortujemy po rozmiarze malejaco — najwiekszy niepobrany to zwykle ten,
    ktory hf_transfer aktualnie ciagnie.
    """
    for filename, _size in sorted(repo_files.items(), key=lambda x: -x[1]):
        if not (model_dir / filename).exists():
            return filename
    return "<finalizing>"


# --- Status kolejki: co gotowe, co w toku, co czeka ------------------------

def classify_model_status(
    subdir: str,
    models_dir: Path,
    subdir_to_repo: dict[str, str],
    api_cache: dict[str, dict[str, int]],
) -> str:
    """
    Zwraca 'complete' / 'downloading' / 'queued'.

    complete  = katalog istnieje, ma >=1 safetensors, zero .incomplete
    downloading = katalog ma aktywne .incomplete
    queued = katalog nie istnieje lub pusty
    """
    target = models_dir / subdir
    if not target.exists():
        return "queued"
    download_dir = target / ".cache" / "huggingface" / "download"
    has_incomplete = download_dir.exists() and any(download_dir.glob("*.incomplete"))
    if has_incomplete:
        return "downloading"
    safetensors_count = sum(1 for _ in target.glob("*.safetensors"))
    if safetensors_count > 0:
        return "complete"
    return "queued"


def render_queue(
    current_subdir: str,
    ordered_subdirs: list[str],
    models_dir: Path,
    subdir_to_repo: dict[str, str],
    api_cache: dict[str, dict[str, int]],
) -> None:
    """Wyswietla liste wszystkich modeli z kolejki ze statusami."""
    print()
    print(" Queue status:")
    print(" " + "-" * 62)
    for subdir in ordered_subdirs:
        status = classify_model_status(subdir, models_dir, subdir_to_repo, api_cache)
        if status == "complete":
            marker = "[DONE]    "
            label = "transfer complete"
        elif status == "downloading" or subdir == current_subdir:
            marker = "[ACTIVE]  "
            label = "downloading now"
        else:
            marker = "[QUEUED]  "
            label = "in queue"
        print(f"  {marker}{subdir:<35s} {label}")


# --- Formatowanie -----------------------------------------------------------

def format_bytes(n: float) -> str:
    """1073741824 -> ' 1.00 GB'"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:7.2f} {unit}"
        n /= 1024
    return f"{n:7.2f} PB"


def format_duration(seconds: float) -> str:
    """3725 -> '01:02:05'. Nierealne -> placeholder."""
    if seconds < 0 or seconds > 99 * 3600:
        return "--:--:--"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_bar(current: int, total: int) -> str:
    """Pasek ASCII z procentem."""
    if total <= 0:
        return "[" + "?" * BAR_WIDTH + "]  (rozmiar nieznany)"
    ratio = min(current / total, 1.0)
    filled = int(ratio * BAR_WIDTH)
    bar = "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)
    return f"[{bar}] {ratio * 100:5.1f}%"


# --- Logika pomiaru predkosci -----------------------------------------------

def compute_rate_bps(history: deque[Snapshot]) -> float:
    """
    Srednia predkosc z okna czasowego w history (bajty/s).

    hf_transfer pobiera 100 MB chunki i flushuje je atomicznie co ~60s,
    wiec pomiar instant (delta miedzy dwoma snapshotami) oscyluje
    miedzy 0 a 200 MB/s. Srednia z 30s wygladza te skoki.
    """
    if len(history) < 2:
        return 0.0
    oldest, newest = history[0], history[-1]
    if oldest.model_dir != newest.model_dir:
        return 0.0
    delta_t = newest.at_time - oldest.at_time
    delta_bytes = newest.downloaded - oldest.downloaded
    if delta_t <= 0 or delta_bytes < 0:
        return 0.0
    return delta_bytes / delta_t


def compute_eta_sec(snap: Snapshot, rate_bps: float) -> Optional[float]:
    """Sekundy do konca. None jesli nie da sie obliczyc."""
    if snap.expected <= 0 or rate_bps <= 0:
        return None
    return (snap.expected - snap.downloaded) / rate_bps


# --- Render -----------------------------------------------------------------

def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")


def render_idle() -> None:
    clear_screen()
    print("=" * 64)
    print(" HF Download Monitor")
    print("=" * 64)
    print("\n  Brak aktywnych pobran w ~/models/")
    print(f"\n  (Ctrl+C = exit, odswiezanie co {REFRESH_INTERVAL_SEC:.0f}s)")
    sys.stdout.flush()


def render_active(
    snap: Snapshot,
    rate_bps: float,
    repo_id: str,
    ordered_subdirs: list[str],
    models_dir: Path,
    subdir_to_repo: dict[str, str],
    api_cache: dict[str, dict[str, int]],
) -> None:
    clear_screen()
    print("=" * 64)
    print(" HF Download Monitor - Ctrl+C by wyjsc")
    print("=" * 64)
    print(f" Model:      {snap.model_dir.name}")
    print(f" Repo:       {repo_id}")
    print(f" Plik:       {snap.active_file}")
    print()
    print(" " + render_bar(snap.downloaded, snap.expected))
    print()
    print(f" Pobrano:    {format_bytes(snap.downloaded)}  /  {format_bytes(snap.expected)}")
    print(f" Zostalo:    {format_bytes(max(snap.expected - snap.downloaded, 0))}")
    print(f" Predkosc:   {format_bytes(rate_bps)}/s")
    eta = compute_eta_sec(snap, rate_bps)
    print(f" ETA:        {format_duration(eta) if eta else '--:--:--'}")
    render_queue(
        snap.model_dir.name, ordered_subdirs, models_dir, subdir_to_repo, api_cache,
    )
    print(f"\n (odswiezanie co {REFRESH_INTERVAL_SEC:.0f}s)")
    sys.stdout.flush()


# --- Main loop --------------------------------------------------------------

def take_snapshot(model_dir: Path, repo_files: dict[str, int]) -> Snapshot:
    """Jedno zdjecie stanu pobierania modelu."""
    return Snapshot(
        model_dir=model_dir,
        downloaded=count_downloaded(model_dir),
        expected=sum(repo_files.values()),
        active_file=find_active_filename(model_dir, repo_files),
        at_time=time.time(),
    )


def monitor_loop(
    models_dir: Path,
    subdir_to_repo: dict[str, str],
    interval: float,
) -> None:
    """Glowna petla z cache'em HF API i oknem predkosci 30s."""
    # maxlen=15 * 2s interval = 30s okno pomiarowe
    history: deque[Snapshot] = deque(maxlen=15)
    api_cache: dict[str, dict[str, int]] = {}
    ordered_subdirs = get_download_order(subdir_to_repo)

    while True:
        active = find_active_model(models_dir)
        if active is None:
            render_idle()
            history.clear()
            time.sleep(interval)
            continue

        subdir = active.name
        repo_id = subdir_to_repo.get(subdir, "<unknown>")

        # Fetch API raz na model, potem cache
        if repo_id != "<unknown>" and subdir not in api_cache:
            api_cache[subdir] = fetch_repo_files(repo_id)

        repo_files = api_cache.get(subdir, {})
        curr = take_snapshot(active, repo_files)

        # Reset okna przy zmianie modelu
        if history and history[-1].model_dir != curr.model_dir:
            history.clear()
        history.append(curr)

        rate = compute_rate_bps(history)
        render_active(
            curr, rate, repo_id, ordered_subdirs, models_dir, subdir_to_repo, api_cache,
        )
        time.sleep(interval)


def main() -> int:
    subdir_to_repo = load_subdir_to_repo(SCRIPT_PATH)
    if not subdir_to_repo:
        print(f"[!] Nie udalo sie wczytac mappingu z {SCRIPT_PATH}", file=sys.stderr)
        return 1
    try:
        monitor_loop(MODELS_DIR, subdir_to_repo, REFRESH_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n\nZakonczono.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
