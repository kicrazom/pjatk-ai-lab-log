#!/usr/bin/env bash
# =============================================================================
# download_models.sh
# Pobiera pełny zestaw benchmarkowy (12 modeli) do ~/models/
# Target: lokalny dysk NVMe, zgodny z protokołem badania (jedno źródło prawdy)
# Wymaga: pip install -U "huggingface_hub[hf_transfer]"
# =============================================================================
set -euo pipefail

# --- Konfiguracja -----------------------------------------------------------
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
export HF_HUB_ENABLE_HF_TRANSFER=1       # kluczowe: ~10x szybsze pobieranie
export HF_HUB_DISABLE_XET=1         # wymusza stary LFS CDN (EU), obejscie wolnego us-east-1 XET
export HF_HUB_DOWNLOAD_TIMEOUT=60        # timeout pojedynczej operacji (s)

# Uwaga licencyjna:
# - Bielik 11B v2.3 = Apache 2.0
# - PLLuM warianty "-nc-" = CC-BY-NC-4.0 (tylko research/UMB OK)
# - PLLuM warianty bez "-nc-" = Apache 2.0 (produkcyjne)

# --- Lista modeli -----------------------------------------------------------
# Format: "repo_id|local_subdir|rozmiar_approx|komentarz"
# Kolejność: od najmniejszych (warmup) do największych (finale)
declare -a MODELS=(
    # --- Warmup: małe modele, szybkie pobieranie, pierwsze testy -----------
    "speakleash/Bielik-11B-v2.3-Instruct-AWQ|bielik-11b-v23-awq|~6GB|Bielik AWQ - flagowy polski, gotowy TP=1"
    "CYFRAGOVPL/Llama-PLLuM-8B-instruct|llama-pllum-8b-instruct|~16GB|PLLuM na bazie Llama 3.1, BF16"
    "Qwen/Qwen2.5-7B-Instruct|qwen25-7b-instruct|~15GB|Qwen 7B FP16 (referencja TP=1)"

    # --- Midsize: 11-24 GB, TP=1 jeszcze się mieści -----------------------
    "speakleash/Bielik-11B-v2.3-Instruct|bielik-11b-v23|~22GB|Bielik FP16 - porównanie z AWQ"
    "CYFRAGOVPL/PLLuM-12B-chat|pllum-12b-chat|~24GB|PLLuM 12B BF16 (Mistral base)"
    "mistralai/Mistral-Nemo-Instruct-2407|mistral-nemo-instruct-2407|~24GB|Mistral-Nemo 12B BF16"

    # --- Large AWQ: TP=2 konieczne lub opłacalne --------------------------
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ|mixtral-8x7b-awq|~25GB|Mixtral MoE AWQ, TP=2"
    "Qwen/Qwen2.5-72B-Instruct-AWQ|qwen25-72b-awq|~40GB|Qwen 72B AWQ - flagowy TP=2 (cel dnia)"

    # --- Tryptyk PLLuM 70B: base vs instruct vs chat ----------------------
    # Ten sam model bazowy (Llama 3.1 70B), różne post-training
    # Łączny rozmiar: 3x ~142GB = 426GB, planowana kwantyzacja AWQ potem
    "CYFRAGOVPL/Llama-PLLuM-70B-base|llama-pllum-70b-base|~142GB|PLLuM 70B base - sam pretraining"
    "CYFRAGOVPL/Llama-PLLuM-70B-instruct|llama-pllum-70b-instruct|~142GB|PLLuM 70B instruct - SFT"
    "CYFRAGOVPL/Llama-PLLuM-70B-chat|llama-pllum-70b-chat|~142GB|PLLuM 70B chat - SFT + RLHF"
)

# --- Funkcje pomocnicze -----------------------------------------------------

### Sprawdza, czy katalog modelu jest w pełni pobrany
### Pełny = ma config.json + co najmniej jeden .safetensors + zero .incomplete
is_complete() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] || return 1
    local safetensors_count
    safetensors_count=$(find "$dir" -maxdepth 1 -name "*.safetensors" -type f 2>/dev/null | wc -l)
    [[ "$safetensors_count" -gt 0 ]] || return 1
    local incomplete_count
    incomplete_count=$(find "$dir" -name "*.incomplete" -type f 2>/dev/null | wc -l)
    [[ "$incomplete_count" -eq 0 ]] || return 1
    return 0
}

### Sprawdza, czy katalog istnieje ale jest tylko szkieletem (pusty lub same metadane)
### Szkielet = brak .safetensors niezależnie od obecności configów
is_skeleton() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    local safetensors_count
    safetensors_count=$(find "$dir" -maxdepth 1 -name "*.safetensors" -type f 2>/dev/null | wc -l)
    [[ "$safetensors_count" -eq 0 ]] && return 0
    return 1
}

# --- Sanity checks ----------------------------------------------------------
echo "=== download_models.sh — 12 modeli benchmarkowych ==="
echo "Target directory: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

### Miejsce na dysku — zestaw waży ~438 GB, potrzebujemy rezerwy
echo ""
echo "--- Miejsce na dysku ---"
df -hT "$MODELS_DIR" | tail -1

### hf CLI — powinno być zainstalowane jako część huggingface_hub[hf_transfer]
if ! command -v hf >/dev/null 2>&1; then
    echo "[!] Brak komendy 'hf'. Instaluje..."
    pip install -U "huggingface_hub[hf_transfer]"
fi

### Login — opcjonalny dla publicznych, ale token jest fine-grained
### Do wariantów "-nc-" gated nie będziemy schodzić, więc publiczny OK
if ! hf auth whoami >/dev/null 2>&1; then
    echo "[!] Nie zalogowany do HF. Jesli model jest gated, zrob: hf auth login"
fi

# --- Pobieranie -------------------------------------------------------------
echo ""
echo "=== Rozpoczynamy pobieranie ==="

for entry in "${MODELS[@]}"; do
    ### IFS='|' = separator pól; read -r rozbija linię na 4 zmienne
    IFS='|' read -r repo_id subdir size comment <<< "$entry"
    target="$MODELS_DIR/$subdir"

    echo ""
    echo "================================================================"
    echo "Model:     $repo_id"
    echo "Rozmiar:   $size"
    echo "Komentarz: $comment"
    echo "Target:    $target"
    echo "================================================================"

    ### Jeśli katalog istnieje i model jest kompletny — pomijamy
    if is_complete "$target"; then
        echo "[skip] Model kompletny (config + safetensors + brak .incomplete)."
        continue
    fi

    ### Jeśli katalog jest szkieletem (puste/same configi) — kasujemy przed pobraniem
    ### hf download lubi pustą przestrzeń roboczą; szkielety z poprzednich sesji
    ### mogą mieć nieaktualne configi względem aktualnych wag na HF
    if is_skeleton "$target"; then
        echo "[clean] Katalog jest szkieletem bez wag — czyszczę przed pobraniem."
        rm -rf "$target"
    fi

    ### Pobieranie: --max-workers 2 (było 8) dla stabilności na długich 140GB pobieraniach
    ### hf_transfer + 4 workery = wciąż szybkie, ale mniej podatne na timeouty CDN
    ### Jeśli pobieranie padnie w połowie, skrypt można re-run — hf wznowi od .incomplete
    hf download "$repo_id" \
        --local-dir "$target" \
        --max-workers 2

    ### Walidacja post-download — błąd fatalny, jeśli model nie jest kompletny po udanym hf download
    if ! is_complete "$target"; then
        echo "[!] UWAGA: $repo_id pobrany, ale walidacja nie przeszła"
        echo "    Sprawdź ręcznie: ls -lah $target"
        ### Nie exitujemy (set -e złapie, jeśli hf download zwróci błąd) — tylko ostrzegamy
    else
        echo "[ok] $repo_id → $target (kompletny)"
    fi
done

echo ""
echo "=== Podsumowanie: zawartość $MODELS_DIR ==="
du -sh "$MODELS_DIR"/*/ 2>/dev/null || true

echo ""
echo "=== Gotowe. Modele w: $MODELS_DIR ==="
