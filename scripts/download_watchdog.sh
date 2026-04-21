#!/usr/bin/env bash
# =============================================================================
# download_watchdog.sh
# Monitoruje pobieranie modeli HF, auto-restart przy zwisie.
#
# Logika:
#   - Co 30s sprawdza, czy aktywny .incomplete rosnie
#   - Gdy > 120s bez ruchu -> zwis
#   - 3x gentle restart (pkill hf download -> skrypt padnie -> watchdog go wznowi)
#   - 4-ty raz: heavy restart (kill wszystkiego + cooldown + ping HF + restart)
#
# Uzycie:
#   nohup ~/pjatk-ai-lab-log/scripts/download_watchdog.sh \
#       > ~/pjatk-ai-lab-log/logs/watchdog_$(date +%Y%m%d_%H%M).log 2>&1 &
#   disown
#
# Zatrzymanie:
#   pkill -f "download_watchdog.sh"
# =============================================================================

# UWAGA: bez set -e — watchdog musi przezyc bledy pgrep/find itp.
set -uo pipefail

# --- Konfiguracja -----------------------------------------------------------
STUCK_THRESHOLD_SEC=300         # Ile sekund bez ruchu = zwis
CHECK_INTERVAL_SEC=30           # Co ile sekund sprawdzamy
MAX_GENTLE_RESTARTS=2           # Ile razy gentle, potem heavy
HEAVY_COOLDOWN_SEC=60           # Pauza recovery w heavy restarcie

MODELS_DIR="$HOME/models"
SCRIPT_PATH="$HOME/pjatk-ai-lab-log/scripts/download_models.sh"
LOG_DIR="$HOME/pjatk-ai-lab-log/logs"

# --- Stan (zmienne globalne pętli) ------------------------------------------
last_size=0
last_change_time=$(date +%s)
last_incomplete_path=""
last_model_dir=""
restart_counter=0
last_restart_time=0

# --- Logowanie --------------------------------------------------------------

### Timestamp + komunikat na stdout; log leci do pliku via nohup
log_event() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# --- Detekcja stanu pobierania ----------------------------------------------

### Zwraca sciezke KATALOGU MODELU, ktory aktywnie pobiera
### (czyli ten z najnowszym .incomplete; hf_transfer ma 4 shardy naraz,
### ale wszystkie w tym samym modelu)
find_active_model_dir() {
    local latest_incomplete
    latest_incomplete=$(find "$MODELS_DIR" \
        -path "*/.cache/huggingface/download/*.incomplete" \
        -type f 2>/dev/null \
        | xargs -r ls -t 2>/dev/null \
        | head -1)
    [[ -z "$latest_incomplete" ]] && return
    # Sciezka: <MODELS>/<MODEL>/.cache/huggingface/download/<hash>.incomplete
    # dirname x4 -> katalog modelu
    dirname "$(dirname "$(dirname "$(dirname "$latest_incomplete")")")"
}

### Suma bajtow w katalogu modelu: wszystkie safetensors + wszystkie .incomplete
### To stabilna miara postepu (nie skacze miedzy shardami)
count_model_bytes() {
    local model_dir="$1"
    [[ -d "$model_dir" ]] || { echo 0; return; }
    local total=0
    local final_bytes incomplete_bytes
    ### Final files w katalogu modelu (ukonczone safetensors + configi)
    final_bytes=$(find "$model_dir" -maxdepth 1 -type f -printf "%s\n" 2>/dev/null \
        | awk "{ sum += \$1 } END { print sum+0 }")
    ### Aktywne .incomplete w .cache/huggingface/download/
    incomplete_bytes=$(find "$model_dir/.cache/huggingface/download" \
        -maxdepth 1 -name "*.incomplete" -type f -printf "%s\n" 2>/dev/null \
        | awk "{ sum += \$1 } END { print sum+0 }")
    echo $((final_bytes + incomplete_bytes))
}

### true/false: czy jakis hf download zyje
is_hf_running() {
    pgrep -f "hf download" >/dev/null 2>&1
}

### true/false: czy download_models.sh zyje
is_script_running() {
    pgrep -f "download_models.sh" >/dev/null 2>&1
}

# --- Akcje restartu ---------------------------------------------------------

### Uruchamia download_models.sh w tle (nohup, disown)
### Nowy log z timestampem dla audytu
spawn_download_script() {
    local new_log
    new_log="$LOG_DIR/download_$(date +%Y%m%d_%H%M)_watchdog.log"
    log_event "  -> start: $SCRIPT_PATH"
    log_event "  -> log:   $new_log"
    nohup "$SCRIPT_PATH" > "$new_log" 2>&1 &
    disown
    log_event "  -> PID:   $!"
}

### Gentle restart: tylko hf download
### Skrypt ma set -e, wiec sam padnie; watchdog wykryje i wznowi
gentle_restart() {
    local attempt=$((restart_counter + 1))
    log_event "=== GENTLE RESTART ${attempt}/${MAX_GENTLE_RESTARTS} ==="
    log_event "  -> pkill -9 hf download"
    pkill -9 -f "hf download" 2>/dev/null || true
    sleep 3

    if is_script_running; then
        log_event "  -> skrypt nadal zyje, pozwalam mu kontynuowac"
    else
        log_event "  -> skrypt padl (set -e), wznawiam"
        spawn_download_script
    fi

    restart_counter=$attempt
}

### Heavy restart: zabijamy wszystko + cooldown + ping + restart
heavy_restart() {
    log_event "=== HEAVY RESTART (gentle ${MAX_GENTLE_RESTARTS}x nie pomogl) ==="

    log_event "  -> pkill -9 hf download + download_models.sh"
    pkill -9 -f "hf download" 2>/dev/null || true
    pkill -9 -f "download_models.sh" 2>/dev/null || true
    sleep 5

    log_event "  -> cooldown ${HEAVY_COOLDOWN_SEC}s (recovery sieci)"
    sleep "$HEAVY_COOLDOWN_SEC"

    ### Test lacznosci — czy warto w ogole restartowac
    if ping -c 2 -W 3 huggingface.co >/dev/null 2>&1; then
        log_event "  -> siec OK (ping HF przeszedl)"
    else
        log_event "  -> SIEC NIEDOSTEPNA, kolejne ${HEAVY_COOLDOWN_SEC}s i probujemy"
        sleep "$HEAVY_COOLDOWN_SEC"
    fi

    spawn_download_script
    restart_counter=0  # reset licznika, nowy cykl
}

### Decyzja: gentle czy heavy
handle_stuck() {
    if [[ $restart_counter -lt $MAX_GENTLE_RESTARTS ]]; then
        gentle_restart
    else
        heavy_restart
    fi
    ### Zerowanie stanu detekcji + zapis czasu restartu dla reset-countera
    last_size=0
    last_change_time=$(date +%s)
    last_incomplete_path=""
    last_model_dir=""
    last_restart_time=$(date +%s)
}

# --- Obsluga progresu -------------------------------------------------------

### Reset timera + log zmiany + zerowanie counter-a po dlugim sukcesie
handle_progress() {
    local current_size="$1"
    local delta=$((current_size - last_size))
    log_event "Postep: +$(numfmt --to=iec "$delta" 2>/dev/null || echo "${delta}B")"

    last_size=$current_size
    last_change_time=$(date +%s)

    ### Po 10 min stabilnego postepu od ostatniego restartu -> zerujemy counter
    ### Liczymy od last_restart_time, nie od startu watchdoga (inaczej falszywy reset)
    local since_restart=$(( $(date +%s) - last_restart_time ))
    if [[ $restart_counter -gt 0 && $last_restart_time -gt 0 && $since_restart -gt 600 ]]; then
        log_event "10+ min stabilnego postepu od restartu — zeruje gentle counter"
        restart_counter=0
    fi
}

# --- Main loop --------------------------------------------------------------

main() {
    log_event "=========================================="
    log_event "Download watchdog startuje (PID $$)"
    log_event "  MODELS_DIR:        $MODELS_DIR"
    log_event "  SCRIPT_PATH:       $SCRIPT_PATH"
    log_event "  stuck threshold:   ${STUCK_THRESHOLD_SEC}s"
    log_event "  check interval:    ${CHECK_INTERVAL_SEC}s"
    log_event "  gentle before heavy: ${MAX_GENTLE_RESTARTS}x"
    log_event "=========================================="

    while true; do
        ### Sanity: czy skrypt w ogole zyje? Jesli nie — uruchamiamy
        if ! is_script_running; then
            log_event "download_models.sh nie zyje — wznawiam"
            spawn_download_script
            sleep "$CHECK_INTERVAL_SEC"
            continue
        fi

        ### Sledzimy CALY katalog modelu (nie pojedynczy shard)
        ### hf_transfer ma 4 workery, wiec skakanie miedzy .incomplete to norma
        local current_model_dir current_bytes now stuck_for
        current_model_dir=$(find_active_model_dir)

        if [[ -z "$current_model_dir" ]]; then
            ### Brak aktywnego pobierania = transition miedzy modelami lub koniec
            sleep "$CHECK_INTERVAL_SEC"
            continue
        fi

        current_bytes=$(count_model_bytes "$current_model_dir")
        now=$(date +%s)

        ### Zmiana modelu (poprzedni sie skonczyl, nowy zaczyna) = reset detekcji
        if [[ "$current_model_dir" != "$last_model_dir" ]]; then
            log_event "Nowy model: $(basename "$current_model_dir") ($(numfmt --to=iec "$current_bytes" 2>/dev/null || echo "${current_bytes}B"))"
            last_model_dir="$current_model_dir"
            last_size=$current_bytes
            last_change_time=$now
            sleep "$CHECK_INTERVAL_SEC"
            continue
        fi

        ### Rosnie? (suma wszystkich shardow modelu)
        if [[ $current_bytes -gt $last_size ]]; then
            handle_progress "$current_bytes"
        else
            stuck_for=$((now - last_change_time))
            if [[ $stuck_for -ge $STUCK_THRESHOLD_SEC ]]; then
                log_event "!!! ZWIS: ${stuck_for}s bez ruchu w $(basename "$current_model_dir") przy $(numfmt --to=iec "$current_bytes" 2>/dev/null || echo "${current_bytes}B")"
                handle_stuck
            fi
        fi

        sleep "$CHECK_INTERVAL_SEC"
    done
}

main
