#!/usr/bin/env bash
# Crash-sichere, chunk-basierte Testausfuehrung fuer grosse Testmengen.
# Idee: mehrere kurze pytest-Prozesse statt ein langer Prozess -> kein RSS-Aufstauen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ROOT_PATH="${1:-tests/unit}"
shift || true

CHUNK_FILES="${AURIK_BATCH_FILES:-8}"
CHUNK_LIMIT="${AURIK_CHUNK_LIMIT:-0}"
CHUNK_START="${AURIK_CHUNK_START:-0}"
LOG_FILE="${AURIK_CHUNK_LOG_FILE:-${SCRIPT_DIR}/logs/pytest_chunked_safe.log}"
OOM_RETRY_MEM_GB="${AURIK_OOM_RETRY_MEM_GB:-20}"
OOM_RETRY_SWAP_MB="${AURIK_OOM_RETRY_SWAP_MB:-6144}"
ULTRA_QUIET_RAW="${AURIK_ULTRA_QUIET:-0}"
ULTRA_QUIET="0"
PROGRESS_EVERY="${AURIK_ULTRA_QUIET_PROGRESS_EVERY:-10}"

case "${ULTRA_QUIET_RAW,,}" in
    1|true|yes|on)
        ULTRA_QUIET="1"
        ;;
    *)
        ULTRA_QUIET="0"
        ;;
esac

mkdir -p "$(dirname "$LOG_FILE")"

if [[ ! -d "$ROOT_PATH" ]]; then
    echo "[chunked-safe] Fehler: Pfad nicht gefunden: $ROOT_PATH" >&2
    exit 2
fi

mapfile -t TEST_FILES < <(find "$ROOT_PATH" -type f -name "test_*.py" | sort)
TOTAL_FILES="${#TEST_FILES[@]}"

if [[ "$TOTAL_FILES" -eq 0 ]]; then
    echo "[chunked-safe] Keine Testdateien unter $ROOT_PATH gefunden." >&2
    exit 2
fi

TOTAL_CHUNKS=$(( (TOTAL_FILES + CHUNK_FILES - 1) / CHUNK_FILES ))

echo "══════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo " Aurik Chunked Safe Test Runner" | tee -a "$LOG_FILE"
echo " Root         : $ROOT_PATH" | tee -a "$LOG_FILE"
echo " Dateien      : $TOTAL_FILES" | tee -a "$LOG_FILE"
echo " Chunk-Groesse: $CHUNK_FILES" | tee -a "$LOG_FILE"
echo " Chunks gesamt: $TOTAL_CHUNKS" | tee -a "$LOG_FILE"
echo " Chunk-Limit  : $CHUNK_LIMIT (0 = alle)" | tee -a "$LOG_FILE"
echo " Chunk-Start  : $CHUNK_START (0 = ab Chunk 1)" | tee -a "$LOG_FILE"
if [[ "$ULTRA_QUIET" == "1" ]]; then
    echo " Ultra-Quiet  : aktiv (reduzierte Terminal-Ausgabe)" | tee -a "$LOG_FILE"
fi
echo " Extra-Args   : $*" | tee -a "$LOG_FILE"
echo "══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

chunk_idx=0
file_idx=0

while [[ "$file_idx" -lt "$TOTAL_FILES" ]]; do
    chunk_idx=$((chunk_idx + 1))
    if [[ "$CHUNK_START" -gt 0 && "$chunk_idx" -lt "$CHUNK_START" ]]; then
        # Chunks vor CHUNK_START überspringen (Resume nach behobenem Fehler).
        file_idx=$((file_idx + CHUNK_FILES))
        [[ "$file_idx" -gt "$TOTAL_FILES" ]] && file_idx="$TOTAL_FILES"
        continue
    fi
    if [[ "$CHUNK_LIMIT" -gt 0 && "$chunk_idx" -gt "$CHUNK_LIMIT" ]]; then
        echo "[chunked-safe] Chunk-Limit erreicht ($CHUNK_LIMIT). Stoppe planmaessig." | tee -a "$LOG_FILE"
        break
    fi

    end_idx=$((file_idx + CHUNK_FILES))
    if [[ "$end_idx" -gt "$TOTAL_FILES" ]]; then
        end_idx="$TOTAL_FILES"
    fi

    CHUNK=("${TEST_FILES[@]:file_idx:end_idx-file_idx}")
    _chunk_start_msg="[chunked-safe] Starte Chunk ${chunk_idx}/${TOTAL_CHUNKS} (${#CHUNK[@]} Dateien)"
    echo "${_chunk_start_msg}" >> "$LOG_FILE"
    if [[ "$ULTRA_QUIET" != "1" || "$chunk_idx" -eq 1 || "$chunk_idx" -eq "$TOTAL_CHUNKS" || $((chunk_idx % PROGRESS_EVERY)) -eq 0 ]]; then
        echo "${_chunk_start_msg}"
    fi

    # Jede Charge in frischem, isoliertem Prozess starten.
    # Wichtig: Exitcode direkt auswerten (kein `if ! ...; then rc=$?`), sonst wird rc=0.
    if "$SCRIPT_DIR/run_tests_safe.sh" "${CHUNK[@]}" "$@"; then
        :
    else
        rc=$?
        if [[ "$rc" -eq 137 ]]; then
            echo "[chunked-safe] Chunk ${chunk_idx}/${TOTAL_CHUNKS} OOM/kill (Exit 137). Starte Fallback pro Datei." | tee -a "$LOG_FILE"
            for _file in "${CHUNK[@]}"; do
                echo "[chunked-safe]   Fallback-Datei: ${_file}" | tee -a "$LOG_FILE"
                if "$SCRIPT_DIR/run_tests_safe.sh" "${_file}" "$@"; then
                    :
                else
                    file_rc=$?
                    if [[ "$file_rc" -eq 137 ]]; then
                        echo "[chunked-safe]   OOM in Fallback-Datei ${_file}. Retry mit Budget ${OOM_RETRY_MEM_GB}G/${OOM_RETRY_SWAP_MB}M." | tee -a "$LOG_FILE"
                        if AURIK_MEM_GB="$OOM_RETRY_MEM_GB" AURIK_SWAP_MB="$OOM_RETRY_SWAP_MB" "$SCRIPT_DIR/run_tests_safe.sh" "${_file}" "$@"; then
                            echo "[chunked-safe]   Retry erfolgreich: ${_file}" | tee -a "$LOG_FILE"
                        else
                            retry_rc=$?
                            echo "[chunked-safe]   FEHLER in Fallback-Datei ${_file} nach OOM-Retry (Exit ${retry_rc})." | tee -a "$LOG_FILE"
                            exit "$retry_rc"
                        fi
                    else
                        echo "[chunked-safe]   FEHLER in Fallback-Datei ${_file} (Exit ${file_rc})." | tee -a "$LOG_FILE"
                        exit "$file_rc"
                    fi
                fi
            done
            echo "[chunked-safe] Chunk ${chunk_idx}/${TOTAL_CHUNKS} per Datei-Fallback erfolgreich." | tee -a "$LOG_FILE"
        else
            echo "[chunked-safe] FEHLER in Chunk ${chunk_idx}/${TOTAL_CHUNKS} (Exit ${rc})." | tee -a "$LOG_FILE"
            exit "$rc"
        fi
    fi

    _chunk_ok_msg="[chunked-safe] Chunk ${chunk_idx}/${TOTAL_CHUNKS} erfolgreich."
    echo "${_chunk_ok_msg}" >> "$LOG_FILE"
    if [[ "$ULTRA_QUIET" != "1" || "$chunk_idx" -eq 1 || "$chunk_idx" -eq "$TOTAL_CHUNKS" || $((chunk_idx % PROGRESS_EVERY)) -eq 0 ]]; then
        echo "${_chunk_ok_msg}"
    fi
    file_idx="$end_idx"
done

echo "[chunked-safe] Fertig." | tee -a "$LOG_FILE"
