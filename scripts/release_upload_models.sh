#!/usr/bin/env bash
# scripts/release_upload_models.sh — Resumierbarer Upload der großen ML-Modelle.
#
# Release-Strategie 2026-09-23: Modelle > 40 MB werden als GitHub-Release-
# Assets ausgeliefert (LFS im Repo nur für den Offline-Kern <= 40 MB).
#
# Eigenschaften:
#   - Retry pro Asset (MAX_RETRIES, 30 s Backoff)
#   - Bereits hochgeladene Assets werden übersprungen (Resume nach Abbruch:
#     Skript einfach erneut starten)
#   - Dateien > 1.9 GB werden in Parts gesplittet (GitHub-Limit: 2 GB/Asset)
#   - Asset-Namen deterministisch: Pfad mit "__" normalisiert
#
# Nutzung:
#   export PATH="$HOME/.local/bin:$PATH"
#   bash scripts/release_upload_models.sh
#
# Umgebungsvariablen: TAG (Default models-10.2.0), REPO, WORKDIR, MAX_RETRIES
set -u

TAG="${TAG:-models-10.2.0}"
REPO="${REPO:-michaelarnold2307/Aurik_10.0.20}"
WORKDIR="${WORKDIR:-/tmp/aurik_release_chunks}"
CHUNK_BYTES=$((1900 * 1024 * 1024)) # 1.9 GB < 2-GB-GitHub-Limit
MAX_RETRIES="${MAX_RETRIES:-3}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$WORKDIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Release anlegen, falls nicht vorhanden
if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    log "Lege Release $TAG an"
    gh release create "$TAG" --repo "$REPO" \
        --title "ML-Modelle 10.2.0 (große Dateien > 40 MB)" \
        --notes "Release-Strategie 2026-09-23: Modelle > 40 MB werden als Release-Assets ausgeliefert (model_downloader §13.3, models/manifest.json). LFS im Repo nur für den Offline-Kern (<= 40 MB)." \
        || { log "Release-Anlage fehlgeschlagen"; exit 1; }
fi

uploaded="$(gh release view "$TAG" --repo "$REPO" --json assets --jq '.assets[].name' 2>/dev/null)"
is_uploaded() { printf '%s\n' "$uploaded" | grep -qxF "$1"; }

upload_asset() {
    local src="$1" name="$2"
    if is_uploaded "$name"; then
        log "SKIP (bereits hochgeladen): $name"
        return 0
    fi
    local attempt
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        log "UPLOAD $name (Versuch $attempt/$MAX_RETRIES, $(du -h "$src" | cut -f1))"
        if gh release upload "$TAG" "$src#$name" --repo "$REPO" --clobber; then
            uploaded="$(printf '%s\n%s' "$uploaded" "$name")"
            log "OK: $name"
            return 0
        fi
        log "FEHLER bei $name (Versuch $attempt) — warte 30 s"
        sleep 30
    done
    log "AUFGEGEBEN: $name — Skript später erneut starten (Resume überspringt Erledigtes)"
    return 1
}

failed=0

# Alle Modelle > 40 MB durchlaufen
while IFS= read -r file; do
    rel="${file#"$ROOT_DIR"/}"
    asset_base="$(echo "$rel" | tr '/' '__')"
    size=$(stat -c %s "$file")
    if [ "$size" -gt "$CHUNK_BYTES" ]; then
        prefix="$WORKDIR/$asset_base.part"
        rm -f "$prefix"* 2>/dev/null
        log "SPLIT $rel ($(numfmt --to=iec "$size"))"
        if ! split -b "$CHUNK_BYTES" -d "$file" "$prefix"; then
            log "Split fehlgeschlagen: $rel"
            failed=1
            continue
        fi
        for part in "$prefix"*; do
            idx="${part##*.part}"
            name="${asset_base}.part${idx}"
            if ! upload_asset "$part" "$name"; then
                failed=1
            fi
        done
        rm -f "$prefix"*
    else
        if ! upload_asset "$file" "$asset_base"; then
            failed=1
        fi
    fi
done < <(cd "$ROOT_DIR" && find models -type f -size +40M \
    \( -name "*.onnx" -o -name "*.pt" -o -name "*.pth" -o -name "*.bin" \
       -o -name "*.safetensors" -o -name "*.ckpt" -o -name "*.th" \) -print | sort)

log "FERTIG (failed=$failed)"
exit "$failed"
