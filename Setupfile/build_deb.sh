#!/usr/bin/env bash
# build_deb.sh — baut aurik_10.2.0_amd64.deb in Setupfile/
#
# Enthalten (Laufzeit): backend/ plugins/ Aurik10/ cli/ denker/ dsp/
#                        forensics/ models/ (ALLE ML-Modelle) requirements_aurik.txt
# Ausgeschlossen (Programmier-Ballast): tests/ scripts/ .github/ .agents/
#   docs/ benchmarks/ audit/ forensics-Dev-Tools, scratch-Ordner (logs/output/
#   export/data/analysis_results/corpus/golden_samples/imports/learning/
#   memories/media-configs/policy*/chain_templates/config*/processing/
#   node_modules/output_audio/temp_repro/), __pycache__, .git, .venv*, Root-Markdown.
#
# Nutzung: ./Setupfile/build_deb.sh          (voll, mit Modellen, ~29 GB)
#          ./Setupfile/build_deb.sh --ohne-modelle   (Schnelltest der Struktur)
set -euo pipefail

# Der Projekt-Mount ignoriert chmod (alles 777) — unter fakeroot sieht
# dpkg-deb die gesetzten Rechte (stat wird gefälscht). Re-Exec unter fakeroot.
if [ -z "${FAKEROOTKEY:-}" ] && command -v fakeroot >/dev/null 2>&1; then
    exec fakeroot "$0" "$@"
fi

cd "$(dirname "$0")/.."

# dpkg-deb baut das data.tar.zst in $TMPDIR — die Systemdisk (/tmp) hat nur
# ~4 GB frei, die 29-GB-Paketierung braucht den großen Mount als Temp.
export TMPDIR="$PWD/Setupfile/tmp"
mkdir -p "$TMPDIR"

WITH_MODELS=1
NUR_PAKETIEREN=0
case "${1:-}" in
    --ohne-modelle) WITH_MODELS=0 ;;
    --nur-paketieren) NUR_PAKETIEREN=1 ;;
esac

DEST="Setupfile/aurik"
DEB="Setupfile/aurik_10.2.0_amd64.deb"

if [ "$NUR_PAKETIEREN" -eq 0 ]; then
    rm -rf "$DEST"
fi
mkdir -p \
    "$DEST/DEBIAN" \
    "$DEST/opt/aurik" \
    "$DEST/usr/bin" \
    "$DEST/usr/share/applications" \
    "$DEST/usr/share/doc/aurik"

if [ "$NUR_PAKETIEREN" -eq 0 ]; then
    echo "== Kopiere Laufzeit-Kern =="
    for d in backend plugins Aurik10 cli denker dsp forensics requirements; do
        rsync -a \
            --exclude '__pycache__' \
            --exclude '*.pyc' \
            --exclude '.git' \
            "$d/" "$DEST/opt/aurik/$d/"
    done

    if [ "$WITH_MODELS" -eq 1 ]; then
        echo "== Kopiere ALLE ML-Modelle (dauert) =="
        rsync -a \
            --exclude '__pycache__' \
            --exclude '*.pyc' \
            models/ "$DEST/opt/aurik/models/"
    else
        mkdir -p "$DEST/opt/aurik/models"
    fi

    # Nur die Laufzeit-Requirements ins Paket (Dev/Optimierung = Ballast)
    find "$DEST/opt/aurik/requirements" -type f ! -name 'requirements_aurik.txt' -delete

    cp LICENSE "$DEST/usr/share/doc/aurik/copyright"
else
    echo "== Baum wiederverwendet (--nur-paketieren) =="
fi

# __pycache__/pyc sind Ballast und können während langer Läufe inkonsistent
# werden (parallel laufende Python-Prozesse) — vor dem Paketieren entfernen.
find "$DEST" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$DEST" -name '*.pyc' -delete 2>/dev/null || true

echo "== Schreibe DEBIAN-Metadaten =="
cp Setupfile/DEBIAN_TEMPLATE/control  "$DEST/DEBIAN/control"
cp Setupfile/DEBIAN_TEMPLATE/postinst "$DEST/DEBIAN/postinst"
cp Setupfile/DEBIAN_TEMPLATE/postrm   "$DEST/DEBIAN/postrm"
chmod 755 "$DEST/DEBIAN"
chmod 644 "$DEST/DEBIAN/control"
chmod 755 "$DEST/DEBIAN/postinst" "$DEST/DEBIAN/postrm"
cp Setupfile/DEBIAN_TEMPLATE/aurik          "$DEST/usr/bin/aurik"
chmod 755 "$DEST/usr/bin/aurik"
cp Setupfile/DEBIAN_TEMPLATE/aurik.desktop  "$DEST/usr/share/applications/aurik.desktop"

echo "== Baue Paket (zstd) =="
rm -f "$DEB"
dpkg-deb --build --root-owner-group -Zzstd "$DEST" "$DEB"
echo "== Fertig: $DEB =="
ls -lh "$DEB"
