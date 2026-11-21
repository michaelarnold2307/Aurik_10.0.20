#!/bin/bash
# rx_autopilot/vm_setup.sh — Windows-VM für den RX-Autopiloten (ein Befehl, kein Mensch).
#
# Baut eine QEMU/KVM-VM mit Windows 11 Enterprise Evaluation. Transfer
# Host<->VM läuft über HTTP (rx_autopilot/http_share.py, Port 8090,
# aus der VM: http://10.0.2.2:8090) — kein Samba/Root nötig. In der VM
# läuft nach der unbeaufsichtigten Installation der windows_agent.py als
# HTTP-Watch-Queue — RX-Bearbeitung damit vollständig automatisiert.
#
# Voraussetzungen (einmalig, kein Know-how):
#   1. Windows-Eval-ISO: <repo>/../rx_vm/Win11_Enterprise_Eval.iso
#      (Download: https://go.microsoft.com/fwlink/p/?LinkID=2195404 — ohne Konto)
#   2. RX-Installer (optional): <repo>/../rx_vm/RX_Installer.exe in den Share
#      legen — das Firstboot-Skript installiert ihn still (/VERYSILENT).
#      RX selbst hängt am iZotope-Portal (einmaliger Lizenz-Akt, kein Bedien-Know-how).
#
# Aufruf:  bash rx_autopilot/vm_setup.sh   (idempotent — bestehende VM wird genutzt)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VM_DIR="${REPO}/../rx_vm"
ISO="${VM_DIR}/Win11_Enterprise_Eval.iso"
SHARE="${VM_DIR}/share"
DISK="${VM_DIR}/win11.qcow2"
AGENT_SRC="${REPO}/rx_autopilot"
AGENT_DST="${SHARE}/agent"

mkdir -p "${VM_DIR}" "${SHARE}" "${AGENT_DST}"

if [ ! -f "${ISO}" ]; then
    echo "FEHLT: ${ISO} — Download: https://go.microsoft.com/fwlink/p/?LinkID=2195404" >&2
    exit 2
fi
if [ ! -e /dev/kvm ]; then
    echo "FEHLT: /dev/kvm (KVM nicht verfügbar)" >&2
    exit 2
fi

# Agent + Presets + Bootstrap in den Share legen (die VM lädt sie über HTTP)
cp -f "${AGENT_SRC}/windows_agent.py" "${AGENT_DST}/"
cp -f "${AGENT_SRC}/rx_presets.json" "${AGENT_DST}/"

# Bootstrap für die VM: lädt RX-Installer/Agent über HTTP, installiert still,
# registriert den Agenten als geplante Aufgabe (HTTP-Modus).
cat > "${AGENT_DST}/bootstrap.cmd" <<'EOF'
@echo off
set BASE=http://10.0.2.2:8090
mkdir C:\rx_autopilot 2>nul
powershell -NoProfile -Command "try { Invoke-WebRequest '%BASE%/files/RX_Installer.exe' -OutFile 'C:\rx_autopilot\RX_Installer.exe' } catch {}"
if exist C:\rx_autopilot\RX_Installer.exe C:\rx_autopilot\RX_Installer.exe /VERYSILENT /NORESTART /SUPPRESSMSGBOXES
powershell -NoProfile -Command "Invoke-WebRequest '%BASE%/agent/windows_agent.py' -OutFile 'C:\rx_autopilot\windows_agent.py'; Invoke-WebRequest '%BASE%/agent/rx_presets.json' -OutFile 'C:\rx_autopilot\rx_presets.json'"
(
  echo @echo off
  echo set RXP=^"C:\Program Files\iZotope\RX 11 Audio Editor\python\python.exe^"
  echo if not exist %%RXP%% set RXP=^"C:\Program Files\iZotope\RX 12 Audio Editor\python\python.exe^"
  echo if not exist %%RXP%% exit /b 1
  echo %%RXP%% C:\rx_autopilot\windows_agent.py --http http://10.0.2.2:8090
) > C:\rx_autopilot\run_agent.cmd
schtasks /Create /SC ONSTART /RU SYSTEM /TN "RXAutopilot" /TR "C:\rx_autopilot\run_agent.cmd" /F
schtasks /Run /TN "RXAutopilot"
EOF

# HTTP-Share (Host) starten, falls nicht aktiv
if ! pgrep -f "rx_autopilot/http_share.py" >/dev/null; then
    nohup "${REPO}/.venv_aurik/bin/python" -B "${AGENT_SRC}/http_share.py" --share "${SHARE}" > "${VM_DIR}/http_share.log" 2>&1 &
    echo "HTTP-Share gestartet: http://127.0.0.1:8090 -> ${SHARE}"
fi

# Unbeaufsichtigte Installation: autounattend auf eine kleine ISO (floppy-artig)
if [ ! -f "${VM_DIR}/autounattend.iso" ]; then
    python3 - <<PYEOF
import os, subprocess, tempfile
src = "${AGENT_SRC}/autounattend.xml"
dst_dir = tempfile.mkdtemp(prefix="au_")
dst = os.path.join(dst_dir, "Autounattend.xml")
with open(src, encoding="utf-8") as f:
    inhalt = f.read()
with open(dst, "w", encoding="utf-8") as f:
    f.write(inhalt)
subprocess.run(["genisoimage", "-o", "${VM_DIR}/autounattend.iso", "-J", "-R", dst], check=True)
print("autounattend.iso gebaut")
PYEOF
fi

if [ ! -f "${DISK}" ]; then
    qemu-img create -f qcow2 "${DISK}" 60G
fi

echo "Starte VM (VNC :5901). Erster Start installiert Windows unbeaufsichtigt."
echo "Transfer: HTTP http://10.0.2.2:8090 (Agent verarbeitet queue/ automatisch)."

exec qemu-system-x86_64 \
    -enable-kvm -cpu host -smp 4 -m 8192 \
    -drive file="${DISK}",if=virtio,format=qcow2 \
    -cdrom "${ISO}" \
    -drive file="${VM_DIR}/autounattend.iso",if=virtio,readonly=on \
    -netdev user,id=n0 -device virtio-net-pci,netdev=n0 \
    -daemonize -display none -vnc :1
