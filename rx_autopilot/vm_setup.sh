#!/bin/bash
# rx_autopilot/vm_setup.sh — Windows-VM für den RX-Autopiloten (ein Befehl, kein Mensch).
#
# Baut eine QEMU/KVM-VM mit Windows 11 Enterprise Evaluation und einem
# QEMU-SMB-Share (Host: <repo>/../rx_vm/share, in der VM \\\\10.0.2.4\\qemu).
# In der VM läuft nach der unbeaufsichtigten Installation der
# windows_agent.py als Watch-Queue — RX-Bearbeitung damit vollständig
# automatisiert.
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

# Agent + Presets in den Share legen (die VM sieht sie über QEMU-SMB)
cp -f "${AGENT_SRC}/windows_agent.py" "${AGENT_DST}/"
cp -f "${AGENT_SRC}/rx_presets.json" "${AGENT_DST}/"

# Firstboot-Skript für die VM (wird per autounattend FirstLogonCommands gestartet)
cat > "${SHARE}/firstboot.cmd" <<'EOF'
@echo off
rem RX still installieren, falls der Installer im Share liegt
if exist Z:\RX_Installer.exe Z:\RX_Installer.exe /VERYSILENT /NORESTART /SUPPRESSMSGBOXES
rem Runner auf C: ablegen — er löst das RX-Python auf und startet den Agenten
mkdir C:\rx_autopilot 2>nul
copy /Y Z:\agent\windows_agent.py C:\rx_autopilot\ >nul
copy /Y Z:\agent\rx_presets.json C:\rx_autopilot\ >nul
(
  echo @echo off
  echo set RXP=^"C:\Program Files\iZotope\RX 11 Audio Editor\python\python.exe^"
  echo if not exist %%RXP%% set RXP=^"C:\Program Files\iZotope\RX 12 Audio Editor\python\python.exe^"
  echo if not exist %%RXP%% exit /b 1
  echo %%RXP%% C:\rx_autopilot\windows_agent.py --watch --queue Z:\queue --out Z:\output
) > C:\rx_autopilot\run_agent.cmd
rem Agent-Autostart als geplante Aufgabe (läuft dauerhaft als Watch-Queue)
schtasks /Create /SC ONSTART /RU SYSTEM /TN "RXAutopilot" /TR "C:\rx_autopilot\run_agent.cmd" /F
schtasks /Run /TN "RXAutopilot"
EOF

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
echo "Share: QEMU-SMB \\\\10.0.2.4\\qemu -> ${SHARE} (Agent verarbeitet queue/ automatisch)."

exec qemu-system-x86_64 \
    -enable-kvm -cpu host -smp 4 -m 8192 \
    -drive file="${DISK}",if=virtio,format=qcow2 \
    -cdrom "${ISO}" \
    -drive file="${VM_DIR}/autounattend.iso",if=virtio,readonly=on \
    -netdev user,id=n0,smb="${SHARE}" -device virtio-net-pci,netdev=n0 \
    -daemonize -display none -vnc :1
