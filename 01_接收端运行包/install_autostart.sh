#!/bin/bash
set -e

RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"

chmod +x "$RUN_ROOT/auto_start_receiver.sh"
mkdir -p /home/pi/.config/autostart
cp "$RUN_ROOT/jamsystem_receiver.desktop" /home/pi/.config/autostart/jamsystem_receiver.desktop

sudo mkdir -p /etc/lightdm/lightdm.conf.d
sudo tee /etc/lightdm/lightdm.conf.d/50-autologin.conf >/dev/null <<'EOF'
[Seat:*]
autologin-user=pi
autologin-user-timeout=0
EOF

echo "JamSystem Receiver autostart installed."
