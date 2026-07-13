#!/bin/bash
set -e

RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"

chmod +x "$RUN_ROOT/build_tx.sh" "$RUN_ROOT/start_tx_gui.sh" "$RUN_ROOT/autostart_tx_gnome.sh"
mkdir -p "$HOME/.config/systemd/user"
cp "$RUN_ROOT/jamsystem-tx.service" "$HOME/.config/systemd/user/jamsystem-tx.service"
systemctl --user daemon-reload
systemctl --user enable --now jamsystem-tx.service
