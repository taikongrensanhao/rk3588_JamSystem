#!/bin/bash

systemctl --user stop jamsystem-tx.service 2>/dev/null || true
pkill -f transmitter_qt_gui || true
pkill -f ad9361_rk3588 || true
