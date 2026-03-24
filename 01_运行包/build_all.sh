#!/bin/bash
set -e

RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"
export JAMSYSTEM_BASE_PATH="${JAMSYSTEM_BASE_PATH:-$RUN_ROOT/JamSystem}"
export JAMSYSTEM_AD9361_EXE="${JAMSYSTEM_AD9361_EXE:-$JAMSYSTEM_BASE_PATH/ad9361_rk3588}"

cd "$JAMSYSTEM_BASE_PATH"
gcc ad9361_rk3588.c -o ad9361_rk3588 -liio -lm

cd "$RUN_ROOT/rk3588_gui"
qmake rk3588_gui.pro
make -j4
