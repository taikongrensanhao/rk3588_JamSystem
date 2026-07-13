#!/bin/bash
set -e

RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$RUN_ROOT/JamSystem"
gcc ad9361_rk3588.c -o ad9361_rk3588 -liio -lm -pthread

cd "$RUN_ROOT/transmitter_qt_gui"
qmake transmitter_qt_gui.pro
make -j"$(nproc)"
