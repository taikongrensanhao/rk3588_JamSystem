#!/bin/bash
set -e

RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"

export JAMSYSTEM_BASE_PATH="${JAMSYSTEM_BASE_PATH:-$RUN_ROOT/JamSystem}"
export JAMSYSTEM_AD9361_EXE="${JAMSYSTEM_AD9361_EXE:-$JAMSYSTEM_BASE_PATH/ad9361_rk3588}"

if [ -z "${JAMSYSTEM_PYTHON_EXE}" ]; then
  if [ -x "$JAMSYSTEM_BASE_PATH/jam_env/bin/python3" ]; then
    export JAMSYSTEM_PYTHON_EXE="$JAMSYSTEM_BASE_PATH/jam_env/bin/python3"
  else
    export JAMSYSTEM_PYTHON_EXE="python3"
  fi
fi

cd "$RUN_ROOT/rk3588_gui"
./rk3588_gui
