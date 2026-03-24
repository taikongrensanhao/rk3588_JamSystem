#!/bin/bash
set -e

RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"
JAM_ROOT="${JAMSYSTEM_BASE_PATH:-$RUN_ROOT/JamSystem}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$JAM_ROOT"

if [ ! -d "jam_env" ]; then
  "$PYTHON_BIN" -m venv jam_env
fi

. "$JAM_ROOT/jam_env/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$JAM_ROOT/requirements.txt"

echo "Python 虚拟环境已准备完成: $JAM_ROOT/jam_env"
