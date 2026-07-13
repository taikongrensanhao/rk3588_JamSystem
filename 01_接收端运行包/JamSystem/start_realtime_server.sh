#!/usr/bin/env bash
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${JAMSYSTEM_PYTHON:-$BASE_DIR/jam_env/bin/python3}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${JAMSYSTEM_PYTHON:-python3}"
fi

HOST="${JAMSYSTEM_SERVER_HOST:-0.0.0.0}"
PORT="${JAMSYSTEM_SERVER_PORT:-8008}"
DATA_DIR="${JAMSYSTEM_REALTIME_DATA_DIR:-$BASE_DIR/dataset_realtime}"
BASE_DATA_DIR="${JAMSYSTEM_REALWORLD_DATA_DIR:-$BASE_DIR/dataset_realworld}"
SIM_DATA_DIR="${JAMSYSTEM_SIM_DATA_DIR:-${MIX_SIM_DATA_DIR:-}}"
EPOCHS="${JAMSYSTEM_TRAIN_EPOCHS:-3}"
BATCH="${JAMSYSTEM_TRAIN_BATCH:-16}"
RKNN_COMMAND="${JAMSYSTEM_RKNN_COMMAND:-}"
if [ -z "$RKNN_COMMAND" ] && [ "${JAMSYSTEM_ENABLE_RKNN_CONVERT:-0}" = "1" ]; then
  RKNN_COMMAND="$PYTHON $BASE_DIR/convert_latest_model_to_rknn.py"
fi

mkdir -p "$DATA_DIR"
mkdir -p "$BASE_DATA_DIR"

echo "[SERVER] python: $PYTHON"
echo "[SERVER] listen: $HOST:$PORT"
echo "[SERVER] base data dir: $BASE_DATA_DIR"
echo "[SERVER] data dir: $DATA_DIR"
if [ -n "$SIM_DATA_DIR" ]; then
  echo "[SERVER] sim data dir: $SIM_DATA_DIR"
fi
echo "[SERVER] epochs: $EPOCHS batch: $BATCH"
if [ -n "$RKNN_COMMAND" ]; then
  echo "[SERVER] rknn command: $RKNN_COMMAND"
fi

ARGS=(
  "$BASE_DIR/server_realtime_train.py"
  --host "$HOST" \
  --port "$PORT" \
  --base-data-dir "$BASE_DATA_DIR" \
  --data-dir "$DATA_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH"
)

if [ -n "$SIM_DATA_DIR" ]; then
  ARGS+=(--sim-data-dir "$SIM_DATA_DIR")
fi

if [ -n "$RKNN_COMMAND" ]; then
  ARGS+=(--rknn-command "$RKNN_COMMAND")
fi

exec "$PYTHON" "${ARGS[@]}"
