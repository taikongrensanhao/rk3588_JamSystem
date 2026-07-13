#!/usr/bin/env bash
set -u

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
IP_URI="${1:-ip:192.168.1.10}"
MODULATION="${2:-digital_qpsk}"
COLLECT_FRAMES="${3:-200}"
DEFAULT_REPEATS="${4:-5}"
RECORD_ROOT="${JAMSYSTEM_RECORD_DIR:-/mnt/usb/JamRecords}"
MAX_FILES="${JAMSYSTEM_MAX_COLLECT_FILES:-40}"

CLASSES=(
  none
  single_tone
  narrowband
  wideband_barrage
  comb
  white_noise
  noise_fm
)

repeat_for_class() {
  local cls="$1"
  case "$cls" in
    none) echo $((DEFAULT_REPEATS + 1)) ;;
    single_tone) echo $((DEFAULT_REPEATS + 2)) ;;
    comb) echo $((DEFAULT_REPEATS + 2)) ;;
    *) echo "$DEFAULT_REPEATS" ;;
  esac
}

count_record_files() {
  find "$RECORD_ROOT" -maxdepth 1 -type f -name "*.bin" 2>/dev/null | wc -l
}

mkdir -p "$RECORD_ROOT"

echo "[AUTO] base dir: $BASE_DIR"
echo "[AUTO] ip uri: $IP_URI"
echo "[AUTO] modulation: $MODULATION"
echo "[AUTO] collect frames/file: $COLLECT_FRAMES"
echo "[AUTO] base repeats/class: $DEFAULT_REPEATS"
echo "[AUTO] max bin files: $MAX_FILES"
echo "[AUTO] record root: $RECORD_ROOT"
echo
echo "[AUTO] plan:"
for cls in "${CLASSES[@]}"; do
  echo "  - $cls: $(repeat_for_class "$cls") run(s)"
done
echo

cd "$BASE_DIR" || exit 1

for cls in "${CLASSES[@]}"; do
  repeats="$(repeat_for_class "$cls")"
  for ((i = 1; i <= repeats; i++)); do
    current_files="$(count_record_files)"
    if [ "$current_files" -ge "$MAX_FILES" ]; then
      echo "[AUTO] reached max file limit: $current_files/$MAX_FILES, stop collecting."
      break 2
    fi

    echo "============================================================"
    echo "[AUTO] collecting class=$cls repeat=$i/$repeats current_files=$current_files/$MAX_FILES"
    echo "[AUTO] command: JAMSYSTEM_COLLECT_ONLY=1 JAMSYSTEM_COLLECT_FRAMES=$COLLECT_FRAMES ./ad9361_rk3588 $cls $IP_URI $MODULATION"
    echo "============================================================"

    export JAMSYSTEM_RECORD_DIR="$RECORD_ROOT"
    export JAMSYSTEM_COLLECT_ONLY=1
    export JAMSYSTEM_COLLECT_FRAMES="$COLLECT_FRAMES"
    ./ad9361_rk3588 "$cls" "$IP_URI" "$MODULATION"

    echo "[AUTO] finished class=$cls repeat=$i"
    sleep 2
  done
done

echo
echo "[AUTO] all done. Saved files:"
find "$RECORD_ROOT" -maxdepth 1 -type f -name "*.bin" -printf "%f %s bytes\n" 2>/dev/null | sort
