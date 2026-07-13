#!/bin/bash
set -e

cd "$(dirname "$0")"

DURATION="${1:-30}"
AD9361_URI="${2:-ip:192.168.1.10}"
LOG_FILE="/tmp/jamsystem_speed_benchmark.log"

rm -f "$LOG_FILE"
echo "[BENCH] duration=${DURATION}s uri=${AD9361_URI}"
echo "[BENCH] collecting..."

set +e
timeout "${DURATION}s" ./speed_rx_test.sh "${AD9361_URI}" > "$LOG_FILE" 2>&1
set -e

awk -v duration="$DURATION" '
  function add_stage(name, value) {
    stage_n[name]++;
    stage_sum[name] += value;
    if (stage_min[name] == "" || value < stage_min[name]) stage_min[name] = value;
    if (stage_max[name] == "" || value > stage_max[name]) stage_max[name] = value;
  }
  /RECOGNITION_TIME_MS:/ {
    split($0, a, "RECOGNITION_TIME_MS:");
    v=a[2]+0;
    n++;
    sum+=v;
    if (min==0 || v<min) min=v;
    if (v>max) max=v;
  }
  /STAGE_WAIT_FRAME_MS:/ {
    split($0, a, "STAGE_WAIT_FRAME_MS:");
    add_stage("wait_frame", a[2]+0);
  }
  /STAGE_POWER_GATE_MS:/ {
    split($0, a, "STAGE_POWER_GATE_MS:");
    add_stage("power_gate", a[2]+0);
  }
  /STAGE_STFT_MS:/ {
    split($0, a, "STAGE_STFT_MS:");
    add_stage("stft", a[2]+0);
  }
  /STAGE_RKNN_MS:/ {
    split($0, a, "STAGE_RKNN_MS:");
    add_stage("rknn", a[2]+0);
  }
  /STAGE_POST_MS:/ {
    split($0, a, "STAGE_POST_MS:");
    add_stage("post", a[2]+0);
  }
  /STAGE_TOTAL_MS:/ {
    split($0, a, "STAGE_TOTAL_MS:");
    add_stage("total", a[2]+0);
  }
  /预测=/ {
    pred++;
    if (last_ts > 0) {
      interval_count++;
    }
  }
  END {
    if (n > 0) {
      printf("[BENCH] model_count=%d\n", n);
      printf("[BENCH] model_avg_ms=%.3f\n", sum/n);
      printf("[BENCH] model_min_ms=%.3f\n", min);
      printf("[BENCH] model_max_ms=%.3f\n", max);
    } else {
      print("[BENCH] model_count=0");
    }
    printf("[BENCH] predict_lines=%d\n", pred);
    if (pred > 0) {
      printf("[BENCH] end_to_end_avg_ms=%.3f\n", duration * 1000.0 / pred);
      printf("[BENCH] end_to_end_rate_hz=%.3f\n", pred / duration);
    }
    order[1]="wait_frame";
    label[1]="wait_frame";
    order[2]="power_gate";
    label[2]="power_gate";
    order[3]="stft";
    label[3]="stft";
    order[4]="rknn";
    label[4]="rknn";
    order[5]="post";
    label[5]="post";
    order[6]="total";
    label[6]="total";
    print("[BENCH] stage timing avg/min/max ms:");
    for (i=1; i<=6; i++) {
      key=order[i];
      if (stage_n[key] > 0) {
        printf("[BENCH] stage_%s_avg_ms=%.3f min=%.3f max=%.3f n=%d\n",
               label[i], stage_sum[key]/stage_n[key], stage_min[key], stage_max[key], stage_n[key]);
      }
    }
  }
' "$LOG_FILE"

echo "[BENCH] result distribution:"
grep -a "预测=" "$LOG_FILE" | sed -n 's/.*预测=\([^ ]*\).*/\1/p' | sort | uniq -c || true
echo "[BENCH] raw log: $LOG_FILE"
