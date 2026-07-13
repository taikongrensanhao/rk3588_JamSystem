# JamSystem Final Stable Config - 2026-06-24

## Receiver

- Board role: receiver only
- RF center frequency: 200 MHz
- AD9361 URI: ip:192.168.1.10
- RX gain default: 45 dB
- Fast backend: fast_rx_rknn
- Fast STFT stride: 2
- Fast single-tone/narrowband hard refine: disabled
- No-signal gate default in GUI launcher: disabled

Recommended speed test:

```bash
cd /home/pi/Desktop/run/JamSystem
JAMSYSTEM_RX_GAIN_DB=45 \
JAMSYSTEM_FAST_STFT_STRIDE=2 \
./speed_benchmark.sh 20
```

The benchmark prints per-stage timing:

- wait_frame: waiting for a fresh AD9361 frame
- power_gate: power estimation and no-signal gate
- stft: IQ to STFT image preprocessing
- rknn: RKNN inference
- post: postprocess / label refinement
- total: wait_frame plus processing time

## Transmitter

- Board role: transmitter only
- Default switch period: 25 ms
- Loop mode keeps normal visible sequence in GUI.
- Internal loop weighting increases wideband_barrage occurrence for speed display.

## Demo Notes

- Stable switching display: 25-30 ms.
- Do not run SSH backend and GUI backend at the same time.
- If recognition power looks wrong, check RF cable first, then RX gain.
- If ADC clips, reduce RX gain. Check with edge_count near zero.
