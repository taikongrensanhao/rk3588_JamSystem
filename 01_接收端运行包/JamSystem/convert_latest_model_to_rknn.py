#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run(cmd):
    print("[AUTO_RKNN]", " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(BASE_DIR))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export latest JamSystem PTH model and convert it to RKNN.")
    parser.add_argument("--weights", default=str(BASE_DIR / "MobileNetV2_Mixed.pth"))
    parser.add_argument("--onnx", default=str(BASE_DIR / "mobilenet_interference.onnx"))
    parser.add_argument("--rknn", default=str(BASE_DIR / "mobilenet_interference_rk3588.rknn"))
    parser.add_argument("--dataset", default=str(BASE_DIR / "quant_dataset.txt"))
    parser.add_argument("--sim-data-dir", default=os.environ.get("JAMSYSTEM_SIM_DATA_DIR", os.environ.get("MIX_SIM_DATA_DIR", "")))
    parser.add_argument("--base-data-dir", default=os.environ.get("JAMSYSTEM_REALWORLD_DATA_DIR", str(BASE_DIR / "dataset_realworld")))
    parser.add_argument("--data-dir", default=os.environ.get("JAMSYSTEM_REALTIME_DATA_DIR", str(BASE_DIR / "dataset_realtime")))
    parser.add_argument("--target", default=os.environ.get("JAMSYSTEM_RKNN_TARGET", "rk3588"))
    parser.add_argument("--max-quant-samples", type=int, default=int(os.environ.get("JAMSYSTEM_RKNN_QUANT_SAMPLES", "70")))
    parser.add_argument("--no-quant", action="store_true", default=os.environ.get("JAMSYSTEM_RKNN_NO_QUANT", "0") == "1")
    args = parser.parse_args()

    py = sys.executable
    run([py, str(BASE_DIR / "export_mobilenet_to_onnx.py"), "--weights", args.weights, "--output", args.onnx])

    convert_cmd = [
        py,
        str(BASE_DIR / "convert_onnx_to_rknn.py"),
        "--onnx",
        args.onnx,
        "--output",
        args.rknn,
        "--target",
        args.target,
    ]
    if args.no_quant:
        convert_cmd.append("--no-quant")
    else:
        quant_cmd = [
            py,
            str(BASE_DIR / "prepare_rknn_quant_dataset.py"),
            "--output",
            args.dataset,
            "--base-data-dir",
            args.base_data_dir,
            "--data-dir",
            args.data_dir,
            "--max-samples",
            str(args.max_quant_samples),
        ]
        if args.sim_data_dir:
            quant_cmd.extend(["--sim-data-dir", args.sim_data_dir])
        ret = subprocess.run(quant_cmd, cwd=str(BASE_DIR))
        if ret.returncode == 0:
            convert_cmd.extend(["--dataset", args.dataset])
        else:
            print("[AUTO_RKNN][WARN] quant dataset unavailable, converting without quantization")
            convert_cmd.append("--no-quant")

    run(convert_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
