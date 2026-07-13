#!/usr/bin/env python3
import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def build_rknn(onnx: Path, output: Path, dataset: Path, target: str, do_quant: bool) -> int:
    try:
        from rknn.api import RKNN
    except Exception as exc:
        print(f"[RKNN][ERR] rknn-toolkit2 is not available: {exc}")
        print("[RKNN][ERR] Install RKNN-Toolkit2 in this Python environment, or run conversion on a Linux/WSL/XTerminal environment that has it.")
        return 10

    rknn = RKNN(verbose=False)

    print("[RKNN] config")
    ret = rknn.config(
        target_platform=target,
        quantized_dtype="asymmetric_quantized-8",
        optimization_level=3,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config failed: {ret}")

    print("[RKNN] load onnx")
    ret = rknn.load_onnx(model=str(onnx))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed: {ret}")

    if do_quant:
        print(f"[RKNN] build with quant dataset: {dataset}")
        ret = rknn.build(do_quantization=True, dataset=str(dataset))
    else:
        print("[RKNN] build without quantization")
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"rknn.build failed: {ret}")

    print(f"[RKNN] export: {output}")
    ret = rknn.export_rknn(str(output))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed: {ret}")

    rknn.release()
    print("[RKNN] done")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert JamSystem ONNX model to RKNN for RK3588.")
    parser.add_argument("--onnx", default=str(BASE_DIR / "mobilenet_interference.onnx"))
    parser.add_argument("--output", default=str(BASE_DIR / "mobilenet_interference_rk3588.rknn"))
    parser.add_argument("--dataset", default=str(BASE_DIR / "quant_dataset.txt"))
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--fallback-no-quant", action="store_true", default=True)
    args = parser.parse_args()

    onnx = Path(args.onnx)
    output = Path(args.output)
    dataset = Path(args.dataset)
    if not onnx.exists():
        raise FileNotFoundError(onnx)

    output.parent.mkdir(parents=True, exist_ok=True)
    do_quant = (not args.no_quant) and dataset.exists()
    try:
        return build_rknn(onnx, output, dataset, args.target, do_quant)
    except Exception as exc:
        if do_quant and args.fallback_no_quant:
            print(f"[RKNN][WARN] quantized build failed, retry without quantization: {exc}")
            return build_rknn(onnx, output, dataset, args.target, False)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
