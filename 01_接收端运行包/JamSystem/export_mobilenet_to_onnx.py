#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from model_v2 import MobileNetV2


BASE_DIR = Path(__file__).resolve().parent
INPUT_SHAPE = (1, 1, 1024, 81)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export JamSystem MobileNetV2 .pth to ONNX.")
    parser.add_argument("--weights", default=str(BASE_DIR / "MobileNetV2_Mixed.pth"))
    parser.add_argument("--output", default=str(BASE_DIR / "mobilenet_interference.onnx"))
    args = parser.parse_args()

    weights = Path(args.weights)
    output = Path(args.output)
    if not weights.exists():
        raise FileNotFoundError(weights)

    model = MobileNetV2(num_classes=7)
    state = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*INPUT_SHAPE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[EXPORT] weights: {weights}")
    print(f"[EXPORT] onnx: {output}")
    print(f"[EXPORT] input shape: {INPUT_SHAPE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
