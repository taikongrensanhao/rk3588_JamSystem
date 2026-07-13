#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from IQ_STFT_python import IQSTFTAnalyzer


BASE_DIR = Path(__file__).resolve().parent
CLASSES = [
    "none",
    "single_tone",
    "narrowband",
    "wideband_barrage",
    "comb",
    "white_noise",
    "noise_fm",
]
INPUT_SHAPE = (1, 1, 1024, 81)


def infer_label(path: Path) -> str:
    text = "/".join(part.lower() for part in path.parts)
    for label in CLASSES:
        if label in text:
            return label
    return ""


def load_bin_iq(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.int16)
    if raw.size < 2:
        raise ValueError(f"empty IQ file: {path}")
    if raw.size % 2:
        raw = raw[:-1]
    iq = raw.reshape(-1, 2).astype(np.float32)
    return iq[:, 0] + 1j * iq[:, 1]


def stft_from_bin(path: Path) -> np.ndarray:
    analyzer = IQSTFTAnalyzer(fs=6.4e6, nperseg=1024, noverlap=512)
    iq = load_bin_iq(path)
    result = analyzer.stft_analysis(iq)
    return result["magnitude_db"].astype(np.float32)


def as_model_input(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == (1024, 81):
        arr = arr[np.newaxis, np.newaxis, :, :]
    elif arr.shape == (1, 1024, 81):
        arr = arr[np.newaxis, :, :, :]
    elif arr.shape == INPUT_SHAPE:
        pass
    else:
        raise ValueError(f"unsupported calibration shape: {arr.shape}")
    return arr.astype(np.float32)


def collect_cached_stft(root: Path) -> List[Tuple[str, Path]]:
    if not root.exists():
        return []
    samples: List[Tuple[str, Path]] = []
    for cache_dir in root.rglob("stft_cache*"):
        if not cache_dir.is_dir():
            continue
        for path in cache_dir.rglob("*.npy"):
            label = infer_label(path)
            if label:
                samples.append((label, path))
    return samples


def collect_bin_samples(root: Path) -> List[Tuple[str, Path]]:
    if not root.exists():
        return []
    samples: List[Tuple[str, Path]] = []

    meta_path = root / "metadata.csv"
    if meta_path.exists():
        iq_roots = [root / "templates", root / "iq_data", root]
        with meta_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                label = str(row.get("interference_type", "")).strip()
                iq_file = str(row.get("iq_file", "")).strip()
                if label not in CLASSES or not iq_file:
                    continue
                for iq_root in iq_roots:
                    path = iq_root / iq_file
                    if path.exists() and path.suffix.lower() == ".bin":
                        samples.append((label, path))
                        break

    for path in root.rglob("*.bin"):
        if "stft_cache" in {p.lower() for p in path.parts}:
            continue
        label = infer_label(path)
        if label:
            samples.append((label, path))
    return samples


def pick_balanced(samples: List[Tuple[str, Path]], max_total: int, max_per_class: int) -> List[Tuple[str, Path]]:
    rng = random.Random(2026)
    by_class: Dict[str, List[Path]] = {label: [] for label in CLASSES}
    for label, path in samples:
        if label in by_class:
            by_class[label].append(path)

    picked: List[Tuple[str, Path]] = []
    per_class = max(1, min(max_per_class, max_total // max(len(CLASSES), 1)))
    for label in CLASSES:
        paths = sorted(set(by_class[label]))
        rng.shuffle(paths)
        picked.extend((label, path) for path in paths[:per_class])

    if len(picked) < max_total:
        remaining = [(label, path) for label, paths in by_class.items() for path in sorted(set(paths))]
        seen = {path for _, path in picked}
        remaining = [(label, path) for label, path in remaining if path not in seen]
        rng.shuffle(remaining)
        picked.extend(remaining[: max_total - len(picked)])
    return picked[:max_total]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare RKNN calibration .npy dataset for JamSystem.")
    parser.add_argument("--output", default=str(BASE_DIR / "quant_dataset.txt"))
    parser.add_argument("--quant-dir", default=str(BASE_DIR / "rknn_quant_data"))
    parser.add_argument("--sim-data-dir", default="")
    parser.add_argument("--base-data-dir", default=str(BASE_DIR / "dataset_realworld"))
    parser.add_argument("--data-dir", default=str(BASE_DIR / "dataset_realtime"))
    parser.add_argument("--max-samples", type=int, default=70)
    parser.add_argument("--max-per-class", type=int, default=10)
    args = parser.parse_args()

    roots = [Path(args.base_data_dir), Path(args.data_dir)]
    if args.sim_data_dir:
        roots.insert(0, Path(args.sim_data_dir))

    cached: List[Tuple[str, Path]] = []
    bins: List[Tuple[str, Path]] = []
    for root in roots:
        cached.extend(collect_cached_stft(root))
        bins.extend(collect_bin_samples(root))

    picked_cached = pick_balanced(cached, args.max_samples, args.max_per_class)
    need = args.max_samples - len(picked_cached)
    picked_bins = pick_balanced(bins, need, args.max_per_class) if need > 0 else []

    quant_dir = Path(args.quant_dir)
    quant_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for idx, (label, path) in enumerate(picked_cached + picked_bins):
        try:
            arr = np.load(str(path)) if path.suffix.lower() == ".npy" else stft_from_bin(path)
            arr = as_model_input(arr)
        except Exception as exc:
            print(f"[QUANT][WARN] skip {path}: {exc}")
            continue
        out = quant_dir / f"{idx:04d}_{label}.npy"
        np.save(str(out), arr)
        written.append(out.resolve())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(path) for path in written) + ("\n" if written else ""), encoding="utf-8")

    print(f"[QUANT] cached candidates: {len(cached)}")
    print(f"[QUANT] bin candidates: {len(bins)}")
    print(f"[QUANT] written: {len(written)}")
    print(f"[QUANT] dataset: {output}")
    return 0 if written else 2


if __name__ == "__main__":
    raise SystemExit(main())
