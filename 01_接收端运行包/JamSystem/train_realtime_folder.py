#!/usr/bin/env python3
"""
Incrementally train MobileNetV2 from a fixed folder of new IQ bin files.

Dataset layout:
  dataset_realworld/
    none/*.bin
    single_tone/*.bin
    ...

  dataset_realtime/
    none/*.bin
    single_tone/*.bin
    narrowband/*.bin
    wideband_barrage/*.bin
    comb/*.bin
    white_noise/*.bin
    noise_fm/*.bin

This script is intentionally conservative: it combines historical realworld
data and newly uploaded realtime data, starts from the current
MobileNetV2_Mixed.pth, trains for a few epochs, and replaces the model only
when validation accuracy does not drop.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from IQ_STFT_python import IQSTFTAnalyzer
from model_v2 import MobileNetV2


LABEL_MAP = {
    "none": 0,
    "single_tone": 1,
    "narrowband": 2,
    "wideband_barrage": 3,
    "comb": 4,
    "white_noise": 5,
    "noise_fm": 6,
}
ID_MAP = {value: key for key, value in LABEL_MAP.items()}
CLASSES = list(LABEL_MAP.keys())
FRAME_IQ_LEN = 40960
FRAME_INT16_LEN = FRAME_IQ_LEN * 2
FRAME_BYTES = FRAME_INT16_LEN * np.dtype(np.int16).itemsize

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DATA_DIR = BASE_DIR / "dataset_realworld"
DEFAULT_DATA_DIR = BASE_DIR / "dataset_realtime"
DEFAULT_MODEL_PATH = BASE_DIR / "MobileNetV2_Mixed.pth"
DEFAULT_OUTPUT_PATH = BASE_DIR / "MobileNetV2_Mixed.pth"
DEFAULT_MANIFEST_PATH = BASE_DIR / "model_version.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def replace_file_allow_readonly(src: Path, dst: Path) -> None:
    if dst.exists():
        try:
            dst.chmod(0o666)
        except OSError:
            pass
    try:
        src.replace(dst)
        return
    except PermissionError:
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))


def bin_frame_count(file_path: Path) -> int:
    size = file_path.stat().st_size
    if size <= 0:
        return 1
    return max(1, size // FRAME_BYTES)


def load_bin_iq(file_path: Path, frame_index: int = 0, target_len: int = FRAME_IQ_LEN) -> np.ndarray:
    raw = np.fromfile(
        str(file_path),
        dtype=np.int16,
        count=target_len * 2,
        offset=frame_index * FRAME_BYTES,
    ).astype(np.float32)
    if raw.size < 2:
        raise ValueError(f"empty IQ frame: {file_path} frame={frame_index}")
    iq = (raw[0::2] + 1j * raw[1::2]) / 32767.0
    iq = iq - np.mean(iq)
    iq = iq[:target_len]
    if len(iq) < target_len:
        iq = np.pad(iq, (0, target_len - len(iq)))
    return iq.astype(np.complex64)


def build_cache(data_dir: Path, cache_dir: Path) -> List[Tuple[Path, int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    analyzer = IQSTFTAnalyzer(fs=6.4e6, nperseg=1024, noverlap=512)
    samples: List[Tuple[Path, int]] = []

    if not data_dir.exists():
        return samples

    for label in CLASSES:
        cls_dir = data_dir / label
        cls_cache = cache_dir / label
        cls_cache.mkdir(parents=True, exist_ok=True)
        if not cls_dir.exists():
            continue

        for bin_path in sorted(cls_dir.glob("*.bin")):
            frame_count = bin_frame_count(bin_path)
            if frame_count > 1:
                print(f"[TRAIN] expand {label}/{bin_path.name}: {frame_count} frames")
            for frame_index in range(frame_count):
                if frame_count == 1:
                    cache_name = bin_path.stem + ".npy"
                else:
                    cache_name = f"{bin_path.stem}_f{frame_index:04d}.npy"
                cache_path = cls_cache / cache_name
                if not cache_path.exists() or cache_path.stat().st_mtime < bin_path.stat().st_mtime:
                    iq = load_bin_iq(bin_path, frame_index=frame_index)
                    result = analyzer.stft_analysis(iq)
                    tmp_path = cache_path.with_suffix(".npy.tmp")
                    np.save(str(tmp_path), result["magnitude_db"].astype(np.float32))
                    saved = Path(str(tmp_path) + ".npy")
                    if saved.exists():
                        saved.replace(cache_path)
                    elif tmp_path.exists():
                        tmp_path.replace(cache_path)
                samples.append((cache_path, LABEL_MAP[label]))

    return samples


class STFTFolderDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        arr = np.load(path).astype(np.float32, copy=False)
        return torch.from_numpy(arr).unsqueeze(0), label


def split_samples(samples: List[Tuple[Path, int]], val_ratio: float, seed: int):
    by_label: Dict[int, List[Tuple[Path, int]]] = {}
    for item in samples:
        by_label.setdefault(item[1], []).append(item)

    rng = random.Random(seed)
    train, val = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) >= 4 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    if not train:
        train = samples
    if not val:
        val = train[:]
    return train, val


def make_sampler(samples: Sequence[Tuple[Path, int]]) -> WeightedRandomSampler:
    counts: Dict[int, int] = {}
    for _, label in samples:
        counts[label] = counts.get(label, 0) + 1
    weights = [1.0 / counts[label] for _, label in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    per_label: Dict[int, List[int]] = {}
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = torch.argmax(model(imgs), dim=1)
            total += labels.numel()
            correct += int((pred == labels).sum().item())
            for gt, pd in zip(labels.cpu().tolist(), pred.cpu().tolist()):
                per_label.setdefault(gt, [0, 0])
                per_label[gt][1] += 1
                if gt == pd:
                    per_label[gt][0] += 1
    return 100.0 * correct / max(total, 1), per_label


def write_manifest(path: Path, model_path: Path, counts: Dict[str, int], val_acc: float, args) -> None:
    manifest = {
        "version": int(time.time()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": model_path.name,
        "sha256": sha256_file(model_path),
        "size": model_path.stat().st_size,
        "classes": CLASSES,
        "class_counts": counts,
        "val_acc": round(float(val_acc), 4),
        "baseline_val_acc": round(float(getattr(args, "baseline_val_acc", -1.0)), 4),
        "epochs": args.epochs,
        "lr": args.lr,
        "freeze_features": bool(args.freeze_features),
        "base_data_dir": str(args.base_data_dir),
        "realtime_data_dir": str(args.data_dir),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 from a fixed realtime dataset folder.")
    parser.add_argument("--base-data-dir", default=str(DEFAULT_BASE_DATA_DIR), help="Historical realworld dataset folder.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--pretrained", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("JAMSYSTEM_TRAIN_EPOCHS", "3")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("JAMSYSTEM_TRAIN_BATCH", "16")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("JAMSYSTEM_TRAIN_LR", "0.0001")))
    parser.add_argument("--val-ratio", type=float, default=float(os.environ.get("JAMSYSTEM_TRAIN_VAL_RATIO", "0.15")))
    parser.add_argument("--min-files", type=int, default=int(os.environ.get("JAMSYSTEM_TRAIN_MIN_FILES", "7")))
    parser.add_argument(
        "--min-acc-delta",
        type=float,
        default=float(os.environ.get("JAMSYSTEM_TRAIN_MIN_ACC_DELTA", "0.0")),
        help="Replace the model only when new_val_acc >= baseline_val_acc + this delta.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-features", action="store_true", default=os.environ.get("JAMSYSTEM_TRAIN_FREEZE", "0") == "1")
    args = parser.parse_args()

    base_data_dir = Path(args.base_data_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else data_dir / "stft_cache"
    base_cache_dir = base_data_dir / "stft_cache"
    pretrained = Path(args.pretrained).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()

    args.base_data_dir = base_data_dir
    args.data_dir = data_dir

    print(f"[TRAIN] base realworld dir: {base_data_dir}")
    print(f"[TRAIN] realtime data dir: {data_dir}")
    print(f"[TRAIN] base cache dir: {base_cache_dir}")
    print(f"[TRAIN] realtime cache dir: {cache_dir}")
    print(f"[TRAIN] pretrained: {pretrained if pretrained.exists() else 'not found, train from scratch'}")
    print(f"[TRAIN] output: {output}")

    base_samples = build_cache(base_data_dir, base_cache_dir)
    realtime_samples = build_cache(data_dir, cache_dir)
    samples = base_samples + realtime_samples

    base_counts = {label: 0 for label in CLASSES}
    realtime_counts = {label: 0 for label in CLASSES}
    counts = {label: 0 for label in CLASSES}
    for _, label_id in base_samples:
        base_counts[ID_MAP[label_id]] += 1
        counts[ID_MAP[label_id]] += 1
    for _, label_id in realtime_samples:
        realtime_counts[ID_MAP[label_id]] += 1
        counts[ID_MAP[label_id]] += 1
    print("[TRAIN] class counts:")
    for label in CLASSES:
        print(f"  {label:<18} base={base_counts[label]:>4} realtime={realtime_counts[label]:>4} total={counts[label]:>4}")

    if len(samples) < args.min_files:
        print(f"[TRAIN] not enough samples: {len(samples)} < {args.min_files}, skip")
        return 2

    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2(num_classes=len(CLASSES)).to(device)
    loaded_pretrained = False
    if pretrained.exists():
        state = torch.load(str(pretrained), map_location=device)
        model.load_state_dict(state, strict=False)
        loaded_pretrained = True
        print("[TRAIN] loaded pretrained weights")

    if args.freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
        print("[TRAIN] frozen feature extractor, training classifier only")

    train_ds = STFTFolderDataset(train_samples)
    val_ds = STFTFolderDataset(val_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train_samples),
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    baseline_acc = -1.0
    if loaded_pretrained:
        baseline_acc, _ = evaluate(model, val_loader, device)
        print(f"[TRAIN] baseline val_acc={baseline_acc:.2f}%")
    args.baseline_val_acc = baseline_acc

    best_state = None
    best_acc = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_acc, per_label = evaluate(model, val_loader, device)
        print(
            f"[TRAIN] epoch {epoch + 1}/{args.epochs} "
            f"loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_acc={val_acc:.2f}% time={time.time() - t0:.1f}s"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    replace_threshold = baseline_acc + args.min_acc_delta if baseline_acc >= 0 else -1.0
    if best_acc < replace_threshold:
        print(
            f"[TRAIN] skip replace: best_val_acc={best_acc:.2f}% "
            f"< threshold={replace_threshold:.2f}%"
        )
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if output.exists():
        backup = output.with_name(output.stem + f"_backup_{int(time.time())}" + output.suffix)
        shutil.copy2(output, backup)
        print(f"[TRAIN] backup old model: {backup}")

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    torch.save(best_state, str(tmp_output))
    replace_file_allow_readonly(tmp_output, output)
    write_manifest(manifest, output, counts, best_acc, args)
    print(f"[TRAIN] saved model: {output}")
    print(f"[TRAIN] saved manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
