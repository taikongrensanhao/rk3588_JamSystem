#!/usr/bin/env python3
"""
OrthoReg incremental learning entry point for the JamSystem online-learning demo.

Modes:
  class_increment  - expand a 6-class QPSK base model to 7 classes by learning noise_fm.
  domain_increment - adapt the 7-class model to a new modulation domain, such as FM.

The original teacher scripts used hard-coded /root/autodl-tmp paths. This version
keeps the OrthoReg idea but reads the package folders used by the RK/PC workflow.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from IQ_STFT_python import IQSTFTAnalyzer
from model_v2 import MobileNetV2


LABEL_MAP_7 = {
    "none": 0,
    "single_tone": 1,
    "narrowband": 2,
    "wideband_barrage": 3,
    "comb": 4,
    "white_noise": 5,
    "noise_fm": 6,
}
LABEL_MAP_6 = {k: v for k, v in LABEL_MAP_7.items() if k != "noise_fm"}
ID_MAP_7 = {v: k for k, v in LABEL_MAP_7.items()}
CLASSES_7 = list(LABEL_MAP_7.keys())

BASE_DIR = Path(__file__).resolve().parent


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


def load_bin_iq(file_path: Path, target_len: int = 40960) -> np.ndarray:
    raw = np.fromfile(str(file_path), dtype=np.int16).astype(np.float32)
    if raw.size < 2:
        raise ValueError(f"empty IQ file: {file_path}")
    iq = (raw[0::2] + 1j * raw[1::2]) / 32767.0
    iq = iq - np.mean(iq)
    iq = iq[:target_len]
    if len(iq) < target_len:
        iq = np.pad(iq, (0, target_len - len(iq)))
    return iq.astype(np.complex64)


def cache_stft(bin_path: Path, cache_root: Path, source_root: Path) -> Path:
    rel = bin_path.relative_to(source_root)
    cache_path = cache_root / rel.with_suffix(".npy")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_mtime >= bin_path.stat().st_mtime:
        return cache_path

    analyzer = IQSTFTAnalyzer(fs=6.4e6, nperseg=1024, noverlap=512)
    iq = load_bin_iq(bin_path)
    result = analyzer.stft_analysis(iq)
    tmp = cache_path.with_suffix(".npy.tmp")
    np.save(str(tmp), result["magnitude_db"].astype(np.float32))
    saved = Path(str(tmp) + ".npy")
    if saved.exists():
        saved.replace(cache_path)
    elif tmp.exists():
        tmp.replace(cache_path)
    return cache_path


def infer_label(path: Path) -> str:
    parts = [p.lower() for p in path.parts] + [path.name.lower()]
    text = "/".join(parts)
    for label in CLASSES_7:
        if label in text:
            return label
    return ""


def collect_metadata_samples(root: Path, cache_name: str) -> List[Tuple[Path, int]]:
    root = root.resolve()
    meta_path = root / "metadata.csv"
    if not meta_path.exists():
        return []

    import pandas as pd

    df = pd.read_csv(meta_path)
    if "iq_file" not in df.columns or "interference_type" not in df.columns:
        print(f"[INC] simulation metadata missing required columns: {meta_path}")
        return []

    iq_roots = [root / "templates", root / "iq_data", root]
    cache_root = root / cache_name
    samples: List[Tuple[Path, int]] = []
    for _, row in df.iterrows():
        label = str(row["interference_type"])
        if label not in LABEL_MAP_7:
            continue
        rel = Path(str(row["iq_file"]))
        bin_path = None
        for iq_root in iq_roots:
            candidate = iq_root / rel
            if candidate.exists():
                bin_path = candidate
                break
        if bin_path is None:
            continue
        cache_path = cache_stft(bin_path, cache_root, root)
        samples.append((cache_path, LABEL_MAP_7[label]))
    print(f"[INC] loaded simulation/ideal samples: {len(samples)} from {root}")
    return samples


def collect_samples(roots: Sequence[Path], cache_name: str, mode: str, repeat: int = 1) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    repeat = max(1, int(repeat))
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        cache_root = root / cache_name
        for bin_path in sorted(root.rglob("*.bin")):
            if cache_name in bin_path.parts:
                continue
            label = infer_label(bin_path)
            if label not in LABEL_MAP_7:
                continue
            if mode == "class_increment" and label == "noise_fm":
                pass
            cache_path = cache_stft(bin_path, cache_root, root)
            samples.extend([(cache_path, LABEL_MAP_7[label])] * repeat)
    return samples


def split_samples(samples: List[Tuple[Path, int]], val_ratio: float, seed: int):
    by_label: Dict[int, List[Tuple[Path, int]]] = {}
    for item in samples:
        by_label.setdefault(item[1], []).append(item)

    rng = random.Random(seed)
    train: List[Tuple[Path, int]] = []
    val: List[Tuple[Path, int]] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) >= 4 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    if not train:
        train = samples[:]
    if not val:
        val = train[:]
    return train, val


class STFTDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        arr = np.load(path).astype(np.float32, copy=False)
        return torch.from_numpy(arr).unsqueeze(0), int(label)


def make_sampler(samples: Sequence[Tuple[Path, int]]) -> WeightedRandomSampler:
    counts: Dict[int, int] = {}
    for _, label in samples:
        counts[label] = counts.get(label, 0) + 1
    weights = [1.0 / counts[label] for _, label in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def load_matching_state(model: nn.Module, weights: Path, device: torch.device) -> None:
    if not weights.exists():
        print(f"[INC] weights not found, train from scratch: {weights}")
        return
    state = torch.load(str(weights), map_location=device)
    model_state = model.state_dict()
    merged = copy.deepcopy(model_state)
    loaded = 0
    skipped = []
    for key, value in state.items():
        if key not in model_state:
            skipped.append(key)
            continue
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            merged[key] = value
            loaded += 1
            continue
        if value.ndim == target.ndim and all(value.shape[i] <= target.shape[i] for i in range(value.ndim)):
            padded = target.detach().clone()
            slices = tuple(slice(0, value.shape[i]) for i in range(value.ndim))
            padded[slices] = value.to(padded.device)
            merged[key] = padded
            loaded += 1
            print(f"[INC] partially loaded {key}: {tuple(value.shape)} -> {tuple(target.shape)}")
            continue
        skipped.append(key)
    model.load_state_dict(merged, strict=False)
    print(f"[INC] loaded {loaded} tensors from {weights}")
    if skipped:
        print(f"[INC] skipped shape-mismatch tensors: {', '.join(skipped[:6])}")


def expand_state_for_orthoreg(init_state: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    expanded = {}
    current = model.state_dict()
    for name, value in current.items():
        if name not in init_state:
            expanded[name] = torch.zeros_like(value)
            continue
        old = init_state[name]
        if tuple(old.shape) == tuple(value.shape):
            expanded[name] = old.detach().clone()
            continue
        padded = torch.zeros_like(value)
        slices = tuple(slice(0, min(a, b)) for a, b in zip(old.shape, value.shape))
        padded[slices] = old[slices].to(padded.device)
        expanded[name] = padded
    return expanded


def ortho_reg_loss(model: nn.Module, init_state: Dict[str, torch.Tensor], lambda_ortho: float) -> torch.Tensor:
    reg_loss = None
    num_layers = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or "weight" not in name or param.dim() < 2:
            continue
        init = init_state.get(name)
        if init is None:
            init = torch.zeros_like(param)
        else:
            init = init.to(param.device)
        delta = param - init
        delta_2d = delta.view(delta.size(0), -1) if delta.dim() == 4 else delta
        gram = delta_2d.T @ delta_2d
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        layer_loss = torch.norm(gram - identity, p="fro") ** 2 / max(gram.size(0), 1)
        reg_loss = layer_loss if reg_loss is None else reg_loss + layer_loss
        num_layers += 1
    if reg_loss is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return lambda_ortho * reg_loss / max(num_layers, 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    matrix = np.zeros((7, 7), dtype=int)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = torch.argmax(model(imgs), dim=1)
            total += labels.numel()
            correct += int((pred == labels).sum().item())
            for gt, pd in zip(labels.cpu().tolist(), pred.cpu().tolist()):
                if 0 <= gt < 7 and 0 <= pd < 7:
                    matrix[gt, pd] += 1
    return 100.0 * correct / max(total, 1), matrix


def print_matrix(matrix: np.ndarray) -> None:
    print("[INC] confusion matrix")
    print("[INC] " + "".join(f"{name[:7]:>8}" for name in CLASSES_7))
    for idx, name in ID_MAP_7.items():
        row = "".join(f"{int(matrix[idx, j]):>8}" for j in range(7))
        total = max(int(matrix[idx].sum()), 1)
        acc = matrix[idx, idx] / total * 100
        print(f"[INC] {name:>16}{row}  {acc:.1f}%")


def class_counts(samples: Sequence[Tuple[Path, int]]) -> Dict[str, int]:
    counts = {name: 0 for name in CLASSES_7}
    for _, label in samples:
        counts[ID_MAP_7[label]] += 1
    return counts


def train(args) -> int:
    mode = args.mode
    if mode not in {"class_increment", "domain_increment"}:
        raise ValueError(f"unsupported mode: {mode}")

    samples = []
    if args.sim_data_dir:
        samples.extend(collect_metadata_samples(Path(args.sim_data_dir), "stft_cache_incremental"))
    samples.extend(collect_samples([Path(args.base_data_dir)], "stft_cache_incremental", mode))
    realtime_samples = collect_samples(
        [Path(args.data_dir)],
        "stft_cache_incremental",
        mode,
        repeat=args.realtime_repeat,
    )
    samples.extend(realtime_samples)
    counts = class_counts(samples)
    print(f"[INC] mode: {mode}")
    print(f"[INC] realtime repeat: {args.realtime_repeat}")
    print(f"[INC] sample counts: {counts}")
    if len(samples) < args.min_files:
        print(f"[INC] not enough files: {len(samples)} < {args.min_files}")
        return 2
    if mode == "class_increment" and counts.get("noise_fm", 0) < max(1, args.min_new_class_files):
        print(f"[INC] not enough noise_fm files for class increment: {counts.get('noise_fm', 0)}")
        return 2

    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    train_loader = DataLoader(
        STFTDataset(train_samples),
        batch_size=args.batch_size,
        sampler=make_sampler(train_samples),
        num_workers=0,
    )
    val_loader = DataLoader(STFTDataset(val_samples), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2(num_classes=7).to(device)

    if mode == "class_increment":
        start_weights = Path(args.class_base_weights)
    else:
        start_weights = Path(args.domain_base_weights)
        if not start_weights.exists():
            start_weights = Path(args.output)
    load_matching_state(model, start_weights, device)
    init_state = expand_state_for_orthoreg(copy.deepcopy(model.state_dict()), model)

    baseline_acc, baseline_matrix = evaluate(model, val_loader, device)
    print(f"[INC] baseline val_acc={baseline_acc:.2f}%")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_acc = -1.0
    best_state = None
    best_matrix = None

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            ce = criterion(logits, labels)
            ortho = ortho_reg_loss(model, init_state, args.lambda_ortho)
            loss = ce + ortho
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        val_acc, matrix = evaluate(model, val_loader, device)
        print(
            f"[INC] epoch {epoch + 1}/{args.epochs} "
            f"loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_acc={val_acc:.2f}% time={time.time() - t0:.1f}s"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_matrix = matrix

    threshold = baseline_acc + args.min_acc_delta
    if best_acc < threshold:
        print(f"[INC] skip replace: best={best_acc:.2f}% < threshold={threshold:.2f}%")
        print_matrix(best_matrix if best_matrix is not None else baseline_matrix)
        return 3

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        backup = output.with_name(output.stem + f"_backup_{int(time.time())}" + output.suffix)
        shutil.copy2(output, backup)
        print(f"[INC] backup old output: {backup}")
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(best_state, str(tmp))
    replace_file_allow_readonly(tmp, output)

    named_output = Path(args.class_output if mode == "class_increment" else args.domain_output)
    if named_output != output:
        shutil.copy2(output, named_output)

    manifest = {
        "version": int(time.time()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "model_file": output.name,
        "sha256": sha256_file(output),
        "baseline_val_acc": round(float(baseline_acc), 4),
        "best_val_acc": round(float(best_acc), 4),
        "counts": counts,
        "classes": CLASSES_7,
        "lambda_ortho": args.lambda_ortho,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    manifest_path = Path(args.manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INC] saved model: {output}")
    print(f"[INC] saved mode output: {named_output}")
    print(f"[INC] saved manifest: {manifest_path}")
    print_matrix(best_matrix if best_matrix is not None else baseline_matrix)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="OrthoReg class/domain incremental learning for JamSystem.")
    parser.add_argument("--mode", choices=["class_increment", "domain_increment"], required=True)
    parser.add_argument("--base-data-dir", default=str(BASE_DIR / "dataset_realworld"))
    parser.add_argument("--data-dir", default=str(BASE_DIR / "dataset_realtime"))
    parser.add_argument("--sim-data-dir", default=os.environ.get("MIX_SIM_DATA_DIR", ""))
    parser.add_argument("--class-base-weights", default=str(BASE_DIR / "MobileNetV2_QPSKDataTrain_6classes_IncludeNone.pth"))
    parser.add_argument("--domain-base-weights", default=str(BASE_DIR / "final_incremental_model.pth"))
    parser.add_argument("--output", default=str(BASE_DIR / "MobileNetV2_Mixed.pth"))
    parser.add_argument("--class-output", default=str(BASE_DIR / "final_incremental_model.pth"))
    parser.add_argument("--domain-output", default=str(BASE_DIR / "final_merged_model.pth"))
    parser.add_argument("--manifest", default=str(BASE_DIR / "model_version.json"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("JAMSYSTEM_INC_EPOCHS", "10")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("JAMSYSTEM_INC_BATCH", "32")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("JAMSYSTEM_INC_LR", "0.0001")))
    parser.add_argument("--lambda-ortho", type=float, default=float(os.environ.get("JAMSYSTEM_INC_LAMBDA", "10")))
    parser.add_argument("--val-ratio", type=float, default=float(os.environ.get("JAMSYSTEM_INC_VAL_RATIO", "0.15")))
    parser.add_argument("--min-files", type=int, default=int(os.environ.get("JAMSYSTEM_INC_MIN_FILES", "7")))
    parser.add_argument("--min-new-class-files", type=int, default=int(os.environ.get("JAMSYSTEM_INC_MIN_NEW_CLASS", "1")))
    parser.add_argument("--min-acc-delta", type=float, default=float(os.environ.get("JAMSYSTEM_INC_MIN_ACC_DELTA", "0.0")))
    parser.add_argument("--realtime-repeat", type=int, default=int(os.environ.get("JAMSYSTEM_REALTIME_REPEAT", "8")))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(train(parse_args()))
