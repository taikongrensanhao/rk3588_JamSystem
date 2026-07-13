"""
混合训练：仿真数据 + SDR 实采数据

默认目录约定：
1. 本脚本位于 `.../rk3588/train_realworld.py`
2. 仿真数据位于同级上层目录下的 `干扰样式识别流程图及参考代码/`
   其中包含：
   - metadata.csv
   - templates/*.bin
3. 实采数据位于本脚本同级目录下的 `dataset_realworld/`

如果实际目录不同，可通过环境变量覆盖：
- MIX_SIM_DATA_DIR
- MIX_REAL_DATA_DIR
- MIX_PRETRAINED_PATH
- MIX_SAVE_PATH
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIM_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "干扰样式识别流程图及参考代码"))
REAL_DATA_DIR = os.environ.get("MIX_REAL_DATA_DIR", os.path.join(BASE_DIR, "dataset_realworld"))
SAVE_PATH = os.environ.get("MIX_SAVE_PATH", os.path.join(BASE_DIR, "MobileNetV2_Mixed.pth"))

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
NUM_WORKERS = 0
TEST_SIZE = 0.1
RANDOM_STATE = 42


def resolve_sim_paths() -> Tuple[str, str, str, str]:
    """自动识别当前要使用的仿真数据目录。"""
    candidates = [
        os.environ.get("MIX_SIM_DATA_DIR"),
        DEFAULT_SIM_ROOT,
        BASE_DIR,
    ]

    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)

        direct_csv = os.path.join(candidate, "metadata.csv")
        direct_templates = os.path.join(candidate, "templates")
        if os.path.exists(direct_csv) and os.path.isdir(direct_templates):
            return (
                candidate,
                direct_csv,
                direct_templates,
                os.path.join(candidate, "stft_cache"),
            )

        legacy_root = os.path.join(candidate, "rf_interference_dataset")
        legacy_csv = os.path.join(legacy_root, "metadata.csv")
        legacy_iq = os.path.join(legacy_root, "iq_data")
        if os.path.exists(legacy_csv) and os.path.isdir(legacy_iq):
            return (
                legacy_root,
                legacy_csv,
                legacy_iq,
                os.path.join(legacy_root, "stft_cache"),
            )

    raise FileNotFoundError(
        "未找到可用的仿真数据目录。请确认 metadata.csv 与 templates/ 已生成，"
        "或通过环境变量 MIX_SIM_DATA_DIR 指定目录。"
    )


SIM_DATA_DIR, SIM_CSV, SIM_IQ_DIR, SIM_CACHE_DIR = resolve_sim_paths()
REAL_CACHE_DIR = os.path.join(REAL_DATA_DIR, "stft_cache")


def resolve_pretrained_path() -> Optional[str]:
    candidates = [
        os.environ.get("MIX_PRETRAINED_PATH"),
        os.path.join(BASE_DIR, "MobileNetV2_NewData_Bin.pth"),
        os.path.join(SIM_DATA_DIR, "MobileNetV2_NewDataV2.pth"),
        os.path.join(BASE_DIR, "MobileNetV2_Mixed.pth"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


PRETRAINED_PATH = resolve_pretrained_path()


def load_bin_iq(file_path: str, target_len: int = 40960) -> np.ndarray:
    raw = np.fromfile(file_path, dtype=np.int16).astype(np.float32)
    iq = (raw[0::2] + 1j * raw[1::2]) / 32767.0
    iq = iq - np.mean(iq)
    iq = iq[:target_len]
    if len(iq) < target_len:
        iq = np.pad(iq, (0, target_len - len(iq)))
    return iq.astype(np.complex64)


def build_realworld_cache(data_dir: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    analyzer = IQSTFTAnalyzer(fs=6.4e6, nperseg=1024, noverlap=512)
    rows: List[Dict[str, str]] = []
    total_new = 0

    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"  [WARN] 不存在目录: {cls_dir}")
            continue

        cls_cache = os.path.join(cache_dir, cls)
        os.makedirs(cls_cache, exist_ok=True)
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".bin")])

        for fname in files:
            rel = f"{cls}/{fname}"
            rows.append({"iq_file": rel, "interference_type": cls, "source": "sdr_realworld"})

            cache_path = os.path.join(cls_cache, fname.replace(".bin", ".npy"))
            if os.path.exists(cache_path):
                continue

            iq = load_bin_iq(os.path.join(cls_dir, fname))
            result = analyzer.stft_analysis(iq)
            np.save(cache_path, result["magnitude_db"].astype(np.float32))
            total_new += 1

        print(f"  {cls:<18}: {len(files)} 个实采文件")

    meta_path = os.path.join(data_dir, "metadata_generated.csv")
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    print(f"  新增缓存 {total_new} 个，metadata 写入: {meta_path}")
    return meta_path


def ensure_sim_cache(csv_path: str, iq_dir: str, cache_dir: str) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    metadata = pd.read_csv(csv_path)
    analyzer = IQSTFTAnalyzer(fs=6.4e6, nperseg=1024, noverlap=512)

    if "iq_file" not in metadata.columns:
        raise KeyError(f"{csv_path} 中缺少 iq_file 列")

    need = any(
        not os.path.exists(os.path.join(cache_dir, str(row["iq_file"]).replace(".bin", ".npy")))
        for _, row in metadata.iterrows()
    )
    if not need:
        print(f"  仿真缓存已存在（{len(metadata)} 个），跳过计算")
        return metadata

    print(f"  开始计算仿真 STFT 缓存（{len(metadata)} 个）...")
    t0 = time.time()
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), ncols=80):
        iq_name = str(row["iq_file"])
        cache_path = os.path.join(cache_dir, iq_name.replace(".bin", ".npy"))
        if os.path.exists(cache_path):
            continue

        iq_path = os.path.join(iq_dir, iq_name)
        iq = load_bin_iq(iq_path)
        result = analyzer.stft_analysis(iq)
        np.save(cache_path, result["magnitude_db"].astype(np.float32))

    print(f"  仿真缓存完成，耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return metadata


class CachedSTFTDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, cache_dir: str):
        self.meta = meta.reset_index(drop=True)
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int):
        row = self.meta.iloc[index]
        cache_file = os.path.join(self.cache_dir, str(row["iq_file"]).replace(".bin", ".npy"))
        stft = np.load(cache_file)
        label = LABEL_MAP[str(row["interference_type"])]
        return torch.FloatTensor(stft).unsqueeze(0), label


def resample_per_class(meta: pd.DataFrame, target: int) -> pd.DataFrame:
    parts = []
    for cls in CLASSES:
        rows = meta[meta["interference_type"] == cls]
        count = len(rows)
        if count == 0:
            continue
        if count >= target:
            parts.append(rows.sample(target, random_state=RANDOM_STATE))
            continue

        reps = target // count
        rem = target % count
        block = [rows] * reps
        if rem > 0:
            block.append(rows.sample(rem, random_state=RANDOM_STATE))
        parts.append(pd.concat(block, ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def print_confusion_matrix(labels: List[int], preds: List[int]) -> None:
    names = [ID_MAP[i] for i in range(len(CLASSES))]
    cm = np.zeros((len(names), len(names)), dtype=int)
    for label, pred in zip(labels, preds):
        cm[label][pred] += 1

    print(f"\n{'':>18}" + "".join(f"{name[:7]:>8}" for name in names))
    for i, name in enumerate(names):
        total = max(cm[i].sum(), 1)
        acc = cm[i][i] / total * 100
        row = f"{name:>18}" + "".join(f"{cm[i][j]:>8}" for j in range(len(names)))
        print(f"{row}  {acc:.0f}%")
    print(f"总体准确率: {np.trace(cm) / max(cm.sum(), 1) * 100:.1f}%\n")


def make_mixed_dataset(meta: pd.DataFrame) -> ConcatDataset:
    parts = []
    sim_meta = meta[meta["source"] == "simulation"]
    real_meta = meta[meta["source"] == "sdr_realworld"]

    if len(sim_meta) > 0:
        parts.append(CachedSTFTDataset(sim_meta, SIM_CACHE_DIR))
    if len(real_meta) > 0:
        parts.append(CachedSTFTDataset(real_meta, REAL_CACHE_DIR))

    if not parts:
        raise RuntimeError("没有可用于训练的数据")
    return ConcatDataset(parts)


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    print("\n[0] 检查路径")
    print(f"  仿真目录: {SIM_DATA_DIR}")
    print(f"  仿真 metadata: {SIM_CSV}")
    print(f"  仿真 IQ 目录: {SIM_IQ_DIR}")
    print(f"  实采目录: {REAL_DATA_DIR}")
    print(f"  预训练权重: {PRETRAINED_PATH if PRETRAINED_PATH else '未找到，将从零训练'}")

    if not os.path.exists(SIM_CSV):
        raise FileNotFoundError(f"仿真 metadata 不存在: {SIM_CSV}")
    if not os.path.exists(SIM_IQ_DIR):
        raise FileNotFoundError(f"仿真 IQ 目录不存在: {SIM_IQ_DIR}")
    if not os.path.exists(REAL_DATA_DIR):
        raise FileNotFoundError(f"实采目录不存在: {REAL_DATA_DIR}")

    print("\n[1] 构建 STFT 缓存")
    print("  仿真缓存:")
    sim_meta = ensure_sim_cache(SIM_CSV, SIM_IQ_DIR, SIM_CACHE_DIR)
    print("  实采缓存:")
    real_csv = build_realworld_cache(REAL_DATA_DIR, REAL_CACHE_DIR)

    sim_meta = sim_meta.copy()
    sim_meta["source"] = "simulation"
    real_meta = pd.read_csv(real_csv)
    if "source" not in real_meta.columns:
        real_meta["source"] = "sdr_realworld"

    print("\n[2] 数据统计")
    print(f"  仿真样本数: {len(sim_meta)}")
    print(f"  实采样本数: {len(real_meta)}")
    for cls in CLASSES:
        sim_count = len(sim_meta[sim_meta["interference_type"] == cls])
        real_count = len(real_meta[real_meta["interference_type"] == cls])
        print(f"    {cls:<18} 仿真={sim_count:>4}  实采={real_count:>4}")

    sim_target = int(sim_meta["interference_type"].value_counts().max())
    real_balanced = resample_per_class(real_meta, sim_target)
    print(f"\n  实采平衡到每类 {sim_target} 个: {len(real_meta)} -> {len(real_balanced)}")

    all_meta = pd.concat([sim_meta, real_balanced], ignore_index=True)
    print(f"  混合总数: {len(all_meta)}")

    train_idx, val_idx = train_test_split(
        range(len(all_meta)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_meta["interference_type"],
    )
    train_meta = all_meta.iloc[train_idx].reset_index(drop=True)
    val_meta = all_meta.iloc[val_idx].reset_index(drop=True)

    train_ds = make_mixed_dataset(train_meta)
    val_ds = make_mixed_dataset(val_meta)
    print(f"\n  训练集: {len(train_ds)}")
    print(f"  验证集: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print("\n[3] 构建模型")
    model = MobileNetV2(num_classes=len(CLASSES)).to(device)
    if PRETRAINED_PATH:
        state = torch.load(PRETRAINED_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"  已加载预训练权重: {PRETRAINED_PATH}")
    else:
        print("  未找到可用预训练权重，将从零开始训练")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 100)

    best_acc = 0.0
    patience = 8
    no_improve = 0

    print(f"\n[4] 开始训练（{EPOCHS} 轮）")
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch + 1:02d}/{EPOCHS}", ncols=80):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        model.eval()
        all_labels: List[int] = []
        all_preds: List[int] = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = torch.argmax(model(imgs), dim=1)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        val_acc = 100.0 * np.mean(np.array(all_labels) == np.array(all_preds))
        train_acc = 100.0 * train_correct / max(train_total, 1)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(
            f"Ep {epoch + 1:02d}  loss={avg_loss:.4f}  "
            f"train={train_acc:.1f}%  val={val_acc:.1f}%  "
            f"time={time.time() - t0:.0f}s"
        )
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  已保存最佳权重: {SAVE_PATH} (best={best_acc:.1f}%)")
            print_confusion_matrix(all_labels, all_preds)
        else:
            no_improve += 1
            print(f"  未提升 ({no_improve}/{patience})")
            if no_improve >= patience:
                print("触发早停")
                break

    print(f"\n训练完成，最佳验证准确率: {best_acc:.1f}%")
    print(f"权重已保存到: {SAVE_PATH}")


if __name__ == "__main__":
    train()
