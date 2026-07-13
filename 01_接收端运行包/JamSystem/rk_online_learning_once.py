#!/usr/bin/env python3
"""
RK one-shot online learning client.

This script is designed to be launched by the Qt "在线学习" button:
  1. connect to the PC training service through the second Ethernet port;
  2. upload new local .bin samples, if any;
  3. ask the PC to train MobileNetV2 now;
  4. wait until training finishes;
  5. download and atomically deploy updated .rknn/.pth model files on RK.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CLASSES = [
    "wideband_barrage",
    "single_tone",
    "white_noise",
    "narrowband",
    "noise_fm",
    "comb",
    "none",
]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE = BASE_DIR / "rk_online_learning_state.json"
DEFAULT_PTH = BASE_DIR / "MobileNetV2_Mixed.pth"
DEFAULT_RKNN = BASE_DIR / "mobilenet_interference_rk3588.rknn"


def log(message: str) -> None:
    print(f"[ONLINE] {message}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"uploaded": {}, "pth_sha256": "", "rknn_sha256": ""}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def http_json(url: str, timeout: int) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def infer_label(path: Path) -> str:
    text = "/".join([p.lower() for p in path.parts] + [path.name.lower()])
    for label in CLASSES:
        if label in text:
            return label
    return ""


def upload_file(server: str, path: Path, label: str, mode: str, timeout: int) -> bool:
    params = urlencode({"label": label, "filename": path.name, "mode": mode})
    req = Request(
        f"{server.rstrip('/')}/upload?{params}",
        data=path.read_bytes(),
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return bool(payload.get("ok"))


def upload_new_samples(args, state: dict) -> int:
    upload_dir = Path(args.upload_dir)
    if not upload_dir.exists():
        log(f"新数据目录不存在，跳过上传: {upload_dir}")
        return 0

    sent = 0
    for path in sorted(upload_dir.rglob("*.bin")):
        digest = sha256_file(path)
        key = str(path)
        if state["uploaded"].get(key) == digest:
            continue

        label = args.label or infer_label(path)
        if label not in CLASSES:
            print(f"[WARN] 无法从文件名或目录识别类别，跳过: {path}", flush=True)
            continue

        log(f"上传新样本 {label}: {path.name}")
        if upload_file(args.server, path, label, args.mode, args.timeout):
            state["uploaded"][key] = digest
            sent += 1
    log(f"上传完成，新增 {sent} 个文件")
    return sent


def trigger_train(args) -> None:
    log(f"请求电脑端立即训练，模式: {args.mode}")
    params = urlencode({"mode": args.mode})
    payload = http_json(f"{args.server.rstrip('/')}/train-now?{params}", args.timeout)
    if not payload.get("ok"):
        raise RuntimeError(f"train-now failed: {payload}")
    log(f"电脑端训练状态: {payload.get('status', '')}")


def wait_train_finished(args) -> dict:
    deadline = time.time() + args.max_wait
    last_status = ""
    seen_training = False
    while time.time() < deadline:
        info = http_json(f"{args.server.rstrip('/')}/model-info", args.timeout)
        status = str(info.get("status", ""))
        training = bool(info.get("training", False))
        if status != last_status:
            log(f"训练服务: {status}, training={training}")
            last_status = status
        if training:
            seen_training = True
        elif seen_training or status in {"trained", "skip: validation accuracy dropped", "skip: not enough files"}:
            return info
        time.sleep(args.poll_interval)
    raise TimeoutError("等待电脑端训练超时")


def download_model(server: str, url_path: str, target: Path, timeout: int) -> str:
    url = f"{server.rstrip('/')}{url_path}"
    with urlopen(url, timeout=timeout) as resp:
        expected_sha = resp.headers.get("X-SHA256", "")
        tmp = target.with_suffix(target.suffix + ".download")
        with tmp.open("wb") as f:
            shutil.copyfileobj(resp, f)

    actual_sha = sha256_file(tmp)
    if expected_sha and expected_sha != actual_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"模型校验失败: {target.name}")

    if target.exists() and sha256_file(target) != actual_sha:
        backup = target.with_name(target.stem + f"_backup_{int(time.time())}" + target.suffix)
        shutil.copy2(target, backup)
        log(f"已备份旧模型: {backup.name}")

    tmp.replace(target)
    log(f"已部署模型: {target}")
    return actual_sha


def deploy_models(args, state: dict, info: dict) -> int:
    changed = 0
    if info.get("has_rknn"):
        sha = download_model(
            args.server,
            "/download/mobilenet_interference_rk3588.rknn",
            Path(args.rknn_path),
            args.timeout,
        )
        if sha != state.get("rknn_sha256"):
            state["rknn_sha256"] = sha
            changed += 1
    else:
        log("电脑端没有生成RKNN，跳过RKNN部署")

    if info.get("has_pth"):
        sha = download_model(
            args.server,
            "/download/MobileNetV2_Mixed.pth",
            Path(args.pth_path),
            args.timeout,
        )
        if sha != state.get("pth_sha256"):
            state["pth_sha256"] = sha
            changed += 1
    else:
        log("电脑端没有PTH模型，跳过PTH部署")

    return changed


def parse_args():
    parser = argparse.ArgumentParser(description="Trigger PC training and deploy updated model to RK.")
    parser.add_argument("--server", required=True, help="Example: http://192.168.137.2:8008")
    parser.add_argument("--upload-dir", default="/mnt/usb/JamRecords")
    parser.add_argument(
        "--mode",
        choices=["class_increment", "domain_increment", "mixed_train", "retrain"],
        default="class_increment",
        help="Online learning mode selected by the Qt test page.",
    )
    parser.add_argument("--label", default="", help="Force label when upload-dir is a flat single-class folder.")
    parser.add_argument("--pth-path", default=str(DEFAULT_PTH))
    parser.add_argument("--rknn-path", default=str(DEFAULT_RKNN))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-wait", type=int, default=1800)
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--train-now", action="store_true")
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_path = Path(args.state)
    state = load_state(state_path)

    try:
        health = http_json(f"{args.server.rstrip('/')}/health", args.timeout)
        log(f"已连接电脑训练服务: {health}")
        upload_new_samples(args, state)
        if args.train_now:
            trigger_train(args)
        info = wait_train_finished(args) if args.wait else http_json(f"{args.server.rstrip('/')}/model-info", args.timeout)
        status = str(info.get("status", ""))
        if status.startswith("train failed") or status.startswith("rknn convert failed"):
            raise RuntimeError(status)
        if status.startswith("skip:"):
            log(f"电脑端未产生新模型，跳过部署: {status}")
            save_state(state_path, state)
            return 2
        changed = deploy_models(args, state, info)
        save_state(state_path, state)
        log(f"在线学习流程完成，模型更新数量: {changed}")
        return 0
    except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
        print(f"[ERR] 在线学习失败: {exc}", flush=True)
        save_state(state_path, state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
