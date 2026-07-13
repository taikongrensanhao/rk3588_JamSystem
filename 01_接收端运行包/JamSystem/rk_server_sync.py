#!/usr/bin/env python3
"""
RK side client: upload new bin files to the training server and pull updated models.

Typical use on RK:
  ./jam_env/bin/python3 rk_server_sync.py \
    --server http://192.168.137.2:8008 \
    --upload-dir /mnt/usb/JamRecords \
    --watch
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional
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
DEFAULT_STATE = BASE_DIR / "rk_server_sync_state.json"
DEFAULT_MODEL = BASE_DIR / "MobileNetV2_Mixed.pth"
DEFAULT_RKNN = BASE_DIR / "mobilenet_interference_rk3588.rknn"


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
    return {"uploaded": {}, "model_sha256": "", "rknn_sha256": ""}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def http_json(url: str, timeout: int = 10) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def infer_label(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts] + [path.name.lower()]
    joined = "/".join(parts)
    for label in CLASSES:
        if label in joined:
            return label
    return None


def upload_one(server: str, path: Path, label: str, timeout: int) -> bool:
    params = urlencode({"label": label, "filename": path.name})
    url = f"{server.rstrip('/')}/upload?{params}"
    data = path.read_bytes()
    req = Request(url, data=data, method="POST", headers={"Content-Type": "application/octet-stream"})
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return bool(payload.get("ok"))


def download_file(url: str, target: Path, timeout: int) -> Optional[str]:
    try:
        with urlopen(url, timeout=timeout) as resp:
            expected_sha = resp.headers.get("X-SHA256", "")
            tmp = target.with_suffix(target.suffix + ".download")
            with tmp.open("wb") as f:
                shutil.copyfileobj(resp, f)
            actual_sha = sha256_file(tmp)
            if expected_sha and expected_sha != actual_sha:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"sha256 mismatch for {target.name}")
            if target.exists():
                backup = target.with_name(target.stem + f"_backup_{int(time.time())}" + target.suffix)
                shutil.copy2(target, backup)
            tmp.replace(target)
            return actual_sha
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def sync_uploads(args, state: dict) -> int:
    upload_dir = Path(args.upload_dir)
    if not upload_dir.exists():
        return 0

    sent = 0
    for path in sorted(upload_dir.rglob("*.bin")):
        digest = sha256_file(path)
        key = str(path)
        if state["uploaded"].get(key) == digest:
            continue
        label = args.label or infer_label(path)
        if label not in CLASSES:
            print(f"[SYNC] skip no label: {path}")
            continue
        print(f"[SYNC] upload {label}: {path}")
        if upload_one(args.server, path, label, args.timeout):
            state["uploaded"][key] = digest
            sent += 1
    return sent


def sync_models(args, state: dict) -> int:
    info = http_json(f"{args.server.rstrip('/')}/model-info", timeout=args.timeout)
    manifest = info.get("manifest") or {}
    changed = 0

    if info.get("has_rknn"):
        sha = download_file(
            f"{args.server.rstrip('/')}/download/mobilenet_interference_rk3588.rknn",
            Path(args.rknn_path),
            args.timeout,
        )
        if sha and sha != state.get("rknn_sha256"):
            state["rknn_sha256"] = sha
            changed += 1
            print(f"[SYNC] updated RKNN: {args.rknn_path}")

    if info.get("has_pth"):
        remote_sha = manifest.get("sha256", "")
        if remote_sha and remote_sha == state.get("model_sha256"):
            return changed
        sha = download_file(
            f"{args.server.rstrip('/')}/download/MobileNetV2_Mixed.pth",
            Path(args.model_path),
            args.timeout,
        )
        if sha:
            state["model_sha256"] = sha
            changed += 1
            print(f"[SYNC] updated PTH: {args.model_path}")

    return changed


def run_once(args) -> None:
    state_path = Path(args.state)
    state = load_state(state_path)
    health = http_json(f"{args.server.rstrip('/')}/health", timeout=args.timeout)
    print(f"[SYNC] server ok: {health}")
    sent = sync_uploads(args, state) if args.upload_dir else 0
    changed = sync_models(args, state)
    save_state(state_path, state)
    print(f"[SYNC] done upload={sent} model_changed={changed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload RK records and pull updated JamSystem models.")
    parser.add_argument("--server", required=True, help="Example: http://192.168.137.2:8008")
    parser.add_argument("--upload-dir", default=os.environ.get("JAMSYSTEM_UPLOAD_DIR", ""))
    parser.add_argument("--label", default="", help="Force label for flat upload directory.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--rknn-path", default=str(DEFAULT_RKNN))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--interval", type=int, default=int(os.environ.get("JAMSYSTEM_SYNC_INTERVAL", "60")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("JAMSYSTEM_SYNC_TIMEOUT", "60")))
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        try:
            run_once(args)
        except (URLError, HTTPError, TimeoutError, RuntimeError, OSError) as exc:
            print(f"[SYNC] failed: {exc}")
        if not args.watch:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
