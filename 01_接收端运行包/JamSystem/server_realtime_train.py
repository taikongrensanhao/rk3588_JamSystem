#!/usr/bin/env python3
"""
Small HTTP server for realtime data upload and autonomous model training.

Endpoints:
  GET  /health
  GET  /model-info
  GET  /download/MobileNetV2_Mixed.pth
  GET  /download/mobilenet_interference_rk3588.rknn
  POST /upload?label=narrowband&filename=sample.bin

The server also watches a fixed dataset folder. Whenever new .bin files are
added, it runs train_realtime_folder.py and updates model_version.json.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse


CLASSES = [
    "none",
    "single_tone",
    "narrowband",
    "wideband_barrage",
    "comb",
    "white_noise",
    "noise_fm",
]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "dataset_realtime"
DEFAULT_BASE_DATA_DIR = BASE_DIR / "dataset_realworld"
DEFAULT_MODEL_PATH = BASE_DIR / "MobileNetV2_Mixed.pth"
DEFAULT_BASELINE_MODEL_PATH = BASE_DIR / "MobileNetV2_Baseline.pth"
DEFAULT_RKNN_PATH = BASE_DIR / "mobilenet_interference_rk3588.rknn"
DEFAULT_MANIFEST_PATH = BASE_DIR / "model_version.json"
DEFAULT_TRAIN_SCRIPT = BASE_DIR / "train_mixed_sim_realworld.py"
DEFAULT_REALTIME_RETRAIN_SCRIPT = BASE_DIR / "train_realtime_folder.py"
DEFAULT_INCREMENTAL_SCRIPT = BASE_DIR / "train_incremental_orthoreg.py"
FRAME_BYTES = 40960 * 2 * 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(name: str) -> str:
    name = Path(unquote(name)).name
    keep = []
    for ch in name:
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or f"upload_{int(time.time())}.bin"


def count_bins(data_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in CLASSES:
        total = 0
        for bin_path in (data_dir / label).glob("*.bin"):
            total += max(1, bin_path.stat().st_size // FRAME_BYTES)
        counts[label] = total
    return counts


def rknn_is_current(rknn_path: Path, pth_path: Path) -> bool:
    if not rknn_path.exists():
        return False
    if not pth_path.exists():
        return True
    return rknn_path.stat().st_mtime >= pth_path.stat().st_mtime


class TrainState:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.training = False
        self.last_train_time = 0.0
        self.last_signature = ""
        self.last_status = "idle"
        self.last_log = ""
        self.last_mode = "class_increment"

    def signature(self) -> str:
        parts = []
        for root_name, root in (("base", Path(self.args.base_data_dir)), ("realtime", Path(self.args.data_dir))):
            for label in CLASSES:
                cls_dir = root / label
                for p in sorted(cls_dir.glob("*.bin")):
                    st = p.stat()
                    parts.append(f"{root_name}/{label}/{p.name}:{st.st_size}:{int(st.st_mtime)}")
        raw = "\n".join(parts).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def should_train(self) -> bool:
        sig = self.signature()
        total = sum(count_bins(Path(self.args.base_data_dir)).values()) + sum(count_bins(Path(self.args.data_dir)).values())
        if total < self.args.min_files:
            return False
        if sig == self.last_signature:
            return False
        if time.time() - self.last_train_time < self.args.cooldown:
            return False
        return True

    def train_async(self, reason: str, mode: str = "class_increment") -> bool:
        with self.lock:
            if self.training:
                return False
            self.last_mode = mode
            self.training = True
            self.last_status = f"training({mode}): {reason}"
        t = threading.Thread(target=self._run_train, args=(reason, mode), daemon=True)
        t.start()
        return True

    def _run_train(self, reason: str, mode: str) -> None:
        try:
            if mode in {"class_increment", "domain_increment"}:
                cmd = [
                    sys.executable,
                    str(self.args.incremental_script),
                    "--mode",
                    mode,
                    "--base-data-dir",
                    str(self.args.base_data_dir),
                    "--data-dir",
                    str(self.args.data_dir),
                    "--output",
                    str(self.args.model_path),
                    "--manifest",
                    str(self.args.manifest_path),
                    "--epochs",
                    str(self.args.inc_epochs),
                    "--batch-size",
                    str(self.args.batch_size),
                    "--min-files",
                    str(self.args.min_files),
                    "--realtime-repeat",
                    str(self.args.realtime_repeat),
                ]
                if self.args.sim_data_dir:
                    cmd.extend(["--sim-data-dir", str(self.args.sim_data_dir)])
            elif mode in {"mixed_train"}:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("MIX_REAL_DATA_DIR", str(self.args.base_data_dir))
                env.setdefault("MIX_SAVE_PATH", str(self.args.model_path))
                if self.args.sim_data_dir:
                    env.setdefault("MIX_SIM_DATA_DIR", str(self.args.sim_data_dir))
                cmd = [
                    sys.executable,
                    str(self.args.train_script),
                ]
            else:
                retrain_pretrained = (
                    self.args.baseline_model_path
                    if self.args.baseline_model_path.exists()
                    else self.args.model_path
                )
                cmd = [
                    sys.executable,
                    str(self.args.realtime_retrain_script),
                    "--base-data-dir",
                    str(self.args.base_data_dir),
                    "--data-dir",
                    str(self.args.data_dir),
                    "--pretrained",
                    str(retrain_pretrained),
                    "--output",
                    str(self.args.model_path),
                    "--manifest",
                    str(self.args.manifest_path),
                    "--epochs",
                    str(self.args.epochs),
                    "--batch-size",
                    str(self.args.batch_size),
                    "--min-files",
                    str(self.args.min_files),
                ]
                if self.args.freeze_features:
                    cmd.append("--freeze-features")

            if "env" not in locals():
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
            print(f"[SERVER] start training, mode={mode}, reason={reason}")
            proc = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.last_log = proc.stdout[-8000:]
            if proc.returncode == 0:
                self.last_signature = self.signature()
                self.last_status = f"trained: {mode}"
                print("[SERVER] training done")
                self._try_convert_rknn()
            elif proc.returncode == 2:
                self.last_status = "skip: not enough files"
                print("[SERVER] training skipped: not enough files")
            elif proc.returncode == 3:
                self.last_signature = self.signature()
                self.last_status = f"skip: validation accuracy dropped ({mode})"
                print("[SERVER] training skipped: validation accuracy dropped")
            else:
                self.last_status = f"train failed: {proc.returncode}"
                print(self.last_log)
        except Exception as exc:
            self.last_status = f"train exception: {exc}"
            self.last_log = repr(exc)
            print(f"[SERVER] train exception: {exc}")
        finally:
            self.last_train_time = time.time()
            with self.lock:
                self.training = False

    def _try_convert_rknn(self) -> None:
        command = self.args.rknn_command
        if not command:
            return
        print("[SERVER] start RKNN convert command")
        self.last_status = f"converting rknn: {self.last_mode}"
        proc = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            shell=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.last_log = (self.last_log + "\n[RKNN]\n" + proc.stdout)[-8000:]
        if proc.returncode != 0:
            self.last_status = f"rknn convert failed: {proc.returncode}"
            print(proc.stdout)
            return
        self.last_status = f"trained+rknn: {self.last_mode}"
        print("[SERVER] RKNN convert done")


def make_handler(state: TrainState):
    args = state.args

    class Handler(BaseHTTPRequestHandler):
        server_version = "JamRealtimeTrain/1.0"

        def log_message(self, fmt, *items):
            print("[HTTP]", self.address_string(), fmt % items)

        def send_json(self, obj, status=200):
            data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_file(self, path: Path):
            if not path.exists():
                self.send_json({"ok": False, "error": f"not found: {path.name}"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-SHA256", sha256_file(path))
            self.end_headers()
            with path.open("rb") as f:
                shutil.copyfileobj(f, self.wfile)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_json(
                    {
                        "ok": True,
                        "status": state.last_status,
                        "training": state.training,
                        "mode": state.last_mode,
                        "sim_data_dir": str(args.sim_data_dir) if args.sim_data_dir else "",
                    }
                )
                return

            if parsed.path == "/model-info":
                manifest = {}
                manifest_path = Path(args.manifest_path)
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                pth_path = Path(args.model_path)
                rknn_path = Path(args.rknn_path)
                self.send_json(
                    {
                        "ok": True,
                        "status": state.last_status,
                        "training": state.training,
                        "mode": state.last_mode,
                        "base_counts": count_bins(Path(args.base_data_dir)),
                        "realtime_counts": count_bins(Path(args.data_dir)),
                        "sim_data_dir": str(args.sim_data_dir) if args.sim_data_dir else "",
                        "rknn_command_enabled": bool(args.rknn_command),
                        "has_sim_metadata": bool(args.sim_data_dir) and Path(args.sim_data_dir, "metadata.csv").exists(),
                        "manifest": manifest,
                        "has_pth": pth_path.exists(),
                        "has_rknn": rknn_is_current(rknn_path, pth_path),
                        "rknn_stale": rknn_path.exists() and not rknn_is_current(rknn_path, pth_path),
                    }
                )
                return

            if parsed.path == "/train-now":
                qs = parse_qs(parsed.query)
                mode = qs.get("mode", ["class_increment"])[0]
                if mode not in {"class_increment", "domain_increment", "mixed_train", "retrain", "realtime_retrain"}:
                    self.send_json({"ok": False, "error": f"bad mode: {mode}"}, 400)
                    return
                started = state.train_async("manual http request", mode)
                self.send_json({"ok": True, "started": started, "status": state.last_status, "mode": mode})
                return

            if parsed.path == "/last-log":
                self.send_response(200)
                data = state.last_log.encode("utf-8", errors="replace")
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if parsed.path == "/download/MobileNetV2_Mixed.pth":
                self.send_file(Path(args.model_path))
                return

            if parsed.path == "/download/mobilenet_interference_rk3588.rknn":
                self.send_file(Path(args.rknn_path))
                return

            self.send_json({"ok": False, "error": "not found"}, 404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/upload":
                self.send_json({"ok": False, "error": "not found"}, 404)
                return

            qs = parse_qs(parsed.query)
            label = qs.get("label", [""])[0]
            filename = safe_filename(qs.get("filename", [f"upload_{int(time.time())}.bin"])[0])
            if label not in CLASSES:
                self.send_json({"ok": False, "error": f"bad label: {label}"}, 400)
                return

            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                self.send_json({"ok": False, "error": "empty body"}, 400)
                return

            out_dir = Path(args.data_dir) / label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename
            if out_path.exists():
                out_path = out_dir / f"{out_path.stem}_{int(time.time())}{out_path.suffix}"

            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with tmp_path.open("wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            tmp_path.replace(out_path)
            mode = qs.get("mode", [""])[0]
            if mode in {"class_increment", "domain_increment", "mixed_train", "retrain", "realtime_retrain"}:
                state.train_async(f"upload {label}/{out_path.name}", mode)
            self.send_json({"ok": True, "path": str(out_path), "size": out_path.stat().st_size})

    return Handler


def watch_loop(state: TrainState) -> None:
    while True:
        time.sleep(state.args.watch_interval)
        if state.should_train():
            state.train_async("folder changed")


def parse_args():
    parser = argparse.ArgumentParser(description="JamSystem realtime training server.")
    parser.add_argument("--host", default=os.environ.get("JAMSYSTEM_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("JAMSYSTEM_SERVER_PORT", "8008")))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--base-data-dir", default=str(DEFAULT_BASE_DATA_DIR))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--baseline-model-path", default=str(DEFAULT_BASELINE_MODEL_PATH))
    parser.add_argument("--rknn-path", default=str(DEFAULT_RKNN_PATH))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--train-script", default=str(DEFAULT_TRAIN_SCRIPT))
    parser.add_argument("--realtime-retrain-script", default=str(DEFAULT_REALTIME_RETRAIN_SCRIPT))
    parser.add_argument("--incremental-script", default=str(DEFAULT_INCREMENTAL_SCRIPT))
    parser.add_argument("--sim-data-dir", default=os.environ.get("MIX_SIM_DATA_DIR", ""))
    parser.add_argument("--watch-interval", type=int, default=int(os.environ.get("JAMSYSTEM_WATCH_INTERVAL", "30")))
    parser.add_argument("--cooldown", type=int, default=int(os.environ.get("JAMSYSTEM_TRAIN_COOLDOWN", "60")))
    parser.add_argument("--min-files", type=int, default=int(os.environ.get("JAMSYSTEM_TRAIN_MIN_FILES", "7")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("JAMSYSTEM_TRAIN_EPOCHS", "3")))
    parser.add_argument("--inc-epochs", type=int, default=int(os.environ.get("JAMSYSTEM_INC_EPOCHS", "10")))
    parser.add_argument("--realtime-repeat", type=int, default=int(os.environ.get("JAMSYSTEM_REALTIME_REPEAT", "8")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("JAMSYSTEM_TRAIN_BATCH", "16")))
    parser.add_argument("--freeze-features", action="store_true", default=os.environ.get("JAMSYSTEM_TRAIN_FREEZE", "0") == "1")
    parser.add_argument(
        "--rknn-command",
        default=os.environ.get("JAMSYSTEM_RKNN_COMMAND", ""),
        help="Optional shell command to export ONNX/RKNN after .pth training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data_dir = Path(args.data_dir).resolve()
    args.base_data_dir = Path(args.base_data_dir).resolve()
    args.model_path = Path(args.model_path).resolve()
    args.baseline_model_path = Path(args.baseline_model_path).resolve()
    args.rknn_path = Path(args.rknn_path).resolve()
    args.manifest_path = Path(args.manifest_path).resolve()
    args.train_script = Path(args.train_script).resolve()
    args.realtime_retrain_script = Path(args.realtime_retrain_script).resolve()
    args.incremental_script = Path(args.incremental_script).resolve()
    args.sim_data_dir = Path(args.sim_data_dir).resolve() if args.sim_data_dir else None

    for label in CLASSES:
        (args.data_dir / label).mkdir(parents=True, exist_ok=True)

    state = TrainState(args)
    state.last_signature = state.signature()
    threading.Thread(target=watch_loop, args=(state,), daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[SERVER] listening on http://{args.host}:{args.port}")
    print(f"[SERVER] data dir: {args.data_dir}")
    print(f"[SERVER] base data dir: {args.base_data_dir}")
    print(f"[SERVER] baseline model: {args.baseline_model_path}")
    print(f"[SERVER] model: {args.model_path}")
    print(f"[SERVER] rknn: {args.rknn_path}")
    print(f"[SERVER] mixed train script: {args.train_script}")
    print(f"[SERVER] sim data dir: {args.sim_data_dir if args.sim_data_dir else 'auto'}")
    print(f"[SERVER] incremental script: {args.incremental_script}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
