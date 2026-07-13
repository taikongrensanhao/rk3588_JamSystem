#!/usr/bin/env python3
import html
import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent
BIN_PATH = BASE_DIR / "ad9361_rk3588"
DEFAULT_AD_IP = "192.168.1.10"
DEFAULT_SWITCH_MS = "1000"
DEFAULT_SEQ = "narrowband,wideband_barrage,comb,white_noise,noise_fm,single_tone"

JAMMERS = [
    ("narrowband", "窄带干扰"),
    ("wideband_barrage", "宽带阻塞"),
    ("comb", "梳状干扰"),
    ("white_noise", "白噪声"),
    ("noise_fm", "噪声调频"),
    ("single_tone", "单频干扰"),
]

MODES = [
    ("digital_qpsk", "数字调制 QPSK"),
    ("analog_fm", "模拟调制 FM"),
]


class TxState:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.log_lines = []
        self.last_cmd = ""
        self.last_status = "idle"
        self.started_at = None
        self.reader_thread = None

    def append_log(self, line):
        line = line.rstrip("\n")
        with self.lock:
            self.log_lines.append(line)
            self.log_lines = self.log_lines[-300:]

    def snapshot(self):
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            if self.proc is not None and self.proc.poll() is not None and self.last_status == "running":
                self.last_status = f"exited({self.proc.returncode})"
            return {
                "running": running,
                "status": self.last_status if not running else "running",
                "last_cmd": self.last_cmd,
                "started_at": self.started_at,
                "logs": list(self.log_lines[-160:]),
            }

    def stop(self):
        with self.lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            with self.lock:
                self.proc = None
                self.last_status = "idle"
            return {"ok": True, "message": "当前没有运行中的发射任务"}

        self.append_log("[UI] stopping transmitter ...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=5)

        with self.lock:
            self.proc = None
            self.last_status = "stopped"
        self.append_log("[UI] transmitter stopped")
        return {"ok": True, "message": "发射已停止"}

    def start(self, ad_ip, modulation, jammer, switch_ms, seq):
        if modulation not in dict(MODES):
            return {"ok": False, "message": "调制方式不合法"}
        if jammer not in dict(JAMMERS):
            return {"ok": False, "message": "干扰类型不合法"}

        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return {"ok": False, "message": "已有发射任务正在运行，请先停止"}

        if not BIN_PATH.exists():
            return {"ok": False, "message": f"找不到发射程序: {BIN_PATH}"}

        env = os.environ.copy()
        env["JAMSYSTEM_SELFTEST_SWITCH_MS"] = str(switch_ms or DEFAULT_SWITCH_MS)
        env["JAMSYSTEM_SELFTEST_SWITCH_SEQ"] = seq or DEFAULT_SEQ

        cmd = [
            str(BIN_PATH),
            jammer,
            f"ip:{ad_ip}",
            modulation,
            "selftest_switch",
        ]

        self.append_log("[UI] starting: " + " ".join(cmd))
        self.append_log("[UI] switch_ms=" + env["JAMSYSTEM_SELFTEST_SWITCH_MS"])
        self.append_log("[UI] switch_seq=" + env["JAMSYSTEM_SELFTEST_SWITCH_SEQ"])

        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )

        with self.lock:
            self.proc = proc
            self.last_cmd = " ".join(cmd)
            self.last_status = "running"
            self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")

        def reader():
            try:
                for line in proc.stdout:
                    self.append_log(line)
            finally:
                code = proc.poll()
                with self.lock:
                    if self.proc is proc:
                        self.last_status = f"exited({code})"
                self.append_log(f"[UI] transmitter exited: {code}")

        self.reader_thread = threading.Thread(target=reader, daemon=True)
        self.reader_thread.start()
        return {"ok": True, "message": "发射已启动"}

    def check(self, ad_ip):
        if not BIN_PATH.exists():
            return {"ok": False, "message": f"找不到发射程序: {BIN_PATH}", "output": ""}
        cmd = [str(BIN_PATH), "--check", f"ip:{ad_ip}"]
        self.append_log("[UI] check: " + " ".join(cmd))
        try:
            out = subprocess.check_output(
                cmd,
                cwd=str(BASE_DIR),
                stderr=subprocess.STDOUT,
                text=True,
                timeout=12,
            )
            for line in out.splitlines():
                self.append_log(line)
            return {"ok": True, "message": "AD9361 连接检测完成", "output": out}
        except subprocess.CalledProcessError as exc:
            out = exc.output or ""
            for line in out.splitlines():
                self.append_log(line)
            return {"ok": False, "message": f"检测失败: {exc.returncode}", "output": out}
        except Exception as exc:
            self.append_log(f"[UI] check failed: {exc}")
            return {"ok": False, "message": f"检测失败: {exc}", "output": ""}


STATE = TxState()


def page():
    jammer_options = "\n".join(
        f'<option value="{html.escape(value)}">{html.escape(label)}</option>'
        for value, label in JAMMERS
    )
    mode_options = "\n".join(
        f'<option value="{html.escape(value)}">{html.escape(label)}</option>'
        for value, label in MODES
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>发射板控制台</title>
<style>
body {{ margin:0; font-family: Arial, "Noto Sans CJK SC", sans-serif; background:#f4f6f8; color:#18212f; }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 18px; }}
.top {{ display:flex; justify-content:space-between; align-items:center; gap:14px; margin-bottom:14px; }}
h1 {{ font-size:28px; margin:0; }}
.status {{ padding:10px 14px; border-radius:8px; background:#dfe7f3; font-weight:700; }}
.status.running {{ background:#d9f7e7; color:#096b3a; }}
.grid {{ display:grid; grid-template-columns: 380px 1fr; gap:14px; }}
.panel {{ background:white; border:1px solid #d8dee8; border-radius:8px; padding:14px; }}
label {{ display:block; font-size:15px; font-weight:700; margin-top:12px; }}
input, select {{ box-sizing:border-box; width:100%; margin-top:6px; padding:12px; font-size:18px; border:1px solid #b9c2cf; border-radius:6px; background:white; }}
.row {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
button {{ width:100%; padding:16px 12px; margin-top:14px; font-size:19px; font-weight:800; border:0; border-radius:8px; color:white; background:#1f6feb; }}
button.stop {{ background:#d1242f; }}
button.check {{ background:#57606a; }}
button:active {{ transform: translateY(1px); }}
.hint {{ color:#5d6877; font-size:14px; line-height:1.55; margin-top:10px; }}
pre {{ margin:0; height:560px; overflow:auto; white-space:pre-wrap; word-break:break-word; font-size:14px; line-height:1.45; background:#101820; color:#d6e2ff; border-radius:8px; padding:12px; }}
.cmd {{ margin-top:8px; color:#465366; font-size:14px; word-break:break-all; }}
@media (max-width: 850px) {{ .grid {{ grid-template-columns: 1fr; }} pre {{ height:420px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <h1>发射板控制台</h1>
    <div id="status" class="status">idle</div>
  </div>
  <div class="grid">
    <div class="panel">
      <label>AD9361 地址</label>
      <input id="ad_ip" value="{DEFAULT_AD_IP}">
      <div class="row">
        <div>
          <label>调制方式</label>
          <select id="modulation">{mode_options}</select>
        </div>
        <div>
          <label>起始干扰</label>
          <select id="jammer">{jammer_options}</select>
        </div>
      </div>
      <label>切换周期 ms</label>
      <input id="switch_ms" type="number" min="100" step="100" value="{DEFAULT_SWITCH_MS}">
      <label>切换序列</label>
      <input id="seq" value="{DEFAULT_SEQ}">
      <button class="check" onclick="checkAd()">检测 AD9361</button>
      <button onclick="startTx()">开始动态发射</button>
      <button class="stop" onclick="stopTx()">停止发射</button>
      <div class="hint">
        发射板只负责控制本机 AD9361 发射。接收板请使用“开始外部切换跟踪”观察识别结果和切换速度。
      </div>
      <div id="cmd" class="cmd"></div>
    </div>
    <div class="panel">
      <pre id="logs"></pre>
    </div>
  </div>
</div>
<script>
async function post(path, data) {{
  const res = await fetch(path, {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify(data || {{}})
  }});
  const js = await res.json();
  if (!js.ok) alert(js.message || '操作失败');
  return js;
}}
function payload() {{
  return {{
    ad_ip: document.getElementById('ad_ip').value.trim(),
    modulation: document.getElementById('modulation').value,
    jammer: document.getElementById('jammer').value,
    switch_ms: document.getElementById('switch_ms').value,
    seq: document.getElementById('seq').value.trim()
  }};
}}
async function checkAd() {{ await post('/api/check', payload()); refresh(); }}
async function startTx() {{ await post('/api/start', payload()); refresh(); }}
async function stopTx() {{ await post('/api/stop', {{}}); refresh(); }}
async function refresh() {{
  const res = await fetch('/api/status');
  const js = await res.json();
  const status = document.getElementById('status');
  status.textContent = js.status;
  status.className = 'status' + (js.running ? ' running' : '');
  document.getElementById('cmd').textContent = js.last_cmd || '';
  const logs = document.getElementById('logs');
  logs.textContent = (js.logs || []).join('\\n');
  logs.scrollTop = logs.scrollHeight;
}}
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, page(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(200, STATE.snapshot())
        else:
            self._send(404, {"ok": False, "message": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}
        ad_ip = data.get("ad_ip") or DEFAULT_AD_IP

        if path == "/api/check":
            self._send(200, STATE.check(ad_ip))
        elif path == "/api/start":
            self._send(200, STATE.start(
                ad_ip=ad_ip,
                modulation=data.get("modulation") or "digital_qpsk",
                jammer=data.get("jammer") or "narrowband",
                switch_ms=data.get("switch_ms") or DEFAULT_SWITCH_MS,
                seq=data.get("seq") or DEFAULT_SEQ,
            ))
        elif path == "/api/stop":
            self._send(200, STATE.stop())
        else:
            self._send(404, {"ok": False, "message": "not found"})

    def log_message(self, fmt, *args):
        return


def main():
    import argparse

    parser = argparse.ArgumentParser(description="JamSystem transmitter touch UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    print(f"[TX-UI] base dir: {BASE_DIR}")
    print(f"[TX-UI] open: http://127.0.0.1:{args.port}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STATE.stop()


if __name__ == "__main__":
    main()
