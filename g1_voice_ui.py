#!/usr/bin/env python3
"""
Web UI for the Unitree G1 voice assistant.

    python g1_voice_ui.py            # then open http://127.0.0.1:8080

* Start / Stop buttons run and end the assistant (models stay loaded, so a
  second Start is fast).
* The conversation (you and the Unitree G1) appears live, sentence by
  sentence, while the robot is speaking.
* A diagnostics panel shows the log (warnings, errors, startup steps).

Only the Python standard library is used for the UI (HTTP server + Server-Sent
Events). All assistant options from g1_voice_assistant.py are accepted too,
e.g.  --interface eno1 --whisper-model small.en --voice en-US-AriaNeural
"""

from __future__ import annotations

import json
import logging
import queue
import signal
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import g1_voice_assistant as core

log = logging.getLogger("g1voice.ui")


# ============================================================
# EVENT HUB (fan-out to every open browser tab)
# ============================================================

class Hub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: set[queue.Queue] = set()
        self._history: deque = deque(maxlen=1000)   # user/robot/log events
        self.state = "idle"
        self.detail = ""

    def publish(self, event: dict) -> None:
        event = {**event, "ts": time.time()}
        with self._lock:
            if event["type"] == "state":
                if (event["state"], event.get("detail", "")) == (self.state, self.detail):
                    return
                self.state = event["state"]
                self.detail = event.get("detail", "")
            else:
                self._history.append(event)
            clients = list(self._clients)
        for q in clients:
            q.put(event)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            clients = list(self._clients)
        for q in clients:
            q.put({"type": "clear", "ts": time.time()})

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        q: queue.Queue = queue.Queue()
        with self._lock:
            snapshot = [
                {"type": "reset"},
                {"type": "state", "state": self.state, "detail": self.detail, "ts": time.time()},
                *self._history,
            ]
            self._clients.add(q)
        return q, snapshot

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(q)


class HubLogHandler(logging.Handler):
    """Mirrors the assistant's log lines into the UI diagnostics panel."""

    def __init__(self, hub: Hub) -> None:
        super().__init__(level=logging.INFO)
        self.hub = hub
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.hub.publish({"type": "log", "level": record.levelname, "text": self.format(record)})
        except Exception:
            pass


# ============================================================
# CONTROLLER (Start / Stop)
# ============================================================

class Controller:
    def __init__(self, cfg: core.Config, hub: Hub) -> None:
        self.hub = hub
        self.assistant = core.VoiceAssistant(cfg, on_event=hub.publish)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self.assistant.reset_stop()
            self.hub.publish({"type": "state", "state": "loading", "detail": "Starting..."})
            self._thread = threading.Thread(target=self._worker, name="assistant", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            self.hub.publish({"type": "state", "state": "stopping", "detail": ""})
            self.assistant.request_stop()
            return True

    def _worker(self) -> None:
        a = self.assistant
        failed = False
        try:
            a.load()
            if not a.stop_requested:
                a.begin_listening()
                a.set_state("listening")
                a.run()
        except Exception as error:
            failed = True
            log.exception("Assistant stopped because of an error: %s", error)
            self.hub.publish({"type": "state", "state": "error", "detail": str(error)})
        finally:
            try:
                a.end_listening()
            except Exception:
                log.exception("Error while releasing the microphone")
            if not failed:
                self.hub.publish({"type": "state", "state": "idle", "detail": ""})

    def shutdown(self) -> None:
        if self.running:
            self.assistant.request_stop()
            self._thread.join(timeout=15)


# ============================================================
# HTTP SERVER
# ============================================================

class Handler(BaseHTTPRequestHandler):
    server_version = "G1VoiceUI/1.0"

    def log_message(self, fmt, *args) -> None:      # keep the console quiet
        pass

    # -- helpers -------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    # -- routes --------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/events":
            self._events()
        elif path == "/state":
            hub = self.server.hub
            self._json({"state": hub.state, "detail": hub.detail})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        # A custom header forces a CORS preflight, so other websites cannot
        # press the buttons from a visitor's browser.
        if self.headers.get("X-Requested-With") != "g1ui":
            self._json({"ok": False, "error": "forbidden"}, 403)
            return
        ctl: Controller = self.server.controller
        path = self.path.split("?", 1)[0]
        if path == "/start":
            self._json({"ok": ctl.start()})
        elif path == "/stop":
            self._json({"ok": ctl.stop()})
        elif path == "/clear":
            self.server.hub.clear_history()
            self._json({"ok": True})
        else:
            self._json({"ok": False, "error": "unknown"}, 404)

    def _write_event(self, event: dict) -> None:
        data = json.dumps(event, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _events(self) -> None:
        hub: Hub = self.server.hub
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q, snapshot = hub.subscribe()
        try:
            for event in snapshot:
                self._write_event(event)
            while True:
                try:
                    event = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self._write_event(event)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, hub: Hub, controller: Controller) -> None:
        super().__init__(address, Handler)
        self.hub = hub
        self.controller = controller


# ============================================================
# THE PAGE
# ============================================================

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unitree G1 Voice</title>
<style>
:root{
  --bg1:#080818; --bg2:#120e36;
  --glass:rgba(255,255,255,.06); --glass2:rgba(255,255,255,.10); --line:rgba(255,255,255,.13);
  --text:#eef0ff; --muted:#9ea3cf;
  --violet:#7c5cff; --cyan:#22d3ee; --pink:#f472b6; --green:#34d399; --amber:#fbbf24; --red:#fb7185;
  --st:#8b8fb8;                      /* colour of the current state */
}
body[data-state="idle"]{--st:#8b8fb8}
body[data-state="loading"]{--st:#fbbf24}
body[data-state="listening"]{--st:#34d399}
body[data-state="transcribing"]{--st:#22d3ee}
body[data-state="thinking"]{--st:#a78bfa}
body[data-state="speaking"]{--st:#f472b6}
body[data-state="stopping"]{--st:#fbbf24}
body[data-state="error"],body[data-state="offline"]{--st:#fb7185}

*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0; color:var(--text); background:var(--bg1);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  display:flex; flex-direction:column; height:100dvh; overflow:hidden;
}

/* ---------- decorative background ---------- */
.bg{position:fixed; inset:0; z-index:0; overflow:hidden; pointer-events:none;
  background:
    radial-gradient(900px 600px at 12% -10%, rgba(124,92,255,.35), transparent 60%),
    radial-gradient(800px 600px at 105% 10%, rgba(34,211,238,.22), transparent 60%),
    radial-gradient(900px 700px at 50% 120%, rgba(244,114,182,.22), transparent 60%),
    linear-gradient(160deg,var(--bg1),var(--bg2));
}
.bg::before{                       /* faint tech grid */
  content:""; position:absolute; inset:0; opacity:.5;
  background-image:
    linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px);
  background-size:44px 44px;
  -webkit-mask-image:radial-gradient(ellipse at 50% 30%, #000 20%, transparent 75%);
          mask-image:radial-gradient(ellipse at 50% 30%, #000 20%, transparent 75%);
}
.orb{position:absolute; border-radius:50%; filter:blur(70px); opacity:.45; animation:float 18s ease-in-out infinite}
.orb.a{width:340px; height:340px; background:var(--violet); left:-80px; top:18%}
.orb.b{width:300px; height:300px; background:var(--cyan); right:-60px; top:46%; animation-delay:-6s}
.orb.c{width:260px; height:260px; background:var(--pink); left:38%; bottom:-90px; animation-delay:-12s}
@keyframes float{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(40px,-30px) scale(1.1)}66%{transform:translate(-30px,30px) scale(.92)}}
.star{position:absolute; width:3px; height:3px; border-radius:50%; background:#fff; opacity:.5; animation:twinkle 4s ease-in-out infinite}
@keyframes twinkle{0%,100%{opacity:.15; transform:scale(.8)}50%{opacity:.9; transform:scale(1.4)}}

header,.stage,main,footer{position:relative; z-index:1}

/* ---------- header ---------- */
header{
  display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;
  margin:12px 14px 0; padding:10px 16px; border-radius:18px;
  background:var(--glass); border:1px solid var(--line);
  -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px);
  box-shadow:0 8px 32px rgba(0,0,0,.35);
}
.brand{display:flex; align-items:center; gap:12px; min-width:0}
.logo{
  width:42px; height:42px; border-radius:13px; flex:none; display:grid; place-items:center;
  background:linear-gradient(135deg,var(--violet),var(--cyan)); box-shadow:0 0 22px rgba(124,92,255,.65);
}
.logo svg{width:26px; height:26px}
h1{
  margin:0; font-size:18px; font-weight:750; line-height:1.15; letter-spacing:.2px;
  background:linear-gradient(90deg,#c4b5fd,#67e8f9 55%,#f9a8d4);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.sub{margin:2px 0 0; font-size:12.5px; color:var(--muted); display:flex; gap:6px; flex-wrap:wrap; align-items:center}
.chip{
  display:inline-block; padding:1px 9px; border-radius:999px; font-size:11px; font-weight:600; letter-spacing:.3px;
  color:#fff; background:linear-gradient(90deg,rgba(124,92,255,.55),rgba(34,211,238,.45)); border:1px solid var(--line);
}
.controls{display:flex; align-items:center; gap:8px; flex-wrap:wrap}
.pill{
  display:inline-flex; align-items:center; gap:8px; padding:6px 13px; border-radius:999px; font-size:13px; font-weight:650;
  color:var(--text);
  border:1px solid color-mix(in srgb, var(--st) 55%, transparent);
  background:color-mix(in srgb, var(--st) 15%, transparent);
  transition:all .3s;
}
.pill i{width:9px; height:9px; border-radius:50%; background:var(--st); box-shadow:0 0 10px var(--st); display:block}
.pill.anim i{animation:pulse 1.1s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1; transform:scale(1)}50%{opacity:.4; transform:scale(.7)}}

.btn{
  display:inline-flex; align-items:center; gap:8px; border:1px solid transparent; border-radius:12px; color:#fff;
  padding:9px 18px; font:inherit; font-weight:700; cursor:pointer; transition:transform .12s, filter .15s, box-shadow .2s, opacity .15s;
}
.btn svg{width:16px; height:16px; fill:currentColor}
.btn:hover:not(:disabled){transform:translateY(-1px); filter:brightness(1.1)}
.btn:active:not(:disabled){transform:translateY(0)}
.btn:disabled{opacity:.35; cursor:not-allowed; box-shadow:none !important}
.btn:focus-visible{outline:2px solid var(--cyan); outline-offset:2px}
.btn.start{background:linear-gradient(135deg,#10b981,#06b6d4); box-shadow:0 4px 20px rgba(16,185,129,.45)}
.btn.stop{background:linear-gradient(135deg,#f43f5e,#f97316); box-shadow:0 4px 20px rgba(244,63,94,.45)}
.btn.ghost{background:var(--glass); color:var(--muted); border-color:var(--line); font-weight:550; padding:9px 14px}
.btn.ghost:hover:not(:disabled){color:var(--text); background:var(--glass2)}

#banner{display:none; position:relative; z-index:2; margin:10px 14px 0; border-radius:12px; text-align:center;
  padding:7px 12px; font-size:13px; color:#fff; background:linear-gradient(90deg,#f43f5e,#f97316)}

/* ---------- robot stage ---------- */
.stage{
  margin:12px 14px 0; padding:14px 20px; border-radius:22px; display:flex; align-items:center; gap:22px;
  background:linear-gradient(135deg,rgba(124,92,255,.16),rgba(34,211,238,.08) 55%,rgba(244,114,182,.14));
  border:1px solid var(--line);
  -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px);
  box-shadow:0 10px 40px rgba(0,0,0,.35);
}
.avatar{position:relative; flex:none; width:118px; height:118px; display:grid; place-items:center}
.avatar::before{                   /* spinning ring */
  content:""; position:absolute; inset:-5px; border-radius:50%;
  background:conic-gradient(from 0deg,var(--st),transparent 45%,var(--st) 80%,transparent);
  -webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 4px),#000 calc(100% - 3px));
          mask:radial-gradient(farthest-side,transparent calc(100% - 4px),#000 calc(100% - 3px));
  opacity:.55; transition:opacity .3s;
}
.avatar::after{                    /* glow */
  content:""; position:absolute; inset:10px; border-radius:50%; z-index:0;
  box-shadow:0 0 46px 6px var(--st); opacity:.28; transition:opacity .3s, box-shadow .3s;
}
body[data-state="loading"] .avatar::before,
body[data-state="transcribing"] .avatar::before,
body[data-state="thinking"] .avatar::before,
body[data-state="stopping"] .avatar::before{opacity:1; animation:spin 1.3s linear infinite}
body[data-state="listening"] .avatar::after{opacity:.5; animation:breathe 2.4s ease-in-out infinite}
body[data-state="speaking"] .avatar::after{opacity:.75; animation:breathe .8s ease-in-out infinite}
body[data-state="speaking"] .avatar::before{opacity:1}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes breathe{0%,100%{transform:scale(.92)}50%{transform:scale(1.1)}}
.robot{position:relative; z-index:1; width:104px; height:104px; filter:drop-shadow(0 6px 14px rgba(0,0,0,.45))}
.robot .ant{fill:var(--st); filter:drop-shadow(0 0 5px var(--st)); transition:fill .3s}
.robot .eye{fill:var(--st); filter:drop-shadow(0 0 4px var(--st)); transition:fill .3s;
  transform-box:fill-box; transform-origin:center; animation:blink 5s infinite}
.robot .eye:nth-child(2){animation-delay:.02s}
@keyframes blink{0%,92%,100%{transform:scaleY(1)}95%{transform:scaleY(.1)}}
.robot .mb{fill:var(--st); transform-box:fill-box; transform-origin:center; transition:fill .3s}
body[data-state="speaking"] .robot .mb{animation:talk .45s ease-in-out infinite}
.robot .mb:nth-child(2){animation-delay:.08s}.robot .mb:nth-child(3){animation-delay:.16s}
.robot .mb:nth-child(4){animation-delay:.04s}.robot .mb:nth-child(5){animation-delay:.12s}
@keyframes talk{0%,100%{transform:scaleY(1)}50%{transform:scaleY(4.5)}}
body[data-state="thinking"] .robot .eye{animation:look 1.6s ease-in-out infinite}
@keyframes look{0%,100%{transform:translateX(-2px)}50%{transform:translateX(2px)}}

.info{min-width:0; flex:1}
.info .state{
  margin:0; font-size:24px; font-weight:800; line-height:1.15; letter-spacing:.2px;
  background:linear-gradient(90deg,var(--st),#fff 140%); -webkit-background-clip:text; background-clip:text; color:transparent;
}
.info .hint{margin:3px 0 8px; font-size:13.5px; color:var(--muted); min-height:20px}
.wave{display:flex; align-items:center; gap:3px; height:44px}
.wave span{
  width:4px; height:6px; border-radius:4px; flex:none;
  background:linear-gradient(180deg,var(--cyan),var(--violet) 55%,var(--pink)); opacity:.85;
}
body[data-state="listening"] .wave span{animation:eq 2.2s ease-in-out infinite; --h:12px}
body[data-state="transcribing"] .wave span,
body[data-state="thinking"] .wave span{animation:eq 1.1s ease-in-out infinite; --h:22px}
body[data-state="speaking"] .wave span{animation:eq .7s ease-in-out infinite}
@keyframes eq{0%,100%{height:5px}50%{height:var(--h,40px)}}

/* ---------- conversation ---------- */
main{flex:1; overflow-y:auto; padding:16px 14px 10px; scroll-behavior:smooth; display:flex; flex-direction:column}
main::-webkit-scrollbar{width:8px}
main::-webkit-scrollbar-thumb{background:rgba(255,255,255,.18); border-radius:8px}
.wrap{width:100%; max-width:860px; margin:0 auto; display:flex; flex-direction:column; gap:16px; flex:1}

.msg{display:flex; gap:10px; max-width:88%; animation:pop .28s ease-out both}
@keyframes pop{from{opacity:0; transform:translateY(10px) scale(.98)}to{opacity:1; transform:none}}
.msg.user{align-self:flex-end; flex-direction:row-reverse}
.msg.bot{align-self:flex-start}
.badge{
  width:34px; height:34px; border-radius:50%; flex:none; display:grid; place-items:center;
  font-size:12px; font-weight:800; color:#fff; box-shadow:0 0 16px rgba(0,0,0,.35);
}
.user .badge{background:linear-gradient(135deg,#f59e0b,#f43f5e)}
.bot .badge{background:linear-gradient(135deg,var(--violet),var(--cyan))}
.col{display:flex; flex-direction:column; gap:4px; min-width:0}
.user .col{align-items:flex-end}
.meta{font-size:11.5px; color:var(--muted); padding:0 6px}
.bubble{padding:11px 16px; border-radius:18px; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word}
.user .bubble{
  color:#fff; border-top-right-radius:6px;
  background:linear-gradient(135deg,#7c5cff,#3b82f6 60%,#22d3ee); box-shadow:0 6px 24px rgba(88,101,255,.4);
}
.bot .bubble{
  border-top-left-radius:6px; border:1px solid var(--line);
  background:linear-gradient(180deg,rgba(255,255,255,.11),rgba(255,255,255,.05));
  box-shadow:inset 3px 0 0 var(--pink), 0 6px 24px rgba(0,0,0,.28);
  -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px);
}
.bot.live .bubble{box-shadow:inset 3px 0 0 var(--cyan), 0 0 26px rgba(244,114,182,.35)}
.bot.live .bubble::after{
  content:""; display:inline-block; width:3px; height:1em; margin-left:4px; vertical-align:-2px; border-radius:2px;
  background:var(--pink); animation:blink2 .9s steps(2) infinite;
}
@keyframes blink2{50%{opacity:0}}
.bot.error .bubble{border-color:var(--red); color:#ffd5db; box-shadow:inset 3px 0 0 var(--red)}

.dots .bubble{display:flex; gap:6px; padding:14px 18px}
.dots .bubble span{width:8px; height:8px; border-radius:50%; animation:bounce 1.2s infinite}
.dots .bubble span:nth-child(1){background:var(--cyan)}
.dots .bubble span:nth-child(2){background:var(--violet); animation-delay:.15s}
.dots .bubble span:nth-child(3){background:var(--pink); animation-delay:.3s}
@keyframes bounce{0%,60%,100%{transform:translateY(0); opacity:.55}30%{transform:translateY(-6px); opacity:1}}
.hidden{display:none !important}

#empty{margin:auto; text-align:center; max-width:460px; padding:18px}
#empty .big{
  font-size:22px; font-weight:800; margin-bottom:6px;
  background:linear-gradient(90deg,#c4b5fd,#67e8f9,#f9a8d4); -webkit-background-clip:text; background-clip:text; color:transparent;
}
#empty p{margin:0 0 14px; color:var(--muted)}
.try{display:flex; flex-wrap:wrap; gap:8px; justify-content:center}
.try span{padding:6px 13px; border-radius:999px; font-size:13px; font-weight:550; border:1px solid var(--line); color:var(--text)}
.try span:nth-child(1){background:rgba(124,92,255,.22)}
.try span:nth-child(2){background:rgba(34,211,238,.18)}
.try span:nth-child(3){background:rgba(244,114,182,.2)}
.try span:nth-child(4){background:rgba(52,211,153,.18)}

#jump{
  position:sticky; bottom:6px; align-self:center; border:1px solid var(--line); color:#fff; border-radius:999px;
  padding:6px 16px; font:inherit; font-size:13px; font-weight:600; cursor:pointer;
  background:linear-gradient(135deg,var(--violet),var(--pink)); box-shadow:0 6px 22px rgba(124,92,255,.5);
}

/* ---------- diagnostics ---------- */
footer{
  margin:0 14px 12px; border-radius:16px; background:var(--glass); border:1px solid var(--line);
  -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px);
}
details{max-width:860px; margin:0 auto; padding:0 16px}
summary{cursor:pointer; padding:9px 0; font-size:13px; font-weight:600; color:var(--muted); user-select:none}
summary:hover{color:var(--text)}
summary .count{
  display:inline-block; min-width:19px; padding:0 6px; margin-left:8px; border-radius:9px; text-align:center;
  background:linear-gradient(135deg,#f59e0b,#f43f5e); color:#fff; font-size:11px;
}
#logs{max-height:170px; overflow-y:auto; padding:2px 0 12px; font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
#logs div{white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted)}
#logs .INFO{color:#a5b4fc}
#logs .WARNING{color:var(--amber)}
#logs .ERROR,#logs .CRITICAL{color:var(--red)}

@media (max-width:640px){
  .stage{flex-direction:column; text-align:center; gap:10px; padding:12px}
  .avatar{width:92px; height:92px}.robot{width:82px; height:82px}
  .info .state{font-size:20px}
  .wave{justify-content:center}
  .msg{max-width:96%}
  .btn{padding:8px 13px}
}
@media (max-height:640px){.stage{display:none}}
@media (prefers-reduced-motion: reduce){
  *{animation:none !important; scroll-behavior:auto !important}
}
</style>
</head>
<body data-state="idle">
<div class="bg" aria-hidden="true">
  <div class="orb a"></div><div class="orb b"></div><div class="orb c"></div>
  <div id="stars"></div>
</div>

<header>
  <div class="brand">
    <div class="logo">
      <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="4" y="7" width="16" height="12" rx="4"/><path d="M12 7V3.5"/><circle cx="12" cy="3" r="1.2" fill="#fff"/>
        <circle cx="9" cy="13" r="1.1" fill="#fff"/><circle cx="15" cy="13" r="1.1" fill="#fff"/>
      </svg>
    </div>
    <div>
      <h1>Unitree G1 Voice</h1>
      <p class="sub"><span class="chip">ENGLISH</span> Talk naturally &mdash; the robot answers out loud</p>
    </div>
  </div>
  <div class="controls">
    <span id="pill" class="pill"><i></i><span id="pillText">Stopped</span></span>
    <button id="btnStart" class="btn start" type="button">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>Start
    </button>
    <button id="btnStop" class="btn stop" type="button" disabled>
      <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>Stop
    </button>
    <button id="btnClear" class="btn ghost" type="button">Clear</button>
  </div>
</header>
<div id="banner">Connection to the server lost. Retrying&hellip;</div>

<section class="stage" aria-live="polite">
  <div class="avatar">
    <svg class="robot" viewBox="0 0 120 120" aria-hidden="true">
      <defs>
        <linearGradient id="hg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#8b6cff"/><stop offset="1" stop-color="#2dd4ff"/>
        </linearGradient>
        <linearGradient id="fg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#1a1650"/><stop offset="1" stop-color="#0b0a26"/>
        </linearGradient>
      </defs>
      <line x1="60" y1="16" x2="60" y2="29" stroke="url(#hg)" stroke-width="4" stroke-linecap="round"/>
      <circle class="ant" cx="60" cy="12" r="6"/>
      <rect x="11" y="60" width="11" height="26" rx="5.5" fill="url(#hg)"/>
      <rect x="98" y="60" width="11" height="26" rx="5.5" fill="url(#hg)"/>
      <rect x="20" y="28" width="80" height="72" rx="24" fill="url(#hg)"/>
      <rect x="28" y="36" width="64" height="56" rx="18" fill="url(#fg)"/>
      <g>
        <rect class="eye" x="40" y="50" width="13" height="17" rx="6.5"/>
        <rect class="eye" x="67" y="50" width="13" height="17" rx="6.5"/>
      </g>
      <g>
        <rect class="mb" x="45" y="76" width="4" height="4" rx="2"/>
        <rect class="mb" x="52" y="76" width="4" height="4" rx="2"/>
        <rect class="mb" x="58" y="76" width="4" height="4" rx="2"/>
        <rect class="mb" x="64" y="76" width="4" height="4" rx="2"/>
        <rect class="mb" x="71" y="76" width="4" height="4" rx="2"/>
      </g>
    </svg>
  </div>
  <div class="info">
    <p id="stageState" class="state">Stopped</p>
    <p id="stageHint" class="hint">Press Start to wake the robot</p>
    <div id="wave" class="wave" aria-hidden="true"></div>
  </div>
</section>

<main id="chat">
  <div class="wrap" id="wrap">
    <div id="empty">
      <div class="big">Say hello to your G1</div>
      <p>Press <b>Start</b>, then speak in English. Everything you say and every reply from the robot shows up here live.</p>
      <div class="try">
        <span>Tell me a story about a brave robot</span>
        <span>Explain how rainbows form in detail</span>
        <span>What is the capital of France?</span>
        <span>Tell me a joke</span>
      </div>
    </div>
    <div id="typing" class="msg hidden dots">
      <div class="badge">G1</div>
      <div class="col"><div class="bubble"><span></span><span></span><span></span></div></div>
    </div>
  </div>
  <button id="jump" class="hidden" type="button">&darr; Latest</button>
</main>

<footer>
  <details id="diag">
    <summary>Diagnostics<span id="warnCount" class="count hidden">0</span></summary>
    <div id="logs"></div>
  </details>
</footer>

<script>
(() => {
  const $ = id => document.getElementById(id);
  const chat = $('chat'), wrap = $('wrap'), empty = $('empty'), typing = $('typing');
  const logs = $('logs'), jump = $('jump'), banner = $('banner');
  const btnStart = $('btnStart'), btnStop = $('btnStop');

  // decorative: twinkling stars and the equaliser bars
  const stars = $('stars');
  for (let i = 0; i < 40; i++) {
    const s = document.createElement('span');
    s.className = 'star';
    s.style.left = Math.random() * 100 + '%';
    s.style.top = Math.random() * 100 + '%';
    s.style.animationDelay = (Math.random() * 4).toFixed(2) + 's';
    s.style.animationDuration = (3 + Math.random() * 4).toFixed(2) + 's';
    stars.appendChild(s);
  }
  const wave = $('wave');
  for (let i = 0; i < 32; i++) {
    const b = document.createElement('span');
    b.style.setProperty('--h', (14 + Math.random() * 28).toFixed(0) + 'px');
    b.style.animationDelay = (i * 0.045).toFixed(3) + 's';
    wave.appendChild(b);
  }

  const STATES = {
    idle:         {label: 'Stopped',       hint: 'Press Start to wake the robot',  anim: false},
    loading:      {label: 'Starting\u2026',     hint: 'Getting everything ready',       anim: true},
    listening:    {label: 'Listening',     hint: 'Go ahead \u2014 I am listening',      anim: true},
    transcribing: {label: 'Hearing you\u2026',  hint: 'Catching every word',            anim: true},
    thinking:     {label: 'Thinking\u2026',     hint: 'Putting my thoughts together',   anim: true},
    speaking:     {label: 'Speaking',      hint: 'Talking to you now',             anim: true},
    stopping:     {label: 'Stopping\u2026',     hint: 'Wrapping up',                    anim: true},
    error:        {label: 'Error',         hint: 'Something went wrong',           anim: false},
    offline:      {label: 'Offline',       hint: 'Cannot reach the server',        anim: false}
  };

  let state = 'idle';
  let bots = {};          // turn -> bot message element
  let warns = 0;
  let pinned = true;

  const atBottom = () => chat.scrollHeight - chat.scrollTop - chat.clientHeight < 60;
  function scrollEnd(force) { if (force || pinned) chat.scrollTop = chat.scrollHeight; }
  chat.addEventListener('scroll', () => {
    pinned = atBottom();
    jump.classList.toggle('hidden', pinned || !hasMessages());
  });
  jump.addEventListener('click', () => { pinned = true; scrollEnd(true); jump.classList.add('hidden'); });

  const hasMessages = () => wrap.querySelector('.msg:not(#typing)') !== null;
  const clock = ts => new Date((ts || Date.now() / 1000) * 1000)
      .toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});

  function setState(s, detail) {
    if (!STATES[s]) s = 'idle';
    state = s;
    const info = STATES[s];
    document.body.dataset.state = s;
    $('pill').className = 'pill' + (info.anim ? ' anim' : '');
    $('pillText').textContent = info.label;
    $('stageState').textContent = info.label;
    $('stageHint').textContent = detail || info.hint;
    const running = !['idle', 'error', 'offline'].includes(s);
    btnStart.disabled = running || s === 'offline';
    btnStop.disabled = !running || s === 'stopping';
    updateTyping();
  }

  function updateTyping() {
    const show = state === 'transcribing' || state === 'thinking';
    typing.classList.toggle('hidden', !show);
    typing.classList.toggle('user', state === 'transcribing');
    typing.classList.toggle('bot', state === 'thinking');
    typing.querySelector('.badge').textContent = state === 'transcribing' ? 'YOU' : 'G1';
    if (show) { wrap.appendChild(typing); scrollEnd(); }
  }

  function addMessage(kind, who, badge, text, ts) {
    empty.classList.add('hidden');
    const el = document.createElement('div');
    el.className = 'msg ' + kind;
    const b = document.createElement('div');
    b.className = 'badge'; b.textContent = badge;
    const col = document.createElement('div');
    col.className = 'col';
    const m = document.createElement('div');
    m.className = 'meta'; m.textContent = who + ' \u00b7 ' + clock(ts);
    const bub = document.createElement('div');
    bub.className = 'bubble'; bub.textContent = text;
    col.append(m, bub);
    el.append(b, col);
    wrap.insertBefore(el, typing);
    return el;
  }

  function onUser(ev) {
    addMessage('user', 'You', 'YOU', ev.text, ev.ts);
    scrollEnd(true); pinned = true;
  }

  function onRobot(ev) {
    typing.classList.add('hidden');
    let el = bots[ev.turn];
    if (!el) {
      el = addMessage('bot live', 'Unitree G1', 'G1', ev.text, ev.ts);
      bots[ev.turn] = el;
    } else {
      el.querySelector('.bubble').textContent += ' ' + ev.text;
    }
    if (ev.error) el.classList.add('error');
    scrollEnd();
  }

  function onRobotDone(ev) {
    const el = bots[ev.turn];
    if (el) el.classList.remove('live');
  }

  function addLog(level, text, ts) {
    const row = document.createElement('div');
    row.className = level;
    row.textContent = clock(ts) + '  ' + level + '  ' + text;
    const stick = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 30;
    logs.appendChild(row);
    while (logs.children.length > 300) logs.removeChild(logs.firstChild);
    if (stick) logs.scrollTop = logs.scrollHeight;
    if (level === 'WARNING' || level === 'ERROR' || level === 'CRITICAL') {
      warns += 1;
      const c = $('warnCount');
      c.textContent = warns; c.classList.remove('hidden');
    }
  }

  function clearChat() {
    wrap.querySelectorAll('.msg:not(#typing)').forEach(n => n.remove());
    bots = {};
    empty.classList.remove('hidden');
    jump.classList.add('hidden');
  }

  function resetAll() {
    clearChat();
    logs.textContent = '';
    warns = 0;
    $('warnCount').classList.add('hidden');
  }

  function handle(ev) {
    switch (ev.type) {
      case 'reset':      resetAll(); break;
      case 'clear':      clearChat(); break;
      case 'state':      setState(ev.state, ev.detail); break;
      case 'user':       onUser(ev); break;
      case 'robot':      onRobot(ev); break;
      case 'robot_done': onRobotDone(ev); break;
      case 'log':        addLog(ev.level, ev.text, ev.ts); break;
    }
  }

  async function post(path) {
    try {
      const r = await fetch(path, {method: 'POST', headers: {'X-Requested-With': 'g1ui'}});
      if (!r.ok) throw new Error('HTTP ' + r.status);
    } catch (e) {
      addLog('ERROR', 'Request failed: ' + e.message);
    }
  }
  btnStart.addEventListener('click', () => post('/start'));
  btnStop.addEventListener('click', () => post('/stop'));
  $('btnClear').addEventListener('click', () => post('/clear'));

  const es = new EventSource('/events');
  es.onopen = () => { banner.style.display = 'none'; };
  es.onmessage = e => { try { handle(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  es.onerror = () => {
    banner.style.display = 'block';
    setState('offline', '');
  };

  setState('idle');
})();
</script>
</body>
</html>
"""


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    cfg = core.Config()
    parser = core.build_parser(cfg)
    parser.description = "Web UI for the Unitree G1 voice assistant"
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to serve on (use 0.0.0.0 to open it from another device; there is no login)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--autostart", action="store_true", help="start the assistant immediately")
    args = parser.parse_args()
    core.apply_args(cfg, args)
    core.setup_logging(args.verbose)

    hub = Hub()
    logging.getLogger("g1voice").addHandler(HubLogHandler(hub))
    controller = Controller(cfg, hub)
    server = UIServer((args.host, args.port), hub, controller)

    signal.signal(signal.SIGTERM, core._raise_keyboard_interrupt)

    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    url = f"http://{shown}:{args.port}"
    log.info("UI running at %s  (Ctrl+C to quit)", url)
    if args.host == "0.0.0.0":
        log.warning("Serving on all interfaces without a login: anyone on your network can "
                    "see the conversation and press Start/Stop.")

    if args.autostart:
        controller.start()
    if not args.no_browser:
        try:
            threading.Timer(0.8, webbrowser.open, args=(url,)).start()
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        controller.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()


#  cd /home/abhay/Projects/Voice/g1_realtime_voice && source .venv_api/bin/activate && python g1_voice_ui.py
