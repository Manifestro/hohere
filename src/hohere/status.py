"""Progress tracking, the `/api/status` JSON and the phone-friendly status page.

`Progress` is the only mutable shared thing in the pipeline: worker threads call
`add(stage, seconds)`, `done_chunk(hours)`, `error(...)` and `shard_uploaded(...)`,
the HTTP thread serialises a snapshot. Nothing here touches the models, so the page
works in Part A against whatever the state DB already knows.

The JSON is built once, by `Progress.snapshot`, and used by both the server and
`hohere status --state` — offline and online views can never drift apart.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import threading
import time
from collections import deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import order

WINDOW_S = 600.0  # rolling speed window: 10 minutes
RECENT_SHARDS = 10
RECENT_ERRORS = 20
STAGES = ("download", "decode", "vad", "wavlm", "mimi", "write")


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat(timespec="seconds")


class Progress:
    """Thread-safe live progress of one run (optionally backed by the state DB)."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        target: int | str = "full",
        started_at: float | None = None,
        state: Any = None,
    ) -> None:
        self.lock = threading.RLock()
        self.device = device
        self.target = target
        self.started_at = started_at if started_at is not None else time.time()
        self.state = state
        self.stage_totals: dict[str, list[float]] = {stage: [0.0, 0.0] for stage in STAGES}
        self.files_total = 0
        self.files_done = 0
        self.source_hours = 0.0
        self.source_hours_done = 0.0
        self.done_chunks = 0
        self.done_hours = 0.0
        self.shards_uploaded = 0
        self.bytes_uploaded = 0
        self.errors_count = 0
        self.errors: deque[str] = deque(maxlen=RECENT_ERRORS)
        self.recent_shards: deque[dict[str, Any]] = deque(maxlen=RECENT_SHARDS)
        self._window: deque[tuple[float, float]] = deque()  # (wall time, chunk hours)

    # -- updates -----------------------------------------------------------------
    def set_totals(self, files: int, source_hours: float) -> None:
        with self.lock:
            self.files_total = int(files)
            self.source_hours = float(source_hours)

    def add(self, stage: str, seconds: float) -> None:
        """Record the duration of one stage for one chunk (ML stages arrive in Part B)."""
        with self.lock:
            entry = self.stage_totals.setdefault(stage, [0.0, 0.0])
            entry[0] += 1
            entry[1] += float(seconds)

    def done_chunk(self, hours: float) -> None:
        with self.lock:
            self.done_chunks += 1
            self.done_hours += float(hours)
            self._window.append((time.time(), float(hours)))
            self._trim_window(time.time())

    def done_file(self, source_hours: float) -> None:
        with self.lock:
            self.files_done += 1
            self.source_hours_done += float(source_hours)

    def error(self, message: str) -> None:
        with self.lock:
            self.errors_count += 1
            self.errors.append(str(message)[:500])
        self._persist_errors()

    def shard_uploaded(self, shard: dict[str, Any]) -> None:
        with self.lock:
            self.shards_uploaded += 1
            self.bytes_uploaded += int(shard.get("bytes") or 0)
            self.recent_shards.append(dict(shard))

    def _persist_errors(self) -> None:
        if self.state is None:
            return
        try:
            self.state.set_meta("errors_count", self.errors_count)
            self.state.set_meta("errors_last", list(self.errors))
        except Exception:  # progress reporting must never break the pipeline
            pass

    def _trim_window(self, now: float) -> None:
        while self._window and now - self._window[0][0] > WINDOW_S:
            self._window.popleft()

    # -- views -------------------------------------------------------------------
    def rolling_speed(self, now: float | None = None) -> float:
        """Audio hours per wall hour over the last 10 minutes."""
        now = time.time() if now is None else now
        with self.lock:
            self._trim_window(now)
            if not self._window:
                return 0.0
            span = now - self._window[0][0]
            hours = sum(hours for _, hours in self._window)
        if span <= 0:  # a single sample: assume it took nothing and report 0
            return 0.0
        return hours / (span / 3600.0)

    def snapshot(self, state: Any = None) -> dict[str, Any]:
        """The `/api/status` payload; `state` supplies the durable numbers."""
        state = state if state is not None else self.state
        now = time.time()
        with self.lock:
            stage_times = {
                stage: {
                    "n": int(count),
                    "total_s": round(total, 3),
                    "mean_s": round(total / count, 4) if count else 0.0,
                }
                for stage, (count, total) in self.stage_totals.items()
            }
            rolling = self.rolling_speed(now)
            errors_count = self.errors_count
            errors = list(self.errors)
            shards = list(self.recent_shards)
            local = {
                "files_total": self.files_total,
                "files_done": self.files_done,
                "source_hours": self.source_hours,
                "source_hours_done": self.source_hours_done,
                "remaining_files": max(0, self.files_total - self.files_done),
                "remaining_hours": max(0.0, self.source_hours - self.source_hours_done),
                "chunks": self.done_chunks,
                "hours": self.done_hours,
                "shards_uploaded": self.shards_uploaded,
                "bytes_uploaded": self.bytes_uploaded,
            }
            device = self.device
            target = self.target
            started_at = self.started_at

        summary = dict(local)
        if state is not None:
            durable = state.summary()
            summary.update({k: v for k, v in durable.items() if v is not None})
            if not shards:
                shards = state.last_shards(RECENT_SHARDS)
            if not errors_count:
                errors_count = int(state.get_meta("errors_count", 0) or 0)
                errors = list(state.get_meta("errors_last", []) or [])
            stored_start = state.get_meta("started_at")
            if stored_start and self.state is None:  # offline: there is no live run to time
                with contextlib.suppress(ValueError):
                    started_at = datetime.fromisoformat(stored_start).timestamp()
            stored_target = state.get_meta("target_hours")
            if stored_target is not None:
                target = stored_target

        uptime = max(0.0, now - started_at)
        done_hours = float(summary.get("hours") or 0.0)
        source_done = float(summary.get("source_hours_done") or 0.0)
        # A live run is timed on what this process did, not on what earlier runs left behind.
        run_hours = self.done_hours if self.state is not None else done_hours
        run_source = self.source_hours_done if self.state is not None else source_done
        overall = run_hours / (uptime / 3600.0) if uptime > 0 else 0.0
        kept_rate = overall
        source_rate = run_source / (uptime / 3600.0) if uptime > 0 else 0.0
        remaining_source = float(summary.get("remaining_hours") or 0.0)

        eta_target: float | None = None
        progress_fraction = 0.0
        if isinstance(target, (int, float)) and not isinstance(target, bool):
            missing = max(0.0, float(target) - done_hours)
            eta_target = missing / kept_rate * 3600.0 if kept_rate > 0 else None
            progress_fraction = min(1.0, done_hours / float(target)) if target else 0.0
        elif summary.get("files_total"):
            progress_fraction = float(summary["files_done"]) / float(summary["files_total"])
        eta_full = remaining_source / source_rate * 3600.0 if source_rate > 0 else None

        return {
            "started_at": iso(started_at),
            "uptime_s": round(uptime, 1),
            "device": device,
            "target": target,
            "totals": {
                "files": int(summary.get("files_total") or 0),
                "source_hours": round(float(summary.get("source_hours") or 0.0), 3),
            },
            "done": {
                "files": int(summary.get("files_done") or 0),
                "chunks": int(summary.get("chunks") or 0),
                "hours": round(done_hours, 3),
                "shards_uploaded": int(summary.get("shards_uploaded") or 0),
                "bytes_uploaded": int(summary.get("bytes_uploaded") or 0),
            },
            "remaining": {
                "files": int(summary.get("remaining_files") or 0),
                "hours": round(remaining_source, 3),
            },
            "speed": {
                "window_s": WINDOW_S,
                "rolling_10min": round(rolling, 3),
                "overall": round(overall, 3),
            },
            "eta_s": {
                "target": None if eta_target is None else round(eta_target, 1),
                "full": None if eta_full is None else round(eta_full, 1),
            },
            "progress": {"fraction": round(progress_fraction, 4)},
            "stage_times": stage_times,
            "tier": {
                "reached": order.tier_of(done_hours),
                "hours": round(done_hours, 3),
            },
            "errors": {"count": int(errors_count), "last": errors[-RECENT_ERRORS:]},
            "shards": shards[-RECENT_SHARDS:],
        }


# -- HTTP ------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>hohere</title>
<style>
:root{color-scheme:dark light}
body{font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;background:#111;color:#eee}
h1{font-size:18px;margin:0 0 12px}
.bar{height:18px;background:#2a2a2a;border-radius:9px;overflow:hidden;margin:10px 0}
#fill{height:100%;width:0;background:linear-gradient(90deg,#3b82f6,#22c55e);transition:width .4s}
.row{display:flex;justify-content:space-between;gap:8px;padding:3px 0;border-bottom:1px solid #222}
.k{color:#9aa}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px;color:#888}
#err{color:#f88}
</style></head>
<body>
<h1>hohere · <span id="stage">connecting…</span></h1>
<div class="bar"><div id="fill"></div></div>
<div id="rows"></div>
<p id="err"></p>
<pre id="raw"></pre>
<script>
const q=location.search;
function human(sec){
  if(sec===null||sec===undefined||!isFinite(sec))return "—";
  if(sec<90)return Math.round(sec)+" s";
  if(sec<5400)return Math.round(sec/60)+" min";
  if(sec<172800)return (sec/3600).toFixed(1)+" h";
  return (sec/86400).toFixed(1)+" d";
}
function rows(s){
  const e=s.eta_s||{}, sp=s.speed||{}, d=s.done||{}, t=s.totals||{}, r=s.remaining||{};
  return [
    ["target", String(s.target)],
    ["done", d.hours+" h · "+d.chunks+" chunks · "+d.files+"/"+t.files+" files"],
    ["remaining", r.hours+" h · "+r.files+" files"],
    ["speed (10 min / all)", sp.rolling_10min+" / "+sp.overall+" audio h per wall h"],
    ["eta target / full", human(e.target)+" / "+human(e.full)],
    ["tier reached", String((s.tier||{}).reached)],
    ["uploaded", (d.shards_uploaded||0)+" shards · "+((d.bytes_uploaded||0)/1e9).toFixed(2)+" GB"],
    ["uptime", human(s.uptime_s)+" · "+s.device],
    ["errors", String((s.errors||{}).count||0)],
  ].map(([k,v])=>'<div class="row"><span class="k">'+k+'</span><span>'+v+'</span></div>').join("");
}
async function tick(){
  try{
    const res=await fetch('/api/status'+q,{cache:'no-store'});
    if(!res.ok)throw new Error('HTTP '+res.status);
    const s=await res.json();
    document.getElementById('fill').style.width=Math.round(100*(s.progress||{}).fraction||0)+'%';
    document.getElementById('rows').innerHTML=rows(s);
    const last=(s.errors&&s.errors.last)||[];
    document.getElementById('stage').textContent=last.length?'errors':'running';
    document.getElementById('err').textContent=last.slice(-3).join(' | ');
    document.getElementById('raw').textContent=JSON.stringify({stage_times:s.stage_times,shards:s.shards.slice(-3)},null,1);
  }catch(err){
    document.getElementById('stage').textContent='offline';
    document.getElementById('err').textContent=String(err);
  }
}
tick();setInterval(tick,5000);
</script>
</body></html>
"""


def _handler(progress: Progress, state: Any, token: str | None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "hohere/2"

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _allowed(self, query: str) -> bool:
            if token is None:
                return True
            given = parse_qs(query).get("token", [""])[0]
            return hmac.compare_digest(given, token)

        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            if not self._allowed(parts.query):
                self._send(403, "text/plain; charset=utf-8", b"forbidden\n")
                return
            if parts.path == "/api/status":
                payload = json.dumps(progress.snapshot(state)).encode("utf-8")
                self._send(200, "application/json", payload)
            elif parts.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found\n")

        def do_HEAD(self) -> None:
            self.do_GET()

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the console clean
            del fmt, args

    return Handler


class StatusServer:
    """`http.server` in a daemon thread; bind 0.0.0.0 so a phone can read it."""

    def __init__(
        self,
        progress: Progress,
        state: Any = None,
        *,
        host: str = "0.0.0.0",
        port: int = 0,
        token: str | None = None,
    ) -> None:
        self.progress = progress
        self.state = state
        self.host = host
        self.token = token
        self._server = ThreadingHTTPServer((host, port), _handler(progress, state, token))
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        suffix = f"?token={self.token}" if self.token else ""
        return f"http://127.0.0.1:{self.port}/{suffix}"

    def start(self) -> StatusServer:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="hohere-status", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> StatusServer:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def serve(
    progress: Progress,
    state: Any = None,
    *,
    port: int = 0,
    host: str = "0.0.0.0",
    token: str | None = None,
) -> StatusServer:
    """Start the page and return the running server (never blocks the caller)."""
    return StatusServer(progress, state, host=host, port=port, token=token).start()
