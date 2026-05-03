#!/usr/bin/env python3
"""JARVIS — Web AI Assistant."""
import os
import json
import time
import base64
import socket
import logging
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("jarvis")

app = Flask(__name__)
CORS(app)

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")
OPENWEATHER_KEY     = os.environ.get("OPENWEATHER_KEY", "")
SERPER_API_KEY      = os.environ.get("SERPER_API_KEY", "")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
SESSION_TTL  = 60 * 60  # 1 hour
MAX_SESSIONS = 500

chat_histories = {}  # sid -> {"history": [...], "ts": float}
_lock = threading.Lock()

MUSIC_GENRES = [
    "pop", "hip-hop", "rock", "jazz", "electronic", "lo-fi", "classical",
    "indie", "r&b", "ambient", "metal", "reggae", "soul", "blues", "latin",
    "country", "house", "afrobeats", "kpop", "dance", "punjabi", "bollywood",
]


def cleanup_sessions():
    now = time.time()
    with _lock:
        stale = [sid for sid, v in chat_histories.items() if now - v["ts"] > SESSION_TTL]
        for sid in stale:
            del chat_histories[sid]
        if len(chat_histories) > MAX_SESSIONS:
            sorted_sids = sorted(chat_histories.items(), key=lambda kv: kv[1]["ts"])
            for sid, _ in sorted_sids[:len(chat_histories) - MAX_SESSIONS]:
                del chat_histories[sid]


def search_itunes(term, limit=8):
    try:
        r = requests.get(
            "https://itunes.apple.com/search",
            params={"term": term, "media": "music", "limit": limit, "entity": "song"},
            timeout=10,
        )
        r.raise_for_status()
        return [{
            "title":   t.get("trackName", ""),
            "artist":  t.get("artistName", ""),
            "album":   t.get("collectionName", ""),
            "image":   t.get("artworkUrl100", "").replace("100x100", "300x300"),
            "url":     t.get("trackViewUrl", ""),
            "preview": t.get("previewUrl", ""),
        } for t in r.json().get("results", [])]
    except requests.RequestException as e:
        log.warning("iTunes search failed: %s", e)
        return []


def do_search(q):
    if not SERPER_API_KEY:
        return ""
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            data=json.dumps({"q": q}),
            timeout=8,
        )
        r.raise_for_status()
        return " | ".join(o.get("snippet", "") for o in r.json().get("organic", [])[:3])
    except requests.RequestException as e:
        log.warning("Serper search failed: %s", e)
        return ""


def do_weather(city="Bangalore"):
    if not OPENWEATHER_KEY:
        return {"temp": "--", "feels_like": "--", "humidity": "--", "description": "no api key", "city": city}
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": OPENWEATHER_KEY, "units": "metric"},
            timeout=8,
        )
        r.raise_for_status()
        d = r.json()
        return {
            "temp": round(d["main"]["temp"]),
            "feels_like": round(d["main"]["feels_like"]),
            "humidity": d["main"]["humidity"],
            "description": d["weather"][0]["description"],
            "city": d.get("name", city),
        }
    except requests.RequestException as e:
        log.warning("Weather failed: %s", e)
        return {"temp": "--", "feels_like": "--", "humidity": "--", "description": "unavailable", "city": city}


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    cleanup_sessions()
    data = request.json or {}
    user_msg = (data.get("message") or "").strip()
    sid = data.get("session_id", "default")
    if not user_msg:
        return jsonify({"error": "empty message"}), 400
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500

    with _lock:
        if sid not in chat_histories:
            chat_histories[sid] = {"history": [], "ts": time.time()}
        chat_histories[sid]["ts"] = time.time()
        history = list(chat_histories[sid]["history"])

    ctx = do_search(user_msg) if len(user_msg) > 5 else ""
    user_content = ("Context: {}\n\n".format(ctx) if ctx else "") + "Question: {}".format(user_msg)
    msgs = history[-12:] + [{"role": "user", "content": user_content}]

    def generate():
        full = ""
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": 1024,
                    "stream": True,
                    "system": (
                        "You are JARVIS, Tony Stark's AI assistant. Be concise, helpful, "
                        "slightly witty. Use markdown when useful (bold, code blocks, lists). "
                        "Address the user as 'sir' occasionally."
                    ),
                    "messages": msgs,
                },
                stream=True,
                timeout=60,
            )
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                try:
                    ev = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "content_block_delta":
                    token = ev.get("delta", {}).get("text", "")
                    full += token
                    yield "data: {}\n\n".format(json.dumps({"token": token}))

            with _lock:
                chat_histories[sid]["history"].append({"role": "user", "content": user_msg})
                chat_histories[sid]["history"].append({"role": "assistant", "content": full})
                chat_histories[sid]["history"] = chat_histories[sid]["history"][-20:]
                chat_histories[sid]["ts"] = time.time()
            yield "data: {}\n\n".format(json.dumps({"done": True, "full": full}))
        except requests.RequestException as e:
            log.error("Chat stream error: %s", e)
            yield "data: {}\n\n".format(json.dumps({"error": str(e)}))

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/chat/clear", methods=["POST"])
def chat_clear():
    sid = (request.json or {}).get("session_id", "default")
    with _lock:
        chat_histories.pop(sid, None)
    return jsonify({"ok": True})


@app.route("/api/chat/export")
def chat_export():
    sid = request.args.get("session_id", "default")
    with _lock:
        h = chat_histories.get(sid, {}).get("history", [])
    return jsonify({"session_id": sid, "messages": h, "exported_at": int(time.time())})


@app.route("/api/tts", methods=["POST"])
def tts():
    text = ((request.json or {}).get("text") or "").strip()
    if not text:
        return jsonify({"error": "no text"}), 400
    if not (ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID):
        return jsonify({"error": "tts not configured"}), 503
    try:
        r = requests.post(
            "https://api.elevenlabs.io/v1/text-to-speech/{}".format(ELEVENLABS_VOICE_ID),
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text[:500],
                "model_id": "eleven_turbo_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=20,
        )
        if r.status_code == 200:
            return jsonify({"audio": base64.b64encode(r.content).decode()})
        return jsonify({"error": "ElevenLabs {}".format(r.status_code)}), 502
    except requests.RequestException as e:
        log.error("TTS error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/weather")
def weather():
    return jsonify(do_weather(request.args.get("city", "Bangalore")))


@app.route("/api/music/search")
def music_search():
    query = (request.args.get("q") or "").strip()
    genre = (request.args.get("genre") or "").strip()
    try:
        limit = max(1, min(int(request.args.get("limit", "10")), 25))
    except ValueError:
        limit = 10
    if query:
        results = search_itunes(query, limit)
    elif genre:
        results = search_itunes(genre + " music", limit) or search_itunes(genre, limit)
    else:
        results = search_itunes("top hits 2024", limit)
    return jsonify(results)


@app.route("/api/music/genres")
def music_genres():
    return jsonify(MUSIC_GENRES)


@app.route("/api/sysinfo")
def sysinfo():
    if not HAS_PSUTIL:
        return jsonify({"available": False})
    try:
        return jsonify({
            "available": True,
            "cpu": round(psutil.cpu_percent(interval=None)),
            "memory": round(psutil.virtual_memory().percent),
            "disk": round(psutil.disk_usage("/").percent),
        })
    except Exception as e:
        log.warning("sysinfo failed: %s", e)
        return jsonify({"available": False})


@app.route("/api/status")
def status():
    return jsonify({
        "anthropic": bool(ANTHROPIC_API_KEY),
        "elevenlabs": bool(ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID),
        "weather": bool(OPENWEATHER_KEY),
        "search": bool(SERPER_API_KEY),
        "itunes": True,
    })


@app.route("/")
def index():
    return HTML


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JARVIS — AI Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
<style>
:root{
  --bg:#020912; --bg2:#040e1a; --panel:rgba(8,20,31,.72); --card:#0b1a2e;
  --cyan:#00d4ff; --cyan-dim:#0088aa; --blue:#0055ff; --gold:#ffc200;
  --red:#ff2244; --green:#00ff88; --text:#d4eeff; --dim:#3a6080;
  --border:#0d2a44; --border-bright:#1e4a7a;
  --shadow:0 0 30px rgba(0,212,255,.12);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  font-family:'JetBrains Mono',ui-monospace,monospace;
  background:radial-gradient(ellipse at top,#0a1f3a 0%,var(--bg) 50%,#000 100%);
  color:var(--text); height:100vh; overflow:hidden;
  display:flex; flex-direction:column;
}
body::before{
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:linear-gradient(rgba(0,212,255,.04) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,212,255,.04) 1px,transparent 1px);
  background-size:48px 48px;
  animation:gridScroll 24s linear infinite;
  mask-image:radial-gradient(ellipse at center,#000 30%,transparent 80%);
}
body::after{
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:radial-gradient(circle at 20% 30%,rgba(0,85,255,.08),transparent 40%),
             radial-gradient(circle at 80% 70%,rgba(0,212,255,.06),transparent 40%);
}
@keyframes gridScroll{to{background-position:48px 48px}}

/* HUD corner brackets */
.hud{position:relative}
.hud::before,.hud::after{
  content:''; position:absolute; width:12px; height:12px;
  border:1px solid var(--cyan); pointer-events:none;
}
.hud::before{top:-1px;left:-1px;border-right:none;border-bottom:none}
.hud::after{bottom:-1px;right:-1px;border-left:none;border-top:none}

#topbar{
  position:relative; z-index:10;
  display:flex; align-items:center; justify-content:space-between;
  padding:0 24px; height:60px;
  background:linear-gradient(90deg,var(--bg2) 0%,rgba(0,85,255,.1) 50%,var(--bg2) 100%);
  border-bottom:1px solid var(--cyan);
  box-shadow:0 0 30px rgba(0,212,255,.18);
}
.logo{
  font-family:'Orbitron',sans-serif; font-size:24px; font-weight:900;
  color:var(--cyan); letter-spacing:6px;
  text-shadow:0 0 24px rgba(0,212,255,.9), 0 0 4px rgba(0,212,255,1);
}
.logo .sub{font-size:9px;letter-spacing:3px;color:var(--dim);font-weight:400;display:block;margin-top:-2px}
.meta{display:flex;gap:20px;align-items:center}
#clock{font-size:16px;color:var(--gold);font-variant-numeric:tabular-nums}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);animation:pulse 2s infinite}
#stext{color:var(--green);font-size:10px;letter-spacing:2px}
.iconbtn{
  background:transparent;border:1px solid var(--border-bright);color:var(--cyan);
  font-family:inherit;font-size:11px;padding:6px 10px;border-radius:3px;cursor:pointer;
  transition:all .15s;letter-spacing:1px;
}
.iconbtn:hover{background:rgba(0,212,255,.1);border-color:var(--cyan)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

#main{position:relative;z-index:1;flex:1;display:flex;gap:10px;padding:10px;overflow:hidden}

.panel{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:4px;
  backdrop-filter:blur(8px);
  -webkit-backdrop-filter:blur(8px);
  display:flex;flex-direction:column;overflow:hidden;
  box-shadow:var(--shadow);
}
.ptitle{
  font-size:9px;letter-spacing:3px;color:var(--cyan-dim);
  padding:10px 14px;
  border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;
  text-transform:uppercase;
}

#left{width:220px;display:flex;flex-direction:column;gap:8px}
.sp{
  background:var(--panel); border:1px solid var(--border); border-radius:4px;
  padding:12px; backdrop-filter:blur(8px);
}
.sp h3{font-size:9px;letter-spacing:3px;color:var(--cyan-dim);margin-bottom:10px;border-bottom:1px solid var(--border);padding-bottom:6px}
#radarCanvas{display:block;margin:0 auto}
.sr{display:flex;align-items:center;gap:8px;font-size:9px;margin-bottom:7px}
.sl{color:var(--dim);width:74px;flex-shrink:0;letter-spacing:1px}
.sb{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden;position:relative}
.sf{height:100%;border-radius:2px;transition:width .8s ease;box-shadow:0 0 8px currentColor}
.sp2{color:var(--cyan);font-size:9px;width:30px;text-align:right;font-variant-numeric:tabular-nums}
.qb{
  display:flex;align-items:center;gap:8px;width:100%;
  background:rgba(0,0,0,.3); border:1px solid var(--border); color:var(--dim);
  font-family:inherit; font-size:10px; padding:7px 10px; cursor:pointer;
  border-radius:3px; margin-bottom:4px; transition:all .15s; text-decoration:none;
  letter-spacing:1px;
}
.qb:hover{border-color:var(--cyan);color:var(--cyan);background:rgba(0,212,255,.05);transform:translateX(2px)}
.qb-icon{width:14px;height:14px;flex-shrink:0;opacity:.7}

#center{flex:1;min-width:0}
#chatbox{flex:1;overflow-y:auto;padding:18px;scroll-behavior:smooth;scrollbar-width:thin}
#chatbox::-webkit-scrollbar{width:5px}
#chatbox::-webkit-scrollbar-thumb{background:var(--border-bright);border-radius:3px}
#chatbox::-webkit-scrollbar-track{background:transparent}

.empty{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:100%;text-align:center;gap:24px;color:var(--dim);
}
.empty .lg{font-family:'Orbitron',sans-serif;font-size:48px;font-weight:900;letter-spacing:10px;color:var(--cyan);text-shadow:0 0 30px var(--cyan);opacity:.4}
.empty .tag{font-size:11px;letter-spacing:3px;text-transform:uppercase}
.suggest{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;max-width:520px;width:100%}
.scard{
  background:rgba(0,212,255,.04);border:1px solid var(--border);
  padding:12px 14px;border-radius:4px;cursor:pointer;text-align:left;
  font-size:11px;color:var(--text);transition:all .15s;line-height:1.5;
}
.scard:hover{border-color:var(--cyan);background:rgba(0,212,255,.08);transform:translateY(-1px)}
.scard b{color:var(--cyan);font-weight:600;display:block;margin-bottom:3px;letter-spacing:1px;font-size:10px}

.msg{margin-bottom:18px;display:flex;gap:12px;animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.av{
  width:32px;height:32px;border-radius:4px;display:flex;align-items:center;
  justify-content:center;font-size:11px;flex-shrink:0;margin-top:2px;
  font-weight:600;letter-spacing:1px;
}
.msg.j .av{background:rgba(0,212,255,.12);border:1px solid var(--cyan);color:var(--cyan);box-shadow:0 0 12px rgba(0,212,255,.3)}
.msg.u .av{background:rgba(255,194,0,.12);border:1px solid var(--gold);color:var(--gold)}
.msg.s .av{background:rgba(0,255,136,.08);border:1px solid var(--green);color:var(--green)}
.mb{flex:1;min-width:0}
.mm{font-size:9px;color:var(--dim);margin-bottom:5px;display:flex;align-items:center;gap:8px;letter-spacing:1px}
.spk{font-size:10px;font-weight:600;letter-spacing:2px}
.msg.j .spk{color:var(--cyan)}.msg.u .spk{color:var(--gold)}.msg.s .spk{color:var(--green)}
.mt{font-size:13px;line-height:1.7;color:rgba(212,238,255,.95);word-wrap:break-word}
.mt p{margin-bottom:8px}.mt p:last-child{margin-bottom:0}
.mt strong{color:var(--cyan);font-weight:600}
.mt em{color:var(--gold);font-style:italic}
.mt code{background:rgba(0,0,0,.4);padding:2px 6px;border-radius:3px;font-size:12px;color:var(--green);border:1px solid var(--border)}
.mt pre{background:rgba(0,0,0,.5);border:1px solid var(--border);padding:10px 12px;border-radius:4px;overflow-x:auto;margin:8px 0}
.mt pre code{background:none;border:none;padding:0;color:var(--text);font-size:12px}
.mt ul,.mt ol{padding-left:20px;margin:6px 0}.mt li{margin-bottom:3px}
.mt a{color:var(--cyan);text-decoration:underline;text-underline-offset:2px}
.mt h1,.mt h2,.mt h3{color:var(--cyan);margin:8px 0 4px;font-family:'Orbitron',sans-serif;letter-spacing:1px}
.tc{display:inline-block;width:7px;height:13px;background:var(--cyan);animation:bl .7s step-end infinite;vertical-align:text-bottom;margin-left:2px;box-shadow:0 0 8px var(--cyan)}
@keyframes bl{0%,100%{opacity:1}50%{opacity:0}}
.copybtn{
  background:transparent;border:1px solid var(--border);color:var(--dim);
  font-family:inherit;font-size:8px;padding:2px 6px;cursor:pointer;border-radius:2px;
  letter-spacing:1px;transition:all .15s;opacity:0;
}
.msg:hover .copybtn{opacity:1}
.copybtn:hover{border-color:var(--cyan);color:var(--cyan)}

.interim{color:var(--dim);font-style:italic;opacity:.7}

#waveCanvas{width:100%;height:54px;display:block}
#irow{display:flex;gap:8px;padding:12px 14px;border-top:1px solid var(--border);background:rgba(0,0,0,.3)}
#minput{
  flex:1;background:rgba(0,0,0,.4);border:1px solid var(--border);border-radius:3px;
  color:var(--text);font-family:inherit;font-size:13px;padding:10px 14px;outline:none;
  transition:all .15s;min-width:0;
}
#minput:focus{border-color:var(--cyan);box-shadow:0 0 0 1px var(--cyan),0 0 16px rgba(0,212,255,.2)}
#minput::placeholder{color:var(--dim)}
.btn{
  font-family:inherit;font-size:11px;padding:10px 16px;border:none;border-radius:3px;
  cursor:pointer;transition:all .15s;letter-spacing:2px;white-space:nowrap;font-weight:600;
}
#sbtn{background:var(--cyan);color:var(--bg);font-weight:700;box-shadow:0 0 12px rgba(0,212,255,.3)}
#sbtn:hover{background:#33ddff;box-shadow:0 0 20px rgba(0,212,255,.5)}
#sbtn:disabled{opacity:.4;cursor:not-allowed;box-shadow:none}
#mbtn{background:transparent;color:var(--cyan);border:1px solid var(--cyan)}
#mbtn:hover{background:rgba(0,212,255,.1)}
#mbtn.active{background:var(--red);color:#fff;border-color:var(--red);animation:micPulse 1.2s ease infinite}
@keyframes micPulse{0%,100%{box-shadow:0 0 0 0 rgba(255,34,68,.6)}50%{box-shadow:0 0 0 8px rgba(255,34,68,0)}}
#stpbtn{background:transparent;color:var(--red);border:1px solid var(--red)}
#stpbtn:hover{background:rgba(255,34,68,.1)}

#right{width:300px;display:flex;flex-direction:column;gap:8px;overflow-y:auto;scrollbar-width:thin}
#right::-webkit-scrollbar{width:5px}
#right::-webkit-scrollbar-thumb{background:var(--border-bright);border-radius:3px}
.wc,.apic{background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:14px;backdrop-filter:blur(8px)}
.wc h3,.apic h3{font-size:9px;letter-spacing:3px;color:var(--cyan-dim);margin-bottom:10px;display:flex;justify-content:space-between;align-items:center}
.wcity{font-size:9px;color:var(--dim);text-transform:none;letter-spacing:1px}
.wtemp{font-size:32px;color:var(--gold);font-family:'Orbitron',sans-serif;font-weight:700;text-shadow:0 0 20px rgba(255,194,0,.4)}
.wdesc{color:var(--text);font-size:10px;text-transform:uppercase;letter-spacing:2px;margin-top:2px}
.wmeta{display:flex;gap:16px;margin-top:8px;font-size:9px;color:var(--dim);letter-spacing:1px}
.ar{display:flex;justify-content:space-between;font-size:9px;margin-bottom:5px;letter-spacing:1px}
.ok{color:var(--green)}.bad{color:var(--red)}

#vbadge{
  display:none;position:fixed;top:74px;left:50%;transform:translateX(-50%);
  background:rgba(255,34,68,.95);color:#fff;padding:8px 24px;border-radius:24px;
  font-size:11px;letter-spacing:3px;z-index:100;font-weight:600;
  box-shadow:0 4px 24px rgba(255,34,68,.4);
}
#vbadge.on{display:flex;align-items:center;gap:8px}
#vbadge::before{content:'';width:8px;height:8px;border-radius:50%;background:#fff;animation:pulse 1s infinite}

#toast{
  position:fixed;bottom:90px;left:50%;transform:translateX(-50%);
  background:rgba(0,212,255,.95);color:var(--bg);padding:10px 24px;border-radius:24px;
  font-size:11px;letter-spacing:2px;opacity:0;transition:opacity .3s;
  pointer-events:none;z-index:200;font-weight:600;
  box-shadow:0 4px 24px rgba(0,212,255,.4);
}
#toast.show{opacity:1}
#toast.err{background:rgba(255,34,68,.95);color:#fff;box-shadow:0 4px 24px rgba(255,34,68,.4)}

#musicSection{background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:14px;flex:1;display:flex;flex-direction:column;overflow:hidden;min-height:0;backdrop-filter:blur(8px)}
#musicSection h3{font-size:9px;letter-spacing:3px;color:var(--cyan-dim);margin-bottom:10px}
#musicSearch{display:flex;gap:6px;margin-bottom:8px}
#musicInput{flex:1;background:rgba(0,0,0,.4);border:1px solid var(--border);border-radius:3px;color:var(--text);font-family:inherit;font-size:10px;padding:6px 10px;outline:none}
#musicInput:focus{border-color:var(--green)}
#musicSearchBtn{background:transparent;border:1px solid var(--green);color:var(--green);font-family:inherit;font-size:10px;padding:5px 12px;cursor:pointer;border-radius:3px;letter-spacing:1px}
#musicSearchBtn:hover{background:rgba(0,255,136,.1)}
#genreRow{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px}
.gtag{background:rgba(0,0,0,.3);border:1px solid var(--border);color:var(--dim);font-family:inherit;font-size:9px;padding:4px 8px;cursor:pointer;border-radius:2px;transition:all .15s;white-space:nowrap;letter-spacing:1px}
.gtag:hover,.gtag.active{border-color:var(--green);color:var(--green);background:rgba(0,255,136,.06)}
#songList{overflow-y:auto;flex:1;scrollbar-width:thin}
#songList::-webkit-scrollbar{width:3px}
#songList::-webkit-scrollbar-thumb{background:var(--border-bright)}
.songCard{display:flex;align-items:center;gap:10px;padding:7px 4px;border-bottom:1px solid rgba(13,42,68,.4);transition:background .15s}
.songCard:hover{background:rgba(0,212,255,.03)}
.songImg{width:40px;height:40px;border-radius:3px;object-fit:cover;background:var(--border);flex-shrink:0}
.songInfo{flex:1;overflow:hidden}
.songTitle{font-size:11px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.songArtist{font-size:9px;color:var(--dim);margin-top:2px}
.songBtns{display:flex;gap:4px;flex-shrink:0;align-items:center}
.prevBtn{background:transparent;border:1px solid var(--cyan);color:var(--cyan);font-family:inherit;font-size:9px;padding:3px 8px;cursor:pointer;border-radius:2px;letter-spacing:1px}
.prevBtn:hover{background:rgba(0,212,255,.1)}
.prevBtn.playing{border-color:var(--red);color:var(--red)}
.openBtn{color:var(--green);font-size:14px;text-decoration:none;padding:0 4px}

/* Modal */
.modal{display:none;position:fixed;inset:0;z-index:300;background:rgba(0,0,0,.7);backdrop-filter:blur(4px);align-items:center;justify-content:center}
.modal.on{display:flex}
.modal-card{background:var(--bg2);border:1px solid var(--cyan);border-radius:4px;padding:24px;width:420px;max-width:92vw;box-shadow:0 0 40px rgba(0,212,255,.3)}
.modal-card h2{font-family:'Orbitron',sans-serif;font-size:16px;color:var(--cyan);letter-spacing:3px;margin-bottom:20px;border-bottom:1px solid var(--border);padding-bottom:10px}
.field{margin-bottom:16px}
.field label{display:block;font-size:10px;letter-spacing:2px;color:var(--dim);margin-bottom:6px;text-transform:uppercase}
.field input,.field select{width:100%;background:rgba(0,0,0,.4);border:1px solid var(--border);border-radius:3px;color:var(--text);font-family:inherit;font-size:12px;padding:8px 10px;outline:none}
.field input:focus,.field select:focus{border-color:var(--cyan)}
.field-row{display:flex;align-items:center;gap:10px}
.field-row input[type=checkbox]{width:auto}
.modal-btns{display:flex;gap:8px;justify-content:flex-end;margin-top:20px}

@media(max-width:900px){
  #left,#right{display:none}
  .logo{font-size:18px;letter-spacing:3px}
  .empty .lg{font-size:32px}
  .suggest{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div id="topbar">
  <div class="logo">JARVIS<span class="sub">JUST A RATHER VERY INTELLIGENT SYSTEM</span></div>
  <div class="meta">
    <div class="dot"></div><span id="stext">SYSTEM ONLINE</span>
    <span id="clock">--:--:--</span>
    <button class="iconbtn" id="settingsBtn" title="Settings">⚙ SETTINGS</button>
  </div>
</div>
<div id="vbadge">LISTENING</div>
<div id="toast"></div>

<div id="main">
  <div id="left">
    <div class="sp hud"><h3>NEURAL RADAR</h3><canvas id="radarCanvas" width="190" height="150"></canvas></div>
    <div class="sp hud"><h3>SYSTEM STATUS</h3>
      <div class="sr"><span class="sl">CPU LOAD</span><div class="sb"><div class="sf" id="bar-cpu" style="width:0%;background:var(--cyan);color:var(--cyan)"></div></div><span class="sp2" id="val-cpu">--%</span></div>
      <div class="sr"><span class="sl">MEMORY</span><div class="sb"><div class="sf" id="bar-mem" style="width:0%;background:var(--blue);color:var(--blue)"></div></div><span class="sp2" id="val-mem">--%</span></div>
      <div class="sr"><span class="sl">DISK</span><div class="sb"><div class="sf" id="bar-disk" style="width:0%;background:var(--green);color:var(--green)"></div></div><span class="sp2" id="val-disk">--%</span></div>
      <div class="sr"><span class="sl">NEURAL NET</span><div class="sb"><div class="sf" style="width:93%;background:var(--gold);color:var(--gold)"></div></div><span class="sp2">93%</span></div>
    </div>
    <div class="sp hud"><h3>QUICK LAUNCH</h3>
      <a class="qb" href="https://youtube.com" target="_blank" rel="noopener">▶ YouTube</a>
      <a class="qb" href="https://github.com" target="_blank" rel="noopener">⌥ GitHub</a>
      <a class="qb" href="https://mail.google.com" target="_blank" rel="noopener">✉ Gmail</a>
      <a class="qb" href="https://music.apple.com" target="_blank" rel="noopener">♪ Apple Music</a>
      <a class="qb" href="https://open.spotify.com" target="_blank" rel="noopener">♫ Spotify</a>
    </div>
  </div>

  <div class="panel hud" id="center">
    <div class="ptitle"><span>COMMUNICATION LINK</span><span id="sessInfo" style="color:var(--dim)">SID: --</span></div>
    <div id="chatbox"></div>
    <div style="border-top:1px solid var(--border);padding:6px 14px;background:rgba(0,0,0,.3)"><canvas id="waveCanvas"></canvas></div>
    <div id="irow">
      <button class="btn" id="mbtn" title="Hold or click to speak (Space)">◉ MIC</button>
      <input id="minput" type="text" placeholder="Type a command, /help for shortcuts, or click MIC..." autocomplete="off" autofocus>
      <button class="btn" id="stpbtn" title="Stop voice/audio">■ STOP</button>
      <button class="btn" id="sbtn">SEND →</button>
    </div>
  </div>

  <div id="right">
    <div class="wc hud">
      <h3><span>WEATHER INTEL</span><span class="wcity" id="wcity">Bangalore</span></h3>
      <div class="wtemp" id="wtemp">--°</div>
      <div class="wdesc" id="wdesc">Loading...</div>
      <div class="wmeta"><span id="whum">--</span><span id="wfeel">--</span></div>
    </div>
    <div id="musicSection" class="hud">
      <h3>MUSIC DISCOVER</h3>
      <div id="musicSearch">
        <input id="musicInput" type="text" placeholder="Search song, artist...">
        <button id="musicSearchBtn">FIND</button>
      </div>
      <div id="genreRow"></div>
      <div id="songList"><div style="color:var(--dim);font-size:10px;padding:8px;text-align:center">Pick a genre or search above</div></div>
    </div>
    <div class="apic hud"><h3>API STATUS</h3>
      <div class="ar"><span>Anthropic</span><span id="st-anthropic" class="dim">CHECKING...</span></div>
      <div class="ar"><span>ElevenLabs</span><span id="st-elevenlabs" class="dim">CHECKING...</span></div>
      <div class="ar"><span>Weather</span><span id="st-weather" class="dim">CHECKING...</span></div>
      <div class="ar"><span>Search</span><span id="st-search" class="dim">CHECKING...</span></div>
      <div class="ar"><span>iTunes</span><span id="st-itunes" class="dim">CHECKING...</span></div>
    </div>
  </div>
</div>

<!-- Settings Modal -->
<div class="modal" id="settingsModal">
  <div class="modal-card hud">
    <h2>⚙ SETTINGS</h2>
    <div class="field">
      <label>Weather City</label>
      <input id="setCity" type="text" placeholder="Bangalore">
    </div>
    <div class="field">
      <label>Voice Recognition Language</label>
      <select id="setLang">
        <option value="en-US">English (US)</option>
        <option value="en-IN">English (India)</option>
        <option value="en-GB">English (UK)</option>
        <option value="en-AU">English (Australia)</option>
        <option value="hi-IN">Hindi (India)</option>
        <option value="es-ES">Spanish (Spain)</option>
        <option value="fr-FR">French</option>
        <option value="de-DE">German</option>
        <option value="ja-JP">Japanese</option>
      </select>
    </div>
    <div class="field">
      <div class="field-row"><input type="checkbox" id="setAutoSpeak"><label for="setAutoSpeak" style="margin:0">Auto-speak replies (TTS)</label></div>
    </div>
    <div class="field">
      <div class="field-row"><input type="checkbox" id="setLiveTranscript" checked><label for="setLiveTranscript" style="margin:0">Show live transcript while speaking</label></div>
    </div>
    <div class="modal-btns">
      <button class="btn" id="setCancel" style="background:transparent;border:1px solid var(--border);color:var(--dim)">CANCEL</button>
      <button class="btn" id="setSave" style="background:var(--cyan);color:var(--bg)">SAVE</button>
    </div>
  </div>
</div>

<script>
/* ===========================================================
   STATE
=========================================================== */
const SID = 'sid_' + Math.random().toString(36).slice(2, 10);
let listening = false;
let speaking = false;
let recognition = null;
let curAudio = null;
let prevAudio = null;
let prevBtn = null;
let audioCtx = null;
let analyser = null;
let micStream = null;
let interimEl = null;

const settings = {
  city: localStorage.getItem('jarvis.city') || 'Bangalore',
  lang: localStorage.getItem('jarvis.lang') || 'en-US',
  autoSpeak: localStorage.getItem('jarvis.autoSpeak') === '1',
  liveTranscript: localStorage.getItem('jarvis.liveTranscript') !== '0',
};

document.getElementById('sessInfo').textContent = 'SID: ' + SID.slice(0, 12);

/* ===========================================================
   UI HELPERS
=========================================================== */
function $(id){return document.getElementById(id)}

function toast(msg, opts){
  opts = opts || {};
  const el = $('toast');
  el.textContent = msg;
  el.classList.toggle('err', !!opts.err);
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), opts.duration || 2800);
}

function setStatus(text, color){
  const el = $('stext');
  el.textContent = text;
  el.style.color = color || 'var(--green)';
}

function clock(){
  $('clock').textContent = new Date().toTimeString().slice(0, 8);
}
setInterval(clock, 1000); clock();

/* ===========================================================
   CHAT
=========================================================== */
function clearEmpty(){
  const empty = document.querySelector('.empty');
  if (empty) empty.remove();
}

function renderEmpty(){
  const box = $('chatbox');
  box.innerHTML = `
    <div class="empty">
      <div>
        <div class="lg">JARVIS</div>
        <div class="tag">AI Assistant — Online</div>
      </div>
      <div class="suggest">
        <button class="scard" data-q="What's the weather like today?"><b>WEATHER</b>What's the weather like today?</button>
        <button class="scard" data-q="Tell me a quick joke"><b>SOCIAL</b>Tell me a quick joke</button>
        <button class="scard" data-q="Summarize the latest in AI news"><b>NEWS</b>Summarize the latest in AI news</button>
        <button class="scard" data-q="Write a Python function to reverse a string"><b>CODE</b>Write a Python function to reverse a string</button>
      </div>
      <div style="font-size:10px;color:var(--dim);letter-spacing:1px">TIP: Press SPACE to talk · /help for commands</div>
    </div>`;
  box.querySelectorAll('.scard').forEach(b => {
    b.onclick = () => { const q = b.dataset.q; $('minput').value = q; submitInput(); };
  });
}

function appendMsg(speaker, text, type){
  clearEmpty();
  type = type || 'j';
  const box = $('chatbox');
  const d = document.createElement('div');
  d.className = 'msg ' + type;
  const av = type === 'u' ? 'YOU' : type === 's' ? 'SYS' : 'J';
  const now = new Date().toTimeString().slice(0, 5);
  d.innerHTML = `<div class="av">${av}</div>
    <div class="mb">
      <div class="mm"><span class="spk">${speaker}</span><span>${now}</span></div>
      <div class="mt"></div>
    </div>`;
  box.appendChild(d);
  const mt = d.querySelector('.mt');
  mt.textContent = text;
  box.scrollTop = box.scrollHeight;
  return d;
}

function streamMsg(speaker){
  clearEmpty();
  const box = $('chatbox');
  const d = document.createElement('div');
  d.className = 'msg j';
  const now = new Date().toTimeString().slice(0, 5);
  d.innerHTML = `<div class="av">J</div>
    <div class="mb">
      <div class="mm"><span class="spk">${speaker}</span><span>${now}</span><button class="copybtn" style="margin-left:auto;display:none">COPY</button></div>
      <div class="mt"><span class="tc"></span></div>
    </div>`;
  box.appendChild(d);
  const mt = d.querySelector('.mt');
  const cur = mt.querySelector('.tc');
  const copyBtn = d.querySelector('.copybtn');
  let raw = '';
  box.scrollTop = box.scrollHeight;
  return {
    add(t){
      raw += t;
      cur.remove();
      mt.textContent = raw;
      mt.appendChild(cur);
      box.scrollTop = box.scrollHeight;
    },
    done(full){
      raw = full || raw;
      cur.remove();
      try { mt.innerHTML = marked.parse(raw); } catch(e){ mt.textContent = raw; }
      copyBtn.style.display = '';
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(raw).then(() => toast('Copied'));
      };
      box.scrollTop = box.scrollHeight;
      return raw;
    }
  };
}

async function send(text, speakReply){
  if (!text.trim()) return;

  // Slash commands
  if (text.startsWith('/')) {
    handleSlash(text);
    return;
  }

  setStatus('QUERYING AI...', 'var(--gold)');
  $('sbtn').disabled = true;
  const s = streamMsg('JARVIS');
  let full = '';

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, session_id: SID})
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.token) { full += ev.token; s.add(ev.token); }
          if (ev.done) { s.done(ev.full || full); if (speakReply || settings.autoSpeak) speakText(ev.full || full); }
          if (ev.error) s.done('**Error:** ' + ev.error);
        } catch(e) {}
      }
    }
  } catch(e) {
    s.done('**Connection error:** ' + e.message);
  }

  setStatus('SYSTEM ONLINE', 'var(--green)');
  $('sbtn').disabled = false;
}

function handleSlash(cmd){
  const c = cmd.trim().toLowerCase();
  if (c === '/clear') {
    fetch('/api/chat/clear', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:SID})});
    $('chatbox').innerHTML = '';
    renderEmpty();
    toast('Conversation cleared');
  } else if (c === '/help') {
    appendMsg('SYSTEM',
      '/clear — Clear conversation\n' +
      '/export — Download conversation as JSON\n' +
      '/help — Show this help\n' +
      'SPACE — Toggle microphone\n' +
      'ENTER — Send message',
      's');
  } else if (c === '/export') {
    fetch('/api/chat/export?session_id=' + SID).then(r => r.json()).then(d => {
      const blob = new Blob([JSON.stringify(d, null, 2)], {type:'application/json'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'jarvis_chat_' + Date.now() + '.json';
      a.click();
      URL.revokeObjectURL(url);
      toast('Exported');
    });
  } else {
    appendMsg('SYSTEM', 'Unknown command. Try /help', 's');
  }
}

function submitInput(){
  const i = $('minput');
  const t = i.value.trim();
  i.value = '';
  if (!t) return;
  if (!t.startsWith('/')) appendMsg('YOU', t, 'u');
  send(t, false);
}

$('sbtn').addEventListener('click', submitInput);
$('minput').addEventListener('keydown', e => { if (e.key === 'Enter') submitInput(); });

/* ===========================================================
   TTS
=========================================================== */
function stopSpeaking(){
  if (curAudio) { curAudio.pause(); curAudio.currentTime = 0; curAudio = null; }
  speaking = false;
  setStatus('SYSTEM ONLINE', 'var(--green)');
}

async function speakText(text){
  if (!text) return;
  // strip markdown for TTS
  const clean = text.replace(/```[\s\S]*?```/g,' code block ').replace(/[*_`#>]/g,'').slice(0, 480);
  stopSpeaking();
  speaking = true;
  setStatus('SPEAKING...', 'var(--cyan)');
  try {
    const r = await fetch('/api/tts', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:clean})});
    const d = await r.json();
    if (d.audio) {
      const bytes = atob(d.audio);
      const buf = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
      const url = URL.createObjectURL(new Blob([buf], {type:'audio/mpeg'}));
      curAudio = new Audio(url);
      curAudio.onended = () => { speaking = false; setStatus('SYSTEM ONLINE','var(--green)'); URL.revokeObjectURL(url); };
      curAudio.onerror = () => { speaking = false; setStatus('SYSTEM ONLINE','var(--green)'); };
      curAudio.play();
    } else {
      speaking = false;
      setStatus('SYSTEM ONLINE', 'var(--green)');
      if (d.error) toast('TTS: ' + d.error, {err:true});
    }
  } catch(e) {
    speaking = false;
    setStatus('SYSTEM ONLINE', 'var(--green)');
    toast('TTS error', {err:true});
  }
}

$('stpbtn').addEventListener('click', () => {
  stopSpeaking();
  if (prevAudio) { prevAudio.pause(); prevAudio = null; }
  if (prevBtn) { prevBtn.textContent = 'PLAY'; prevBtn.classList.remove('playing'); prevBtn = null; }
  if (recognition && listening) recognition.stop();
  toast('Stopped');
});

/* ===========================================================
   SPEECH-TO-TEXT  (the main fix)
=========================================================== */
const mbtn = $('mbtn');
const vbadge = $('vbadge');

function speechSupported(){
  return 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
}

async function ensureMicPermission(){
  // Force the browser permission prompt and verify mic actually works.
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Microphone API not available. Use HTTPS or localhost.');
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    }});
    return stream;
  } catch(e) {
    if (e.name === 'NotAllowedError') throw new Error('Microphone permission denied. Enable it in your browser settings.');
    if (e.name === 'NotFoundError')   throw new Error('No microphone found.');
    if (e.name === 'NotReadableError')throw new Error('Microphone is in use by another app.');
    throw new Error('Mic error: ' + e.message);
  }
}

function startMicVisualizer(stream){
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
  } catch(e) { /* visualizer is non-essential */ }
}

function stopMicVisualizer(){
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  analyser = null;
}

function buildRecognition(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const r = new SR();
  r.lang = settings.lang;
  r.continuous = false;
  r.interimResults = settings.liveTranscript;
  r.maxAlternatives = 1;

  r.onstart = () => {
    listening = true;
    mbtn.textContent = '◉ STOP';
    mbtn.classList.add('active');
    vbadge.classList.add('on');
    setStatus('LISTENING...', 'var(--red)');
    if (settings.liveTranscript) {
      interimEl = appendMsg('YOU (voice)', '...', 'u');
      interimEl.querySelector('.mt').classList.add('interim');
    }
  };

  r.onend = () => {
    listening = false;
    mbtn.textContent = '◉ MIC';
    mbtn.classList.remove('active');
    vbadge.classList.remove('on');
    setStatus('SYSTEM ONLINE', 'var(--green)');
    stopMicVisualizer();
    if (interimEl && interimEl.querySelector('.mt').textContent === '...') {
      interimEl.remove();
    }
    interimEl = null;
  };

  r.onerror = ev => {
    listening = false;
    mbtn.textContent = '◉ MIC';
    mbtn.classList.remove('active');
    vbadge.classList.remove('on');
    setStatus('SYSTEM ONLINE', 'var(--green)');
    if (interimEl) { interimEl.remove(); interimEl = null; }
    const msg = {
      'not-allowed':   'Mic permission denied',
      'audio-capture': 'No microphone detected',
      'no-speech':     null,
      'network':       'Network error — speech needs internet',
      'aborted':       null,
    }[ev.error];
    if (msg) toast(msg, {err:true});
  };

  r.onresult = ev => {
    let final = '', interim = '';
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const t = ev.results[i][0].transcript;
      if (ev.results[i].isFinal) final += t;
      else interim += t;
    }
    if (interimEl) {
      const mt = interimEl.querySelector('.mt');
      mt.textContent = (final + interim).trim() || '...';
    }
    if (final) {
      if (interimEl) {
        interimEl.querySelector('.mt').textContent = final;
        interimEl.querySelector('.mt').classList.remove('interim');
        interimEl = null;
      } else {
        appendMsg('YOU (voice)', final, 'u');
      }
      send(final, true);
    }
  };

  return r;
}

async function startListening(){
  if (listening) return;
  if (!speechSupported()) {
    toast('Speech recognition needs Chrome, Edge, or Safari', {err:true, duration:4000});
    return;
  }
  try {
    micStream = await ensureMicPermission();
    startMicVisualizer(micStream);
  } catch(e) {
    toast(e.message, {err:true, duration:4000});
    return;
  }
  if (!recognition) recognition = buildRecognition();
  else { recognition.lang = settings.lang; recognition.interimResults = settings.liveTranscript; }
  try { recognition.start(); }
  catch(e) {
    toast('Could not start mic: ' + e.message, {err:true});
    stopMicVisualizer();
  }
}

mbtn.addEventListener('click', () => {
  if (listening) { recognition && recognition.stop(); }
  else { startListening(); }
});

document.addEventListener('keydown', e => {
  if (e.code === 'Space' &&
      document.activeElement !== $('minput') &&
      document.activeElement !== $('musicInput') &&
      !document.querySelector('.modal.on')) {
    e.preventDefault();
    mbtn.click();
  }
  if (e.key === 'Escape' && document.querySelector('.modal.on')) {
    closeSettings();
  }
});

/* ===========================================================
   MUSIC
=========================================================== */
const genres = ['pop','hip-hop','rock','jazz','electronic','lo-fi','classical','r&b','ambient','metal','reggae','soul','blues','latin','house','afrobeats','kpop','bollywood','punjabi','dance'];
(function(){
  const row = $('genreRow');
  genres.forEach(g => {
    const btn = document.createElement('button');
    btn.className = 'gtag';
    btn.textContent = g;
    btn.onclick = () => {
      document.querySelectorAll('.gtag').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      loadMusic(null, g);
    };
    row.appendChild(btn);
  });
})();

$('musicSearchBtn').addEventListener('click', () => {
  const q = $('musicInput').value.trim();
  if (q) {
    document.querySelectorAll('.gtag').forEach(b => b.classList.remove('active'));
    loadMusic(q, null);
  }
});
$('musicInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    const q = e.target.value.trim();
    if (q) {
      document.querySelectorAll('.gtag').forEach(b => b.classList.remove('active'));
      loadMusic(q, null);
    }
  }
});

async function loadMusic(query, genre){
  const list = $('songList');
  list.innerHTML = '<div style="color:var(--dim);font-size:10px;padding:8px;text-align:center">SEARCHING...</div>';
  let url = '/api/music/search?limit=12';
  if (query) url += '&q=' + encodeURIComponent(query);
  if (genre) url += '&genre=' + encodeURIComponent(genre);
  try {
    const r = await fetch(url);
    const songs = await r.json();
    list.innerHTML = '';
    if (!songs.length) {
      list.innerHTML = '<div style="color:var(--dim);font-size:10px;padding:8px;text-align:center">No results</div>';
      return;
    }
    songs.forEach(s => {
      const div = document.createElement('div');
      div.className = 'songCard';
      const img = s.image ? `<img class="songImg" src="${s.image}" onerror="this.style.display='none'" alt="">` : '<div class="songImg"></div>';
      const prevHtml = s.preview ? `<button class="prevBtn">PLAY</button>` : '';
      const openHtml = s.url ? `<a class="openBtn" href="${s.url}" target="_blank" rel="noopener" title="Open in iTunes">↗</a>` : '';
      div.innerHTML = `${img}<div class="songInfo"><div class="songTitle">${escapeHtml(s.title)}</div><div class="songArtist">${escapeHtml(s.artist)}</div></div><div class="songBtns">${prevHtml}${openHtml}</div>`;
      list.appendChild(div);
      const pb = div.querySelector('.prevBtn');
      if (pb && s.preview) pb.onclick = () => togglePreview(pb, s.preview);
    });
  } catch(e) {
    list.innerHTML = '<div style="color:var(--red);font-size:10px;padding:8px">Error loading songs</div>';
  }
}

function escapeHtml(s){ return (s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function togglePreview(btn, url){
  if (prevAudio && !prevAudio.paused) {
    prevAudio.pause(); prevAudio = null;
    if (prevBtn) { prevBtn.textContent = 'PLAY'; prevBtn.classList.remove('playing'); }
    if (prevBtn === btn) { prevBtn = null; return; }
  }
  prevAudio = new Audio(url);
  prevAudio.play();
  prevAudio.onended = () => { btn.textContent='PLAY'; btn.classList.remove('playing'); prevBtn=null; };
  btn.textContent = 'STOP';
  btn.classList.add('playing');
  prevBtn = btn;
}

/* ===========================================================
   RADAR
=========================================================== */
(function(){
  const cv = $('radarCanvas');
  const rc = cv.getContext('2d');
  let ra = 0;
  let blips = [];
  function draw(){
    const W=190, H=150, cx=W/2, cy=H/2, r=64;
    rc.clearRect(0,0,W,H);
    rc.strokeStyle = '#0d2a44'; rc.lineWidth = 1;
    for (let i = 1; i <= 4; i++) { rc.beginPath(); rc.arc(cx,cy,r*i/4,0,Math.PI*2); rc.stroke(); }
    rc.beginPath(); rc.moveTo(cx-r,cy); rc.lineTo(cx+r,cy); rc.stroke();
    rc.beginPath(); rc.moveTo(cx,cy-r); rc.lineTo(cx,cy+r); rc.stroke();
    for (let j = 0; j < 60; j++) {
      const a = (ra-j)*Math.PI/180, al = ((60-j)/60)*.5;
      rc.strokeStyle = `rgba(0,212,255,${al})`; rc.lineWidth = 1;
      rc.beginPath(); rc.moveTo(cx,cy); rc.lineTo(cx+r*Math.cos(a),cy+r*Math.sin(a)); rc.stroke();
    }
    const a2 = ra*Math.PI/180;
    rc.strokeStyle = '#00d4ff'; rc.lineWidth = 2;
    rc.beginPath(); rc.moveTo(cx,cy); rc.lineTo(cx+r*Math.cos(a2),cy+r*Math.sin(a2)); rc.stroke();
    const now = Date.now();
    blips = blips.filter(b => now - b.t < 4000);
    if (Math.random() < .04) {
      const th = Math.random()*Math.PI*2, bd = Math.random()*(r-12)+6;
      const cols = ['#00ff88','#00d4ff','#ffc200'];
      blips.push({x:cx+bd*Math.cos(th), y:cy+bd*Math.sin(th), t:now, c:cols[Math.floor(Math.random()*3)]});
    }
    blips.forEach(b => {
      const sz = 2 + (now-b.t)/4000*5;
      rc.strokeStyle = b.c; rc.lineWidth = 1;
      rc.beginPath(); rc.arc(b.x,b.y,sz,0,Math.PI*2); rc.stroke();
      rc.fillStyle = b.c; rc.beginPath(); rc.arc(b.x,b.y,2,0,Math.PI*2); rc.fill();
    });
    rc.fillStyle = '#00d4ff'; rc.beginPath(); rc.arc(cx,cy,3,0,Math.PI*2); rc.fill();
    ra = (ra+2) % 360;
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ===========================================================
   WAVEFORM (real mic data when listening)
=========================================================== */
(function(){
  const cv = $('waveCanvas');
  const wc = cv.getContext('2d');
  function draw(){
    const W = cv.offsetWidth || 600;
    const H = 54;
    cv.width = W; cv.height = H;
    wc.clearRect(0,0,W,H);
    const t = Date.now()/1000;
    const active = listening || speaking;
    wc.strokeStyle = listening ? '#ff2244' : (speaking ? '#00ff88' : '#00d4ff');
    wc.lineWidth = 1.6;
    wc.shadowColor = wc.strokeStyle;
    wc.shadowBlur = 6;

    if (analyser && listening) {
      const data = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(data);
      wc.beginPath();
      const step = data.length / W;
      for (let x = 0; x < W; x++) {
        const v = data[Math.floor(x*step)] / 128 - 1;
        const y = H/2 + v * (H/2 - 4);
        if (x === 0) wc.moveTo(x,y); else wc.lineTo(x,y);
      }
      wc.stroke();
    } else {
      wc.beginPath();
      for (let x = 0; x < W; x += 2) {
        const amp = active ? 16 + Math.random()*8 : 3 + Math.random()*2;
        const y = H/2 + amp * Math.sin((active?6:2)*(x/W)*Math.PI*2 + t*4) + (Math.random()-.5)*(active?2:.5);
        if (x === 0) wc.moveTo(x,y); else wc.lineTo(x,y);
      }
      wc.stroke();
    }
    wc.shadowBlur = 0;
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ===========================================================
   WEATHER + SYSINFO + STATUS
=========================================================== */
function loadWeather(){
  fetch('/api/weather?city=' + encodeURIComponent(settings.city))
    .then(r => r.json())
    .then(d => {
      if (typeof d.temp === 'number') {
        $('wtemp').textContent = d.temp + '°';
        $('wdesc').textContent = (d.description||'').toUpperCase();
        $('whum').textContent  = 'HUM ' + d.humidity + '%';
        $('wfeel').textContent = 'FEELS ' + d.feels_like + '°';
        $('wcity').textContent = d.city || settings.city;
      } else {
        $('wtemp').textContent = '--°';
        $('wdesc').textContent = (d.description||'unavailable').toUpperCase();
      }
    }).catch(() => {});
}

function loadSysInfo(){
  fetch('/api/sysinfo').then(r => r.json()).then(d => {
    if (!d.available) return;
    $('val-cpu').textContent  = d.cpu + '%';
    $('val-mem').textContent  = d.memory + '%';
    $('val-disk').textContent = d.disk + '%';
    $('bar-cpu').style.width  = d.cpu + '%';
    $('bar-mem').style.width  = d.memory + '%';
    $('bar-disk').style.width = d.disk + '%';
  }).catch(() => {});
}

function loadStatus(){
  fetch('/api/status').then(r => r.json()).then(d => {
    const set = (id, ok) => {
      const el = $(id);
      el.textContent = ok ? 'CONNECTED' : 'OFFLINE';
      el.className = ok ? 'ok' : 'bad';
    };
    set('st-anthropic', d.anthropic);
    set('st-elevenlabs', d.elevenlabs);
    set('st-weather', d.weather);
    set('st-search', d.search);
    set('st-itunes', d.itunes);
  }).catch(() => {});
}

setInterval(loadSysInfo, 3000);
setInterval(loadWeather, 5*60*1000);

/* ===========================================================
   SETTINGS MODAL
=========================================================== */
const modal = $('settingsModal');
function openSettings(){
  $('setCity').value = settings.city;
  $('setLang').value = settings.lang;
  $('setAutoSpeak').checked = settings.autoSpeak;
  $('setLiveTranscript').checked = settings.liveTranscript;
  modal.classList.add('on');
}
function closeSettings(){ modal.classList.remove('on'); }
$('settingsBtn').addEventListener('click', openSettings);
$('setCancel').addEventListener('click', closeSettings);
modal.addEventListener('click', e => { if (e.target === modal) closeSettings(); });
$('setSave').addEventListener('click', () => {
  settings.city = $('setCity').value.trim() || 'Bangalore';
  settings.lang = $('setLang').value;
  settings.autoSpeak = $('setAutoSpeak').checked;
  settings.liveTranscript = $('setLiveTranscript').checked;
  localStorage.setItem('jarvis.city', settings.city);
  localStorage.setItem('jarvis.lang', settings.lang);
  localStorage.setItem('jarvis.autoSpeak', settings.autoSpeak ? '1' : '0');
  localStorage.setItem('jarvis.liveTranscript', settings.liveTranscript ? '1' : '0');
  if (recognition) { recognition.lang = settings.lang; recognition.interimResults = settings.liveTranscript; }
  loadWeather();
  closeSettings();
  toast('Settings saved');
});

/* ===========================================================
   BOOT
=========================================================== */
renderEmpty();
loadWeather();
loadSysInfo();
loadStatus();
setTimeout(() => { loadMusic(null, 'pop'); const first = document.querySelector('.gtag'); if (first) first.classList.add('active'); }, 500);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except socket.error:
        local_ip = "127.0.0.1"
    port = int(os.environ.get("PORT", 8080))
    print("\n" + "=" * 56)
    print("  JARVIS — ONLINE")
    print("  Local:    http://127.0.0.1:{}".format(port))
    print("  Network:  http://{}:{}".format(local_ip, port))
    print("  psutil:   {}".format("yes" if HAS_PSUTIL else "no — pip install psutil"))
    print("=" * 56 + "\n")
    app.run(host="0.0.0.0", port=port, threaded=True)
