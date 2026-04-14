#!/usr/bin/env python3
import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests, json, random, base64, socket

app = Flask(__name__)
CORS(app)

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")
OPENWEATHER_KEY     = os.environ.get("OPENWEATHER_KEY", "")
SERPER_API_KEY      = os.environ.get("SERPER_API_KEY", "")

chat_histories = {}

MUSIC_GENRES = ["pop","hip-hop","rock","jazz","electronic","lo-fi","classical","indie","r&b","ambient","metal","reggae","soul","blues","latin","country","house","afrobeats","kpop","dance","punjabi","bollywood"]

def search_itunes(term, limit=8):
    try:
        r = requests.get("https://itunes.apple.com/search",
            params={"term": term, "media": "music", "limit": limit, "entity": "song"},
            timeout=10)
        songs = []
        for t in r.json().get("results", []):
            songs.append({
                "title":   t.get("trackName", ""),
                "artist":  t.get("artistName", ""),
                "album":   t.get("collectionName", ""),
                "image":   t.get("artworkUrl100", "").replace("100x100", "300x300"),
                "url":     t.get("trackViewUrl", ""),
                "preview": t.get("previewUrl", ""),
            })
        return songs
    except:
        return []

def do_search(q):
    try:
        r = requests.post("https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            data=json.dumps({"q": q}), timeout=8)
        return " | ".join(o.get("snippet","") for o in r.json().get("organic",[])[:3])
    except: return ""

def do_weather(city="Bangalore"):
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric".format(city, OPENWEATHER_KEY), timeout=8).json()
        return {"temp": round(r["main"]["temp"]), "feels_like": round(r["main"]["feels_like"]), "humidity": r["main"]["humidity"], "description": r["weather"][0]["description"]}
    except: return {"temp":"--","feels_like":"--","humidity":"--","description":"unavailable"}

@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data     = request.json or {}
    user_msg = data.get("message","").strip()
    sid      = data.get("session_id","default")
    if sid not in chat_histories: chat_histories[sid] = []
    history  = chat_histories[sid]
    ctx      = do_search(user_msg) if len(user_msg) > 5 else ""
    msgs     = history[-12:] + [{"role":"user","content":("Context: {}\n\n".format(ctx) if ctx else "")+"Question: {}".format(user_msg)}]
    api_key  = ANTHROPIC_API_KEY

    def generate():
        full = ""
        try:
            r = requests.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key":api_key,"anthropic-version":"2023-06-01","content-type":"application/json"},
                json={"model":"claude-haiku-4-5-20251001","max_tokens":512,"stream":True,
                      "system":"You are JARVIS, Tony Stark's AI. Be concise, helpful, slightly witty.",
                      "messages":msgs}, stream=True, timeout=30)
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            ev = json.loads(line[6:])
                            if ev.get("type") == "content_block_delta":
                                token = ev["delta"].get("text","")
                                full += token
                                yield "data: {}\n\n".format(json.dumps({"token":token}))
                        except: pass
            history.append({"role":"user","content":user_msg})
            history.append({"role":"assistant","content":full})
            chat_histories[sid] = history[-20:]
            yield "data: {}\n\n".format(json.dumps({"done":True,"full":full}))
        except Exception as e:
            yield "data: {}\n\n".format(json.dumps({"error":str(e)}))

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/tts", methods=["POST"])
def tts():
    text = (request.json or {}).get("text","")
    if not text: return jsonify({"error":"no text"}), 400
    try:
        r = requests.post("https://api.elevenlabs.io/v1/text-to-speech/{}".format(ELEVENLABS_VOICE_ID),
            headers={"xi-api-key":ELEVENLABS_API_KEY,"Content-Type":"application/json","Accept":"audio/mpeg"},
            json={"text":text[:500],"model_id":"eleven_turbo_v2","voice_settings":{"stability":0.5,"similarity_boost":0.75}},
            timeout=15)
        if r.status_code == 200: return jsonify({"audio":base64.b64encode(r.content).decode()})
        return jsonify({"error":"ElevenLabs {}".format(r.status_code)}), 500
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/api/weather")
def weather(): return jsonify(do_weather(request.args.get("city","Bangalore")))

@app.route("/api/music/search")
def music_search():
    query = request.args.get("q","").strip()
    genre = request.args.get("genre","").strip()
    limit = int(request.args.get("limit","8"))
    if query:
        results = search_itunes(query, limit)
    elif genre:
        results = search_itunes(genre + " music", limit)
        if not results:
            results = search_itunes(genre, limit)
    else:
        results = search_itunes("top hits 2024", limit)
    return jsonify(results)

@app.route("/api/music/genres")
def music_genres():
    return jsonify(MUSIC_GENRES)

@app.route("/")
def index(): return HTML

HTML = (
"<!DOCTYPE html>"
"<html lang='en'>"
"<head>"
"<meta charset='UTF-8'>"
"<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
"<title>JARVIS</title>"
"<style>"
":root{--bg:#020912;--panel:#08141f;--card:#0b1a2e;--cyan:#00d4ff;--blue:#0055ff;--gold:#ffc200;--red:#ff2244;--green:#00ff88;--text:#d4eeff;--dim:#3a6080;--border:#0d2a44;}"
"*{box-sizing:border-box;margin:0;padding:0;}"
"body{font-family:'Courier New',Courier,monospace;background:var(--bg);color:var(--text);height:100vh;overflow:hidden;display:flex;flex-direction:column;}"
"body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;background-image:linear-gradient(rgba(0,212,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,.03) 1px,transparent 1px);background-size:40px 40px;animation:gridScroll 20s linear infinite;}"
"@keyframes gridScroll{to{background-position:40px 40px;}}"
"#topbar{position:relative;z-index:10;display:flex;align-items:center;justify-content:space-between;padding:0 24px;height:56px;background:linear-gradient(90deg,var(--panel) 0%,rgba(0,85,255,.08) 50%,var(--panel) 100%);border-bottom:1px solid var(--cyan);box-shadow:0 0 30px rgba(0,212,255,.15);}"
".logo{font-size:22px;font-weight:bold;color:var(--cyan);letter-spacing:4px;text-shadow:0 0 20px rgba(0,212,255,.8);}"
".meta{display:flex;gap:24px;align-items:center;}"
"#clock{font-size:17px;color:var(--gold);}"
".dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 2s infinite;}"
"#stext{color:var(--green);font-size:11px;}"
"@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}"
"#main{position:relative;z-index:1;flex:1;display:flex;gap:10px;padding:10px;overflow:hidden;}"
".panel{background:rgba(11,26,46,.82);border:1px solid var(--border);border-radius:4px;backdrop-filter:blur(4px);display:flex;flex-direction:column;overflow:hidden;}"
".ptitle{font-size:9px;letter-spacing:2px;color:var(--dim);padding:8px 12px 4px;border-bottom:1px solid var(--border);}"
"#left{width:210px;display:flex;flex-direction:column;gap:8px;}"
".sp{background:rgba(11,26,46,.88);border:1px solid var(--border);border-radius:4px;padding:10px 12px;}"
".sp h3{font-size:9px;letter-spacing:2px;color:var(--dim);margin-bottom:8px;border-bottom:1px solid var(--border);padding-bottom:4px;}"
"#radarCanvas{display:block;margin:0 auto;}"
".sr{display:flex;align-items:center;gap:6px;font-size:9px;margin-bottom:5px;}"
".sl{color:var(--dim);width:72px;flex-shrink:0;}"
".sb{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden;}"
".sf{height:100%;border-radius:2px;}"
".sp2{color:var(--cyan);font-size:9px;width:28px;text-align:right;}"
".qb{display:flex;width:100%;background:var(--panel);border:1px solid var(--border);color:var(--dim);font-family:'Courier New',monospace;font-size:10px;padding:5px 8px;cursor:pointer;border-radius:2px;margin-bottom:3px;transition:all .15s;text-decoration:none;}"
".qb:hover{border-color:var(--cyan);color:var(--cyan);}"
"#center{flex:1;}"
"#chatbox{flex:1;overflow-y:auto;padding:14px;scroll-behavior:smooth;}"
"#chatbox::-webkit-scrollbar{width:4px;}"
"#chatbox::-webkit-scrollbar-thumb{background:var(--border);}"
".msg{margin-bottom:16px;display:flex;gap:10px;animation:fi .3s ease;}"
"@keyframes fi{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}"
".av{width:28px;height:28px;border-radius:2px;display:flex;align-items:center;justify-content:center;font-size:11px;flex-shrink:0;margin-top:2px;}"
".msg.j .av{background:rgba(0,212,255,.15);border:1px solid var(--cyan);color:var(--cyan);}"
".msg.u .av{background:rgba(255,194,0,.15);border:1px solid var(--gold);color:var(--gold);}"
".msg.s .av{background:rgba(0,255,136,.1);border:1px solid var(--green);color:var(--green);}"
".mb{flex:1;min-width:0;}"
".mm{font-size:9px;color:var(--dim);margin-bottom:4px;}"
".spk{font-size:10px;margin-right:6px;}"
".msg.j .spk{color:var(--cyan);}.msg.u .spk{color:var(--gold);}.msg.s .spk{color:var(--green);}"
".mt{font-size:12px;line-height:1.6;color:rgba(212,238,255,.9);word-wrap:break-word;}"
".tc{display:inline-block;width:8px;height:14px;background:var(--cyan);animation:bl .7s step-end infinite;vertical-align:text-bottom;}"
"@keyframes bl{0%,100%{opacity:1;}50%{opacity:0;}}"
"#waveCanvas{width:100%;height:50px;display:block;}"
"#irow{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border);background:var(--panel);}"
"#minput{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:2px;color:var(--text);font-family:'Courier New',monospace;font-size:13px;padding:8px 12px;outline:none;transition:border-color .15s;min-width:0;}"
"#minput:focus{border-color:var(--cyan);}"
"#minput::placeholder{color:var(--dim);}"
".btn{font-family:'Courier New',monospace;font-size:11px;padding:8px 14px;border:none;border-radius:2px;cursor:pointer;transition:all .15s;letter-spacing:1px;white-space:nowrap;}"
"#sbtn{background:var(--cyan);color:var(--bg);font-weight:bold;}"
"#sbtn:hover{background:var(--blue);color:#fff;}"
"#sbtn:disabled{opacity:0.5;cursor:not-allowed;}"
"#mbtn{background:var(--card);color:var(--cyan);border:1px solid var(--cyan);}"
"#mbtn:hover{background:rgba(0,212,255,.1);}"
"#mbtn.active{background:var(--red);color:#fff;border-color:var(--red);}"
"#stpbtn{background:var(--card);color:var(--red);border:1px solid var(--red);}"
"#stpbtn:hover{background:rgba(255,34,68,.1);}"
"#right{width:280px;display:flex;flex-direction:column;gap:8px;overflow-y:auto;}"
".wc,.apic{background:rgba(11,26,46,.88);border:1px solid var(--border);border-radius:4px;padding:12px;}"
".wc h3,.apic h3{font-size:9px;letter-spacing:2px;color:var(--dim);margin-bottom:8px;}"
".wtemp{font-size:26px;color:var(--gold);}"
".wdesc{color:var(--dim);font-size:10px;text-transform:uppercase;letter-spacing:1px;}"
".wmeta{display:flex;gap:12px;margin-top:6px;font-size:9px;color:var(--dim);}"
".ar{display:flex;justify-content:space-between;font-size:9px;margin-bottom:3px;}"
".ok{color:var(--green);}"
"#vbadge{display:none;position:fixed;top:68px;left:50%;transform:translateX(-50%);background:rgba(255,34,68,.9);color:#fff;padding:6px 20px;border-radius:20px;font-size:11px;letter-spacing:2px;z-index:100;}"
"#vbadge.on{display:block;}"
"#toast{position:fixed;bottom:80px;left:50%;transform:translateX(-50%);background:rgba(0,212,255,.9);color:var(--bg);padding:8px 20px;border-radius:20px;font-size:11px;opacity:0;transition:opacity .3s;pointer-events:none;z-index:200;}"
"#toast.show{opacity:1;}"
"#musicSection{background:rgba(11,26,46,.88);border:1px solid var(--border);border-radius:4px;padding:12px;flex:1;display:flex;flex-direction:column;overflow:hidden;min-height:0;}"
"#musicSection h3{font-size:9px;letter-spacing:2px;color:var(--dim);margin-bottom:8px;}"
"#musicSearch{display:flex;gap:6px;margin-bottom:8px;}"
"#musicInput{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:2px;color:var(--text);font-family:'Courier New',monospace;font-size:10px;padding:5px 8px;outline:none;}"
"#musicInput:focus{border-color:var(--green);}"
"#musicInput::placeholder{color:var(--dim);}"
"#musicSearchBtn{background:none;border:1px solid var(--green);color:var(--green);font-family:'Courier New',monospace;font-size:10px;padding:4px 10px;cursor:pointer;border-radius:2px;}"
"#musicSearchBtn:hover{background:rgba(0,255,136,.1);}"
"#genreRow{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px;}"
".gtag{background:var(--panel);border:1px solid var(--border);color:var(--dim);font-family:'Courier New',monospace;font-size:9px;padding:3px 7px;cursor:pointer;border-radius:2px;transition:all .15s;white-space:nowrap;}"
".gtag:hover,.gtag.active{border-color:var(--green);color:var(--green);background:rgba(0,255,136,.05);}"
"#songList{overflow-y:auto;flex:1;}"
"#songList::-webkit-scrollbar{width:3px;}"
"#songList::-webkit-scrollbar-thumb{background:var(--border);}"
".songCard{display:flex;align-items:center;gap:8px;padding:6px 4px;border-bottom:1px solid rgba(13,42,68,.5);}"
".songImg{width:38px;height:38px;border-radius:3px;object-fit:cover;background:var(--border);flex-shrink:0;}"
".songInfo{flex:1;overflow:hidden;}"
".songTitle{font-size:10px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}"
".songArtist{font-size:9px;color:var(--dim);}"
".songBtns{display:flex;gap:4px;flex-shrink:0;}"
".prevBtn{background:none;border:1px solid var(--cyan);color:var(--cyan);font-family:'Courier New',monospace;font-size:9px;padding:2px 6px;cursor:pointer;border-radius:2px;}"
".prevBtn:hover{background:rgba(0,212,255,.1);}"
".prevBtn.playing{border-color:var(--red);color:var(--red);}"
".openBtn{color:var(--green);font-size:13px;text-decoration:none;}"
"@media(max-width:800px){#left,#right{display:none;}}"
"</style>"
"</head>"
"<body>"
"<div id='topbar'>"
"<div class='logo'>JARVIS</div>"
"<div class='meta'><div class='dot'></div><span id='stext'>SYSTEM ONLINE</span><span id='clock'>--:--:--</span></div>"
"</div>"
"<div id='vbadge'>LISTENING...</div>"
"<div id='toast'></div>"
"<div id='main'>"
"<div id='left'>"
"<div class='sp'><h3>NEURAL RADAR</h3><canvas id='radarCanvas' width='186' height='145'></canvas></div>"
"<div class='sp'><h3>SYSTEM STATUS</h3>"
"<div class='sr'><span class='sl'>CPU LOAD</span><div class='sb'><div class='sf' style='width:62%;background:var(--cyan)'></div></div><span class='sp2'>62%</span></div>"
"<div class='sr'><span class='sl'>MEMORY</span><div class='sb'><div class='sf' style='width:41%;background:var(--blue)'></div></div><span class='sp2'>41%</span></div>"
"<div class='sr'><span class='sl'>NEURAL NET</span><div class='sb'><div class='sf' style='width:93%;background:var(--green)'></div></div><span class='sp2'>93%</span></div>"
"<div class='sr'><span class='sl'>VOICE SYNC</span><div class='sb'><div class='sf' style='width:80%;background:var(--gold)'></div></div><span class='sp2'>80%</span></div>"
"</div>"
"<div class='sp'><h3>QUICK LAUNCH</h3>"
"<a class='qb' href='https://youtube.com' target='_blank'>YouTube</a>"
"<a class='qb' href='https://github.com' target='_blank'>GitHub</a>"
"<a class='qb' href='https://mail.google.com' target='_blank'>Gmail</a>"
"<a class='qb' href='https://music.apple.com' target='_blank'>Apple Music</a>"
"<a class='qb' href='https://open.spotify.com' target='_blank'>Spotify</a>"
"</div>"
"</div>"
"<div class='panel' id='center'>"
"<div class='ptitle'>COMMUNICATION LINK</div>"
"<div id='chatbox'></div>"
"<div style='border-top:1px solid var(--border);padding:6px 12px;background:var(--panel)'><canvas id='waveCanvas'></canvas></div>"
"<div id='irow'>"
"<button class='btn' id='mbtn'>MIC</button>"
"<input id='minput' type='text' placeholder='Type a command or ask anything...' autocomplete='off'>"
"<button class='btn' id='stpbtn'>STOP</button>"
"<button class='btn' id='sbtn'>SEND</button>"
"</div>"
"</div>"
"<div id='right'>"
"<div class='wc'><h3>WEATHER INTEL</h3>"
"<div class='wtemp' id='wtemp'>--C</div>"
"<div class='wdesc' id='wdesc'>Loading...</div>"
"<div class='wmeta'><span id='whum'>--</span><span id='wfeel'>--</span></div>"
"</div>"
"<div id='musicSection'>"
"<h3>MUSIC DISCOVER</h3>"
"<div id='musicSearch'>"
"<input id='musicInput' type='text' placeholder='Search any song, artist...'>"
"<button id='musicSearchBtn'>Search</button>"
"</div>"
"<div id='genreRow'></div>"
"<div id='songList'><div style='color:var(--dim);font-size:10px;padding:8px;text-align:center;'>Select a genre or search above</div></div>"
"</div>"
"<div class='apic'><h3>API STATUS</h3>"
"<div class='ar'><span>Anthropic</span><span class='ok'>CONNECTED</span></div>"
"<div class='ar'><span>ElevenLabs</span><span class='ok'>CONNECTED</span></div>"
"<div class='ar'><span>iTunes</span><span class='ok'>CONNECTED</span></div>"
"<div class='ar'><span>Weather</span><span class='ok'>CONNECTED</span></div>"
"</div>"
"</div>"
"</div>"
"<script>"
"var SID='sid_'+Math.random().toString(36).slice(2);"
"var listening=false,speaking=false,recognition=null,curAudio=null,prevAudio=null,prevBtn=null;"
"setInterval(function(){document.getElementById('clock').textContent=new Date().toTimeString().slice(0,8);},1000);"
"function toast(m,d){var el=document.getElementById('toast');el.textContent=m;el.classList.add('show');setTimeout(function(){el.classList.remove('show');},d||2500);}"
"function setStatus(t,c){var el=document.getElementById('stext');el.textContent=t;el.style.color=c||'var(--green)';}"
"function appendMsg(speaker,text,type){"
"type=type||'j';"
"var box=document.getElementById('chatbox'),d=document.createElement('div');"
"d.className='msg '+type;"
"var av=type==='u'?'YOU':type==='s'?'SYS':'J';"
"var now=new Date().toTimeString().slice(0,5);"
"d.innerHTML='<div class=\"av\">'+av+'</div><div class=\"mb\"><div class=\"mm\"><span class=\"spk\">'+speaker+'</span>'+now+'</div><div class=\"mt\"></div></div>';"
"box.appendChild(d);"
"d.querySelector('.mt').textContent=text;"
"box.scrollTop=box.scrollHeight;"
"}"
"function streamMsg(speaker){"
"var box=document.getElementById('chatbox'),d=document.createElement('div');"
"d.className='msg j';"
"var now=new Date().toTimeString().slice(0,5);"
"d.innerHTML='<div class=\"av\">J</div><div class=\"mb\"><div class=\"mm\"><span class=\"spk\">'+speaker+'</span>'+now+'</div><div class=\"mt\"><span class=\"tc\"></span></div></div>';"
"box.appendChild(d);"
"var mt=d.querySelector('.mt'),cur=mt.querySelector('.tc');"
"box.scrollTop=box.scrollHeight;"
"return{add:function(t){cur.remove();mt.textContent+=t;mt.appendChild(cur);box.scrollTop=box.scrollHeight;},done:function(f){cur.remove();mt.textContent=f;}};"
"}"
"async function send(text,speakReply){"
"if(!text.trim())return;"
"setStatus('QUERYING AI...','var(--gold)');"
"document.getElementById('sbtn').disabled=true;"
"var s=streamMsg('JARVIS'),full='';"
"try{"
"var resp=await fetch('/api/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,session_id:SID})});"
"var reader=resp.body.getReader(),dec=new TextDecoder();"
"while(true){"
"var res=await reader.read();if(res.done)break;"
"var lines=dec.decode(res.value).split('\\n');"
"for(var i=0;i<lines.length;i++){"
"var line=lines[i];if(!line.startsWith('data: '))continue;"
"try{var ev=JSON.parse(line.slice(6));if(ev.token){full+=ev.token;s.add(ev.token);}if(ev.done){s.done(ev.full||full);if(speakReply&&ev.full)speakText(ev.full);}if(ev.error)s.done('Error: '+ev.error);}catch(e){}"
"}"
"}"
"}catch(e){s.done('Connection error: '+e.message);}"
"setStatus('SYSTEM ONLINE','var(--green)');"
"document.getElementById('sbtn').disabled=false;"
"}"
"document.getElementById('sbtn').addEventListener('click',function(){var i=document.getElementById('minput'),t=i.value.trim();i.value='';if(t){appendMsg('YOU',t,'u');send(t,false);}});"
"document.getElementById('minput').addEventListener('keydown',function(e){if(e.key==='Enter'){var t=this.value.trim();this.value='';if(t){appendMsg('YOU',t,'u');send(t,false);}}});"
"function stopSpeaking(){if(curAudio){curAudio.pause();curAudio.currentTime=0;curAudio=null;}speaking=false;setStatus('SYSTEM ONLINE','var(--green)');}"
"async function speakText(text){"
"stopSpeaking();speaking=true;setStatus('SPEAKING...','var(--cyan)');"
"try{"
"var r=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.slice(0,400)})});"
"var d=await r.json();"
"if(d.audio){var bytes=atob(d.audio),buf=new Uint8Array(bytes.length);for(var i=0;i<bytes.length;i++)buf[i]=bytes.charCodeAt(i);var url=URL.createObjectURL(new Blob([buf],{type:'audio/mpeg'}));curAudio=new Audio(url);curAudio.onended=function(){speaking=false;setStatus('SYSTEM ONLINE','var(--green)');};curAudio.play();}else{speaking=false;setStatus('SYSTEM ONLINE','var(--green)');}"
"}catch(e){speaking=false;setStatus('SYSTEM ONLINE','var(--green)');}"
"}"
"document.getElementById('stpbtn').addEventListener('click',function(){stopSpeaking();if(prevAudio){prevAudio.pause();prevAudio=null;}if(prevBtn){prevBtn.textContent='Preview';prevBtn.classList.remove('playing');prevBtn=null;}if(recognition&&listening)recognition.stop();toast('Stopped');});"
"var mbtn=document.getElementById('mbtn'),vbadge=document.getElementById('vbadge');"
"function setupRec(){"
"var SR=window.SpeechRecognition||window.webkitSpeechRecognition;"
"if(!SR){toast('Use Chrome for voice');return null;}"
"var r=new SR();r.lang='en-IN';r.continuous=false;r.interimResults=false;r.maxAlternatives=1;"
"r.onstart=function(){listening=true;mbtn.textContent='STOP';mbtn.classList.add('active');vbadge.classList.add('on');setStatus('LISTENING...','var(--red)');};"
"r.onend=function(){listening=false;mbtn.textContent='MIC';mbtn.classList.remove('active');vbadge.classList.remove('on');setStatus('SYSTEM ONLINE','var(--green)');};"
"r.onerror=function(e){listening=false;mbtn.textContent='MIC';mbtn.classList.remove('active');vbadge.classList.remove('on');setStatus('SYSTEM ONLINE','var(--green)');if(e.error!=='no-speech')toast('Voice error: '+e.error);};"
"r.onresult=function(e){var t=e.results[0][0].transcript;appendMsg('YOU (voice)',t,'u');send(t,true);};"
"return r;"
"}"
"mbtn.addEventListener('click',function(){if(listening){if(recognition)recognition.stop();return;}if(!recognition)recognition=setupRec();if(!recognition)return;try{recognition.start();}catch(e){toast('Mic error: '+e.message);}});"
"document.addEventListener('keydown',function(e){if(e.code==='Space'&&document.activeElement!==document.getElementById('minput')&&document.activeElement!==document.getElementById('musicInput')){e.preventDefault();mbtn.click();}});"
"var genres=['pop','hip-hop','rock','jazz','electronic','lo-fi','classical','r&b','ambient','metal','reggae','soul','blues','latin','house','afrobeats','kpop','bollywood','punjabi','dance'];"
"(function(){"
"var row=document.getElementById('genreRow');"
"genres.forEach(function(g){"
"var btn=document.createElement('button');btn.className='gtag';btn.textContent=g;"
"btn.onclick=function(){document.querySelectorAll('.gtag').forEach(function(b){b.classList.remove('active');});btn.classList.add('active');loadMusic(null,g);};"
"row.appendChild(btn);"
"});"
"})();"
"document.getElementById('musicSearchBtn').addEventListener('click',function(){var q=document.getElementById('musicInput').value.trim();if(q){document.querySelectorAll('.gtag').forEach(function(b){b.classList.remove('active');});loadMusic(q,null);}});"
"document.getElementById('musicInput').addEventListener('keydown',function(e){if(e.key==='Enter'){var q=this.value.trim();if(q){document.querySelectorAll('.gtag').forEach(function(b){b.classList.remove('active');});loadMusic(q,null);}}});"
"async function loadMusic(query,genre){"
"var list=document.getElementById('songList');"
"list.innerHTML='<div style=\"color:var(--dim);font-size:10px;padding:8px;text-align:center;\">Searching...</div>';"
"var url='/api/music/search?limit=10';"
"if(query)url+='&q='+encodeURIComponent(query);"
"if(genre)url+='&genre='+encodeURIComponent(genre);"
"try{"
"var r=await fetch(url),songs=await r.json();"
"list.innerHTML='';"
"if(!songs||!songs.length){list.innerHTML='<div style=\"color:var(--dim);font-size:10px;padding:8px;text-align:center;\">No results found</div>';return;}"
"songs.forEach(function(s){"
"var div=document.createElement('div');div.className='songCard';"
"var img=s.image?'<img class=\"songImg\" src=\"'+s.image+'\" onerror=\"this.style.display=\\'none\\'\" alt=\"\">':'<div class=\"songImg\"></div>';"
"var prevHtml=s.preview?'<button class=\"prevBtn\" onclick=\"togglePreview(this,\\''+s.preview+'\\')\" >Play</button>':'';"
"var openHtml=s.url?'<a class=\"openBtn\" href=\"'+s.url+'\" target=\"_blank\" title=\"Open\">&#9654;</a>':'';"
"div.innerHTML=img+'<div class=\"songInfo\"><div class=\"songTitle\">'+s.title+'</div><div class=\"songArtist\">'+s.artist+'</div></div><div class=\"songBtns\">'+prevHtml+openHtml+'</div>';"
"list.appendChild(div);"
"});"
"}catch(e){list.innerHTML='<div style=\"color:var(--red);font-size:10px;padding:8px;\">Error loading songs</div>';}"
"}"
"function togglePreview(btn,url){"
"if(prevAudio&&!prevAudio.paused){"
"prevAudio.pause();prevAudio=null;"
"if(prevBtn){prevBtn.textContent='Play';prevBtn.classList.remove('playing');}"
"if(prevBtn===btn){prevBtn=null;return;}"
"}"
"prevAudio=new Audio(url);prevAudio.play();"
"prevAudio.onended=function(){btn.textContent='Play';btn.classList.remove('playing');prevBtn=null;};"
"btn.textContent='Stop';btn.classList.add('playing');"
"prevBtn=btn;"
"}"
"(function(){"
"var rc=document.getElementById('radarCanvas').getContext('2d'),ra=0,blips=[];"
"function draw(){var W=186,H=145,cx=W/2,cy=H/2,r=62;rc.clearRect(0,0,W,H);rc.strokeStyle='#0d2a44';rc.lineWidth=1;for(var i=1;i<=4;i++){var rr=r*i/4;rc.beginPath();rc.arc(cx,cy,rr,0,Math.PI*2);rc.stroke();}rc.beginPath();rc.moveTo(cx-r,cy);rc.lineTo(cx+r,cy);rc.stroke();rc.beginPath();rc.moveTo(cx,cy-r);rc.lineTo(cx,cy+r);rc.stroke();for(var j=0;j<60;j++){var a=(ra-j)*Math.PI/180,al=((60-j)/60)*.5;rc.strokeStyle='rgba(0,212,255,'+al+')';rc.lineWidth=1;rc.beginPath();rc.moveTo(cx,cy);rc.lineTo(cx+r*Math.cos(a),cy+r*Math.sin(a));rc.stroke();}var a2=ra*Math.PI/180;rc.strokeStyle='#00d4ff';rc.lineWidth=2;rc.beginPath();rc.moveTo(cx,cy);rc.lineTo(cx+r*Math.cos(a2),cy+r*Math.sin(a2));rc.stroke();var now=Date.now();blips=blips.filter(function(b){return now-b.t<4000;});if(Math.random()<.04){var th=Math.random()*Math.PI*2,bd=Math.random()*(r-12)+6,cols=['#00ff88','#00d4ff','#ffc200'];blips.push({x:cx+bd*Math.cos(th),y:cy+bd*Math.sin(th),t:now,c:cols[Math.floor(Math.random()*3)]});}blips.forEach(function(b){var sz=2+(now-b.t)/4000*5;rc.strokeStyle=b.c;rc.lineWidth=1;rc.beginPath();rc.arc(b.x,b.y,sz,0,Math.PI*2);rc.stroke();rc.fillStyle=b.c;rc.beginPath();rc.arc(b.x,b.y,2,0,Math.PI*2);rc.fill();});rc.fillStyle='#00d4ff';rc.beginPath();rc.arc(cx,cy,3,0,Math.PI*2);rc.fill();ra=(ra+2)%360;requestAnimationFrame(draw);}draw();"
"})();"
"(function(){var cv=document.getElementById('waveCanvas'),wc=cv.getContext('2d');function draw(){var W=cv.offsetWidth||600,H=50;cv.width=W;cv.height=H;wc.clearRect(0,0,W,H);var t=Date.now()/1000,active=listening||speaking;wc.strokeStyle=listening?'#ff2244':(speaking?'#00ff88':'#00d4ff');wc.lineWidth=1.5;wc.beginPath();for(var x=0;x<W;x+=2){var amp=active?(16+Math.random()*8):(3+Math.random()*2),y=H/2+amp*Math.sin((active?6:2)*(x/W)*Math.PI*2+t*4)+(Math.random()-.5)*(active?2:.5);if(x===0)wc.moveTo(x,y);else wc.lineTo(x,y);}wc.stroke();requestAnimationFrame(draw);}draw();})();"
"fetch('/api/weather?city=Bangalore').then(function(r){return r.json();}).then(function(d){if(d.temp){document.getElementById('wtemp').textContent=d.temp+'C';document.getElementById('wdesc').textContent=d.description.toUpperCase();document.getElementById('whum').textContent='Humidity: '+d.humidity+'%';document.getElementById('wfeel').textContent='Feels: '+d.feels_like+'C';}}).catch(function(){});"
"[['SYSTEM','Initializing JARVIS...','s'],['SYSTEM','Music search powered by iTunes. Search anything!','s'],['JARVIS','Good day. Type below or click MIC to speak.','j']].forEach(function(b,i){setTimeout(function(){appendMsg(b[0],b[1],b[2]);},i*500);});"
"setTimeout(function(){loadMusic(null,'pop');document.querySelector('.gtag').classList.add('active');},1000);"
"</script>"
"</body>"
"</html>"
)

if __name__ == "__main__":
    try:    local_ip = socket.gethostbyname(socket.gethostname())
    except: local_ip = "127.0.0.1"
    print("\n" + "="*50)
    print("  JARVIS - ONLINE")
    print("  Local:   http://localhost:8080")
    print("  Network: http://{}:8080".format(local_ip))
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)