# -*- coding: utf-8 -*-
"""Minimal, hardened server for PUBLIC hosting -- safe to expose to the internet.

app.py is the local *developer* tool: it has a key-input box, a corrections writer, batch endpoints,
and a single shared API key -- none of which may face the internet (a stranger could spend your key
or wipe it). This file is the opposite: it mounts ONLY the three surfaces a public visitor needs,

    GET  /app                 the doodle demo page (app.html)
    POST /api/fetch_comments  pull Thai comments from a YouTube/Pantip/Reddit link (host-allowlisted)
    POST /api/escalate        the paid tier of the cascade: WangchanBERTa -> gpt-4.1-mini
    GET  /healthz             liveness

and nothing else. The API key is read from the environment ONLY (never from a request), each request
is size-capped and rate-limited, responses carry hardening headers, and the host is safe-by-default.

Run behind a real WSGI server + HTTPS reverse proxy (see HOSTING.md):
    OPENAI_API_KEY=sk-...  TRUST_PROXY=1  gunicorn -w 2 -b 127.0.0.1:8000 serve_public:app
Direct run (dev check only): python serve_public.py   # binds 127.0.0.1:8000 by default
"""
import ipaddress
import os
import sys
import threading
import time

from flask import Flask, jsonify, request

import fetch_social as fs
import predict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP_HTML = os.path.join(ROOT, "app.html")
WCB_DIR = os.path.join(HERE, "wcb_model")

# ---- config (all via env, safe defaults) ----
API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
TRUST_PROXY = os.environ.get("TRUST_PROXY", "").strip() in ("1", "true", "yes")
MAX_TEXT = int(os.environ.get("PUBLIC_MAX_TEXT", "2000"))       # chars per escalate request
FETCH_LIMIT_CAP = int(os.environ.get("PUBLIC_FETCH_LIMIT", "80"))
DAILY_CAP = int(os.environ.get("PUBLIC_DAILY_LIMIT", "2000"))   # escalate calls/day, everyone combined
IP_HOUR_CAP = int(os.environ.get("PUBLIC_IP_HOURLY_LIMIT", "60"))
WCB_NEG = 0.17   # cascade tier 2 cut-off -- MUST match app.py (finding 21). Below this P(sarcastic),
                 # WangchanBERTa answers "not sarcastic" for free; it is never confidently sarcastic.

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024   # reject oversized POST bodies outright

_usage = {"day": "", "day_count": 0, "ip_hour": {}}
_usage_lock = threading.Lock()
_det = None
_wcb = None


# ---------------------------------------------------------------- helpers
def client_ip():
    """the real client IP. Only trust X-Forwarded-For when TRUST_PROXY is set AND your proxy APPENDS
    the socket peer (nginx $proxy_add_x_forwarded_for): then the rightmost entry is the one your proxy
    observed and a client cannot spoof it. Otherwise use the socket peer directly."""
    if TRUST_PROXY:
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[-1].strip()
    return request.remote_addr or "?"


def rate_limited(ip):
    """return an error string if this IP/the whole service is over quota, else None."""
    day = time.strftime("%Y-%m-%d")
    hour = int(time.time() // 3600)
    with _usage_lock:
        if _usage["day"] != day:
            _usage.update(day=day, day_count=0, ip_hour={})
        if _usage["day_count"] + 1 > DAILY_CAP:
            return f"daily quota reached ({DAILY_CAP}/day) -- try again tomorrow"
        k = (ip, hour)
        if _usage["ip_hour"].get(k, 0) + 1 > IP_HOUR_CAP:
            return f"too many requests this hour ({IP_HOUR_CAP}/hr) -- please slow down"
        _usage["day_count"] += 1
        _usage["ip_hour"][k] = _usage["ip_hour"].get(k, 0) + 1
    return None


def has_wcb():
    return os.path.isdir(WCB_DIR) and os.path.exists(os.path.join(WCB_DIR, "config.json"))


def wcb_prob(text):
    """P(sarcastic) from the deployed WangchanBERTa (lazy-loaded). Mirrors app.py.wcb_prob."""
    global _wcb
    if _wcb is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(WCB_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(WCB_DIR)
        mdl.eval()
        _wcb = (tok, mdl, torch)
    tok, mdl, torch = _wcb
    with torch.no_grad():
        enc = tok([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
        return float(torch.softmax(mdl(**enc).logits[0], -1)[1])


def detector():
    global _det
    if _det is None and API_KEY:
        _det = predict.SarcasmDetector(operating="balanced", api_key=API_KEY)
    return _det


def _dec(label):
    return "sarcasm" if str(label) == "1" else "not_sarcasm"


@app.after_request
def _harden(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    # the page is fully inline + Google Fonts; allow those and nothing else external
    resp.headers.setdefault("Content-Security-Policy",
        "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
    return resp


# ---------------------------------------------------------------- routes
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "wcb": has_wcb(), "llm": bool(API_KEY)})


@app.route("/app")
@app.route("/")
def page():
    with open(APP_HTML, encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/fetch_comments", methods=["POST"])
def api_fetch_comments():
    body = request.get_json(silent=True) or {}
    url = str(body.get("url", "")).strip()
    limit = min(int(body.get("limit", 60) or 60), FETCH_LIMIT_CAP)
    if not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error": "ใส่ลิงก์ให้ถูก (ขึ้นต้น http)"}), 400
    try:
        comments, plat = fs.fetch_any(url, limit)      # host-allowlisted + private-IP guarded (fetch_social)
    except fs.UnsupportedError:
        return jsonify({"error": "รองรับเฉพาะ YouTube · Pantip · Reddit"}), 422
    except fs.FetchError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": type(e).__name__}), 502
    if not comments:
        return jsonify({"error": "ไม่พบคอมเมนต์ภาษาไทยจากลิงก์นี้"}), 404
    return jsonify({"comments": comments, "platform": plat})


@app.route("/api/escalate", methods=["POST"])
def api_escalate():
    """the paid tier, reached only for cue-unsure text. Cascade identical to app.py.api_escalate:
    WangchanBERTa clears confident negatives for free -> gpt-4.1-mini decides the rest.
    Degrades gracefully: no model / no key / any error -> {available: false}."""
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    if len(text) > MAX_TEXT:
        return jsonify({"error": f"text too long (max {MAX_TEXT} chars)"}), 413

    # tier 2: WangchanBERTa, one-sided negative filter (free, offline)
    if has_wcb():
        try:
            pw = wcb_prob(text)
            if pw < WCB_NEG:
                return jsonify({"available": True, "label": "0", "decision": "not_sarcasm",
                                "prob": pw, "by": "WangchanBERTa (ฟรี)", "tier": "wcb"})
        except Exception:
            pass

    # tier 3: the paid model -- key comes from the server env ONLY
    det = detector()
    if det is None:
        return jsonify({"available": False, "reason": "no_key"})
    err = rate_limited(client_ip())
    if err:
        return jsonify({"available": False, "reason": "quota", "error": err})
    try:
        p = det.prob(text)
        label = "1" if p >= det.t else "0"
    except Exception as e:
        return jsonify({"available": False, "reason": "error", "error": type(e).__name__})
    return jsonify({"available": True, "label": label, "decision": _dec(label),
                    "prob": p, "by": "gpt-4.1-mini"})


def _startup_banner():
    print("=" * 60)
    print(" Thai Sarcasm Detector -- PUBLIC server (serve_public.py)")
    print(f"  OPENAI_API_KEY : {'present' if API_KEY else 'MISSING -> escalate stays cue/WCB-only'}")
    print(f"  WangchanBERTa  : {'ready' if has_wcb() else 'no wcb_model/ -> tier 2 off (cue -> LLM)'}")
    print(f"  TRUST_PROXY    : {'on (X-Forwarded-For, rightmost)' if TRUST_PROXY else 'off (socket peer)'}")
    print(f"  limits         : {IP_HOUR_CAP}/hr/IP · {DAILY_CAP}/day total · {MAX_TEXT} chars/req")
    print("  exposed routes : GET /app, POST /api/fetch_comments, POST /api/escalate, GET /healthz")
    print("  NOTE: run behind gunicorn/waitress + HTTPS for real traffic (see HOSTING.md).")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="bind address (default 127.0.0.1 -- safe; set 0.0.0.0 only behind a proxy)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    a = ap.parse_args()
    _startup_banner()
    if a.host == "0.0.0.0" and not TRUST_PROXY:
        print("WARNING: binding 0.0.0.0 without TRUST_PROXY -- rate limits will see the proxy IP, "
              "not real clients. Prefer a reverse proxy + TRUST_PROXY=1.", file=sys.stderr)
    app.run(debug=False, host=a.host, port=a.port)
