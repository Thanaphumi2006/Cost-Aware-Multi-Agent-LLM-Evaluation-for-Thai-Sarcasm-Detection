# -*- coding: utf-8 -*-
"""Experiment web app + comparison of 3 Thai sarcasm detection systems

Run:
  set OPENAI_API_KEY=sk-...            (PowerShell: $env:OPENAI_API_KEY="sk-...")
  C:/Users/thana/pt/Scripts/python.exe app.py
  open http://127.0.0.1:5000

Notes:
- the API key can be set 2 ways: an environment variable or typed on the page (the field at the top)
  both keep it in the process RAM only -- not written to a file, not embedded in code, gone when the server stops
- this app binds to 127.0.0.1; don't expose it to the internet, or the key field becomes a way for others to hit the API with your key
- without a key the app still works, but only WangchanBERTa runs (the one that needs no API)
- the WangchanBERTa the app uses was trained on all of gold -> don't evaluate it on gold
  if you type text that is in gold it will "remember the answer" (the app warns you)
"""
import os
import sys
import threading
import time

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

import envload  # noqa: F401  -- loads OPENAI_API_KEY from .env on import (so escalation works with no manual export)
import baseline
import multiagent
import multiagent_debate
import multiagent_hybrid
from baseline import PRICE_PER_MTOK, metrics

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
WCB_DIR = os.path.join(HERE, "wcb_model")
WCB_NEG = 0.17    # cascade tier 2: below this P(sarcastic), WangchanBERTa answers "not sarcastic" for free.
                  # derived from the out-of-fold probs -- every gold item it answers under this cut is a true 0.
                  # there is deliberately no upper cut: WCB is never confidently *sarcastic* (cascade_eval.py)
IN_P, OUT_P = PRICE_PER_MTOK["gpt"]

app = Flask(__name__)

# ---------- load heavy things once at startup ----------
_gold = pd.read_csv(os.path.join(HERE, "gold.csv"), dtype=str).fillna("")
_gold["label"] = _gold["label"].str.strip()
_gold = _gold[_gold["label"].isin(["0", "1"])].reset_index(drop=True)
GOLD_TEXTS = dict(zip(_gold["text"], _gold["label"]))

_wcb = None       # lazy load (torch is heavy)
_client = None
_api_key = os.environ.get("OPENAI_API_KEY", "").strip()   # in RAM only, never written to disk


def wcb():
    global _wcb
    if _wcb is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(WCB_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(WCB_DIR)
        mdl.eval()
        _wcb = (tok, mdl, torch)
    return _wcb


def client():
    global _client
    if _client is None and _api_key:
        from openai import OpenAI
        _client = OpenAI(api_key=_api_key)
    return _client


def mask_key(k):
    """sk-proj-abcd...wxyz -- enough to tell which key it is without revealing it"""
    return f"{k[:6]}…{k[-4:]}" if len(k) > 14 else "sk-…"


def has_wcb():
    return os.path.isdir(WCB_DIR) and os.path.exists(os.path.join(WCB_DIR, "config.json"))


def cost(i, o):
    return i / 1e6 * IN_P + o / 1e6 * OUT_P


# ---------- real results table on gold (computed from actual CSVs, not hardcoded numbers) ----------
def gold_table():
    files = [
        ("① เอเจนต์เดี่ยว (base)", "baseline_preds_gpt.csv", 127, 0.094, 751),
        ("② Multi-agent (base + ผู้ตรวจ)", "multiagent_preds_gpt_conservative.csv", 183, 0.169, 967),
        ("WangchanBERTa (5-fold CV)", "wangchanberta_preds.csv", 0, 0.0, None),
    ]
    rows = []
    for name, f, calls, c, lat in files:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, dtype=str).fillna("")
        d = d[d["pred"].isin(["0", "1"])]
        acc, prec, rec, f1, (tn, fp, fn, tp) = metrics(d["label"].tolist(), d["pred"].tolist())
        rows.append(dict(name=name, f1=round(f1, 3), prec=round(prec, 3), rec=round(rec, 3),
                         tp=tp, fp=fp, fn=fn, calls=calls, cost=c, lat=lat, n=len(d)))
    return rows


GOLD_ROWS = gold_table()


# ---------- predict ----------
def run_baseline(text):
    c = client()
    if not c:
        return {"pred": "n/a", "note": "ไม่มี OPENAI_API_KEY"}
    r = baseline.predict_one(c, text)
    return {"pred": r["pred"], "latency_ms": r["latency_ms"], "calls": 1,
            "in_tok": r["in_tok"], "out_tok": r["out_tok"],
            "cost": round(cost(r["in_tok"], r["out_tok"]), 6), "err": r.get("err", "")}


def run_multiagent(text):
    """run each stage manually (instead of calling run_pipeline in one shot) to collect token/time/decision "per stage"
    logic must match multiagent.run_pipeline exactly -- uses the same prompt/schema straight from that file"""
    c = client()
    if not c:
        return {"pred": "n/a", "note": "ไม่มี OPENAI_API_KEY"}

    steps = []
    t0 = time.perf_counter()
    det_sys = multiagent.DETECT_SYS + _corrections_block(text)   # teach multi-agent with human corrections (few-shot)
    det, i1, o1 = multiagent._ask(c, det_sys, multiagent.DETECT_SCHEMA, "label", text)
    d_ms = round((time.perf_counter() - t0) * 1000)
    steps.append({
        "role": "ด่าน 1 · พนักงานคัดกรอง (detector)",
        "job": "อ่านข้อความดิบ แล้วชี้ว่า “น่าจะประชด” หรือไม่ เหวี่ยงแหกว้างไว้ก่อน",
        "said": det, "say_txt": {"1": "ประชด", "0": "ไม่ประชด"}.get(det, "ตอบเพี้ยน"),
        "ms": d_ms, "in_tok": i1, "out_tok": o1, "cost": round(cost(i1, o1), 6), "ran": True,
    })

    if det == "1":
        t1 = time.perf_counter()
        ver, i2, o2 = multiagent._ask(c, multiagent.VERIFY_SYS, multiagent.VERIFY_SCHEMA, "verdict", text)
        v_ms = round((time.perf_counter() - t1) * 1000)
        steps.append({
            "role": "ด่าน 2 · หัวหน้า QC (verifier)",
            "job": "ตรวจเฉพาะข้อที่ด่าน 1 ชี้ว่าประชด มีอำนาจ “ปัดตก” อย่างเดียว เพิ่มประชดใหม่ไม่ได้",
            "said": ver, "say_txt": {"1": "ยืนยัน: คงเป็นประชด", "0": "ปัดตก: ไม่ใช่ประชด"}.get(ver, "ตอบเพี้ยน"),
            "ms": v_ms, "in_tok": i2, "out_tok": o2, "cost": round(cost(i2, o2), 6), "ran": True,
        })
        final = ver if ver in ("0", "1") else "err"
        flipped = (det == "1" and ver == "0")
    else:
        steps.append({
            "role": "ด่าน 2 · หัวหน้า QC (verifier)",
            "job": "ตรวจเฉพาะข้อที่ด่าน 1 ชี้ว่าประชด มีอำนาจ “ปัดตก” อย่างเดียว เพิ่มประชดใหม่ไม่ได้",
            "said": "", "say_txt": "ไม่ได้ถูกเรียก (ด่าน 1 ว่าไม่ประชด → ปล่อยผ่านเลย)",
            "ms": 0, "in_tok": 0, "out_tok": 0, "cost": 0.0, "ran": False,
        })
        final = "0" if det == "0" else "err"
        flipped = False
        i2 = o2 = 0

    tot_i, tot_o = i1 + i2, o1 + o2
    return {"pred": final, "latency_ms": round((time.perf_counter() - t0) * 1000),
            "calls": 1 + (1 if det == "1" else 0),
            "in_tok": tot_i, "out_tok": tot_o, "cost": round(cost(tot_i, tot_o), 6),
            "detect": det, "verdict": steps[1]["said"], "flipped": flipped,
            "steps": steps, "err": ""}


def run_debate(text):
    """architecture 2: prosecutor + defender -> judge (can re-decide both ways)"""
    c = client()
    if not c:
        return {"pred": "n/a", "note": "ไม่มี OPENAI_API_KEY"}
    r = multiagent_debate.run_debate(c, text)
    return {"pred": r["pred"], "latency_ms": r["latency_ms"], "calls": r["calls"],
            "in_tok": r["in_tok"], "out_tok": r["out_tok"],
            "cost": round(cost(r["in_tok"], r["out_tok"]), 6),
            "pros": r["pros"], "defe": r["defe"], "judge": r["judge"], "err": r["err"]}


def run_hybrid(text):
    """combined system: detector -> (prosecutor vs defender -> a reject-only judge)"""
    c = client()
    if not c:
        return {"pred": "n/a", "note": "ไม่มี OPENAI_API_KEY"}
    r = multiagent_hybrid.run_hybrid(c, text)
    skipped = (r["detect"] == "0")
    return {"pred": r["pred"], "latency_ms": r["latency_ms"], "calls": r["calls"],
            "in_tok": r["in_tok"], "out_tok": r["out_tok"],
            "cost": round(cost(r["in_tok"], r["out_tok"]), 6),
            "detect": r["detect"], "pros": r["pros"], "defe": r["defe"],
            "judge": r["judge"], "skipped": skipped,
            "flipped": (r["detect"] == "1" and r["pred"] == "0"), "err": r["err"]}


def wcb_prob(text):
    """P(sarcastic) from the deployed WangchanBERTa -- the middle tier of the cascade.
    run_wcb() returns only the winning class's confidence; the router needs the sarcastic side."""
    tok, mdl, torch = wcb()
    with torch.no_grad():
        enc = tok([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
        return float(torch.softmax(mdl(**enc).logits[0], -1)[1])


def run_wcb(text):
    if not has_wcb():
        return {"pred": "n/a", "note": "ยังไม่ได้เทรน -- รัน train_final_wcb.py"}
    tok, mdl, torch = wcb()
    t0 = time.perf_counter()
    with torch.no_grad():
        enc = tok([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
        logits = mdl(**enc).logits[0]
        prob = torch.softmax(logits, -1).tolist()
        pred = int(logits.argmax())
    return {"pred": str(pred), "latency_ms": round((time.perf_counter() - t0) * 1000),
            "calls": 0, "in_tok": 0, "out_tok": 0, "cost": 0.0,
            "conf": round(prob[pred], 3), "err": ""}


@app.route("/api/predict", methods=["POST"])
def api_predict():
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "ข้อความว่าง"}), 400
    in_gold = text in GOLD_TEXTS
    return jsonify({
        "text": text,
        "gold": GOLD_TEXTS.get(text),          # None if not in gold
        "in_gold": in_gold,
        "baseline": run_baseline(text),
        "multiagent": run_multiagent(text),
        "wangchanberta": run_wcb(text),
    })
    # debate/hybrid were removed from the app (they lost in experiments -- code+results kept as evidence in RESULTS.md)
    # run_debate()/run_hybrid() still exist; to re-enable, just plug them back in


_detectors = {}       # cache a detector per operating point (load the client once)


def detector(op):
    """the production-ready detector from predict.py -- uses the key the user entered + a shared cache"""
    import predict
    key = (op, _api_key)
    if key not in _detectors:
        _detectors[key] = predict.SarcasmDetector(operating=op, api_key=_api_key)
    return _detectors[key]


# ---------- guardrails: prevent others from burning the owner's key when deployed for others ----------
# tunable via env: PUBLIC_DAILY_LIMIT (items/day, everyone combined), PUBLIC_IP_HOURLY_LIMIT (items/hr/IP)
# the local machine (127.0.0.1) is unlimited -> the owner is unrestricted; limits apply only to remote users
DAILY_CAP = int(os.environ.get("PUBLIC_DAILY_LIMIT", "2000"))
IP_HOUR_CAP = int(os.environ.get("PUBLIC_IP_HOURLY_LIMIT", "200"))
_usage_lock = threading.Lock()
_usage = {"day": "", "day_count": 0, "ip_hour": {}}


def _guard(n_items, ip):
    """check + count quota, return an error message if exceeded, else None (localhost unlimited)"""
    if ip in ("127.0.0.1", "::1", "localhost", None):
        return None
    n_items = max(int(n_items or 1), 1)
    day = time.strftime("%Y-%m-%d")
    hour = int(time.time() // 3600)
    with _usage_lock:
        if _usage["day"] != day:
            _usage.update(day=day, day_count=0, ip_hour={})
        if _usage["day_count"] + n_items > DAILY_CAP:
            return f"วันนี้เต็มโควตารวมแล้ว ({DAILY_CAP} ข้อ/วัน) ลองใหม่พรุ่งนี้นะ"
        key = (ip, hour)
        used = _usage["ip_hour"].get(key, 0)
        if used + n_items > IP_HOUR_CAP:
            return f"คุณใช้เยอะไปในชั่วโมงนี้ ({IP_HOUR_CAP} ข้อ/ชม.) พักแป๊บแล้วค่อยลองใหม่"
        _usage["day_count"] += n_items
        _usage["ip_hour"][key] = used + n_items
    return None


# systems selectable on the /app page (each has a mascot)
MODELS_PUBLIC = ("balanced", "high_recall", "multiagent", "wangchanberta")
MODEL_LABEL = {"balanced": "gpt-4.1-mini", "high_recall": "gpt-4o",
               "multiagent": "multi-agent (2 ตัว)", "wangchanberta": "WangchanBERTa (ฟรี)"}


def _dec(pred):
    return "sarcasm" if pred == "1" else ("not_sarcasm" if pred == "0" else "error")


def _corrections_block(text):
    """few-shot human-corrected examples relevant to this text -- appended to every LLM's prompt"""
    import predict
    corr = predict.load_corrections()
    return predict._shots_block(predict._relevant(corr, text)) if corr else ""


def _corr_map():
    """dict {text: human-corrected label} for exact overrides (works with all models incl. WangchanBERTa)"""
    import predict
    return {c["text"]: c["label"] for c in predict.load_corrections()}


def _classify(model, text, review=False):
    """detect one text with the chosen system -> a standard row (multiagent also returns steps to animate)"""
    if model == "multiagent":
        r = run_multiagent(text)
        p = r.get("pred")
        return {"text": text, "label": p if p in ("0", "1") else None, "prob": None,
                "decision": _dec(p), "steps": r.get("steps"), "note": r.get("note")}
    if model == "wangchanberta":
        r = run_wcb(text)
        p = r.get("pred")
        conf = r.get("conf")                                   # confidence of the predicted class
        psarc = conf if p == "1" else (1 - conf if p == "0" and conf is not None else None)
        return {"text": text, "label": p if p in ("0", "1") else None, "prob": psarc,
                "decision": _dec(p), "note": r.get("note")}
    r = detector(model).predict(text, review_band=review)     # balanced / high_recall
    return {"text": text, **r}


def _need_check(model):
    """return an error message if this system can't be used right now (key/model), else None"""
    if model not in MODELS_PUBLIC:
        return f"ไม่รู้จักระบบ: {model}"
    if model != "wangchanberta" and not _api_key:
        return "ยังไม่มี OPENAI_API_KEY (ใส่คีย์ด้านบนก่อน)"
    if model == "wangchanberta" and not has_wcb():
        return "ยังไม่มีโมเดล WangchanBERTa (รัน train_final_wcb.py)"
    return None


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """detect many texts at once -- system selectable (balanced/high_recall/multiagent/wangchanberta)"""
    body = request.json or {}
    texts = [str(t).strip() for t in body.get("texts", []) if str(t).strip()]
    model = body.get("model") or body.get("op") or "balanced"
    review = bool(body.get("review_band", False))
    if not texts:
        return jsonify({"error": "ไม่มีข้อความ"}), 400
    if len(texts) > 500:
        return jsonify({"error": f"มากเกินไป ({len(texts)}) -- จำกัด 500 ข้อ/ครั้ง"}), 400
    err = _need_check(model)
    if err:
        return jsonify({"error": err}), 400
    gerr = _guard(len(texts) * (2 if model == "multiagent" else 1), request.remote_addr)
    if gerr:
        return jsonify({"error": gerr}), 429

    rows = []
    corr = _corr_map()
    for t in texts:
        if t in corr:                                  # this item was corrected directly -> use the human label (all models)
            lab = corr[t]
            r = {"text": t, "label": lab, "prob": 1.0 if lab == "1" else 0.0,
                 "decision": _dec(lab), "from_correction": True}
        else:
            try:
                r = _classify(model, t, review)
            except Exception as e:
                r = {"text": t, "label": None, "prob": None, "decision": f"error: {type(e).__name__}"}
        rows.append({**r, "in_gold": t in GOLD_TEXTS, "gold": GOLD_TEXTS.get(t)})
    summ = {
        "n": len(rows),
        "sarcasm": sum(1 for r in rows if r["decision"] == "sarcasm"),
        "not": sum(1 for r in rows if r["decision"] == "not_sarcasm"),
        "review": sum(1 for r in rows if r["decision"] == "review"),
        "model": MODEL_LABEL.get(model, model), "op": model,
    }
    return jsonify({"rows": rows, "summary": summ})


# lets a static page (GitHub Pages / opened directly) call this machine's helper
# enabled only for the two "public demo" endpoints: fetch comments, and escalate one unsure text to the cheap LLM
_FETCH_ORIGINS = {"https://thanaphumi2006.github.io", "null"}
_CORS_PATHS = {"/api/fetch_comments", "/api/escalate"}


@app.after_request
def _fetch_cors(resp):
    if request.path in _CORS_PATHS:
        origin = request.headers.get("Origin", "")
        if origin in _FETCH_ORIGINS or origin.startswith("http://127.0.0.1") or origin.startswith("http://localhost"):
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/fetch_comments", methods=["POST", "OPTIONS"])
def api_fetch_comments():
    """fetch comments from a link for the static page (no classification, no key -- the browser scores them)
    exists because the browser is CORS-blocked by YouTube/Pantip/Reddit and can't fetch directly"""
    if request.method == "OPTIONS":
        return "", 204
    body = request.json or {}
    url = str(body.get("url", "")).strip()
    limit = min(int(body.get("limit", 60)), 200)
    if not url.startswith("http"):
        return jsonify({"error": "ใส่ลิงก์ให้ถูก (ขึ้นต้น http)"}), 400
    try:
        import fetch_social as fs
    except ImportError:
        return jsonify({"error": "ยังไม่ได้ติดตั้ง yt-dlp (pip install yt-dlp)"}), 500
    try:
        comments, plat = fs.fetch_any(url, limit)
    except fs.UnsupportedError:
        return jsonify({"error": "แพลตฟอร์มนี้ดึงอัตโนมัติไม่ได้ (ต้องล็อกอิน/เสียเงิน API) "
                                 "รองรับเฉพาะ YouTube · Pantip · Reddit"}), 422
    except fs.FetchError as e:                      # supported platform, fixable reason -> show the real cause
        return jsonify({"error": str(e)}), 502      # (the page already prefixes "ดึงคอมเมนต์ไม่สำเร็จ:")
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}"}), 502
    if not comments:
        return jsonify({"error": "ไม่พบคอมเมนต์ภาษาไทยจากลิงก์นี้"}), 404
    return jsonify({"comments": comments, "platform": plat})


@app.route("/api/escalate", methods=["POST", "OPTIONS"])
def api_escalate():
    """cost-aware cascade, tiers 2 and 3. Tier 1 (the lexical cue model) already ran in the browser and
    was UNSURE about this text, so it lands here:
        tier 2  WangchanBERTa on this machine -- free, offline, and only allowed to say "not sarcastic"
        tier 3  gpt-4.1-mini  -- the paid model, reached only by what survived both free tiers
    only the uncertain minority reaches tier 3, so the paid model is used sparingly -- the core idea of the research.
    degrades gracefully at every step: no model / no key / any error -> {available: false} and the page
    simply keeps its free 'บอกไม่ได้' answer."""
    if request.method == "OPTIONS":
        return "", 204
    text = str((request.json or {}).get("text", "")).strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    corr = _corr_map()
    if text in corr:                                    # a human already corrected this exact text -> trust it
        lab = corr[text]
        return jsonify({"available": True, "label": lab, "decision": _dec(lab),
                        "prob": 1.0 if lab == "1" else 0.0, "by": "human"})

    # --- tier 2: WangchanBERTa as a one-sided negative filter (free, offline) ---
    # measured on the leak-free out-of-fold probs (see cascade_eval.py): the only region where WCB is
    # confidently right is the low end -- every item it answers below WCB_NEG is a true "not sarcastic".
    # it is never confidently sarcastic, so there is no upper cut-off. everything else falls through to the LLM.
    if has_wcb():
        try:
            pw = wcb_prob(text)
            if pw < WCB_NEG:
                return jsonify({"available": True, "label": "0", "decision": "not_sarcasm",
                                "prob": pw, "by": "WangchanBERTa (ฟรี)", "tier": "wcb"})
        except Exception:
            pass                                        # model missing/broken -> just fall through to the LLM

    # --- tier 3: the paid LLM, only for what survived both free tiers ---
    if not _api_key:                                    # WCB deferred but there is no key -> stay cue-only
        return jsonify({"available": False, "reason": "no_key"})
    gerr = _guard(1, request.remote_addr)               # remote users are rate-limited; localhost is unlimited
    if gerr:
        return jsonify({"available": False, "reason": "quota", "error": gerr})
    try:
        r = _classify("balanced", text)                 # one gpt-4.1-mini call w/ logprobs -> P(sarcastic)
    except Exception as e:
        return jsonify({"available": False, "reason": "error", "error": type(e).__name__})
    return jsonify({"available": True, "label": r.get("label"), "decision": r.get("decision"),
                    "prob": r.get("prob"), "by": MODEL_LABEL["balanced"]})


@app.route("/api/youtube", methods=["POST"])
def api_youtube():
    """paste a YouTube link -> fetch Thai comments -> detect sarcasm -> return a list (show only the sarcastic ones)
    *** the YouTube domain is not validated yet (see eval_domain.py) -> results are a "guess", warned on the page ***"""
    body = request.json or {}
    url = str(body.get("url", "")).strip()
    model = body.get("model") or body.get("op") or "balanced"
    limit = min(int(body.get("limit", 80)), 200)
    err = _need_check(model)
    if err:
        return jsonify({"error": err}), 400
    if not url.startswith("http"):
        return jsonify({"error": "ใส่ลิงก์ให้ถูก (ขึ้นต้น http)"}), 400
    try:
        import fetch_social as fs
    except ImportError:
        return jsonify({"error": "ยังไม่ได้ติดตั้ง yt-dlp (pip install yt-dlp)"}), 500

    plat = fs.platform_of(url)
    try:
        comments, plat = fs.fetch_any(url, limit)
    except fs.UnsupportedError:
        # platform cannot be auto-fetched (Twitter/IG/etc. need login/API) -> tell them to paste it themselves
        return jsonify({"error": f"ดึงจาก {plat} อัตโนมัติไม่ได้ (แพลตฟอร์มนี้ต้องล็อกอิน/เสียเงิน API) "
                                 f"ก๊อปคอมเมนต์มาวางในแท็บ “อัปโหลดไฟล์” แทน (ใช้ได้กับทุกแพลตฟอร์ม)",
                        "paste_hint": True}), 422
    except fs.FetchError as e:                      # supported platform, fixable reason -> show the real cause
        return jsonify({"error": f"ดึงจาก {plat} ไม่สำเร็จ: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"ดึงไม่สำเร็จ: {type(e).__name__}"}), 502
    if not comments:
        return jsonify({"error": f"ไม่พบคอมเมนต์ภาษาไทยจาก {plat} (อาจปิดคอมเมนต์ หรือคอมเมนต์ไม่ใช่ไทย)"}), 404
    gerr = _guard(len(comments) * (2 if model == "multiagent" else 1), request.remote_addr)
    if gerr:
        return jsonify({"error": gerr}), 429

    rows = []
    corr = _corr_map()
    for c in comments:
        if c in corr:                                  # this item was corrected directly -> use the human label (all models)
            lab = corr[c]
            rows.append({"text": c, "label": lab, "prob": 1.0 if lab == "1" else 0.0,
                         "decision": _dec(lab), "from_correction": True})
            continue
        try:
            rows.append(_classify(model, c))
        except Exception:
            rows.append({"text": c, "label": None, "prob": None, "decision": "error"})
    rows.sort(key=lambda r: (r.get("decision") != "sarcasm", -(r.get("prob") or 0)))   # sarcasm first
    summ = {"n": len(rows), "sarcasm": sum(1 for r in rows if r["decision"] == "sarcasm"),
            "model": MODEL_LABEL.get(model, model), "op": model, "platform": plat}
    return jsonify({"rows": rows, "summary": summ})


@app.route("/api/correct", methods=["POST"])
def api_correct():
    """the user marks 'the model decided wrong' -> store the correct answer and use it as few-shot next time
    honest note: this is in-context learning (teaching via examples in the prompt), not actually retraining the model"""
    import predict
    body = request.json or {}
    text = str(body.get("text", "")).strip()
    label = str(body.get("label", "")).strip()          # the "correct" label (opposite of the model guess)
    if not text or label not in ("0", "1"):
        return jsonify({"error": "ต้องมี text และ label ('0'/'1')"}), 400
    try:
        n = predict.add_correction(text, label)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    for det in _detectors.values():                     # let already-loaded detectors use the new example immediately
        det.reload_corrections()
    return jsonify({"ok": True, "total": n})


@app.route("/api/key", methods=["POST"])
def api_key_set():
    """take the key from the page -> verify it against OpenAI before accepting (models.list is free)
    kept in process RAM only -- gone when the server stops, never written to a file"""
    global _api_key, _client
    key = (request.json or {}).get("key", "").strip()
    if not key:
        return jsonify({"ok": False, "error": "ยังไม่ได้ใส่คีย์"}), 400
    if not key.startswith("sk-"):
        return jsonify({"ok": False, "error": "รูปแบบคีย์ไม่ถูก -- ต้องขึ้นต้นด้วย sk-"}), 400

    from openai import OpenAI
    try:
        OpenAI(api_key=key).models.list()          # check the key actually works
    except Exception as e:
        return jsonify({"ok": False, "error": f"คีย์ใช้ไม่ได้: {type(e).__name__}"}), 400

    _api_key, _client = key, None                  # clear the old client -> rebuild with this key
    _detectors.clear()
    return jsonify({"ok": True, "masked": mask_key(key)})


@app.route("/api/key", methods=["DELETE"])
def api_key_clear():
    global _api_key, _client
    _api_key, _client = "", None
    _detectors.clear()
    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    """count of what the model learned from everyone combined (persisted in one file on the server -> shared by all)"""
    import predict
    return jsonify({"corrections": len(predict.load_corrections())})


@app.route("/api/sample")
def api_sample():
    lab = request.args.get("label")
    pool = _gold[_gold["label"] == lab] if lab in ("0", "1") else _gold
    r = pool.sample(1).iloc[0]
    return jsonify({"text": r["text"], "label": r["label"]})


@app.route("/")
def index():
    return render_template_string(
        PAGE, rows=GOLD_ROWS,
        has_key=bool(_api_key),
        masked=mask_key(_api_key) if _api_key else "",
        from_env=bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        has_wcb=has_wcb(),
    )


@app.route("/app")
def public_app():
    """page for general users: serves the exact same demo published on GitHub (app.html)
    one text-or-link box, cue-only scoring in the browser; links are fetched via /api/fetch_comments"""
    with open(os.path.join(HERE, "..", "app.html"), encoding="utf-8") as f:
        return f.read()


PAGE = r"""
<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ทดลอง & เทียบระบบตรวจจับประชดภาษาไทย</title>
<style>
:root{
  --bg:#eef1f5; --card:#fff; --card2:#f7f9fb; --line:#e3e8ef;
  --ink:#182029; --ink2:#586675; --muted:#8b97a6;
  --brand:#2f6bd6; --brand-d:#2457b3; --brand-soft:#e8effc;
  --sar:#c0392b; --sar-bg:#fdeceb; --not:#1c7a49; --not-bg:#e7f6ee;
  --warn-bg:#fff7e8; --warn-line:#ead9a6; --warn-ink:#8a6d2f;
  --radius:13px; --shadow:0 1px 2px rgba(20,30,45,.04),0 6px 20px rgba(20,30,45,.06);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.55;-webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Tahoma,system-ui,sans-serif}
.wrap{max-width:840px;margin:0 auto;padding:clamp(18px,4vw,40px) clamp(14px,3vw,26px) 80px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
header h1{font-size:clamp(23px,4.5vw,32px);margin:0;letter-spacing:-.02em;font-weight:800;text-wrap:balance}
header .tagline{color:var(--ink2);font-size:clamp(14px,2.2vw,16px);margin:9px 0 0;text-wrap:pretty}
.sub{color:var(--ink2);font-size:13px}
.step{margin-top:28px}
.steplabel{display:flex;align-items:center;gap:9px;font-size:15px;font-weight:750;margin-bottom:11px}
.num{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;
  background:var(--brand);color:#fff;font-size:13px;font-weight:700;flex:none}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
  padding:clamp(15px,3vw,22px);box-shadow:var(--shadow)}
.tabs{display:flex;gap:7px;margin-bottom:13px;flex-wrap:wrap}
.tab{padding:10px 17px;border:1px solid var(--line);background:var(--card);color:var(--ink2);border-radius:10px;
  font-size:14px;font-weight:600;cursor:pointer;font-family:inherit;transition:background .12s}
.tab.active{background:var(--brand);color:#fff;border-color:var(--brand)}
.tab:not(.active):hover{background:var(--brand-soft);color:var(--brand-d)}
textarea,input[type=text],input[type=password]{width:100%;padding:12px;border:1px solid #cdd7e2;border-radius:10px;
  font-family:inherit;font-size:15px;background:var(--card);color:var(--ink)}
textarea{min-height:90px;resize:vertical}
input:focus,textarea:focus{outline:2px solid var(--brand);outline-offset:-1px;border-color:var(--brand)}
input[type=password]{font-family:ui-monospace,Menlo,monospace;font-size:14px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}
button{padding:10px 18px;border:0;border-radius:9px;font-size:14px;cursor:pointer;font-family:inherit;font-weight:600}
.go{background:var(--brand);color:#fff} .go:hover{background:var(--brand-d)} .go:disabled{opacity:.6;cursor:wait}
.ghost{background:#eaeff5;color:#46546a;font-weight:500} .ghost:hover{background:#dfe6ef}
select{padding:8px 10px;border:1px solid #cdd7e2;border-radius:8px;font-family:inherit;font-size:13px;background:var(--card);color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:14px}
.sys{border:1px solid var(--line);border-radius:11px;padding:15px;background:var(--card2)}
.sys h3{margin:0 0 3px;font-size:14px} .sys .tag{font-size:11px;color:var(--muted);margin-bottom:11px}
.verdict{font-size:19px;font-weight:800;padding:10px;border-radius:9px;margin-bottom:11px;text-align:center}
.v1{background:var(--sar-bg);color:var(--sar)} .v0{background:var(--not-bg);color:var(--not)}
.vna{background:#eef1f4;color:var(--muted);font-size:14px;font-weight:600}
.kv{display:flex;justify-content:space-between;font-size:12.5px;padding:3px 0;color:var(--ink2)}
.kv b{color:var(--ink);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 9px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left} th{color:var(--ink2);font-weight:600;font-size:12px}
tr.best td{background:var(--brand-soft)}
.pill{display:inline-block;padding:4px 11px;border-radius:20px;font-size:12.5px;font-weight:700;white-space:nowrap}
.pill.v1{background:var(--sar-bg);color:var(--sar)} .pill.v0{background:var(--not-bg);color:var(--not)}
.pill.vna{background:#eef1f4;color:var(--muted);font-weight:600}
.note{font-size:12.5px;color:var(--warn-ink);background:var(--warn-bg);border:1px solid var(--warn-line);
  border-radius:9px;padding:10px 12px;margin-top:12px;line-height:1.6}
.warn{font-size:12.5px;color:var(--sar);background:var(--sar-bg);border:1px solid #edc4c1;
  border-radius:9px;padding:10px 12px;margin-top:10px;line-height:1.55}
.ok{font-size:12.5px;color:var(--not);background:var(--not-bg);border:1px solid #bfe3ce;
  border-radius:9px;padding:10px 12px;margin-top:10px}
.gold{font-size:13px;padding:9px 12px;border-radius:9px;background:var(--brand-soft);border:1px solid #cbd9f3;margin-top:12px}
.sp{display:inline-block;width:13px;height:13px;border:2px solid #fff;border-top-color:transparent;
  border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:6px}
@keyframes s{to{transform:rotate(360deg)}}
.keyhead{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
details.about{margin-top:30px;background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
details.about>summary{cursor:pointer;padding:16px 20px;font-weight:700;font-size:15px;list-style:none}
details.about>summary::-webkit-details-marker{display:none}
details.about>summary::before{content:"\25B8  ";color:var(--brand)}
details.about[open]>summary::before{content:"\25BE  "}
.about-body{padding:2px 20px 20px}
.about-body h3{font-size:15px;margin:16px 0 3px}
.flow{display:flex;flex-direction:column;align-items:center;margin-top:8px}
.ag{border:1.6px solid #cdd7e2;border-radius:11px;padding:14px;background:var(--card);width:100%;max-width:340px}
.ag.on{border-color:#2f9e5e;background:var(--not-bg)} .ag.off{border-style:dashed;background:var(--card2);opacity:.72}
.ag.kill{border-color:#c96a6a;background:var(--sar-bg)}
.ag .who{font-size:13.5px;font-weight:700;margin-bottom:2px}
.ag .job{font-size:11.5px;color:var(--ink2);line-height:1.5;margin-bottom:10px}
.ag .say{font-size:14px;font-weight:700;padding:8px 10px;border-radius:6px;background:#eef1f5;color:#42505f;text-align:center;margin-bottom:8px}
.ag.on .say{background:#dff0e7;color:var(--not)} .ag.kill .say{background:#f8e4e4;color:var(--sar)}
.ag.off .say{background:#eef1f4;color:var(--muted);font-weight:500;font-size:12.5px}
.ag .meta{font-size:11.5px;color:var(--ink2);display:flex;justify-content:space-between;padding:2px 0}
.arw{display:flex;flex-direction:column;align-items:center;gap:2px;padding:10px 0;font-size:22px;color:var(--muted);text-align:center}
.arw small{font-size:10.5px;color:var(--muted);line-height:1.3}
.fin{margin-top:12px;padding:11px 14px;border-radius:9px;font-size:13.5px;font-weight:600;text-align:center}
.fin.k{background:var(--sar-bg);color:var(--sar)} .fin.p{background:var(--not-bg);color:var(--not)}
.rule{font-size:12px;color:#5c5340;background:var(--warn-bg);border:1px solid var(--warn-line);border-radius:9px;padding:10px 12px;margin-top:12px;line-height:1.65}
</style></head><body><div class="wrap">

<header>
  <h1>ตรวจจับประชดภาษาไทย</h1>
  <p class="tagline">พิมพ์ข้อความ · อัปโหลดไฟล์ · หรือวางลิงก์ YouTube แล้วดูว่า “ประชด” หรือเปล่า</p>
</header>

<div class="step">
  <div class="steplabel"><span class="num">1</span> ใส่กุญแจ OpenAI (ทำครั้งเดียว)</div>
  <div class="card keycard" id="keycard">
    <div class="keyhead">
      <div class="sub" style="max-width:52ch">ระบบที่เป็น AI ต้องมี API key ถึงจะทำงาน ส่วน WangchanBERTa (โมเดลฟรี) ใช้ได้เลยไม่ต้องใส่</div>
      <span class="pill {{ 'v0' if has_key else 'vna' }}" id="keypill">{{ 'พร้อมใช้งาน · ' ~ masked if has_key else 'ยังไม่มีคีย์' }}</span>
    </div>
    <div class="row" id="keyform" {% if has_key %}style="display:none"{% endif %}>
      <input type="password" id="k" placeholder="sk-..." autocomplete="off" spellcheck="false" style="flex:1;min-width:240px">
      <button class="go" id="ksave" onclick="saveKey()">บันทึก</button>
    </div>
    <div class="row" id="keydone" {% if not has_key %}style="display:none"{% endif %}>
      <button class="ghost" onclick="clearKey()">ลบคีย์ออก</button>
      {% if from_env %}<span class="sub">อ่านมาจาก environment variable <code>OPENAI_API_KEY</code></span>{% endif %}
    </div>
    <div id="kmsg"></div>
    <div class="note">คีย์เก็บในหน่วยความจำเซิร์ฟเวอร์เท่านั้น ไม่เขียนลงไฟล์ ปิดเว็บแล้วหาย · เว็บนี้เปิดแค่บนเครื่องคุณ (127.0.0.1) อย่าเปิดออกอินเทอร์เน็ต</div>
  </div>
</div>

{% if not has_wcb %}<div class="warn" style="margin-top:14px"><b>ยังไม่มีโมเดล WangchanBERTa</b> รัน <code>train_final_wcb.py</code> ก่อน (ระบบอื่นใช้ได้ปกติ)</div>{% endif %}

<div class="step">
  <div class="steplabel"><span class="num">2</span> เลือกวิธีใช้</div>
  <div class="tabs">
    <button class="tab active" id="tabbtn-single" onclick="showTab('single')">พิมพ์ทีละข้อความ</button>
    <button class="tab" id="tabbtn-batch" onclick="showTab('batch')">อัปโหลดไฟล์</button>
    <button class="tab" id="tabbtn-yt" onclick="showTab('yt')">จากลิงก์โซเชียล</button>
  </div>

  <div class="card tabpanel" id="tab-single">
    <div class="sub" style="margin-bottom:11px">พิมพ์ข้อความไทย 1 ข้อ แล้วเทียบ 3 ระบบพร้อมกัน (AI เดี่ยว · AI สองชั้น · โมเดลฟรี)</div>
    <textarea id="t" placeholder="เช่น: ขอบคุณมากนะคะที่ให้รอแค่ 2 ชั่วโมง บริการดีเยี่ยมจริงๆ"></textarea>
    <div class="row">
      <button class="go" id="go" onclick="run()">วิเคราะห์</button>
      <button class="ghost" onclick="samp('1')">ตัวอย่างประชด</button>
      <button class="ghost" onclick="samp('0')">ตัวอย่างไม่ประชด</button>
      <button class="ghost" onclick="$('t').value='';$('out').innerHTML='';$('goldbox').innerHTML=''">ล้าง</button>
    </div>
    <div id="goldbox"></div>
    <div id="out"></div>
  </div>

  <div class="card tabpanel" id="tab-batch" style="display:none">
    <div class="sub" style="margin-bottom:10px">อัปโหลดไฟล์ CSV (มีคอลัมน์ <code>text</code>) หรือวางข้อความทีละบรรทัด → ได้ผลเป็นตาราง ดาวน์โหลดได้</div>
    <div class="warn" style="margin-bottom:12px">คำเตือน: โมเดลวัดผลไว้แค่ <b>รีวิวร้าน + ทวีตสั้น</b> (F1~0.72) ข้อความโดเมนอื่นยังไม่ได้ทดสอบ ผลอาจเพี้ยน</div>
    <div class="row" style="margin-top:0;margin-bottom:4px">
      <label class="sub">ความละเอียด:
        <select id="bop">
          <option value="balanced">ปกติ เร็ว/ถูก (gpt-4.1-mini)</option>
          <option value="high_recall">จับให้ครบ แม่นแต่แพงกว่า (gpt-4o)</option>
        </select></label>
      <label class="sub" style="display:inline-flex;align-items:center;gap:5px"><input type="checkbox" id="brev"> ข้อก้ำกึ่งให้คนตัดสิน</label>
    </div>
    <textarea id="btext" placeholder="วางข้อความทีละบรรทัด หรือเลือกไฟล์ด้านล่าง"></textarea>
    <div class="row">
      <button class="go" id="bgo" onclick="runBatch()">ตรวจทั้งหมด</button>
      <input type="file" id="bfile" accept=".csv,.txt" style="font-size:13px;padding:0;border:0;background:none">
      <span class="sub" id="bhint">สูงสุด 500 ข้อ/ครั้ง</span>
    </div>
    <div id="bout"></div>
  </div>

  <div class="card tabpanel" id="tab-yt" style="display:none">
    <div class="sub" style="margin-bottom:10px">วางลิงก์ → ดึงคอมเมนต์ไทย → โชว์เฉพาะคอมเมนต์ที่ระบบคิดว่า “ประชด”</div>
    <div class="note" style="margin-top:0;margin-bottom:10px">
      ดึงฟรีได้ (ไม่ต้องล็อกอิน): <b>YouTube</b> · <b>Pantip</b> (ฟอรัมไทย) · <b>Reddit</b> ·
      แพลตฟอร์มอื่น (Twitter/X, Instagram, TikTok, Facebook) เข้าถึงคอมเมนต์ฟรีไม่ได้ →
      ก๊อปมาวางในแท็บ <b>“อัปโหลดไฟล์”</b> ได้ทุกที่
    </div>
    <div class="warn" style="margin-bottom:12px">คำเตือน: <b>โซเชียลเป็นโดเมนที่ยังไม่ได้ทดสอบ</b> ผลเป็นการเดา มักจับพลาด (คำชมจริงถูกจับเป็นประชดได้บ่อย) กด “ตัดสินผิด” เพื่อสอนได้</div>
    <div class="row" style="margin-top:0">
      <input type="text" id="yurl" placeholder="วางลิงก์ YouTube / Pantip / Reddit ..." style="flex:1;min-width:230px">
      <button class="go" id="ygo" onclick="runYT()">ดึง + วิเคราะห์</button>
    </div>
    <div class="sub" id="yhint" style="margin-top:8px">ดึงสูงสุด ~80 คอมเมนต์ · ใช้เวลาสักครู่</div>
    <div id="yout"></div>
  </div>
</div>

<details class="about">
  <summary>ระบบทำงานยังไง + คะแนนจากงานวิจัย</summary>
  <div class="about-body">
    <h3>ระบบ “AI สองชั้น” ทำงานยังไง</h3>
    <div class="sub" style="margin-bottom:4px">AI เดี่ยว + เพิ่ม <b>ผู้ตรวจสอบ</b> อีกหนึ่งคน แค่นั้น</div>
  <div id="flow">
    <div class="flow">
      <div class="ag">
        <div class="who">คนที่ 1 · <b>ผู้คัดกรอง</b></div>
        <div class="job"><b>คือ ① เอเจนต์เดี่ยว ตัวเดิมเป๊ะ</b><br>
          อ่านข้อความ แล้วตอบว่า “ประชด” หรือ “ไม่ประชด”</div>
        <div class="say">ประชด? → ใช่ / ไม่ใช่</div>
        <div style="font-size:11.5px;color:#5a6472;background:#f6f8fa;border-radius:6px;padding:8px 10px">
          <b>เก่ง:</b> จับประชดได้ครบ ไม่พลาดเลยสักข้อ<br>
          <b>ไม่เก่ง:</b> เหวี่ยงแหกว้างไป ทายว่าประชด 27 ข้อที่จริงไม่ใช่
        </div>
      </div>

      <div class="arw">↓<small>ส่งต่อ<br><b>เฉพาะข้อที่ตอบ “ใช่”</b></small></div>

      <div class="ag">
        <div class="who">คนที่ 2 · <b>ผู้ตรวจสอบ</b></div>
        <div class="job"><b>คนใหม่ที่เพิ่มเข้ามา</b><br>
          ตรวจของที่คนแรกส่งมา ว่าเป็นประชด<b>จริง</b>ไหม</div>
        <div class="say">ยืนยัน / ปัดตก</div>
        <div style="font-size:11.5px;color:#5a6472;background:#f6f8fa;border-radius:6px;padding:8px 10px">
          <b>อำนาจของเขามีแค่อย่างเดียว: ปัดตก</b><br>
          เขา<b>เพิ่ม</b>ประชดใหม่ไม่ได้ ทำได้แค่บอกว่า “อันนี้คนแรกทายผิด”
        </div>
      </div>
    </div>

    <div class="rule">
      <b>ทำไมออกแบบให้เขาปัดตกได้อย่างเดียว?</b><br>
      เพราะคนแรกจับประชดได้<b>ครบอยู่แล้ว</b> ปัญหาเดียวคือมัน “เหวี่ยงแหกว้างเกิน”
      งานที่เหลือจึงมีแค่ <b>ตัดของเกินทิ้ง</b> ไม่ใช่หาเพิ่ม<br>
      ผลพลอยได้: ผู้ตรวจสอบไม่ต้องดูข้อที่คนแรกตอบ “ไม่ใช่” เลย →
      ใช้แค่ <b>183 ครั้ง แทนที่จะเป็น 254</b> (ประหยัด 28%)<br><br>
      <b>กฎข้อเดียวที่ชี้เป็นชี้ตายคือ “เวลาไม่แน่ใจให้ทำยังไง”</b><br>
      • ถ้าสั่งว่า <b>“ไม่แน่ใจ → ปัดตก”</b> → เขาเผลอตัดประชด<b>จริง</b>ทิ้ง 10 ข้อ → F1 แย่ลง<br>
      • ถ้าสั่งว่า <b>“ไม่แน่ใจ → เก็บไว้”</b> → เสียแค่ 1 ข้อ → <b>F1 ดีที่สุด (0.744)</b><br>
      เหตุผล: <b>ประชดที่แนบเนียนมันอ่านได้สองแง่อยู่แล้วโดยธรรมชาติ</b>  
      ถ้าลังเลแปลว่ามัน<b>น่าจะ</b>ประชด ไม่ใช่ไม่ประชด
    </div>

    <div class="note">
      <b>เราลองแบบซับซ้อนกว่านี้แล้ว และมันแพ้</b><br>
      • <b>Debate</b> (อัยการ + ทนาย + ผู้พิพากษา, 3 คน ตัดสินใหม่ได้อิสระ) → F1 <b>0.694</b> · แพงกว่า <b>4.1×</b><br>
      • <b>Hybrid</b> (เอา debate มาเป็นผู้ตรวจสอบ, 4 คน) → F1 <b>0.700</b> · แพงกว่า <b>2.4×</b><br>
      ทั้งคู่<b>แพ้</b>ระบบ 2 คนข้างบน (0.744) ที่ถูกที่สุดและง่ายที่สุด<br>
      <b>บทเรียน: “จำกัดอำนาจ agent ให้ถูกจุด” สำคัญกว่า “มี agent เยอะ”</b>
      <span style="color:#8a94a6">(รายละเอียด + สถิติ: RESULTS.md)</span>
    </div>
  </div>
    <h3 style="margin-top:22px">คะแนนจริง วัดบนข้อมูล 127 ข้อ (ประชดจริง 30)</h3>
    <div class="sub" style="margin-bottom:10px">ทุกระบบวัดบนข้อมูลชุดเดียวกัน ด้วยโค้ดวัดผลตัวเดียวกัน</div>
  <table>
    <tr><th>ระบบ</th><th>F1</th><th>precision</th><th>recall</th><th>TP</th><th>FP</th><th>FN</th>
        <th>LLM calls</th><th>ค่าใช้จ่าย</th></tr>
    {% for r in rows %}
    <tr {% if r.f1 == rows|map(attribute='f1')|max %}class="best"{% endif %}>
      <td>{{ r.name }}</td><td><b>{{ "%.3f"|format(r.f1) }}</b></td>
      <td>{{ "%.3f"|format(r.prec) }}</td><td>{{ "%.3f"|format(r.rec) }}</td>
      <td>{{ r.tp }}</td><td>{{ r.fp }}</td><td>{{ r.fn }}</td>
      <td>{{ r.calls }}</td><td>${{ "%.3f"|format(r.cost) }}</td>
    </tr>{% endfor %}
  </table>
    <div class="note">multi-agent (สองชั้น) ได้ F1 สูงสุด แต่จ่ายแพงสุด · AI เดี่ยวจับประชดครบทุกข้อ (FN=0) แต่เหวี่ยงแหเกิน (FP 27) · WangchanBERTa ฟรีและออฟไลน์ แต่ปล่อยประชดหลุด 9 ข้อ · <b>อย่าดู accuracy</b> ข้อมูลเอียง 76/24</div>
  </div>
</details>

<script>
const $=i=>document.getElementById(i);
function showTab(t){['single','batch','yt'].forEach(function(n){
  $('tab-'+n).style.display=(n===t?'block':'none');
  $('tabbtn-'+n).classList.toggle('active',n===t);});}

function keyUI(on,masked){
  $('keypill').className='pill '+(on?'v0':'vna');
  $('keypill').textContent = on ? 'พร้อมใช้งาน · '+masked : 'ยังไม่มีคีย์';
  $('keyform').style.display = on?'none':'flex';
  $('keydone').style.display = on?'flex':'none';
}
async function saveKey(){
  const key=$('k').value.trim(); if(!key){$('k').focus();return}
  $('ksave').disabled=true; $('ksave').innerHTML='<span class="sp"></span>กำลังตรวจคีย์...';
  $('kmsg').innerHTML='';
  try{
    const r=await fetch('/api/key',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({key})});
    const d=await r.json();
    if(d.ok){
      $('k').value='';                       // ไม่ทิ้งคีย์ไว้ใน DOM
      keyUI(true,d.masked);
      $('kmsg').innerHTML='<div class="ok">ใช้คีย์นี้ได้ ระบบ ① และ ② พร้อมรันแล้ว</div>';
    } else {
      $('kmsg').innerHTML='<div class="warn">'+d.error+'</div>';
    }
  }catch(e){ $('kmsg').innerHTML='<div class="warn">ต่อเซิร์ฟเวอร์ไม่ได้: '+e+'</div>' }
  $('ksave').disabled=false; $('ksave').textContent='บันทึกคีย์';
}
async function clearKey(){
  await fetch('/api/key',{method:'DELETE'});
  keyUI(false,''); $('kmsg').innerHTML='';
}

async function samp(l){
  const r=await fetch('/api/sample?label='+l); const d=await r.json();
  $('t').value=d.text; $('goldbox').innerHTML=''; $('out').innerHTML='';
}
function box(title,tag,d,goldLab){
  let v,cls;
  if(d.pred==='1'){v='ประชด';cls='v1'} else if(d.pred==='0'){v='ไม่ประชด';cls='v0'}
  else {v=d.note||d.err||'ผิดพลาด';cls='vna'}
  let mark='';
  if(goldLab!=null && (d.pred==='0'||d.pred==='1')) mark = d.pred===goldLab?' ✓':' ✗';
  let kv='';
  const add=(k,val)=>kv+=`<div class="kv"><span>${k}</span><b>${val}</b></div>`;
  if(d.pred==='0'||d.pred==='1'){
    if(d.latency_ms!=null) add('เวลา', d.latency_ms+' ms');
    add('LLM calls', d.calls);
    if(d.in_tok) add('token (in/out)', d.in_tok+' / '+d.out_tok);
    add('ค่าใช้จ่าย', d.cost>0 ? '$'+d.cost.toFixed(5) : '$0.00 (ฟรี)');
    if(d.conf!=null) add('ความมั่นใจ', (d.conf*100).toFixed(1)+'%');
    if(d.detect) add('detector ว่า', d.detect==='1'?'ประชด':'ไม่ประชด');
    if(d.verdict) add('verifier ว่า', d.verdict==='1'?'คงไว้':'พลิกทิ้ง');
  }
  return `<div class="sys"><h3>${title}</h3><div class="tag">${tag}</div>
    <div class="verdict ${cls}">${v}${mark}</div>${kv}</div>`;
}
function drawFlow(m){
  if(!m || !m.steps){return}
  const [s1,s2]=m.steps;
  const meta=s=>s.ran
    ? `<div class="meta"><span>ใช้เวลา</span><b>${s.ms} ms</b></div>
       <div class="meta"><span>ค่าใช้จ่าย</span><b>$${s.cost.toFixed(5)}</b></div>`
    : `<div class="meta"><span>ใช้เวลา</span><b>0 ms</b></div>
       <div class="meta"><span>ค่าใช้จ่าย</span><b>$0.00000 ประหยัดได้</b></div>`;
  const said1 = s1.said==='1' ? 'ประชด' : (s1.said==='0' ? 'ไม่ประชด' : 'ตอบเพี้ยน');
  const c1 = s1.said==='1' ? 'on' : '';
  const c2 = !s2.ran ? 'off' : (s2.said==='0' ? 'kill' : 'on');
  const arrow = s2.ran
    ? '↓<small>ส่งไปตรวจ<br><b>เพราะคนแรกตอบ “ประชด”</b></small>'
    : '⇣<small style="color:#c0c6cf">ไม่ต้องตรวจ<br><b>คนแรกตอบ “ไม่ประชด”</b></small>';
  const say2 = !s2.ran
    ? 'ไม่ได้ทำงาน ไม่มีอะไรให้ตรวจ'
    : (s2.said==='0' ? 'ปัดตก คนแรกทายผิด' : 'ยืนยัน คนแรกทายถูก');
  let fin, cls;
  if(m.flipped){ fin='ผลสุดท้าย: <b>ไม่ประชด</b> ผู้ตรวจสอบจับได้ว่าคนแรกทายผิด แล้วปัดตกทิ้ง'; cls='k'; }
  else if(m.pred==='1'){ fin='ผลสุดท้าย: <b>ประชด</b> ผู้ตรวจสอบตรวจแล้ว ยืนยันตามคนแรก'; cls='p'; }
  else { fin='ผลสุดท้าย: <b>ไม่ประชด</b> คนแรกตีตกตั้งแต่ต้น ผู้ตรวจสอบไม่ต้องทำงาน'; cls='p'; }
  document.getElementById('flow').innerHTML=`
    <div class="flow">
      <div class="ag ${c1}">
        <div class="who">คนที่ 1 · <b>ผู้คัดกรอง</b></div>
        <div class="job">คือ ① เอเจนต์เดี่ยว ตัวเดิมเป๊ะ</div>
        <div class="say">พูดว่า: <b>${said1}</b></div>${meta(s1)}
      </div>
      <div class="arw">${arrow}</div>
      <div class="ag ${c2}">
        <div class="who">คนที่ 2 · <b>ผู้ตรวจสอบ</b></div>
        <div class="job">ปัดตกได้อย่างเดียว เพิ่มประชดใหม่ไม่ได้</div>
        <div class="say">${say2}</div>${meta(s2)}
      </div>
    </div>
    <div class="fin ${cls}">${fin}</div>
    <div class="rule">
      ข้อความนี้ใช้ไป <b>${m.calls} ครั้ง</b> · <b>${m.latency_ms} ms</b> · <b>$${m.cost.toFixed(5)}</b>
      &nbsp; &nbsp; เทียบกับเอเจนต์เดี่ยวที่ใช้ 1 ครั้งเสมอ<br>
      <b>สังเกต:</b> ผู้ตรวจสอบ<b>เพิ่มประชดใหม่ไม่ได้</b> ทำได้แค่ปัดตกของคนแรก →
      ระบบนี้จึงซื้อได้แค่ <b>ความแม่นยำ</b> (ลดการทายเกิน) ไม่ได้ซื้อ “ความครบ”
    </div>`;
}
async function run(){
  const text=$('t').value.trim(); if(!text){$('t').focus();return}
  $('go').disabled=true; $('go').innerHTML='<span class="sp"></span>กำลังวิเคราะห์...';
  $('out').innerHTML='';
  try{
    const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text})});
    const d=await r.json();
    let g='';
    if(d.in_gold){
      g=`<div class="gold"><b>ข้อความนี้อยู่ใน gold set</b> คำตอบจริง:
        <b>${d.gold==='1'?'ประชด':'ไม่ประชด'}</b> (✓/✗ ข้างล่างคือถูก/ผิดเทียบคำตอบจริง)<br>
        <span style="color:#a02020">ระวัง: WangchanBERTa เทรนด้วยข้อนี้มาแล้ว → มันจำคำตอบได้ ไม่ใช่ฝีมือจริง
        ตัวเลขจริงของมันอยู่ในตารางข้างล่าง (out-of-fold)</span></div>`;
    }
    $('goldbox').innerHTML=g;
    $('out').innerHTML=`<div class="card"><div class="grid">
      ${box('① เอเจนต์เดี่ยว','คนเดียว ตัดสินจบ',d.baseline,d.gold)}
      ${box('② Multi-agent','ผู้คัดกรอง → ผู้ตรวจสอบ',d.multiagent,d.gold)}
      ${box('③ WangchanBERTa','โมเดลเล็กเทรนเอง · ฟรี · ออฟไลน์',d.wangchanberta,d.gold)}
    </div></div>`;
    drawFlow(d.multiagent);
    var ab=document.querySelector('details.about'); if(ab) ab.open=true;
  }catch(e){ $('out').innerHTML='<div class="card warn">เรียก API ไม่สำเร็จ: '+e+'</div>' }
  $('go').disabled=false; $('go').textContent='วิเคราะห์ด้วยทั้ง 3 ระบบ';
}
$('t').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')run()});
$('k').addEventListener('keydown',e=>{if(e.key==='Enter')saveKey()});

// ---------- YouTube ----------
let _yall=[], _ysum={}, _yonly=true;
async function runYT(){
  const url=$('yurl').value.trim();
  if(!url){$('yurl').focus();return}
  $('ygo').disabled=true; $('ygo').innerHTML='<span class="sp"></span>กำลังดึง+วิเคราะห์...';
  $('yout').innerHTML=''; $('yhint').textContent='กำลังดึงคอมเมนต์ (อาจ 1-2 นาที)...';
  try{
    const r=await fetch('/api/youtube',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url,op:$('bop').value,limit:80})});
    const d=await r.json();
    if(d.error){
      const jump=d.paste_hint?' <button class="go" style="padding:6px 12px;margin-top:8px" onclick="showTab(\'batch\')">ไปแท็บอัปโหลดไฟล์</button>':'';
      $('yout').innerHTML='<div class="warn">'+d.error+jump+'</div>';
    }
    else{ _yall=d.rows; _ysum=d.summary; _yonly=true; _ypage=1; renderYT(); }
  }catch(e){ $('yout').innerHTML='<div class="warn">ผิดพลาด: '+e+'</div>' }
  $('ygo').disabled=false; $('ygo').textContent='ดึง + วิเคราะห์';
  $('yhint').textContent='ดึงสูงสุด ~80 คอมเมนต์ · ใช้เวลาสักครู่ (ดึง+ยิงทีละข้อ)';
}
const YPP=5;                       // คอมเมนต์ต่อหน้า
let _ypage=1;
function ytToggle(only){_yonly=only; _ypage=1; renderYT();}
function ytPage(p){_ypage=p; renderYT(); document.getElementById('tab-yt').scrollIntoView({behavior:'smooth',block:'nearest'});}
let _corrected=0;
async function markWrong(i,btn){
  const r=_yall[i]; if(!r||r.corrected)return;
  const correct = r.decision==='sarcasm' ? '0' : '1';   // ป้ายที่ถูก = ตรงข้ามกับที่โมเดลทาย
  btn.disabled=true; btn.textContent='กำลังจำ...';
  try{
    const resp=await fetch('/api/correct',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:r.text,label:correct})});
    const d=await resp.json();
    if(d.ok){ r.corrected=true; _corrected=d.total; renderYT(); }
    else { btn.disabled=false; btn.textContent='ผิด'; alert(d.error||'ผิดพลาด'); }
  }catch(e){ btn.disabled=false; btn.textContent='ผิด'; }
}
function renderYT(){
  const s=_ysum;
  const rows=(_yonly?_yall.filter(r=>r.decision==='sarcasm'):_yall);
  const pages=Math.max(1,Math.ceil(rows.length/YPP));
  if(_ypage>pages)_ypage=pages; if(_ypage<1)_ypage=1;
  const pageRows=rows.slice((_ypage-1)*YPP,_ypage*YPP);
  const list=pageRows.map(r=>{
    const i=_yall.indexOf(r);
    const wrongLabel = r.decision==='sarcasm' ? 'ไม่ใช่ประชด' : 'จริงๆ ประชด';
    const btn = r.corrected
      ? '<span style="color:#1e7a4b;font-weight:600">✓ จำแล้ว จะใช้เป็นตัวอย่างครั้งต่อไป</span>'
      : `<button class="ghost" style="padding:4px 10px;font-size:11.5px" onclick="markWrong(${i},this)">ตัดสินผิด → ${wrongLabel}</button>`;
    return `<div style="padding:10px 12px;border:1px solid var(--line);border-radius:8px;
      background:${r.corrected?'#f2faf5':(r.decision==='sarcasm'?'#fdf2f2':'var(--card)')};margin-bottom:7px">
      <div style="font-size:14px">${esc(r.text)}</div>
      <div style="font-size:11.5px;color:var(--muted);margin-top:5px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
        ${pill(r.decision)} <span class="mono">P(ประชด)=${r.prob==null?'–':r.prob}</span> ${btn}</div>
    </div>`;
  }).join('') || '<div class="sub" style="padding:8px 0">  ไม่มี  </div>';
  const relearn = _corrected>0
    ? `<button class="go" style="padding:7px 14px" onclick="reanalyzeYT()">วิเคราะห์คอมเมนต์เดิมใหม่ด้วยสิ่งที่แก้ (${_corrected} ข้อ)</button>`
    : '';
  $('yout').innerHTML=`
    <div class="note" style="margin-top:14px">
      ดึงได้ <b>${s.n}</b> คอมเมนต์ · ระบบคิดว่าประชด <b>${s.sarcasm}</b> ข้อ · ${s.model}
      &nbsp; <button class="ghost" onclick="ytToggle(true)">โชว์เฉพาะประชด</button>
      <button class="ghost" onclick="ytToggle(false)">โชว์ทั้งหมด</button>
      <br><span style="color:var(--ink2)">เจอที่ตัดสินผิด? กด “ตัดสินผิด” ที่ข้อนั้น ระบบจะจำ<b>ถาวร (ข้ามเซสชัน)</b>
      เป็นตัวอย่าง (few-shot) แล้วเก่งขึ้นกับข้อความคล้ายๆ กัน · จำได้ไม่จำกัดจำนวน ·
      อยากฝังในโมเดลจริงถาวรใช้ <code>finetune.py</code> (ไม่ใช่การเทรนใหม่ในเว็บ)</span> ${relearn}
    </div>
    <div style="margin-top:10px">${list}</div>
    ${pager(rows.length,pages)}`;
}
function pager(total,pages){
  if(total<=YPP)return '';
  const lo=(_ypage-1)*YPP+1, hi=Math.min(_ypage*YPP,total);
  const prev=_ypage>1?`<button class="ghost" onclick="ytPage(${_ypage-1})">‹ ก่อนหน้า</button>`
    :`<button class="ghost" disabled style="opacity:.4">‹ ก่อนหน้า</button>`;
  const next=_ypage<pages?`<button class="ghost" onclick="ytPage(${_ypage+1})">ถัดไป ›</button>`
    :`<button class="ghost" disabled style="opacity:.4">ถัดไป ›</button>`;
  return `<div class="row" style="justify-content:center;margin-top:14px;align-items:center">
    ${prev}<span class="sub" style="margin:0 4px">${lo}–${hi} จาก ${total} · หน้า ${_ypage}/${pages}</span>${next}</div>`;
}
async function reanalyzeYT(){
  const texts=_yall.map(r=>r.text);
  $('yout').innerHTML='<div class="note" style="margin-top:14px"><span class="sp" style="border-color:#888;border-top-color:transparent"></span>วิเคราะห์คอมเมนต์เดิมใหม่ด้วยสิ่งที่แก้...</div>';
  try{
    const r=await fetch('/api/batch',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({texts,op:$('bop').value})});
    const d=await r.json();
    if(d.error){$('yout').innerHTML='<div class="warn">'+d.error+'</div>';return}
    _yall=d.rows.map(r=>({text:r.text,decision:r.decision,prob:r.prob}));
    _ysum={n:d.summary.n,sarcasm:d.summary.sarcasm,model:d.summary.model};
    _yonly=true; _ypage=1; renderYT();
  }catch(e){ $('yout').innerHTML='<div class="warn">ผิดพลาด: '+e+'</div>' }
}
$('yurl').addEventListener('keydown',e=>{if(e.key==='Enter')runYT()});

// ---------- batch ----------
let _brows=[];
function parseCSV(txt){
  // แยกบรรทัด ตัด header ถ้ามีคอลัมน์ "text" -> เอาคอลัมน์นั้น ไม่งั้นเอาทั้งบรรทัด
  const lines=txt.split(/\r?\n/).filter(l=>l.trim());
  if(!lines.length) return [];
  const head=lines[0].split(',').map(s=>s.trim().toLowerCase());
  const ti=head.indexOf('text');
  if(ti>=0){
    return lines.slice(1).map(l=>{
      const m=l.match(/("([^"]|"")*"|[^,]*)(,|$)/g)||[];
      let c=(m[ti]||'').replace(/,$/,'').trim();
      if(c.startsWith('"')&&c.endsWith('"')) c=c.slice(1,-1).replace(/""/g,'"');
      return c;
    }).filter(Boolean);
  }
  return lines;   // ไม่มี header text -> ทีละบรรทัด
}
$('bfile').addEventListener('change',async e=>{
  const f=e.target.files[0]; if(!f)return;
  const txt=await f.text();
  $('btext').value=parseCSV(txt).join('\n');
  $('bhint').textContent=parseCSV(txt).length+' ข้อจากไฟล์ พร้อมตรวจ';
});
async function runBatch(){
  const texts=$('btext').value.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
  if(!texts.length){$('btext').focus();return}
  $('bgo').disabled=true; $('bgo').innerHTML='<span class="sp"></span>กำลังตรวจ '+texts.length+' ข้อ...';
  $('bout').innerHTML='';
  try{
    const r=await fetch('/api/batch',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({texts,op:$('bop').value,review_band:$('brev').checked})});
    const d=await r.json();
    if(d.error){$('bout').innerHTML='<div class="warn">'+d.error+'</div>'}
    else{ _brows=d.rows; renderBatch(d); }
  }catch(e){ $('bout').innerHTML='<div class="warn">เรียก API ไม่สำเร็จ: '+e+'</div>' }
  $('bgo').disabled=false; $('bgo').textContent='ตรวจทั้งหมด';
}
function pill(dec){
  const m={sarcasm:['ประชด','v1'],not_sarcasm:['ไม่ประชด','v0'],review:['ยกให้คน','vna']};
  const x=m[dec]||[dec,'vna']; return '<span class="pill '+x[1]+'">'+x[0]+'</span>';
}
function renderBatch(d){
  const s=d.summary;
  let rows=_brows.map((r,i)=>{
    const g=r.in_gold?' <span title="อยู่ใน gold โมเดลอาจเคยเห็น" style="color:#a06a00">(gold)</span>':'';
    return `<tr><td>${i+1}</td><td style="text-align:left;max-width:420px">${esc(r.text)}${g}</td>
      <td>${r.prob==null?'–':r.prob}</td><td>${pill(r.decision)}</td></tr>`;
  }).join('');
  $('bout').innerHTML=`
    <div class="note" style="margin-top:14px">
      <b>${s.n} ข้อ</b> · ${s.model} (${s.op}) · ประชด <b>${s.sarcasm}</b> · ไม่ประชด ${s.not}
      ${s.review?'· ยกให้คน '+s.review:''} · จาก cache ${s.cached} (ฟรี)
      &nbsp; <button class="ghost" onclick="dlCSV()">ดาวน์โหลด CSV</button>
    </div>
    <div style="overflow-x:auto;margin-top:10px"><table>
      <tr><th>#</th><th style="text-align:left">ข้อความ</th><th>P(ประชด)</th><th>ผล</th></tr>
      ${rows}</table></div>`;
}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function dlCSV(){
  const q=s=>'"'+String(s==null?'':s).replace(/"/g,'""')+'"';
  const head='text,pred_prob,pred_label,pred_decision,in_gold\n';
  const body=_brows.map(r=>[r.text,r.prob,r.label,r.decision,r.in_gold].map(q).join(',')).join('\n');
  const blob=new Blob(['﻿'+head+body],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='sarcasm_predictions.csv'; a.click(); URL.revokeObjectURL(a.href);
}
</script>
</div></body></html>
"""


if __name__ == "__main__":
    print("=" * 60)
    print("เว็บทดลอง 3 ระบบตรวจจับประชด")
    print(f"  OPENAI_API_KEY : {'พบ' if os.environ.get('OPENAI_API_KEY') else 'ไม่พบ (LLM จะรันไม่ได้)'}")
    print(f"  WangchanBERTa  : {'พร้อม' if has_wcb() else 'ยังไม่ได้เทรน (รัน train_final_wcb.py)'}")
    print(f"  โควตาผู้ใช้รีโมต : {IP_HOUR_CAP} ข้อ/ชม./ไอพี · {DAILY_CAP} ข้อ/วันรวม (127.0.0.1 ไม่จำกัด)")
    print(f"                   ปรับได้: PUBLIC_IP_HOURLY_LIMIT, PUBLIC_DAILY_LIMIT")
    # on deploy: the host sets HOST=0.0.0.0 + PORT as the host dictates (e.g. HF Spaces = 7860)
    # running locally: default 127.0.0.1:5000 (safe, not exposed to the internet)
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"  หน้าผู้ใช้: http://{host}:{port}/app   ·   หน้า dev: http://{host}:{port}/")
    print("=" * 60)
    app.run(debug=False, host=host, port=port)
