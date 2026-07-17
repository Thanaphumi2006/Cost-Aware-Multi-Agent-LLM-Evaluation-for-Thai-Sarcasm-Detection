# -*- coding: utf-8 -*-
"""เว็บทดลอง + เทียบ 3 ระบบตรวจจับประชดภาษาไทย

รัน:
  set OPENAI_API_KEY=sk-...            (PowerShell: $env:OPENAI_API_KEY="sk-...")
  C:/Users/thana/pt/Scripts/python.exe app.py
  เปิด http://127.0.0.1:5000

หมายเหตุ:
- คีย์ API ใส่ได้ 2 ทาง: environment variable หรือ พิมพ์ในหน้าเว็บ (ช่องด้านบน)
  ทั้งสองทางเก็บไว้ใน RAM ของโปรเซสเท่านั้น -- ไม่เขียนลงไฟล์ ไม่ฝังในโค้ด ปิดเซิร์ฟเวอร์แล้วหาย
- เว็บนี้ผูกกับ 127.0.0.1 อย่าเปิดออกสู่เน็ต เพราะช่องใส่คีย์จะกลายเป็นช่องให้คนอื่นยิง API ด้วยคีย์เรา
- ถ้าไม่มีคีย์ เว็บยังใช้ได้ แต่จะรันได้แค่ WangchanBERTa (ตัวที่ไม่ต้องใช้ API)
- WangchanBERTa ที่เว็บใช้เทรนบน gold ครบทุกข้อ -> ห้ามเอาไปวัดผลบน gold
  ถ้าพิมพ์ข้อความที่อยู่ใน gold มันจะ "จำคำตอบได้" (เว็บจะเตือนให้เอง)
"""
import os
import sys
import threading
import time

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

import baseline
import multiagent
import multiagent_debate
import multiagent_hybrid
from baseline import PRICE_PER_MTOK, metrics

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
WCB_DIR = os.path.join(HERE, "wcb_model")
IN_P, OUT_P = PRICE_PER_MTOK["gpt"]

app = Flask(__name__)

# ---------- โหลดของหนักครั้งเดียวตอนสตาร์ท ----------
_gold = pd.read_csv(os.path.join(HERE, "gold.csv"), dtype=str).fillna("")
_gold["label"] = _gold["label"].str.strip()
_gold = _gold[_gold["label"].isin(["0", "1"])].reset_index(drop=True)
GOLD_TEXTS = dict(zip(_gold["text"], _gold["label"]))

_wcb = None       # โหลดแบบ lazy (torch หนัก)
_client = None
_api_key = os.environ.get("OPENAI_API_KEY", "").strip()   # อยู่ใน RAM เท่านั้น ไม่เขียนลงดิสก์


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
    """sk-proj-abcd...wxyz -- พอให้รู้ว่าเป็นคีย์ตัวไหน แต่ไม่เผยคีย์"""
    return f"{k[:6]}…{k[-4:]}" if len(k) > 14 else "sk-…"


def has_wcb():
    return os.path.isdir(WCB_DIR) and os.path.exists(os.path.join(WCB_DIR, "config.json"))


def cost(i, o):
    return i / 1e6 * IN_P + o / 1e6 * OUT_P


# ---------- ตารางผลจริงบน gold (คำนวณจาก CSV จริง ไม่ใช่เลข hardcode) ----------
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


# ---------- ทำนาย ----------
def run_baseline(text):
    c = client()
    if not c:
        return {"pred": "n/a", "note": "ไม่มี OPENAI_API_KEY"}
    r = baseline.predict_one(c, text)
    return {"pred": r["pred"], "latency_ms": r["latency_ms"], "calls": 1,
            "in_tok": r["in_tok"], "out_tok": r["out_tok"],
            "cost": round(cost(r["in_tok"], r["out_tok"]), 6), "err": r.get("err", "")}


def run_multiagent(text):
    """รันทีละด่านเอง (แทนที่จะเรียก run_pipeline รวดเดียว) เพื่อเก็บ token/เวลา/คำตัดสิน "แยกรายด่าน"
    ตรรกะต้องเหมือน multiagent.run_pipeline ทุกประการ -- ใช้ prompt/schema ตัวเดียวกันจากไฟล์นั้นตรงๆ"""
    c = client()
    if not c:
        return {"pred": "n/a", "note": "ไม่มี OPENAI_API_KEY"}

    steps = []
    t0 = time.perf_counter()
    det_sys = multiagent.DETECT_SYS + _corrections_block(text)   # สอน multi-agent ด้วยที่คนแก้ (few-shot)
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
    """สถาปัตยกรรมที่ 2: อัยการ + ทนาย -> ผู้พิพากษา (ตัดสินใหม่ได้ทั้งสองทาง)"""
    c = client()
    if not c:
        return {"pred": "n/a", "note": "ไม่มี OPENAI_API_KEY"}
    r = multiagent_debate.run_debate(c, text)
    return {"pred": r["pred"], "latency_ms": r["latency_ms"], "calls": r["calls"],
            "in_tok": r["in_tok"], "out_tok": r["out_tok"],
            "cost": round(cost(r["in_tok"], r["out_tok"]), 6),
            "pros": r["pros"], "defe": r["defe"], "judge": r["judge"], "err": r["err"]}


def run_hybrid(text):
    """ระบบรวม: detector -> (อัยการ vs ทนาย -> ผู้พิพากษาที่ปัดตกได้อย่างเดียว)"""
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
        "gold": GOLD_TEXTS.get(text),          # None ถ้าไม่ได้อยู่ใน gold
        "in_gold": in_gold,
        "baseline": run_baseline(text),
        "multiagent": run_multiagent(text),
        "wangchanberta": run_wcb(text),
    })
    # debate/hybrid ถูกถอดออกจากเว็บแล้ว (ทดลองแล้วแพ้ -- เก็บโค้ด+ผลไว้เป็นหลักฐานใน RESULTS.md)
    # run_debate()/run_hybrid() ยังอยู่ ถ้าอยากเปิดกลับมาก็เสียบกลับได้ทันที


_detectors = {}       # cache detector ต่อ operating point (โหลด client ครั้งเดียว)


def detector(op):
    """ตัวตรวจจับพร้อมใช้จริงจาก predict.py -- ใช้คีย์ที่ผู้ใช้ใส่ในหน้าเว็บ + cache ร่วมกัน"""
    import predict
    key = (op, _api_key)
    if key not in _detectors:
        _detectors[key] = predict.SarcasmDetector(operating=op, api_key=_api_key)
    return _detectors[key]


# ---------- guardrails: กันคนอื่นเผาคีย์เจ้าของ ตอน deploy ให้คนอื่นใช้ ----------
# ปรับได้ด้วย env: PUBLIC_DAILY_LIMIT (ข้อ/วัน รวมทุกคน), PUBLIC_IP_HOURLY_LIMIT (ข้อ/ชม./ไอพี)
# เครื่องตัวเอง (127.0.0.1) ไม่ติดลิมิต -> เจ้าของใช้ได้ไม่จำกัด ลิมิตมีผลเฉพาะผู้ใช้รีโมต
DAILY_CAP = int(os.environ.get("PUBLIC_DAILY_LIMIT", "2000"))
IP_HOUR_CAP = int(os.environ.get("PUBLIC_IP_HOURLY_LIMIT", "200"))
_usage_lock = threading.Lock()
_usage = {"day": "", "day_count": 0, "ip_hour": {}}


def _guard(n_items, ip):
    """เช็ค+นับโควตา คืน error message ถ้าเกิน ไม่งั้น None (localhost ไม่จำกัด)"""
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


# ระบบที่หน้า /app เลือกได้ (แต่ละตัวมีมาสคอต)
MODELS_PUBLIC = ("balanced", "high_recall", "multiagent", "wangchanberta")
MODEL_LABEL = {"balanced": "gpt-4.1-mini", "high_recall": "gpt-4o",
               "multiagent": "multi-agent (2 ตัว)", "wangchanberta": "WangchanBERTa (ฟรี)"}


def _dec(pred):
    return "sarcasm" if pred == "1" else ("not_sarcasm" if pred == "0" else "error")


def _corrections_block(text):
    """few-shot ตัวอย่างที่คนแก้ ที่เกี่ยวกับข้อความนี้ -- เอาไปต่อท้าย prompt ของ LLM ทุกตัว"""
    import predict
    corr = predict.load_corrections()
    return predict._shots_block(predict._relevant(corr, text)) if corr else ""


def _corr_map():
    """dict {ข้อความ: ป้ายที่คนแก้} สำหรับ override แบบตรงตัว (ใช้ได้กับทุกโมเดล รวม WangchanBERTa)"""
    import predict
    return {c["text"]: c["label"] for c in predict.load_corrections()}


def _classify(model, text, review=False):
    """ตรวจ 1 ข้อความด้วยระบบที่เลือก -> row มาตรฐาน (multiagent แถม steps ไว้ให้ animate)"""
    if model == "multiagent":
        r = run_multiagent(text)
        p = r.get("pred")
        return {"text": text, "label": p if p in ("0", "1") else None, "prob": None,
                "decision": _dec(p), "steps": r.get("steps"), "note": r.get("note")}
    if model == "wangchanberta":
        r = run_wcb(text)
        p = r.get("pred")
        conf = r.get("conf")                                   # ความมั่นใจของคลาสที่ทาย
        psarc = conf if p == "1" else (1 - conf if p == "0" and conf is not None else None)
        return {"text": text, "label": p if p in ("0", "1") else None, "prob": psarc,
                "decision": _dec(p), "note": r.get("note")}
    r = detector(model).predict(text, review_band=review)     # balanced / high_recall
    return {"text": text, **r}


def _need_check(model):
    """คืน error message ถ้าใช้ระบบนี้ไม่ได้ตอนนี้ (คีย์/โมเดล) ไม่งั้น None"""
    if model not in MODELS_PUBLIC:
        return f"ไม่รู้จักระบบ: {model}"
    if model != "wangchanberta" and not _api_key:
        return "ยังไม่มี OPENAI_API_KEY (ใส่คีย์ด้านบนก่อน)"
    if model == "wangchanberta" and not has_wcb():
        return "ยังไม่มีโมเดล WangchanBERTa (รัน train_final_wcb.py)"
    return None


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """ตรวจหลายข้อความรวดเดียว -- เลือกระบบได้ (balanced/high_recall/multiagent/wangchanberta)"""
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
        if t in corr:                                  # เคยแก้ข้อนี้ตรงตัว -> ใช้ป้ายคน (ทุกโมเดล)
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


@app.route("/api/youtube", methods=["POST"])
def api_youtube():
    """วางลิงก์ YouTube -> ดึงคอมเมนต์ไทย -> จับประชด -> คืนรายการ (โชว์เฉพาะที่ประชด)
    *** โดเมน YouTube ยังไม่ได้ validate (ดู eval_domain.py) -> ผลเป็น "เดา" เตือนที่หน้าเว็บ ***"""
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
        # แพลตฟอร์มดึงอัตโนมัติไม่ได้ (Twitter/IG/ฯลฯ ต้องล็อกอิน/API) -> บอกให้วางเอง
        return jsonify({"error": f"ดึงจาก {plat} อัตโนมัติไม่ได้ (แพลตฟอร์มนี้ต้องล็อกอิน/เสียเงิน API) "
                                 f"ก๊อปคอมเมนต์มาวางในแท็บ “อัปโหลดไฟล์” แทน (ใช้ได้กับทุกแพลตฟอร์ม)",
                        "paste_hint": True}), 422
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
        if c in corr:                                  # เคยแก้ข้อนี้ตรงตัว -> ใช้ป้ายคน (ทุกโมเดล)
            lab = corr[c]
            rows.append({"text": c, "label": lab, "prob": 1.0 if lab == "1" else 0.0,
                         "decision": _dec(lab), "from_correction": True})
            continue
        try:
            rows.append(_classify(model, c))
        except Exception:
            rows.append({"text": c, "label": None, "prob": None, "decision": "error"})
    rows.sort(key=lambda r: (r.get("decision") != "sarcasm", -(r.get("prob") or 0)))   # ประชดขึ้นก่อน
    summ = {"n": len(rows), "sarcasm": sum(1 for r in rows if r["decision"] == "sarcasm"),
            "model": MODEL_LABEL.get(model, model), "op": model, "platform": plat}
    return jsonify({"rows": rows, "summary": summ})


@app.route("/api/correct", methods=["POST"])
def api_correct():
    """คนกดว่า 'โมเดลตัดสินผิด' -> เก็บคำตอบที่ถูก แล้วเอาไปเป็น few-shot ให้ครั้งต่อไป
    หมายเหตุซื่อสัตย์: นี่คือ in-context learning (สอนผ่านตัวอย่างในโปรมป์) ไม่ใช่การเทรนโมเดลใหม่จริง"""
    import predict
    body = request.json or {}
    text = str(body.get("text", "")).strip()
    label = str(body.get("label", "")).strip()          # ป้ายที่ "ถูก" (ตรงข้ามกับที่โมเดลทาย)
    if not text or label not in ("0", "1"):
        return jsonify({"error": "ต้องมี text และ label ('0'/'1')"}), 400
    try:
        n = predict.add_correction(text, label)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    for det in _detectors.values():                     # ให้ detector ที่โหลดไว้ใช้ตัวอย่างใหม่ทันที
        det.reload_corrections()
    return jsonify({"ok": True, "total": n})


@app.route("/api/key", methods=["POST"])
def api_key_set():
    """รับคีย์จากหน้าเว็บ -> ยิงเช็คกับ OpenAI จริงก่อนรับ (models.list ไม่คิดเงิน)
    เก็บไว้ใน RAM ของโปรเซสเท่านั้น -- ปิดเซิร์ฟเวอร์แล้วหาย ไม่มีการเขียนลงไฟล์"""
    global _api_key, _client
    key = (request.json or {}).get("key", "").strip()
    if not key:
        return jsonify({"ok": False, "error": "ยังไม่ได้ใส่คีย์"}), 400
    if not key.startswith("sk-"):
        return jsonify({"ok": False, "error": "รูปแบบคีย์ไม่ถูก -- ต้องขึ้นต้นด้วย sk-"}), 400

    from openai import OpenAI
    try:
        OpenAI(api_key=key).models.list()          # ตรวจว่าคีย์ใช้ได้จริง
    except Exception as e:
        return jsonify({"ok": False, "error": f"คีย์ใช้ไม่ได้: {type(e).__name__}"}), 400

    _api_key, _client = key, None                  # ล้าง client เก่า -> สร้างใหม่ด้วยคีย์นี้
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
    """จำนวนที่โมเดลเรียนจากทุกคนรวมกัน (เก็บถาวรในไฟล์เดียวบนเซิร์ฟเวอร์ -> แชร์ทุกคน)"""
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
    """หน้าสำหรับผู้ใช้ทั่วไป: สะอาด ง่าย ผลลัพธ์เดียวชัดๆ (ไม่มีของวิจัย)
    ใช้ backend ตัวเดียวกับหน้า / (predict.py + endpoints เดิม)"""
    return render_template_string(PUBLIC_PAGE, has_key=bool(_api_key))


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


# ================= หน้าสำหรับผู้ใช้ทั่วไป (/app) สะอาด ผลลัพธ์เดียว =================
PUBLIC_PAGE = r"""
<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ประชดหรือเปล่า?</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Itim&family=Mali:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#fbf4e4; --card:#fffdf7; --ink:#34302a; --ink2:#6b6357; --dot:#ece0c6;
  --blue:#4a86e8; --yellow:#ffd84d; --yellow-d:#f2c33a;
  --sar:#e2593f; --sar-bg:#ffe7e1; --not:#38a05d; --not-bg:#e3f4e9;
  --disp:'Itim','Comic Sans MS',cursive; --body:'Mali','Comic Sans MS',cursive;
}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);font-family:var(--body);line-height:1.6;-webkit-font-smoothing:antialiased;
  background-color:var(--paper);
  background-image:radial-gradient(var(--dot) 1.6px,transparent 1.6px);background-size:24px 24px}
.wrap{max-width:640px;margin:0 auto;padding:clamp(26px,7vw,58px) clamp(16px,4vw,24px) 90px}
/* ---- doodle primitives ---- */
.box{background:var(--card);border:2.6px solid var(--ink);
  border-radius:255px 14px 225px 16px/16px 225px 15px 255px;
  box-shadow:5px 5px 0 var(--ink)}
.box.alt{border-radius:14px 235px 16px 225px/220px 15px 235px 16px}
h1{font-family:var(--disp);font-size:clamp(38px,10vw,62px);margin:0;text-align:center;line-height:1;
  color:var(--ink);letter-spacing:.5px}
.squiggle{display:block;width:min(300px,72%);height:20px;margin:2px auto 0}
.tag{font-family:var(--body);color:var(--ink2);text-align:center;margin:14px auto 0;max-width:34ch;
  font-size:clamp(15px,3.4vw,18px)}
.shared{font-family:var(--body);font-size:13px;color:var(--ink);text-align:center;margin:12px auto 0;
  max-width:fit-content;background:var(--yellow);border:2.2px solid var(--ink);box-shadow:2px 2px 0 var(--ink);
  border-radius:14px 8px 14px 8px;padding:6px 14px;display:none}
.shared.on{display:block}
.spark{position:absolute;pointer-events:none}
.head{position:relative}
.pick-title{font-family:var(--disp);font-size:clamp(19px,4.5vw,23px);text-align:center;color:var(--ink);margin-top:34px}
.models{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.model{background:var(--card);cursor:pointer;text-align:center;font-family:var(--body);
  border:2.6px solid var(--ink);border-radius:225px 16px 255px 14px/14px 255px 16px 225px;
  box-shadow:4px 4px 0 var(--ink);padding:16px 10px 13px;transition:transform .09s,box-shadow .09s,border-color .09s}
.model:nth-child(2){border-radius:16px 225px 14px 255px/255px 14px 225px 16px}
.model .mascot{width:66px;height:66px}
.model .mname{font-family:var(--disp);font-size:20px;margin-top:5px;color:var(--ink);line-height:1.1}
.model .mdesc{font-size:12.5px;color:var(--ink2);margin-top:2px}
.model:not(.sel):hover{background:#fffdf0;transform:translateY(-2px)}
.model.sel{border-color:var(--blue);box-shadow:5px 6px 0 var(--blue);transform:translateY(-3px) rotate(-1deg)}
.model .tick{font-family:var(--body);font-weight:700;font-size:12px;color:var(--blue);height:14px;margin-top:5px}
.panel{padding:clamp(20px,4.5vw,30px);margin-top:20px}
/* ---- multi-agent workflow animation ---- */
.flowbox{margin-top:16px;padding:20px 12px;text-align:center}
.flowrow{display:flex;align-items:center;justify-content:center;gap:0;flex-wrap:nowrap}
.node{flex:none;width:104px;padding:12px 8px;background:var(--card);border:2.6px solid var(--ink);
  border-radius:20px 10px 22px 10px/10px 22px 10px 20px;box-shadow:3px 3px 0 var(--ink)}
.node .nface{width:40px;height:40px}
.node .nname{font-family:var(--disp);font-size:14px;color:var(--ink);line-height:1.15;margin-top:3px}
.node .nsay{font-family:var(--body);font-size:12px;margin-top:4px;min-height:16px;color:var(--ink2)}
.node.work{border-color:var(--blue);box-shadow:3px 3px 0 var(--blue);animation:bob .6s ease-in-out infinite}
.node.done1{border-color:var(--sar);box-shadow:3px 3px 0 var(--sar)} .node.done1 .nsay{color:var(--sar);font-weight:700}
.node.done0{border-color:var(--not);box-shadow:3px 3px 0 var(--not)} .node.done0 .nsay{color:var(--not);font-weight:700}
.node.skip{opacity:.5;border-style:dashed}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
.wire{position:relative;flex:none;width:46px;height:26px}
.wire svg{position:absolute;inset:0;width:100%;height:100%}
.packet{position:absolute;top:7px;left:0;width:11px;height:11px;border-radius:50%;background:var(--yellow);
  border:2px solid var(--ink)}
.wire.run .packet{animation:fly 1s linear infinite}
@keyframes fly{0%{left:-2px;opacity:0}15%{opacity:1}85%{opacity:1}100%{left:38px;opacity:0}}
.flowcap{font-family:var(--body);font-size:13px;color:var(--ink2);margin-top:14px;min-height:18px}
.miniflow{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:9px;font-family:var(--body);font-size:12.5px}
.chip{padding:3px 9px;border:2px solid var(--ink);border-radius:10px 6px 10px 6px;background:#fff}
.chip.s1{background:var(--sar-bg);color:var(--sar)} .chip.s0{background:var(--not-bg);color:var(--not)}
.chip.skip{opacity:.55;border-style:dashed}
.arrowc{color:var(--ink2);font-weight:700}
label,.hint{font-family:var(--body)}
textarea,input[type=text],input[type=password]{width:100%;padding:15px;font-family:var(--body);font-size:16px;
  color:var(--ink);background:#fffef9;border:2.4px dashed var(--ink);border-radius:18px 10px 20px 10px/10px 20px 10px 18px}
textarea{min-height:110px;resize:vertical}
input:focus,textarea:focus{outline:none;border-style:solid;border-color:var(--blue);box-shadow:3px 3px 0 var(--blue)}
.hint{font-size:13.5px;color:var(--ink2);margin:9px 4px 0}
.actions{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:16px}
button{font-family:var(--disp);cursor:pointer;border:2.6px solid var(--ink);color:var(--ink);
  padding:12px 24px;font-size:18px;border-radius:16px 9px 18px 9px/9px 18px 9px 16px;
  box-shadow:3px 3px 0 var(--ink);transition:transform .05s,box-shadow .05s}
button:active{transform:translate(3px,3px);box-shadow:0 0 0 var(--ink)}
.go{background:var(--yellow);flex:1;min-width:150px} .go:hover{background:var(--yellow-d)}
.go:disabled{opacity:.6;cursor:wait}
.file{font-family:var(--body);font-size:13px;color:var(--ink2)}
.file::file-selector-button{font-family:var(--body);border:2.2px solid var(--ink);background:#fff;
  border-radius:12px 7px 12px 7px;padding:7px 12px;cursor:pointer;box-shadow:2px 2px 0 var(--ink);margin-right:8px}
/* ---- results ---- */
.result{margin-top:14px;padding:18px 20px;background:var(--card);border:2.6px solid var(--ink);
  border-radius:225px 16px 255px 14px/14px 255px 16px 225px;box-shadow:4px 4px 0 var(--ink)}
.result:nth-child(even){border-radius:16px 225px 14px 255px/255px 14px 225px 16px;transform:rotate(-.5deg)}
.result:nth-child(odd){transform:rotate(.4deg)}
.verdict{display:flex;align-items:center;gap:11px;font-family:var(--disp);font-size:26px}
.dot{width:16px;height:16px;border-radius:50%;border:2.4px solid var(--ink);flex:none}
.sar .dot{background:var(--sar)} .notx .dot{background:var(--not)}
.sar .verdict,.sar.verdict{color:var(--sar)} .notx .verdict,.notx.verdict{color:var(--not)}
.txt{font-size:16px;color:var(--ink);margin:10px 0 0}
.meter{height:12px;border-radius:8px;background:#fff;border:2.2px solid var(--ink);overflow:hidden;margin-top:14px}
.meter>span{display:block;height:100%}
.conf{font-family:var(--body);font-size:13px;color:var(--ink2);margin-top:7px;display:flex;justify-content:space-between}
.rowline{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:10px}
.pill{font-family:var(--disp);display:inline-block;padding:5px 15px;font-size:15px;border:2.4px solid var(--ink);
  border-radius:16px 8px 16px 8px/8px 16px 8px 16px;box-shadow:2px 2px 0 var(--ink)}
.pill.sar{background:var(--sar-bg);color:var(--sar)} .pill.notx{background:var(--not-bg);color:var(--not)}
.wrongbtn{font-family:var(--body);font-weight:600;padding:6px 13px;font-size:13px;background:#fff;color:var(--ink);
  border:2.2px solid var(--ink);border-radius:12px 7px 12px 7px;box-shadow:2px 2px 0 var(--ink);cursor:pointer}
.wrongbtn:active{transform:translate(2px,2px);box-shadow:0 0 0 var(--ink)}
.learned{font-family:var(--body);color:var(--not);font-weight:700;font-size:13.5px}
.fbtns{display:inline-flex;gap:6px}
.fbtn{font-family:var(--body);font-weight:700;font-size:12px;padding:5px 13px;border:2.2px solid var(--ink);
  border-radius:11px 6px 11px 6px;box-shadow:2px 2px 0 var(--ink);cursor:pointer}
.fbtn.yes{background:var(--not-bg);color:var(--not)} .fbtn.no{background:var(--sar-bg);color:var(--sar)}
.fbtn:active{transform:translate(2px,2px);box-shadow:0 0 0 var(--ink)}
.warn{font-family:var(--body);font-size:14px;color:#8a3320;background:var(--sar-bg);border:2.4px solid var(--sar);
  border-radius:16px 10px 16px 10px;padding:13px 15px;margin-top:14px;box-shadow:3px 3px 0 var(--sar)}
.ok{font-family:var(--body);font-size:14px;color:var(--not);background:var(--not-bg);border:2.4px solid var(--not);
  border-radius:16px 10px 16px 10px;padding:13px 15px;margin-top:14px;box-shadow:3px 3px 0 var(--not)}
.keybar{padding:16px 18px;margin-top:24px}
.keybar .actions{margin-top:12px}
.keybar button{font-size:16px;padding:10px 18px}
.pagerow{display:flex;justify-content:center;gap:12px;align-items:center;margin-top:20px}
.pagerow button{background:#fff;font-size:15px;padding:9px 16px}
.pagerow button:disabled{opacity:.4;cursor:default;box-shadow:1px 1px 0 var(--ink)}
.sp{display:inline-block;width:15px;height:15px;border:2.4px solid var(--ink);border-top-color:transparent;
  border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:8px}
@keyframes s{to{transform:rotate(360deg)}}
.foot{font-family:var(--body);text-align:center;color:var(--ink2);font-size:13px;margin-top:36px}
.note{font-family:var(--body);font-size:13.5px;color:var(--ink2);text-align:center;margin-top:18px;
  max-width:40ch;margin-left:auto;margin-right:auto}
/* ---- research panel (toggle) ---- */
.rbtn{display:block;margin:28px auto 0;background:#fff}
.research{display:none;margin-top:16px;padding:clamp(18px,4vw,26px)}
.research.show{display:block}
.rh{font-family:var(--disp);font-size:20px;color:var(--ink);margin:20px 0 5px}
.rh:first-child{margin-top:0}
.rp{font-family:var(--body);font-size:14px;color:var(--ink);text-wrap:pretty}
.rtable{width:100%;border-collapse:collapse;font-family:var(--body);font-size:13.5px;margin-top:4px}
.rtable th,.rtable td{padding:8px 9px;border-bottom:2px dashed var(--ink);text-align:right}
.rtable th:first-child,.rtable td:first-child{text-align:left}
.rtable th{font-family:var(--disp);font-size:15px;color:var(--ink)}
.rtable tr.hi td{background:var(--yellow)}
.rlist{margin:12px 0 0;padding:0;list-style:none;display:grid;gap:10px}
.rlist li{font-family:var(--body);font-size:14px;padding-left:23px;position:relative;color:var(--ink);text-wrap:pretty}
.rlist li::before{content:"";position:absolute;left:2px;top:8px;width:9px;height:9px;border-radius:50%;
  background:var(--blue);border:2px solid var(--ink)}
.rlist.warn-list li::before{background:var(--sar)}
.rsrc{font-family:var(--body);font-size:13px;color:var(--ink2);margin-top:18px;text-align:center}
/* visual bars */
.bars{margin-top:6px}
.brow{display:flex;align-items:center;gap:9px;margin-top:10px}
.bname{flex:0 0 40%;font-family:var(--body);font-size:12.5px;text-align:right;line-height:1.15}
.btrack{flex:1;height:21px;background:#fff;border:2.4px solid var(--ink);border-radius:11px 6px 11px 6px;overflow:hidden}
.bfill{height:100%;border-right:2.4px solid var(--ink)}
.bval{flex:0 0 auto;font-family:var(--disp);font-size:14px;min-width:40px;text-align:left}
.bscale{font-family:var(--body);font-size:11px;color:var(--muted,#93a0b0);text-align:right;margin-top:6px}
/* versus hero */
.vs{display:flex;align-items:stretch;gap:9px;margin-top:8px}
.vscard{flex:1;background:#fff;border:2.6px solid var(--ink);box-shadow:3px 3px 0 var(--ink);
  border-radius:20px 10px 22px 10px/10px 22px 10px 20px;padding:14px 8px;text-align:center}
.vscard.win{background:var(--yellow)}
.vmasc{width:46px;height:46px}
.vsname{font-family:var(--disp);font-size:15px;margin-top:2px;line-height:1.1}
.vsf1{font-family:var(--disp);font-size:24px;margin-top:5px}
.vscost{font-family:var(--body);font-size:12px;color:var(--ink2);margin-top:2px}
.vscard.win .vscost{color:#8a6d1a;font-weight:700}
.vsmid{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:0 0 auto;padding:0 2px}
.vseq{font-family:var(--disp);font-size:28px;color:var(--ink);line-height:1}
.vseqt{font-family:var(--body);font-size:10.5px;color:var(--ink2);max-width:58px;line-height:1.2;margin-top:2px}
</style></head><body><div class="wrap">

<div class="head">
  <svg class="spark" style="left:6%;top:-6px;width:34px;height:34px" viewBox="0 0 40 40"><path d="M20 3 L23 16 L36 20 L23 24 L20 37 L17 24 L4 20 L17 16 Z" fill="none" stroke="#f2c33a" stroke-width="2.6" stroke-linejoin="round"/></svg>
  <svg class="spark" style="right:8%;top:18px;width:26px;height:26px" viewBox="0 0 40 40"><path d="M20 5 L23 17 L35 20 L23 23 L20 35 L17 23 L5 20 L17 17 Z" fill="#ffd84d" stroke="#34302a" stroke-width="2.2" stroke-linejoin="round"/></svg>
  <h1>ประชดหรือเปล่า?</h1>
  <svg class="squiggle" viewBox="0 0 300 20" preserveAspectRatio="none"><path d="M3 12 Q 38 3 74 12 T 148 12 T 222 12 T 297 10" fill="none" stroke="#ffd84d" stroke-width="7" stroke-linecap="round"/></svg>
  <p class="tag">วางข้อความไทย หรือลิงก์ (YouTube · Pantip · Reddit) แล้วมาดูกันว่า “ประชด” ไหม</p>
  <div class="shared" id="shared"></div>
</div>

{% if not has_key %}
<div class="box keybar alt">
  <div class="hint" style="margin:0;font-size:15px">ใส่ OpenAI API key ครั้งเดียวเพื่อเริ่มเล่น</div>
  <div class="actions">
    <input type="password" id="k" placeholder="sk-..." autocomplete="off" style="flex:1">
    <button id="ksave" onclick="saveKey()" style="background:#fff">บันทึก</button>
  </div>
  <div id="kmsg"></div>
</div>
{% endif %}

<div class="pick-title">เลือกผู้ช่วยของคุณ</div>
<div class="models">
  <button class="model sel" data-op="balanced" onclick="pickModel('balanced',this)">
    <svg class="mascot" viewBox="0 0 72 72">
      <path d="M20 22 L15 7 L31 18 Z" fill="#ffd84d" stroke="#34302a" stroke-width="2.4" stroke-linejoin="round"/>
      <path d="M52 22 L57 7 L41 18 Z" fill="#ffd84d" stroke="#34302a" stroke-width="2.4" stroke-linejoin="round"/>
      <circle cx="36" cy="39" r="22" fill="#ffd84d" stroke="#34302a" stroke-width="2.6"/>
      <circle cx="28" cy="37" r="3" fill="#34302a"/><circle cx="44" cy="37" r="3" fill="#34302a"/>
      <path d="M32 45 Q36 49 40 45" fill="none" stroke="#34302a" stroke-width="2.4" stroke-linecap="round"/>
      <path d="M8 39 L23 41 M10 45 L24 45" stroke="#34302a" stroke-width="2" stroke-linecap="round"/>
      <path d="M64 39 L49 41 M62 45 L48 45" stroke="#34302a" stroke-width="2" stroke-linecap="round"/>
    </svg>
    <div class="mname">น้องแมวไว</div>
    <div class="mdesc">เร็ว ประหยัด</div>
    <div class="tick">กำลังใช้</div>
  </button>
  <button class="model" data-op="high_recall" onclick="pickModel('high_recall',this)">
    <svg class="mascot" viewBox="0 0 72 72">
      <path d="M23 15 L18 5 L29 13 Z" fill="#bcd7f7" stroke="#34302a" stroke-width="2.2" stroke-linejoin="round"/>
      <path d="M49 15 L54 5 L43 13 Z" fill="#bcd7f7" stroke="#34302a" stroke-width="2.2" stroke-linejoin="round"/>
      <path d="M36 11 C55 11 58 30 56 44 C54 59 46 63 36 63 C26 63 18 59 16 44 C14 30 17 11 36 11 Z" fill="#d6e8fb" stroke="#34302a" stroke-width="2.6"/>
      <circle cx="27" cy="34" r="10" fill="#fff" stroke="#34302a" stroke-width="2.4"/>
      <circle cx="45" cy="34" r="10" fill="#fff" stroke="#34302a" stroke-width="2.4"/>
      <circle cx="27" cy="34" r="3.4" fill="#34302a"/><circle cx="45" cy="34" r="3.4" fill="#34302a"/>
      <path d="M36 34 L35.5 34" stroke="#34302a" stroke-width="2.4" stroke-linecap="round"/>
      <path d="M32 44 L36 51 L40 44 Z" fill="#ffd84d" stroke="#34302a" stroke-width="2.2" stroke-linejoin="round"/>
    </svg>
    <div class="mname">คุณนกฮูก</div>
    <div class="mdesc">ละเอียด จับครบ</div>
    <div class="tick"></div>
  </button>
  <button class="model" data-op="multiagent" onclick="pickModel('multiagent',this)">
    <svg class="mascot" viewBox="0 0 72 72">
      <circle cx="25" cy="35" r="15" fill="#ffd1a8" stroke="#34302a" stroke-width="2.4"/>
      <circle cx="21" cy="33" r="2.3" fill="#34302a"/><circle cx="29" cy="33" r="2.3" fill="#34302a"/>
      <path d="M21 40 Q25 43 29 40" fill="none" stroke="#34302a" stroke-width="2.2" stroke-linecap="round"/>
      <circle cx="12" cy="47" r="5" fill="none" stroke="#34302a" stroke-width="2.2"/><line x1="15.5" y1="50.5" x2="19" y2="54" stroke="#34302a" stroke-width="2.2" stroke-linecap="round"/>
      <circle cx="48" cy="35" r="15" fill="#a8e0c0" stroke="#34302a" stroke-width="2.4"/>
      <circle cx="44" cy="33" r="2.3" fill="#34302a"/><circle cx="52" cy="33" r="2.3" fill="#34302a"/>
      <path d="M44 40 Q48 43 52 40" fill="none" stroke="#34302a" stroke-width="2.2" stroke-linecap="round"/>
      <path d="M55 45 L59 49 L65 41" fill="none" stroke="#38a05d" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <div class="mname">คู่หูสองตรวจ</div>
    <div class="mdesc">คัดกรอง แล้วตรวจซ้ำ</div>
    <div class="tick"></div>
  </button>
  <button class="model" data-op="wangchanberta" onclick="pickModel('wangchanberta',this)">
    <svg class="mascot" viewBox="0 0 72 72">
      <line x1="36" y1="15" x2="36" y2="7" stroke="#34302a" stroke-width="2.4" stroke-linecap="round"/>
      <circle cx="36" cy="6" r="3.4" fill="#ffd84d" stroke="#34302a" stroke-width="2.2"/>
      <rect x="15" y="16" width="42" height="35" rx="11" fill="#cfe3f5" stroke="#34302a" stroke-width="2.6"/>
      <rect x="9" y="28" width="6" height="11" rx="2" fill="#cfe3f5" stroke="#34302a" stroke-width="2.2"/>
      <rect x="57" y="28" width="6" height="11" rx="2" fill="#cfe3f5" stroke="#34302a" stroke-width="2.2"/>
      <circle cx="28" cy="31" r="4.2" fill="#34302a"/><circle cx="44" cy="31" r="4.2" fill="#34302a"/>
      <rect x="26" y="40" width="20" height="5" rx="2.5" fill="none" stroke="#34302a" stroke-width="2.2"/>
    </svg>
    <div class="mname">น้องหุ่นไทย</div>
    <div class="mdesc">ฟรี ไม่ต้องต่อเน็ต</div>
    <div class="tick"></div>
  </button>
</div>

<div class="box panel">
  <textarea id="inp" placeholder="พิมพ์หรือวางข้อความตรงนี้ (หลายบรรทัดก็ได้) หรือวางลิงก์ YouTube / Pantip / Reddit"></textarea>
  <div class="hint">วางลิงก์ = ดึงคอมเมนต์มาตรวจให้ · หลายบรรทัด = ตรวจทีละบรรทัด</div>
  <div class="actions">
    <button class="go" id="go" onclick="analyze()">ตรวจเลย!</button>
    <input type="file" id="file" accept=".csv,.txt" class="file">
  </div>
  <div id="out"></div>
</div>

<p class="note">ผลลัพธ์เป็น “การเดา” ของ AI ไม่ใช่คำตัดสินสุดท้าย เห็นว่าผิดก็กด “ตัดสินผิด” ช่วยให้มันเก่งขึ้นได้</p>
<button class="rbtn" onclick="toggleResearch(this)">ดูเบื้องหลังงานวิจัย</button>
<div class="research box" id="research">
  <div class="rh">โปรเจกต์นี้ศึกษาอะไร</div>
  <div class="rp">คำถาม: ใช้ AI หลายตัวช่วยกัน (multi-agent) คุ้มกว่า AI ตัวเดียวไหม สำหรับงานตรวจจับประชดภาษาไทย
    วัด 4 อย่างพร้อมกัน คือ ความแม่น (F1), ราคา, เวลา, และจำนวนครั้งที่เรียก AI บนข้อมูลชุดเดียวกัน 127 ข้อ
    (เป็นประชดจริง 30 ข้อ) เทียบด้วยสถิติแบบ paired bootstrap และ McNemar</div>

  <div class="rh">หัวใจของงานวิจัย</div>
  <div class="rp" style="margin-bottom:8px">AI ตัวเดียวที่ถูกกว่า ทำได้พอๆ กับ AI 2 ตัวที่แพงกว่า</div>
  <div class="vs">
    <div class="vscard win">
      <svg class="vmasc" viewBox="0 0 72 72"><path d="M20 22 L15 7 L31 18 Z" fill="#ffd1a8" stroke="#34302a" stroke-width="2.4" stroke-linejoin="round"/><path d="M52 22 L57 7 L41 18 Z" fill="#ffd1a8" stroke="#34302a" stroke-width="2.4" stroke-linejoin="round"/><circle cx="36" cy="39" r="21" fill="#ffd1a8" stroke="#34302a" stroke-width="2.6"/><circle cx="28" cy="37" r="2.7" fill="#34302a"/><circle cx="44" cy="37" r="2.7" fill="#34302a"/><path d="M32 45 Q36 49 40 45" fill="none" stroke="#34302a" stroke-width="2.2" stroke-linecap="round"/></svg>
      <div class="vsname">น้องแมวไว</div>
      <div class="vsf1">0.727</div>
      <div class="vscost">$0.015 ถูกกว่า 11 เท่า</div>
    </div>
    <div class="vsmid"><div class="vseq">&asymp;</div><div class="vseqt">คุณภาพพอกัน</div></div>
    <div class="vscard">
      <svg class="vmasc" viewBox="0 0 72 72"><circle cx="26" cy="36" r="14" fill="#ffd1a8" stroke="#34302a" stroke-width="2.4"/><circle cx="22" cy="34" r="2.2" fill="#34302a"/><circle cx="30" cy="34" r="2.2" fill="#34302a"/><path d="M22 41 Q26 44 30 41" fill="none" stroke="#34302a" stroke-width="2" stroke-linecap="round"/><circle cx="47" cy="36" r="14" fill="#a8e0c0" stroke="#34302a" stroke-width="2.4"/><circle cx="43" cy="34" r="2.2" fill="#34302a"/><circle cx="51" cy="34" r="2.2" fill="#34302a"/><path d="M43 41 Q47 44 51 41" fill="none" stroke="#34302a" stroke-width="2" stroke-linecap="round"/></svg>
      <div class="vsname">AI 2 ตัว</div>
      <div class="vsf1">0.744</div>
      <div class="vscost">$0.169</div>
    </div>
  </div>

  <div class="rh">คะแนนความแม่น (F1) ของแต่ละระบบ</div>
  <div class="bars">
    <div class="brow"><div class="bname">WangchanBERTa (ฟรี)</div><div class="btrack"><div class="bfill" style="width:30%;background:#cfe3f5"></div></div><div class="bval">0.62</div></div>
    <div class="brow"><div class="bname">AI เดี่ยว</div><div class="btrack"><div class="bfill" style="width:61%;background:#ffd1a8"></div></div><div class="bval">0.69</div></div>
    <div class="brow"><div class="bname">Debate (3 ตัว)</div><div class="btrack"><div class="bfill" style="width:63%;background:#f4b6ab"></div></div><div class="bval">0.69</div></div>
    <div class="brow"><div class="bname">AI เดี่ยว + ปรับเกณฑ์</div><div class="btrack"><div class="bfill" style="width:76%;background:#ffd84d"></div></div><div class="bval">0.73</div></div>
    <div class="brow"><div class="bname">AI 2 ตัว (คู่หูสองตรวจ)</div><div class="btrack"><div class="bfill" style="width:84%;background:#a8e0c0"></div></div><div class="bval">0.74</div></div>
    <div class="bscale">ยิ่งแท่งยาว ยิ่งแม่น (แต่ต่างกันน้อยมาก เพราะข้อมูลมีแค่ 127 ข้อ)</div>
  </div>

  <div class="rh">สรุปสั้นๆ</div>
  <ul class="rlist">
    <li><b>เพิ่ม AI ไม่ได้ช่วยชัดเจน</b> AI 2-3 ตัวแพงกว่าหลายเท่า แต่ไม่ได้ชนะ AI เดี่ยวอย่างมีนัยสำคัญ</li>
    <li><b>ตัวที่สำคัญคือ "เลือกโมเดล"</b> ไม่ใช่ "จำนวน AI" โมเดลถูกทำได้พอๆ กับโมเดลแพง</li>
    <li><b>เพดานอยู่ที่ข้อมูล</b> ประชดมีแค่ 30 ข้อ เลยยังฟันธงกันไม่ได้เต็มที่</li>
  </ul>

  <div class="rh">ข้อควรระวัง</div>
  <ul class="rlist warn-list">
    <li>อย่าดู accuracy: ข้อมูลเอียง 76/24 เดาว่า "ไม่ประชด" ทุกข้อก็ได้ 0.76 แล้ว</li>
    <li>ประชดในชุดข้อมูลบางส่วนถูก AI ช่วยขุดมา ทำให้ recall อาจสูงเกินจริง</li>
    <li>วัดผลบนรีวิวร้าน + ทวีตเท่านั้น โดเมนอื่น (YouTube, ข่าว) ยังไม่ได้ทดสอบ</li>
  </ul>

  <div class="rsrc">รายละเอียดเต็ม พร้อมช่วงความเชื่อมั่นและสถิติ อยู่ใน Gold/RESULTS.md (finding 1 ถึง 11)</div>
</div>

<div class="foot">~ ตรวจจับประชดภาษาไทย ~</div>

<script>
const $=i=>document.getElementById(i);
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}

async function saveKey(){
  const key=$('k').value.trim(); if(!key){$('k').focus();return}
  $('ksave').disabled=true; $('ksave').innerHTML='<span class="sp"></span>';
  $('kmsg').innerHTML='';
  try{
    const r=await fetch('/api/key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})});
    const d=await r.json();
    if(d.ok){ $('kmsg').innerHTML='<div class="ok">พร้อมใช้งานแล้ว</div>'; setTimeout(()=>location.reload(),700); }
    else { $('kmsg').innerHTML='<div class="warn">'+d.error+'</div>'; $('ksave').disabled=false; $('ksave').textContent='บันทึก'; }
  }catch(e){ $('kmsg').innerHTML='<div class="warn">ต่อเซิร์ฟเวอร์ไม่ได้</div>'; $('ksave').disabled=false; $('ksave').textContent='บันทึก'; }
}

$('file').addEventListener('change',async e=>{
  const f=e.target.files[0]; if(!f)return;
  const t=await f.text();
  const lines=t.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
  // ตัด header ถ้ามีคอลัมน์ text
  if(lines[0]&&lines[0].toLowerCase().split(',').includes('text')) lines.shift();
  $('inp').value=lines.map(l=>l.replace(/^"|"$/g,'')).join('\n');
});

let _rows=[], _page=1; const PP=5;
let _op='balanced';
function pickModel(op,el){
  _op=op;
  document.querySelectorAll('.model').forEach(m=>{m.classList.remove('sel'); m.querySelector('.tick').textContent='';});
  el.classList.add('sel'); el.querySelector('.tick').textContent='กำลังใช้';
}
function isURL(s){return /^https?:\/\//i.test(s)}

// ---- multi-agent workflow animation (loading) ----
const F1='<svg class="nface" viewBox="0 0 40 40"><circle cx="20" cy="20" r="13" fill="#ffd1a8" stroke="#34302a" stroke-width="2.2"/><circle cx="16" cy="18" r="2" fill="#34302a"/><circle cx="24" cy="18" r="2" fill="#34302a"/><path d="M16 24 Q20 27 24 24" fill="none" stroke="#34302a" stroke-width="2" stroke-linecap="round"/></svg>';
const F2='<svg class="nface" viewBox="0 0 40 40"><circle cx="20" cy="20" r="13" fill="#a8e0c0" stroke="#34302a" stroke-width="2.2"/><circle cx="16" cy="18" r="2" fill="#34302a"/><circle cx="24" cy="18" r="2" fill="#34302a"/><path d="M16 24 Q20 27 24 24" fill="none" stroke="#34302a" stroke-width="2" stroke-linecap="round"/></svg>';
function node(name,face,cls,say){return `<div class="node ${cls||''}"><div>${face}</div><div class="nname">${name}</div><div class="nsay">${say||''}</div></div>`;}
function wire(run){return `<div class="wire ${run?'run':''}"><svg viewBox="0 0 46 26" preserveAspectRatio="none"><path d="M2 13 Q 23 4 44 13" fill="none" stroke="#34302a" stroke-width="2.4" stroke-dasharray="4 4" stroke-linecap="round"/></svg><div class="packet"></div></div>`;}
function showFlowLoading(){
  $('out').innerHTML=`<div class="flowbox box" style="box-shadow:none;border:none">
    <div class="flowrow">
      ${node('ผู้คัดกรอง',F1,'work','กำลังอ่าน...')}
      ${wire(true)}
      ${node('ผู้ตรวจซ้ำ',F2,'work','รอรับงาน')}
    </div>
    <div class="flowcap"><span class="sp" style="border-color:#34302a;border-top-color:transparent"></span>คู่หูสองเกลอกำลังช่วยกันตรวจ...</div>
  </div>`;
}
function miniflow(steps){
  if(!steps||steps.length<2)return '';
  const s1=steps[0], s2=steps[1];
  const t1=s1.said==='1'?'ประชด':'ไม่ประชด', c1=s1.said==='1'?'chip s1':'chip s0';
  let mid;
  if(!s2.ran){ mid='<span class="arrowc">›</span><span class="chip skip">ไม่ต้องตรวจซ้ำ</span>'; }
  else { const t2=s2.said==='0'?'ปัดตก':'ยืนยัน', c2=s2.said==='0'?'chip s0':'chip s1';
    mid='<span class="arrowc">›</span><span class="'+c2+'">ตรวจซ้ำ: '+t2+'</span>'; }
  return '<div class="miniflow"><span class="'+c1+'">คัดกรอง: '+t1+'</span>'+mid+'</div>';
}

async function analyze(){
  const raw=$('inp').value.trim(); if(!raw){$('inp').focus();return}
  $('go').disabled=true; $('go').innerHTML='<span class="sp"></span>กำลังตรวจ...';
  $('out').innerHTML='';
  if(_op==='multiagent') showFlowLoading();
  const lines=raw.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
  try{
    if(lines.length===1 && isURL(lines[0])){
      const r=await fetch('/api/youtube',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({url:lines[0],limit:80,op:_op})});
      const d=await r.json();
      if(d.error){ $('out').innerHTML='<div class="warn">'+d.error+'</div>'; }
      else { _rows=d.rows; _page=1; _labels={}; _lastChanged=-1; renderList(); }
    } else {
      const r=await fetch('/api/batch',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({texts:lines,op:_op})});
      const d=await r.json();
      if(d.error){ $('out').innerHTML='<div class="warn">'+d.error+'</div>'; }
      else { _rows=d.rows; _page=1; renderCards(); }
    }
  }catch(e){ $('out').innerHTML='<div class="warn">ผิดพลาด: '+e+'</div>'; }
  $('go').disabled=false; $('go').textContent='ตรวจ';
}

function verdictBits(r){
  const sar=r.decision==='sarcasm';
  const pct=r.prob==null?null:Math.round((sar?r.prob:1-r.prob)*100);
  return {sar,pct,word:sar?'ประชด':'ไม่ประชด',cls:sar?'sar':'notx',
    col:sar?'var(--sar)':'var(--not)'};
}
async function markWrong(i,btn){
  const r=_rows[i]; if(!r||r.corrected)return;
  const correct=r.decision==='sarcasm'?'0':'1';
  btn.disabled=true; btn.textContent='กำลังจำ...';
  try{
    const resp=await fetch('/api/correct',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:r.text,label:correct})});
    if((await resp.json()).ok){ r.corrected=true; (_isList?renderList:renderCards)(); loadStats(); }
    else { btn.disabled=false; btn.textContent='ตัดสินผิด'; }
  }catch(e){ btn.disabled=false; btn.textContent='ตัดสินผิด'; }
}
// ---- link feedback: ถูก/ผิด -> เก็บ label + วัด F1 + วิเคราะห์ใหม่ ----
let _labels={}, _lastChanged=-1;
async function markFeedback(i,agree){
  const r=_rows[i]; const m=r.decision==='sarcasm'?'1':'0';
  const t=agree?m:(m==='1'?'0':'1');
  _labels[r.text]={m:m,t:t}; r.fb=agree?'agree':'wrong';
  renderList();
  try{ await fetch('/api/correct',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:r.text,label:t})}); loadStats(); }catch(e){}
}
function linkF1(){
  const L=Object.values(_labels); if(!L.length)return null;
  let tp=0,fp=0,fn=0,ok=0;
  L.forEach(x=>{const mp=x.m==='1',tt=x.t==='1';
    if(mp&&tt)tp++; if(mp&&!tt)fp++; if(!mp&&tt)fn++; if(mp===tt)ok++;});
  const f1=(2*tp+fp+fn)?(2*tp/(2*tp+fp+fn)):1;
  return {n:L.length,ok:ok,f1:f1};
}
async function reanalyzeLink(){
  const texts=_rows.map(r=>r.text);
  const old={}; _rows.forEach(r=>old[r.text]=r.decision);
  $('out').innerHTML='<div class="ok"><span class="sp" style="border-color:#38a05d;border-top-color:transparent"></span>วิเคราะห์ใหม่ด้วยสิ่งที่คุณสอน...</div>';
  try{
    const rr=await fetch('/api/batch',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({texts:texts,model:_op})});
    const d=await rr.json();
    if(d.error){ $('out').innerHTML='<div class="warn">'+d.error+'</div>'; return; }
    _rows=d.rows.map(x=>({...x, fb:_labels[x.text]?( _labels[x.text].t===(x.decision==='sarcasm'?'1':'0')?'agree':'wrong'):undefined}));
    let ch=0; _rows.forEach(r=>{ if(old[r.text]!==undefined && old[r.text]!==r.decision) ch++; });
    _lastChanged=ch; _page=1; renderList();
  }catch(e){ $('out').innerHTML='<div class="warn">ผิดพลาด: '+e+'</div>'; }
}
let _isList=false;
function renderCards(){
  _isList=false;
  $('out').innerHTML=_rows.map((r,i)=>{
    const v=verdictBits(r);
    const learned=r.corrected?'<span class="learned">จำแล้ว</span>'
      :`<button class="wrongbtn" onclick="markWrong(${i},this)">ตัดสินผิด</button>`;
    const meter=v.pct==null?'':`<div class="meter"><span style="width:${v.pct}%;background:${v.col}"></span></div>
      <div class="conf" style="justify-content:flex-start">มั่นใจ ${v.pct}%</div>`;
    return `<div class="result">
      <div class="verdict ${v.cls}"><span class="dot"></span>${v.word}</div>
      <div class="txt">“${esc(r.text)}”</div>
      ${miniflow(r.steps)}
      ${meter}
      <div class="conf" style="justify-content:flex-end;margin-top:10px">${learned}</div>
    </div>`;
  }).join('');
}
function renderList(){
  _isList=true;
  const s=_rows.filter(r=>r.decision==='sarcasm').length;
  const pages=Math.max(1,Math.ceil(_rows.length/PP));
  if(_page>pages)_page=pages;
  const slice=_rows.slice((_page-1)*PP,_page*PP);
  const items=slice.map(r=>{
    const i=_rows.indexOf(r); const v=verdictBits(r);
    let fb;
    if(r.fb==='agree') fb='<span class="learned">คุณ: ถูกแล้ว</span>';
    else if(r.fb==='wrong') fb='<span class="learned" style="color:var(--sar)">คุณ: ตัดสินผิด</span>';
    else fb=`<span class="fbtns"><button class="fbtn yes" onclick="markFeedback(${i},true)">ถูก</button>`
      +`<button class="fbtn no" onclick="markFeedback(${i},false)">ผิด</button></span>`;
    return `<div class="result" style="background:${v.sar?'var(--sar-bg)':'#fff'}">
      <div class="txt" style="margin:0">${esc(r.text)}</div>
      ${miniflow(r.steps)}
      <div class="rowline"><span class="pill ${v.cls}">${v.word}</span>
        ${v.pct==null?'':`<span class="conf" style="margin:0">มั่นใจ ${v.pct}%</span>`}${fb}</div>
    </div>`;
  }).join('');
  const pager = _rows.length>PP?`<div class="pagerow">
    <button ${_page<=1?'disabled':''} onclick="_page--;renderList()">ก่อนหน้า</button>
    <span class="conf">หน้า ${_page}/${pages}</span>
    <button ${_page>=pages?'disabled':''} onclick="_page++;renderList()">ถัดไป</button></div>`:'';
  const f=linkF1();
  const f1box = f?`<div class="ok" style="text-align:center">
    คุณตรวจแล้ว <b>${f.n}</b> ข้อ · โมเดลถูก <b>${f.ok}/${f.n}</b> · <b>F1 = ${f.f1.toFixed(2)}</b>
    <div style="font-size:12px;color:var(--ink2);margin-top:4px">กด "ถูก/ผิด" ที่แต่ละข้อเพื่อวัดคะแนน แล้วสอนโมเดลไปในตัว</div></div>`:'';
  const changed = _lastChanged>=0?`<div class="ok" style="background:var(--not-bg);border-color:var(--not);text-align:center">
    วิเคราะห์ใหม่แล้ว เปลี่ยนคำตัดสินไป <b>${_lastChanged}</b> ข้อ หลังเรียนจากที่คุณสอน</div>`:'';
  const relearn = Object.keys(_labels).length?`<button class="go" style="margin-top:12px;width:100%" onclick="reanalyzeLink()">วิเคราะห์ใหม่ด้วยสิ่งที่คุณสอน (${Object.keys(_labels).length} ข้อ)</button>`:'';
  $('out').innerHTML=`<div class="ok" style="background:var(--brand-soft);color:var(--ink);border-color:#cbd9f3;text-align:center">
    ดึงมา ${_rows.length} คอมเมนต์ · ที่คิดว่าประชด ${s} ข้อ</div>
    ${changed}${f1box}
    <div class="list">${items}</div>${pager}${relearn}`;
}
function toggleResearch(btn){
  const on=$('research').classList.toggle('show');
  btn.textContent = on ? 'ซ่อนงานวิจัย' : 'ดูเบื้องหลังงานวิจัย';
  if(on) $('research').scrollIntoView({behavior:'smooth',block:'nearest'});
}
async function loadStats(){
  try{
    const d=await(await fetch('/api/stats')).json();
    const el=$('shared');
    if(d.corrections>0){ el.textContent='โมเดลนี้เรียนจากทุกคนไปแล้ว '+d.corrections+' ครั้ง (จำถาวร แชร์ทุกคน)'; el.classList.add('on'); }
    else el.classList.remove('on');
  }catch(e){}
}
loadStats();
$('inp').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')analyze()});
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
    print("  หน้า dev http://127.0.0.1:5000/  ·  หน้าผู้ใช้ http://127.0.0.1:5000/app")
    print("=" * 60)
    app.run(debug=False, port=5000)
