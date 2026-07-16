# -*- coding: utf-8 -*-
"""เว็บทดลอง + เทียบ 3 ระบบตรวจจับประชดภาษาไทย

รัน:
  set OPENAI_API_KEY=sk-...            (PowerShell: $env:OPENAI_API_KEY="sk-...")
  C:/Users/thana/pt/Scripts/python.exe app.py
  เปิด http://127.0.0.1:5000

หมายเหตุ:
- คีย์ API ใส่ได้ 2 ทาง: environment variable หรือ พิมพ์ในหน้าเว็บ (ช่อง 🔑 ด้านบน)
  ทั้งสองทางเก็บไว้ใน RAM ของโปรเซสเท่านั้น -- ไม่เขียนลงไฟล์ ไม่ฝังในโค้ด ปิดเซิร์ฟเวอร์แล้วหาย
- เว็บนี้ผูกกับ 127.0.0.1 อย่าเปิดออกสู่เน็ต เพราะช่องใส่คีย์จะกลายเป็นช่องให้คนอื่นยิง API ด้วยคีย์เรา
- ถ้าไม่มีคีย์ เว็บยังใช้ได้ แต่จะรันได้แค่ WangchanBERTa (ตัวที่ไม่ต้องใช้ API)
- WangchanBERTa ที่เว็บใช้เทรนบน gold ครบทุกข้อ -> ห้ามเอาไปวัดผลบน gold
  ถ้าพิมพ์ข้อความที่อยู่ใน gold มันจะ "จำคำตอบได้" (เว็บจะเตือนให้เอง)
"""
import os
import sys
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
    det, i1, o1 = multiagent._ask(c, multiagent.DETECT_SYS, multiagent.DETECT_SCHEMA, "label", text)
    d_ms = round((time.perf_counter() - t0) * 1000)
    steps.append({
        "role": "ด่าน 1 · พนักงานคัดกรอง (detector)",
        "job": "อ่านข้อความดิบ แล้วชี้ว่า “น่าจะประชด” หรือไม่ — เหวี่ยงแหกว้างไว้ก่อน",
        "said": det, "say_txt": {"1": "ประชด", "0": "ไม่ประชด"}.get(det, "ตอบเพี้ยน"),
        "ms": d_ms, "in_tok": i1, "out_tok": o1, "cost": round(cost(i1, o1), 6), "ran": True,
    })

    if det == "1":
        t1 = time.perf_counter()
        ver, i2, o2 = multiagent._ask(c, multiagent.VERIFY_SYS, multiagent.VERIFY_SCHEMA, "verdict", text)
        v_ms = round((time.perf_counter() - t1) * 1000)
        steps.append({
            "role": "ด่าน 2 · หัวหน้า QC (verifier)",
            "job": "ตรวจเฉพาะข้อที่ด่าน 1 ชี้ว่าประชด — มีอำนาจ “ปัดตก” อย่างเดียว เพิ่มประชดใหม่ไม่ได้",
            "said": ver, "say_txt": {"1": "ยืนยัน: คงเป็นประชด", "0": "ปัดตก: ไม่ใช่ประชด"}.get(ver, "ตอบเพี้ยน"),
            "ms": v_ms, "in_tok": i2, "out_tok": o2, "cost": round(cost(i2, o2), 6), "ran": True,
        })
        final = ver if ver in ("0", "1") else "err"
        flipped = (det == "1" and ver == "0")
    else:
        steps.append({
            "role": "ด่าน 2 · หัวหน้า QC (verifier)",
            "job": "ตรวจเฉพาะข้อที่ด่าน 1 ชี้ว่าประชด — มีอำนาจ “ปัดตก” อย่างเดียว เพิ่มประชดใหม่ไม่ได้",
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


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """ตรวจหลายข้อความรวดเดียว (เอาไว้อัปโหลด CSV จากหน้าเว็บ) -- ใช้ predict.py ระบบพร้อมใช้จริง"""
    if not _api_key:
        return jsonify({"error": "ยังไม่มี OPENAI_API_KEY (ใส่คีย์ด้านบนก่อน)"}), 400
    body = request.json or {}
    texts = [str(t).strip() for t in body.get("texts", []) if str(t).strip()]
    op = body.get("op", "balanced")
    review = bool(body.get("review_band", False))
    if not texts:
        return jsonify({"error": "ไม่มีข้อความ"}), 400
    if len(texts) > 500:
        return jsonify({"error": f"มากเกินไป ({len(texts)}) -- จำกัด 500 ข้อ/ครั้ง"}), 400
    import predict
    if op not in predict.OPERATING:
        return jsonify({"error": f"operating point ไม่รู้จัก: {op}"}), 400

    det = detector(op)
    h0 = det.hits
    rows = []
    for t in texts:
        try:
            r = det.predict(t, review_band=review)
        except Exception as e:
            r = {"label": None, "prob": None, "decision": f"error: {type(e).__name__}"}
        rows.append({"text": t, **r, "in_gold": t in GOLD_TEXTS, "gold": GOLD_TEXTS.get(t)})
    summ = {
        "n": len(rows),
        "sarcasm": sum(1 for r in rows if r["decision"] == "sarcasm"),
        "not": sum(1 for r in rows if r["decision"] == "not_sarcasm"),
        "review": sum(1 for r in rows if r["decision"] == "review"),
        "cached": det.hits - h0, "model": det.model, "op": op,
    }
    return jsonify({"rows": rows, "summary": summ})


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


PAGE = r"""
<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ทดลอง & เทียบระบบตรวจจับประชดภาษาไทย</title>
<style>
*{box-sizing:border-box} body{margin:0;font-family:"Segoe UI",Tahoma,sans-serif;background:#f5f7fa;color:#1a1f2b}
.wrap{max-width:1080px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px} .sub{color:#6b7482;font-size:13px;margin-bottom:20px}
.card{background:#fff;border:1px solid #e2e6ec;border-radius:10px;padding:18px;margin-bottom:18px}
textarea{width:100%;min-height:96px;padding:12px;border:1px solid #cfd6df;border-radius:8px;
  font-family:inherit;font-size:15px;resize:vertical}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
button{padding:9px 16px;border:0;border-radius:7px;font-size:14px;cursor:pointer;font-family:inherit}
.go{background:#2f6b47;color:#fff;font-weight:600} .go:disabled{background:#9bb5a6;cursor:wait}
.ghost{background:#eef1f5;color:#42505f}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:6px}
.sys{border:1px solid #e2e6ec;border-radius:9px;padding:14px;background:#fff}
.sys h3{margin:0 0 3px;font-size:14px} .sys .tag{font-size:11px;color:#8a94a6;margin-bottom:10px}
.verdict{font-size:19px;font-weight:700;padding:9px 12px;border-radius:7px;margin-bottom:10px;text-align:center}
.v1{background:#fdecec;color:#a02020} .v0{background:#eaf5ee;color:#1e5c3c}
.vna{background:#f0f1f3;color:#8a94a6;font-size:14px;font-weight:500}
.kv{display:flex;justify-content:space-between;font-size:12.5px;padding:3px 0;color:#5a6472}
.kv b{color:#1a1f2b;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 9px;text-align:right;border-bottom:1px solid #eef0f3}
th:first-child,td:first-child{text-align:left} th{color:#6b7482;font-weight:600;font-size:12px}
tr.best td{background:#f2faf5}
.note{font-size:12px;color:#8a6d2f;background:#fff8ec;border:1px solid #e8d9b0;
  border-radius:7px;padding:9px 11px;margin-top:10px}
.warn{font-size:12px;color:#a02020;background:#fdecec;border:1px solid #e8b8b8;
  border-radius:7px;padding:9px 11px;margin-top:10px}
.gold{font-size:13px;padding:8px 11px;border-radius:7px;background:#eef4ff;
  border:1px solid #c9d9f5;margin-top:12px}
.pill{display:inline-block;padding:4px 11px;border-radius:20px;font-size:12.5px;font-weight:700;
  white-space:nowrap}
.pill.v1{background:#fdecec;color:#a02020} .pill.v0{background:#eaf5ee;color:#1e5c3c}
.pill.vna{background:#f0f1f3;color:#8a94a6;font-weight:500}
.sp{display:inline-block;width:13px;height:13px;border:2px solid #fff;border-top-color:transparent;
  border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:6px}
@keyframes s{to{transform:rotate(360deg)}}

/* ---- multi-agent flow ---- */
.flow{display:flex;flex-direction:column;align-items:center;gap:0;margin-top:14px}
.ag{border:1.6px solid #cfd6df;border-radius:10px;padding:14px;background:#fff;
  width:100%;max-width:330px}
.ag.on{border-color:#2f9e5e;background:#f4fbf7}
.ag.off{border-style:dashed;background:#fafbfc;opacity:.72}
.ag.kill{border-color:#c96a6a;background:#fdf5f5}
.ag .who{font-size:13.5px;font-weight:700;margin-bottom:2px}
.ag .job{font-size:11.5px;color:#6b7482;line-height:1.5;margin-bottom:10px}
.ag .say{font-size:14px;font-weight:700;padding:8px 10px;border-radius:6px;background:#eef1f5;
  color:#42505f;text-align:center;margin-bottom:8px}
.ag.on .say{background:#e3f3ea;color:#1e5c3c} .ag.kill .say{background:#f8e4e4;color:#a02020}
.ag.off .say{background:#f0f1f3;color:#98a1ad;font-weight:500;font-size:12.5px}
.ag .meta{font-size:11.5px;color:#6b7482;display:flex;justify-content:space-between;padding:2px 0}
.arw{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;
  padding:10px 0;font-size:22px;color:#8a94a6;text-align:center}
.arw small{font-size:10.5px;color:#8a94a6;display:block;text-align:center;line-height:1.3}
.fin{margin-top:12px;padding:11px 14px;border-radius:8px;font-size:13.5px;font-weight:600;text-align:center}
.fin.k{background:#fdecec;color:#a02020} .fin.p{background:#eaf5ee;color:#1e5c3c}
.rule{font-size:12px;color:#5c5340;background:#fff8ec;border:1px solid #e8d9b0;
  border-radius:7px;padding:10px 12px;margin-top:12px;line-height:1.65}

/* ---- key panel ---- */
.keyhead{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
.keycard input{flex:1;min-width:260px;padding:9px 12px;border:1px solid #cfd6df;border-radius:7px;
  font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px}
.keycard input:focus{outline:2px solid #2f6b47;outline-offset:-1px;border-color:#2f6b47}
.ok{font-size:12px;color:#1e5c3c;background:#eaf5ee;border:1px solid #b8ddc6;
  border-radius:7px;padding:9px 11px;margin-top:10px}
</style></head><body><div class="wrap">

<h1>ทดลอง &amp; เทียบระบบตรวจจับประชดภาษาไทย</h1>
<div class="sub">เอเจนต์เดี่ยว vs Multi-agent vs WangchanBERTa — วัดคุณภาพ ค่าใช้จ่าย และเวลา พร้อมกัน</div>

<div class="card keycard" id="keycard">
  <div class="keyhead">
    <div>
      <h3 style="margin:0 0 3px;font-size:15px">🔑 OpenAI API key</h3>
      <div class="sub" style="margin:0">ระบบที่ใช้ LLM (① และ ②) ต้องมีคีย์ถึงจะรันได้ — WangchanBERTa ไม่ต้องใช้</div>
    </div>
    <span class="pill {{ 'v0' if has_key else 'vna' }}" id="keypill">
      {{ 'พร้อมใช้งาน · ' ~ masked if has_key else 'ยังไม่มีคีย์' }}
    </span>
  </div>

  <div class="row" id="keyform" {% if has_key %}style="display:none"{% endif %}>
    <input type="password" id="k" placeholder="sk-..." autocomplete="off" spellcheck="false">
    <button class="go" id="ksave" onclick="saveKey()">บันทึกคีย์</button>
  </div>
  <div class="row" id="keydone" {% if not has_key %}style="display:none"{% endif %}>
    <button class="ghost" onclick="clearKey()">ลบคีย์ออก</button>
    {% if from_env %}<span class="sub" style="margin:0">อ่านมาจาก environment variable <code>OPENAI_API_KEY</code></span>{% endif %}
  </div>
  <div id="kmsg"></div>

  <div class="note" style="margin-top:12px">
    คีย์ถูกเก็บไว้<b>ในหน่วยความจำของเซิร์ฟเวอร์เท่านั้น</b> — ไม่เขียนลงไฟล์ ไม่ใส่ใน cookie
    ปิดเซิร์ฟเวอร์เมื่อไหร่ก็หายไป ต้องใส่ใหม่<br>
    เว็บนี้เปิดที่ <code>127.0.0.1</code> (เครื่องตัวเองเท่านั้น) — <b>อย่า deploy ขึ้นเน็ตทั้งอย่างนี้</b>
    เพราะใครก็ตามที่เข้าถึงหน้านี้ได้จะยิง API ด้วยคีย์ของคุณได้
  </div>
</div>

{% if not has_wcb %}<div class="warn"><b>ยังไม่มีโมเดล WangchanBERTa</b> — รัน <code>train_final_wcb.py</code> ก่อน</div>{% endif %}

<div class="card">
  <textarea id="t" placeholder="พิมพ์ข้อความภาษาไทย เช่น: ขอบคุณมากนะคะที่ให้รอแค่ 2 ชั่วโมง บริการดีเยี่ยมจริงๆ"></textarea>
  <div class="row">
    <button class="go" id="go" onclick="run()">วิเคราะห์ด้วยทั้ง 3 ระบบ</button>
    <button class="ghost" onclick="samp('1')">สุ่มตัวอย่าง “ประชด”</button>
    <button class="ghost" onclick="samp('0')">สุ่มตัวอย่าง “ไม่ประชด”</button>
    <button class="ghost" onclick="document.getElementById('t').value=''">ล้าง</button>
  </div>
  <div id="goldbox"></div>
</div>

<div id="out"></div>

<div class="card">
  <h2 style="margin:0 0 4px;font-size:17px">📄 ตรวจทั้งไฟล์ (batch) — ระบบพร้อมใช้จริง</h2>
  <div class="sub" style="margin-bottom:12px">
    อัปโหลด CSV (หรือวางข้อความทีละบรรทัด) → ใช้ <code>predict.py</code> ตัวเดียวกับ production
    (เอเจนต์เดี่ยว + threshold, มี cache ยิงซ้ำฟรี)
  </div>
  <div class="warn" style="margin-bottom:12px">
    ⚠ <b>วัดผลไว้แค่บนรีวิวร้าน (Wongnai) + ทวีตสั้น (Wisesight)</b> — F1 ~0.72 ·
    ข้อความโดเมนอื่น (YouTube / ข่าว / ทางการ) <b>ยังไม่ได้ทดสอบ</b> ผลอาจเพี้ยน
    (จะเช็คก่อนก็ได้: <code>eval_domain.py</code>)
  </div>
  <div class="row" style="margin-bottom:10px">
    <label style="font-size:13px;color:#42505f">จุดทำงาน:
      <select id="bop" style="padding:6px 8px;border:1px solid #cfd6df;border-radius:6px;font-family:inherit">
        <option value="balanced">balanced — gpt-4.1-mini (ถูก)</option>
        <option value="high_recall">high_recall — gpt-4o (จับครบ)</option>
      </select>
    </label>
    <label style="font-size:13px;color:#42505f;display:inline-flex;align-items:center;gap:5px">
      <input type="checkbox" id="brev"> ยกข้อก้ำกึ่งให้คนตัดสิน
    </label>
  </div>
  <textarea id="btext" placeholder="วางข้อความทีละบรรทัด แล้วกดปุ่ม — หรือเลือกไฟล์ CSV ด้านล่าง"></textarea>
  <div class="row">
    <button class="go" id="bgo" onclick="runBatch()">ตรวจทั้งหมด</button>
    <input type="file" id="bfile" accept=".csv,.txt" style="font-size:13px">
    <span class="sub" id="bhint" style="margin:0">รองรับสูงสุด 500 ข้อ/ครั้ง</span>
  </div>
  <div id="bout"></div>
</div>

<div class="card">
  <h2 style="margin:0 0 4px;font-size:17px">② Multi-agent ทำงานยังไง</h2>
  <div class="sub" style="margin-bottom:4px">
    เอเจนต์เดี่ยว (①) + เพิ่ม <b>ผู้ตรวจสอบ</b> อีกหนึ่งคน — แค่นั้น
  </div>
  <div id="flow">
    <div class="flow">
      <div class="ag">
        <div class="who">👀 คนที่ 1 · <b>ผู้คัดกรอง</b></div>
        <div class="job"><b>คือ ① เอเจนต์เดี่ยว ตัวเดิมเป๊ะ</b><br>
          อ่านข้อความ แล้วตอบว่า “ประชด” หรือ “ไม่ประชด”</div>
        <div class="say">ประชด? → ใช่ / ไม่ใช่</div>
        <div style="font-size:11.5px;color:#5a6472;background:#f6f8fa;border-radius:6px;padding:8px 10px">
          <b>เก่ง:</b> จับประชดได้ครบ ไม่พลาดเลยสักข้อ<br>
          <b>ไม่เก่ง:</b> เหวี่ยงแหกว้างไป — ทายว่าประชด 27 ข้อที่จริงไม่ใช่
        </div>
      </div>

      <div class="arw">↓<small>ส่งต่อ<br><b>เฉพาะข้อที่ตอบ “ใช่”</b></small></div>

      <div class="ag">
        <div class="who">🔍 คนที่ 2 · <b>ผู้ตรวจสอบ</b></div>
        <div class="job"><b>คนใหม่ที่เพิ่มเข้ามา</b><br>
          ตรวจของที่คนแรกส่งมา ว่าเป็นประชด<b>จริง</b>ไหม</div>
        <div class="say">ยืนยัน / ปัดตก</div>
        <div style="font-size:11.5px;color:#5a6472;background:#f6f8fa;border-radius:6px;padding:8px 10px">
          <b>อำนาจของเขามีแค่อย่างเดียว: ปัดตก</b><br>
          เขา<b>เพิ่ม</b>ประชดใหม่ไม่ได้ — ทำได้แค่บอกว่า “อันนี้คนแรกทายผิด”
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
      เหตุผล: <b>ประชดที่แนบเนียนมันอ่านได้สองแง่อยู่แล้วโดยธรรมชาติ</b> —
      ถ้าลังเลแปลว่ามัน<b>น่าจะ</b>ประชด ไม่ใช่ไม่ประชด
    </div>

    <div class="note">
      <b>เราลองแบบซับซ้อนกว่านี้แล้ว — และมันแพ้</b><br>
      • <b>Debate</b> (อัยการ + ทนาย + ผู้พิพากษา, 3 คน ตัดสินใหม่ได้อิสระ) → F1 <b>0.694</b> · แพงกว่า <b>4.1×</b><br>
      • <b>Hybrid</b> (เอา debate มาเป็นผู้ตรวจสอบ, 4 คน) → F1 <b>0.700</b> · แพงกว่า <b>2.4×</b><br>
      ทั้งคู่<b>แพ้</b>ระบบ 2 คนข้างบน (0.744) ที่ถูกที่สุดและง่ายที่สุด<br>
      <b>บทเรียน: “จำกัดอำนาจ agent ให้ถูกจุด” สำคัญกว่า “มี agent เยอะ”</b>
      <span style="color:#8a94a6">(รายละเอียด + สถิติ: RESULTS.md)</span>
    </div>
  </div>
</div>

<div class="card">
  <h3 style="margin:0 0 3px;font-size:15px">คะแนนจริง วัดบนข้อมูล 127 ข้อ (เป็นประชดจริง 30 ข้อ)</h3>
  <div class="sub" style="margin-bottom:12px">ทุกระบบวัดบนข้อมูลชุดเดียวกัน ด้วยโค้ดวัดผลตัวเดียวกัน</div>
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
  <div class="note">
    <b>อ่านตารางนี้ยังไง:</b> multi-agent v2 ได้ F1 สูงสุด แต่จ่ายแพงสุด (1.80× ของ baseline) ·
    baseline จับประชดครบทุกข้อ (FN=0) แต่เหวี่ยงแหเกิน (FP 27) ·
    WangchanBERTa ฟรีและไม่ต้องต่อเน็ต แต่ปล่อยประชดหลุด 9 ข้อ<br>
    <b>อย่าดู accuracy</b> — ข้อมูลเอียง 76/24 เดาว่า “ไม่ประชด” ทุกข้อก็ได้ 0.764 แล้ว
  </div>
</div>

<script>
const $=i=>document.getElementById(i);

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
      $('kmsg').innerHTML='<div class="ok">ใช้คีย์นี้ได้ — ระบบ ① และ ② พร้อมรันแล้ว</div>';
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
       <div class="meta"><span>ค่าใช้จ่าย</span><b>$0.00000 — ประหยัดได้</b></div>`;
  const said1 = s1.said==='1' ? 'ประชด' : (s1.said==='0' ? 'ไม่ประชด' : 'ตอบเพี้ยน');
  const c1 = s1.said==='1' ? 'on' : '';
  const c2 = !s2.ran ? 'off' : (s2.said==='0' ? 'kill' : 'on');
  const arrow = s2.ran
    ? '↓<small>ส่งไปตรวจ<br><b>เพราะคนแรกตอบ “ประชด”</b></small>'
    : '⇣<small style="color:#c0c6cf">ไม่ต้องตรวจ<br><b>คนแรกตอบ “ไม่ประชด”</b></small>';
  const say2 = !s2.ran
    ? 'ไม่ได้ทำงาน — ไม่มีอะไรให้ตรวจ'
    : (s2.said==='0' ? '❌ ปัดตก — คนแรกทายผิด' : '✅ ยืนยัน — คนแรกทายถูก');
  let fin, cls;
  if(m.flipped){ fin='ผลสุดท้าย: <b>ไม่ประชด</b> — ผู้ตรวจสอบจับได้ว่าคนแรกทายผิด แล้วปัดตกทิ้ง'; cls='k'; }
  else if(m.pred==='1'){ fin='ผลสุดท้าย: <b>ประชด</b> — ผู้ตรวจสอบตรวจแล้ว ยืนยันตามคนแรก'; cls='p'; }
  else { fin='ผลสุดท้าย: <b>ไม่ประชด</b> — คนแรกตีตกตั้งแต่ต้น ผู้ตรวจสอบไม่ต้องทำงาน'; cls='p'; }
  document.getElementById('flow').innerHTML=`
    <div class="flow">
      <div class="ag ${c1}">
        <div class="who">👀 คนที่ 1 · <b>ผู้คัดกรอง</b></div>
        <div class="job">คือ ① เอเจนต์เดี่ยว ตัวเดิมเป๊ะ</div>
        <div class="say">พูดว่า: <b>${said1}</b></div>${meta(s1)}
      </div>
      <div class="arw">${arrow}</div>
      <div class="ag ${c2}">
        <div class="who">🔍 คนที่ 2 · <b>ผู้ตรวจสอบ</b></div>
        <div class="job">ปัดตกได้อย่างเดียว — เพิ่มประชดใหม่ไม่ได้</div>
        <div class="say">${say2}</div>${meta(s2)}
      </div>
    </div>
    <div class="fin ${cls}">${fin}</div>
    <div class="rule">
      ข้อความนี้ใช้ไป <b>${m.calls} ครั้ง</b> · <b>${m.latency_ms} ms</b> · <b>$${m.cost.toFixed(5)}</b>
      &nbsp;—&nbsp; เทียบกับเอเจนต์เดี่ยวที่ใช้ 1 ครั้งเสมอ<br>
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
      g=`<div class="gold"><b>ข้อความนี้อยู่ใน gold set</b> — คำตอบจริง:
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
    document.getElementById('flow').scrollIntoView({behavior:'smooth',block:'nearest'});
  }catch(e){ $('out').innerHTML='<div class="card warn">เรียก API ไม่สำเร็จ: '+e+'</div>' }
  $('go').disabled=false; $('go').textContent='วิเคราะห์ด้วยทั้ง 3 ระบบ';
}
$('t').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')run()});
$('k').addEventListener('keydown',e=>{if(e.key==='Enter')saveKey()});

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
    const g=r.in_gold?' <span title="อยู่ใน gold — โมเดลอาจเคยเห็น" style="color:#a06a00">⚑</span>':'';
    return `<tr><td>${i+1}</td><td style="text-align:left;max-width:420px">${esc(r.text)}${g}</td>
      <td>${r.prob==null?'–':r.prob}</td><td>${pill(r.decision)}</td></tr>`;
  }).join('');
  $('bout').innerHTML=`
    <div class="note" style="margin-top:14px">
      <b>${s.n} ข้อ</b> · ${s.model} (${s.op}) · ประชด <b>${s.sarcasm}</b> · ไม่ประชด ${s.not}
      ${s.review?'· ยกให้คน '+s.review:''} · จาก cache ${s.cached} (ฟรี)
      &nbsp; <button class="ghost" onclick="dlCSV()">⬇ ดาวน์โหลด CSV</button>
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
    print("  เปิด http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, port=5000)
