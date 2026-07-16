# -*- coding: utf-8 -*-
"""ตัวตรวจจับประชดภาษาไทย "พร้อมใช้จริง" — ตกผลึกจาก finding 1-11

สรุป finding: ไม่ต้อง multi-agent. ระบบที่คุ้มสุด = เอเจนต์เดี่ยว 1 call + อ่าน logprob + threshold
  - โมเดล gpt-4.1-mini (ถูกสุด/ดีสุดบน frontier, ~$0.0001/ข้อ)
  - ยิงครั้งเดียว ขอ logprob -> P(ประชด) -> เทียบ threshold ตาม "จุดทำงาน" ที่เลือก

จุดทำงานที่ "ทำได้จริง" บนงานนี้ (เลือกจาก PR curve บน gold 127 ข้อ) — เลือก "โมเดล" ตามงาน:
  balanced    : gpt-4.1-mini t=0.095  P≈0.68 R≈0.83 F1≈0.75  ถูกสุด (~$0.0001/ข้อ) — ค่าเริ่มต้น
  high_recall : gpt-4o       t=0.05   P≈0.43 R≈1.00           "ห้ามพลาด" -> คนรีวิว FP ต่อ (~6x แพงกว่า)
  review_band : 0.05–0.50 = "ส่งให้คนตัดสิน"                   (นอกแถบตอบเองมั่นใจ ในแถบยกให้คน)

*** ข้อจำกัดที่ต้องรู้ก่อน deploy (honest — วัดจากข้อมูลจริง ไม่ใช่เดา) ***
  - **gpt-4.1-mini มีเพดาน recall ~0.83**: ประชด 5/30 ข้อมันให้คะแนน ~0 (มองไม่เห็น) ลด threshold เท่าไรก็ไม่เจอ
    -> งานที่ "พลาดประชดไม่ได้" ต้องใช้ gpt-4o (โหมด high_recall) ไม่ใช่ mini
  - precision เพดาน ~0.68 ทั้งสองโมเดล — ตั้ง "high precision (>0.8)" ไม่ได้ เพราะรีวิวสมดุลก้ำกึ่งจริง
  - recall บน gold สูงเกินจริง (self-selection bias, ดู PROVENANCE.md) -> ของจริงจะต่ำกว่านี้
  - เทรน/วัดบน Wongnai (รีวิว) + Wisesight (ทวีต) — โดเมนอื่นคาดว่าตกลง
  - F1 ที่คาดหวังจริง ~0.70 ไม่ใช่ 0.9x — อย่าสัญญาเกิน

ใช้:
  export OPENAI_API_KEY=sk-...
  python predict.py "ขอบคุณที่ให้รอ 2 ชม. บริการดีจริงๆ"        # ข้อความเดียว
  python predict.py --csv in.csv --out out.csv --text-col text  # ทั้งไฟล์ (batch)
  python predict.py "..." --op high_recall                      # เลือกจุดทำงาน

หรือ import:  from predict import SarcasmDetector
"""
import argparse
import hashlib
import json
import math
import os
import sys
import threading

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, ".predict_cache.json")   # gitignored -- ยิงข้อความซ้ำ = ฟรี
CORR_PATH = os.path.join(HERE, ".predict_corrections.json")  # gitignored -- ที่คนแก้ว่า "โมเดลตัดสินผิด"
MODEL = "gpt-4.1-mini"
DETECT_SYS = (
    'ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่\n'
    "ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ\n\n"
    'ตอบเป็น JSON เท่านั้น: {"label": "1" หรือ "0"}\n1 = ประชด, 0 = ไม่ประชด'
)

# ---------- corrections: เรียนจากที่คนบอกว่า "ตัดสินผิด" (few-shot in-context ไม่ใช่การเทรนใหม่จริง) ----------
# เก็บ correction ได้ "ไม่จำกัด" (permanent, ข้ามเซสชัน) แล้วตอนทำนายค่อยดึงเฉพาะอันที่ "เกี่ยวข้องสุด" มาใส่โปรมป์
_MAX_SHOTS = 10          # จำนวนตัวอย่างที่ใส่ต่อการทำนาย 1 ครั้ง (เลือกจากที่คล้ายที่สุด กันโทเคนบวม)
_corr_lock = threading.Lock()


def _trigrams(s):
    s = "".join(s.split())
    return set(s[i:i+3] for i in range(len(s) - 2)) if len(s) >= 3 else {s}


def _relevant(corr, query, k=_MAX_SHOTS):
    """เลือก correction ที่ "เกี่ยวข้องกับข้อความนี้ที่สุด" (Jaccard ของ char-trigram — ใช้กับไทยได้ ไม่ต้องตัดคำ)
    -> เก็บ correction ไว้เยอะแค่ไหนก็ได้ แต่ใส่โปรมป์เฉพาะอันที่ช่วยข้อนี้ = เรียนถาวรและสเกลได้"""
    if len(corr) <= k:
        return corr
    q = _trigrams(query)
    def sim(c):
        t = _trigrams(c["text"])
        return len(t & q) / (len(t | q) or 1)
    return sorted(corr, key=sim, reverse=True)[:k]


def load_corrections():
    if os.path.exists(CORR_PATH):
        try:
            with open(CORR_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def add_correction(text, correct_label):
    """คนกดว่า 'ผิด' -> เก็บ (text, ป้ายที่ถูก). ป้าย '1'=ประชด '0'=ไม่ประชด
    dedupe ตาม text (ตัวแก้ล่าสุดชนะ) · คืนจำนวน correction ทั้งหมด"""
    text = (text or "").strip()
    correct_label = str(correct_label).strip()
    if not text or correct_label not in ("0", "1"):
        raise ValueError("correct_label ต้องเป็น '0' หรือ '1' และ text ต้องไม่ว่าง")
    with _corr_lock:
        corr = [c for c in load_corrections() if c.get("text") != text]
        corr.append({"text": text, "label": correct_label})
        tmp = CORR_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(corr, f, ensure_ascii=False)
        os.replace(tmp, CORR_PATH)
    return len(corr)


def _shots_block(shots):
    """แปลงรายการ correction (ที่เลือกมาแล้ว) เป็นบล็อก few-shot ต่อท้าย system prompt"""
    if not shots:
        return ""
    lines = ["\n\nตัวอย่างที่คนยืนยันคำตอบที่ถูกแล้ว (ให้ยึดตามนี้กับข้อความคล้ายๆ กัน):"]
    for c in shots:
        t = c["text"].replace("\n", " ")[:160]
        lines.append(f'  "{t}" -> {c["label"]}')
    return "\n".join(lines)


def _corr_sig(corr):
    """ลายเซ็นของชุด corrections *ทั้งหมด* -> เป็นส่วนหนึ่งของ cache key
    (corrections เปลี่ยน -> โปรมป์เปลี่ยน -> prob เก่าใช้ไม่ได้ ต้องแยก namespace)"""
    if not corr:
        return "0"
    raw = "|".join(f'{c["text"]}={c["label"]}' for c in corr)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]

# จุดทำงาน: model+threshold คัดจาก PR curve บน gold — ดู header (เลือกโมเดลตามงาน)
OPERATING = {
    "balanced":    {"model": "gpt-4.1-mini", "t": 0.095, "desc": "F1 สูงสุด ถูกสุด (P≈0.68 R≈0.83)"},
    "high_recall": {"model": "gpt-4o",        "t": 0.050, "desc": "จับครบ R≈1.00 ยอม FP (P≈0.43)"},
}
REVIEW_LO, REVIEW_HI = 0.05, 0.50   # แถบ "ยกให้คนตัดสิน" สำหรับโหมด review_band


class _Cache:
    """cache แบบไฟล์ JSON: (model,text) -> prob. ยิงข้อความเดิมซ้ำ = อ่านจาก cache ไม่เสียเงิน
    key เป็น hash กันไฟล์บวม/ประเด็นอักขระ · เขียนแบบ atomic กันไฟล์พังถ้าปิดกลางคัน"""
    def __init__(self, path=CACHE_PATH):
        self.path, self.lock, self.d = path, threading.Lock(), {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.d = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.d = {}

    @staticmethod
    def key(model, text, sig="0"):
        return hashlib.sha1(f"{model}\x00{sig}\x00{text}".encode()).hexdigest()

    def get(self, model, text, sig="0"):
        return self.d.get(self.key(model, text, sig))

    def put(self, model, text, prob, sig="0"):
        if not self.path:
            return
        with self.lock:
            self.d[self.key(model, text, sig)] = prob
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.d, f)
            os.replace(tmp, self.path)


class SarcasmDetector:
    def __init__(self, operating="balanced", api_key=None, model=None, cache=True):
        from openai import OpenAI
        self.model = model or OPERATING[operating]["model"]
        self.t = OPERATING[operating]["t"]
        self.op = operating
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=3)
        self.cache = _Cache() if cache else None
        self.hits = self.misses = 0
        self.reload_corrections()

    def reload_corrections(self):
        """อ่าน corrections ทั้งหมด (permanent จากไฟล์) เก็บไว้เลือกตอนทำนาย
        เรียกใหม่หลังมีคนกด 'ผิด' เพื่อให้คำทำนายต่อไปใช้ตัวอย่างใหม่"""
        self.corr = load_corrections()
        self.corr_sig = _corr_sig(self.corr)
        self.n_corr = len(self.corr)

    def prob(self, text):
        """คืน P(ประชด) 0..1 จาก logprob ของ token label (1 call) — เช็ค cache ก่อน"""
        if self.cache is not None:
            c = self.cache.get(self.model, text, self.corr_sig)
            if c is not None:
                self.hits += 1
                return c
        self.misses += 1
        p = self._call(text)
        if self.cache is not None and p == p:      # ไม่ cache ค่า NaN
            self.cache.put(self.model, text, p, self.corr_sig)
        return p

    def _call(self, text):
        # เลือก correction ที่เกี่ยวข้องกับข้อความนี้ที่สุด แล้วต่อท้าย prompt (retrieval per-query)
        sys_prompt = DETECT_SYS + _shots_block(_relevant(self.corr, text))
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=20, response_format={"type": "json_object"},
            logprobs=True, top_logprobs=20,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": f"ข้อความ: {text}"}])
        for tok in (r.choices[0].logprobs.content or []):
            if tok.token.strip().strip('"') not in ("0", "1"):
                continue
            p0 = p1 = 0.0
            for alt in tok.top_logprobs:
                t = alt.token.strip().strip('"')
                if t == "1": p1 += math.exp(alt.logprob)
                elif t == "0": p0 += math.exp(alt.logprob)
            if p0 + p1 > 0:
                return p1 / (p0 + p1)
        # อ่าน logprob ไม่ได้ -> ใช้คำตอบ hard
        try:
            return 1.0 if str(json.loads(r.choices[0].message.content or "{}").get("label", "")) == "1" else 0.0
        except json.JSONDecodeError:
            return float("nan")

    def predict(self, text, review_band=False):
        """คืน dict: label, prob, decision. review_band=True เปิดโหมด 'ยกให้คน' ในแถบก้ำกึ่ง"""
        p = self.prob(text)
        if p != p:                      # NaN
            return {"label": None, "prob": None, "decision": "error"}
        if review_band and REVIEW_LO <= p <= REVIEW_HI:
            return {"label": None, "prob": round(p, 3), "decision": "review"}
        label = 1 if p >= self.t else 0
        return {"label": label, "prob": round(p, 3),
                "decision": "sarcasm" if label else "not_sarcasm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="ข้อความเดียว")
    ap.add_argument("--csv", help="ไฟล์ input (batch)")
    ap.add_argument("--out", help="ไฟล์ output (คู่กับ --csv)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--op", default="balanced", choices=list(OPERATING))
    ap.add_argument("--review-band", action="store_true", help="ยกข้อก้ำกึ่งให้คนตัดสิน")
    a = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ต้องมี OPENAI_API_KEY (export OPENAI_API_KEY=sk-...)")
    det = SarcasmDetector(operating=a.op)
    print(f"[predict] {det.model} · จุดทำงาน '{a.op}' (t={det.t}) · {OPERATING[a.op]['desc']}", file=sys.stderr)
    print("[predict] ⚠ วัดผลไว้แค่บนรีวิว/ทวีต (F1~0.72) — โดเมนอื่น (YouTube/ข่าว/ทางการ) ยังไม่เทสต์",
          file=sys.stderr)

    if a.csv:
        import pandas as pd
        df = pd.read_csv(a.csv, dtype=str).fillna("")
        if a.text_col not in df.columns:
            sys.exit(f"ไม่มีคอลัมน์ '{a.text_col}' (มี: {list(df.columns)})")
        res = [det.predict(t, review_band=a.review_band) for t in df[a.text_col]]
        df["pred_label"] = [r["label"] for r in res]
        df["pred_prob"] = [r["prob"] for r in res]
        df["pred_decision"] = [r["decision"] for r in res]
        out = a.out or (os.path.splitext(a.csv)[0] + "_pred.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        n = len(df); ns = sum(1 for r in res if r["decision"] == "sarcasm")
        nr = sum(1 for r in res if r["decision"] == "review")
        print(f"เขียน {out} · {n} ข้อ · ประชด {ns}" + (f" · ยกให้คน {nr}" if a.review_band else "")
              + f" · cache hit {det.hits}/{det.hits+det.misses}")
    elif a.text:
        r = det.predict(a.text, review_band=a.review_band)
        print(json.dumps(r, ensure_ascii=False))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
