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
import json
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

MODEL = "gpt-4.1-mini"
DETECT_SYS = (
    'ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่\n'
    "ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ\n\n"
    'ตอบเป็น JSON เท่านั้น: {"label": "1" หรือ "0"}\n1 = ประชด, 0 = ไม่ประชด'
)

# จุดทำงาน: model+threshold คัดจาก PR curve บน gold — ดู header (เลือกโมเดลตามงาน)
OPERATING = {
    "balanced":    {"model": "gpt-4.1-mini", "t": 0.095, "desc": "F1 สูงสุด ถูกสุด (P≈0.68 R≈0.83)"},
    "high_recall": {"model": "gpt-4o",        "t": 0.050, "desc": "จับครบ R≈1.00 ยอม FP (P≈0.43)"},
}
REVIEW_LO, REVIEW_HI = 0.05, 0.50   # แถบ "ยกให้คนตัดสิน" สำหรับโหมด review_band


class SarcasmDetector:
    def __init__(self, operating="balanced", api_key=None, model=None):
        from openai import OpenAI
        self.model = model or OPERATING[operating]["model"]
        self.t = OPERATING[operating]["t"]
        self.op = operating
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=3)

    def prob(self, text):
        """คืน P(ประชด) 0..1 จาก logprob ของ token label (1 call)"""
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=20, response_format={"type": "json_object"},
            logprobs=True, top_logprobs=20,
            messages=[{"role": "system", "content": DETECT_SYS},
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
        print(f"เขียน {out} · {n} ข้อ · ประชด {ns}" + (f" · ยกให้คน {nr}" if a.review_band else ""))
    elif a.text:
        r = det.predict(a.text, review_band=a.review_band)
        print(json.dumps(r, ensure_ascii=False))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
