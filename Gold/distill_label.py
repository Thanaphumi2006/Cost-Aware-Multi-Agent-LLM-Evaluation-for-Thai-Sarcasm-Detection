# -*- coding: utf-8 -*-
"""Distillation ขั้น 1 — ให้ "teacher" ติดป้าย silver ให้ข้อความที่ยังไม่มีป้าย

แนวคิด: เอาระบบ GPT (teacher) ไปติดป้ายข้อความดิบเยอะๆ ได้ "silver data" (ป้ายที่อาจผิดบ้าง)
แล้วเอาไปเทรน WangchanBERTa (student) ต่อ -> หวังปิดช่องว่าง F1 0.62 -> 0.74 แบบต้นทุนต่อข้อ = 0

*** จุดสำคัญที่ทำให้ silver ไม่พังเพราะ teacher ไม่แม่น (precision teacher ~0.68) ***
teacher มี "ความมั่นใจ" (P(ประชด) จาก logprob) -> เก็บเฉพาะข้อที่มั่นใจสูงสองฝั่ง ทิ้งช่วงกลาง
  ประชด    : prob >= --pos-conf (เช่น 0.90)
  ไม่ประชด : prob <= --neg-conf (เช่น 0.05)
-> ลด noise ของป้าย (ช่วงก้ำกึ่งคือที่ teacher ผิดบ่อยสุด)

ทำไม teacher เป็น "single agent" (predict.py/batch_eval.py) ไม่ใช่ pipeline v2:
  - pipeline ให้แต่ป้าย hard ไม่มีความมั่นใจ -> กรอง noise ไม่ได้
  - single agent มี logprob (คัดความมั่นใจได้) + precision สูงกว่า (0.68 vs 0.60) = teacher ที่ดีกว่าสำหรับ distill

ขั้นตอน:
  1) ให้ teacher (ถูกสุด) ติดป้ายข้อความดิบก่อน — แนะนำ batch API ลดครึ่งราคา:
       python batch_eval.py --csv harvest_to_review.csv --out harvest_pred.csv
  2) กรองเป็น silver:
       python distill_label.py --pred harvest_pred.csv --out silver.csv --pos-conf 0.9 --neg-conf 0.05 --balance
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV จาก batch_eval.py/predict.py (ต้องมี text + pred_prob)")
    ap.add_argument("--out", default="silver.csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--pos-conf", type=float, default=0.90, help="เก็บเป็นประชด เมื่อ prob >= ค่านี้")
    ap.add_argument("--neg-conf", type=float, default=0.05, help="เก็บเป็นไม่ประชด เมื่อ prob <= ค่านี้")
    ap.add_argument("--max-pos", type=int, default=0, help="จำกัดจำนวน silver ประชด (0=ไม่จำกัด)")
    ap.add_argument("--max-neg", type=int, default=0, help="จำกัดจำนวน silver ไม่ประชด (0=ไม่จำกัด)")
    ap.add_argument("--balance", action="store_true", help="ตัด negative ให้เท่า positive (กัน silver เอียงไปทางลบ)")
    a = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(a.pred, dtype=str).fillna("")
    if a.text_col not in df.columns or "pred_prob" not in df.columns:
        sys.exit(f"ต้องมีคอลัมน์ '{a.text_col}' และ 'pred_prob' (รัน batch_eval.py/predict.py --csv ก่อน)")
    df = df[df["pred_prob"] != ""].copy()
    df["p"] = df["pred_prob"].astype(float)

    pos = df[df["p"] >= a.pos_conf].copy(); pos["silver_label"] = 1
    neg = df[df["p"] <= a.neg_conf].copy(); neg["silver_label"] = 0
    pos = pos.sort_values("p", ascending=False)   # มั่นใจสุดก่อน (เผื่อ cap)
    neg = neg.sort_values("p", ascending=True)
    if a.max_pos:
        pos = pos.head(a.max_pos)
    if a.max_neg:
        neg = neg.head(a.max_neg)
    if a.balance:
        k = min(len(pos), len(neg))
        pos, neg = pos.head(k), neg.head(k)

    out = pd.concat([pos, neg], ignore_index=True).rename(columns={a.text_col: "text"})
    out = out[["text", "silver_label", "p"]].rename(columns={"p": "teacher_prob"})
    out["teacher_prob"] = out["teacher_prob"].round(3)
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""].drop_duplicates(subset="text").reset_index(drop=True)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")

    npos = int((out["silver_label"] == 1).sum()); nneg = int((out["silver_label"] == 0).sum())
    print(f"เขียน {a.out} · silver {len(out)} ข้อ (ประชด {npos} / ไม่ประชด {nneg}) · "
          f"ทิ้งช่วงไม่มั่นใจ {len(df) - len(out)} ข้อ (pos>={a.pos_conf}, neg<={a.neg_conf})")
    print("เตือน: 'silver' = teacher อาจผิด -- precision teacher จำกัด ประชด(positive)อาจปนเปื้อน "
          "เอาไปเทรนแล้ววัดผลด้วย distill_train_eval.py (OOF ไม่ leak) เท่านั้น")


if __name__ == "__main__":
    main()
