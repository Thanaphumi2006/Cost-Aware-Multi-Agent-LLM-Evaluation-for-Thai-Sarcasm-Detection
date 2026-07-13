# -*- coding: utf-8 -*-
"""
สัปดาห์ 2 — Baseline: เอเจนต์เดี่ยว LLM zero-shot classifier

แนวคิด: ให้ LLM ตัวเดียว 1 คอลตัดสิน "ประชดไหม" แบบ zero-shot (ไม่มีตัวอย่าง)
        แล้ววัดผลเทียบ gold.csv -> ได้ accuracy / precision / recall / F1
        ตัวเลขนี้คือ "เส้นฐาน" ไว้เทียบกับระบบที่ซับซ้อนกว่า (เช่น multi-agent) ทีหลัง

อินพุต : gold.csv   (จาก human_review.py — ต้องมีคอลัมน์ text, label โดย label ∈ {0,1})
เอาต์พุต:
  - baseline_preds.csv : ทุกข้อ + pred (คำทำนายของ baseline) + ถูก/ผิด
  - พิมพ์รายงาน metric + confusion matrix ออกจอ

หมายเหตุ: baseline นี้จงใจใช้ prompt "เรียบๆ" (ไม่ยัดกฎเยอะเหมือนตอน pre-label)
          เพื่อให้เป็นเส้นฐานที่ยุติธรรม — ระบบที่ซับซ้อนกว่าควรทำได้ดีกว่าเส้นนี้

ติดตั้ง:  pip install openai pandas scikit-learn
ตั้งคีย์:  export OPENAI_API_KEY="sk-..."
รัน:       python baseline_zeroshot.py
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# ================== ปรับได้ ==================
GOLD_CSV = "gold.csv"
PRED_CSV = "baseline_preds.csv"
MODEL = "gpt-4o"
SLEEP_SEC = 0.3
POSITIVE = "1"          # คลาสบวก = ประชด
# =============================================

client = OpenAI()

# prompt เรียบๆ ตั้งใจ (เส้นฐานที่ยุติธรรม) — บอกแค่นิยาม ไม่ยัดกฎย่อยเยอะ
SYSTEM = """ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่
ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ

ตอบเป็น JSON เท่านั้น: {"label": "1" หรือ "0"}
1 = ประชด, 0 = ไม่ประชด"""


def predict_one(text):
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=20,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"ข้อความ: {text}"},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        lab = str(json.loads(raw).get("label", "0")).strip()
        return lab if lab in ("0", "1") else "0"
    except Exception:
        return "0"


def main():
    if not os.path.exists(GOLD_CSV):
        print(f"ยังไม่มี {GOLD_CSV} — รัน human_review.py ให้ได้ gold ก่อนนะครับ")
        return

    # โหลด: ทำต่อจากเดิมได้ถ้ามี pred อยู่แล้ว
    if os.path.exists(PRED_CSV):
        df = pd.read_csv(PRED_CSV)
    else:
        df = pd.read_csv(GOLD_CSV)
        df["pred"] = ""
    df["label"] = df["label"].astype(str).str.strip()
    df["pred"] = df["pred"].fillna("").astype(str)

    # เอาเฉพาะข้อที่ label เป็น 0/1 (เผื่อมี X หลุดมา)
    df = df[df["label"].isin(["0", "1"])].reset_index(drop=True)

    todo = df.index[~df["pred"].isin(["0", "1"])].tolist()
    print(f"Baseline zero-shot: ต้องทำนายอีก {len(todo)} จาก {len(df)} ข้อ")

    for i, idx in enumerate(todo, 1):
        df.at[idx, "pred"] = predict_one(str(df.at[idx, "text"]))
        if i % 20 == 0 or i == len(todo):
            df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
            print(f"  ...{i}/{len(todo)}")
        time.sleep(SLEEP_SEC)

    df["correct"] = df["pred"] == df["label"]
    df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    # ---- วัดผล ----
    y_true = df["label"].tolist()
    y_pred = df["pred"].tolist()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["0", "1"])

    print("\n" + "═" * 55)
    print(f"BASELINE (LLM zero-shot, {MODEL}) — n = {len(df)}")
    print("═" * 55)
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}   (ประชด=1 เป็นคลาสบวก)")
    print(f"Recall   : {rec:.3f}")
    print(f"F1       : {f1:.3f}")
    print("\nConfusion matrix  (แถว=จริง, คอลัมน์=ทำนาย)")
    print("            pred:0   pred:1")
    print(f"  true:0     {cm[0][0]:>5}    {cm[0][1]:>5}")
    print(f"  true:1     {cm[1][0]:>5}    {cm[1][1]:>5}")
    print(f"\nบันทึกคำทำนายที่: {PRED_CSV}")
    print("→ เก็บ F1 นี้ไว้เป็นเส้นฐาน เทียบกับระบบ multi-agent ทีหลัง")


if __name__ == "__main__":
    main()
