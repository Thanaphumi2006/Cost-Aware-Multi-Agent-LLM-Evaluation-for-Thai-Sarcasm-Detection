# -*- coding: utf-8 -*-
"""
Week 2 -- Baseline: single-agent LLM zero-shot classifier

Idea: have a single LLM decide "sarcasm?" zero-shot in one call (no examples)
        then evaluate against gold.csv -> get accuracy / precision / recall / F1
        this number is the "baseline" to compare against more complex systems (e.g. multi-agent) later

Input : gold.csv   (from human_review.py -- must have columns text, label with label in {0,1})
Output:
  - baseline_preds.csv : every item + pred (the baseline prediction) + correct/wrong
  - prints a metric report + confusion matrix to the screen

Note: this baseline deliberately uses a "plain" prompt (not stuffed with rules like the pre-label step)
          to be a fair baseline -- more complex systems should beat this line

Install:  pip install openai pandas scikit-learn
Set key:  export OPENAI_API_KEY="sk-..."
Run:       python baseline_zeroshot.py
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# ================== tunable ==================
GOLD_CSV = "gold.csv"
PRED_CSV = "baseline_preds.csv"
MODEL = "gpt-4o"
SLEEP_SEC = 0.3
POSITIVE = "1"          # positive class = sarcasm
# =============================================

client = OpenAI()

# deliberately plain prompt (a fair baseline) -- just the definition, no piled-on sub-rules
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
        print(f"{GOLD_CSV} not found yet -- run human_review.py to produce gold first")
        return

    # load: can resume if pred already exists
    if os.path.exists(PRED_CSV):
        df = pd.read_csv(PRED_CSV)
    else:
        df = pd.read_csv(GOLD_CSV)
        df["pred"] = ""
    df["label"] = df["label"].astype(str).str.strip()
    df["pred"] = df["pred"].fillna("").astype(str)

    # keep only items with label 0/1 (in case an X slipped in)
    df = df[df["label"].isin(["0", "1"])].reset_index(drop=True)

    todo = df.index[~df["pred"].isin(["0", "1"])].tolist()
    print(f"Baseline zero-shot: {len(todo)} items left to predict out of {len(df)}")

    for i, idx in enumerate(todo, 1):
        df.at[idx, "pred"] = predict_one(str(df.at[idx, "text"]))
        if i % 20 == 0 or i == len(todo):
            df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
            print(f"  ...{i}/{len(todo)}")
        time.sleep(SLEEP_SEC)

    df["correct"] = df["pred"] == df["label"]
    df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    # ---- evaluate ----
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
    print(f"Precision: {prec:.3f}   (sarcasm=1 is the positive class)")
    print(f"Recall   : {rec:.3f}")
    print(f"F1       : {f1:.3f}")
    print("\nConfusion matrix  (rows=true, cols=predicted)")
    print("            pred:0   pred:1")
    print(f"  true:0     {cm[0][0]:>5}    {cm[0][1]:>5}")
    print(f"  true:1     {cm[1][0]:>5}    {cm[1][1]:>5}")
    print(f"\nsaved predictions to: {PRED_CSV}")
    print("-> keep this F1 as the baseline to compare against multi-agent systems later")


if __name__ == "__main__":
    main()
