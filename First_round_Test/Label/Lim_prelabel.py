# -*- coding: utf-8 -*-
"""
Option 2 - part 1: have the LLM draft labels (pre-label)  [GPT / OpenAI version]

Input : to_label.csv        (from the round-1 script)
Output: to_label_prelabeled.csv  (adds columns llm_label, llm_reason)

- uses a GPT model to label item by item, with a short reason
- saves incrementally + rerunnable (skips already-labeled items) to avoid losing work on a hang
- these labels are only a "draft"; a human must review/fix them in part 2

Install:  pip install openai pandas
Set key:  export OPENAI_API_KEY="sk-..."          (Windows: set OPENAI_API_KEY=...)
Run:       python llm_prelabel.py
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI

# ================== tunable ==================
INPUT_CSV = "to_label.csv"
OUTPUT_CSV = "to_label_prelabeled.csv"
MODEL = "gpt-4o"          # change to a model you have access to (e.g. a newer one)
SLEEP_SEC = 0.3           # delay to avoid hitting rate limits
# =============================================

client = OpenAI()  # reads the key from OPENAI_API_KEY automatically

SYSTEM = """คุณเป็นผู้ช่วยติดป้ายว่า "ข้อความภาษาไทยนี้ประชด/เสียดสีหรือไม่"

นิยาม: ประชด = ความหมายจริงที่ผู้เขียนตั้งใจ ต่าง/ตรงข้าม กับความหมายผิวเผิน
โดยจงใจให้ผู้อ่านจับได้ เพื่อเหน็บ บ่น หรือแสดงความไม่พอใจ

เส้นแบ่งสำคัญ:
- ตำหนิตรงๆ (ไม่เสแสร้งชม) = ไม่ประชด  เช่น "บริการแย่มาก รอนาน"
- ชมจริงใจ = ไม่ประชด  เช่น "อร่อยมากประทับใจ"
- รีวิวยาวที่มีทั้งข้อดีข้อเสียตามจริง = ไม่ประชด
- ชม/ขอบคุณ แต่สื่อว่าแย่จริง = ประชด

วิธีคิด: (1) ผิวเผินสื่ออะไร (2) เจตนาจริงคืออะไร (3) ต่างกันเพื่อเหน็บไหม

ตอบเป็น JSON เท่านั้น รูปแบบ:
{"label": "1" หรือ "0" หรือ "X", "reason": "เหตุผลสั้นมากไม่เกิน 15 คำ"}
- 1 = ประชด, 0 = ไม่ประชด, X = ตัดสินไม่ได้/ต้องใช้บริบทนอกข้อความ"""


def label_one(text):
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=150,
        response_format={"type": "json_object"},   # force a JSON response
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"ข้อความ: {text}"},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        obj = json.loads(raw)
        lab = str(obj.get("label", "X")).strip().upper()
        if lab not in ("1", "0", "X"):
            lab = "X"
        return lab, str(obj.get("reason", ""))[:120]
    except Exception:
        return "X", "parse_error:" + raw[:60]


# ---- load (if the output file already exists = resume) ----
if os.path.exists(OUTPUT_CSV):
    df = pd.read_csv(OUTPUT_CSV)
else:
    df = pd.read_csv(INPUT_CSV)
    df["llm_label"] = ""
    df["llm_reason"] = ""

# ---- label only the still-empty items ----
todo = df.index[df["llm_label"].astype(str).str.len() == 0].tolist()
print(f"{len(todo)} items left to label out of {len(df)}")

for i, idx in enumerate(todo, 1):
    lab, reason = label_one(str(df.at[idx, "text"]))
    df.at[idx, "llm_label"] = lab
    df.at[idx, "llm_reason"] = reason
    if i % 10 == 0 or i == len(todo):
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")  # save every 10 items
        print(f"  ...{i}/{len(todo)}  (latest: [{lab}] {reason[:40]})")
    time.sleep(SLEEP_SEC)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("\ndone! saved to:", OUTPUT_CSV)
print("distribution of the LLM draft labels:")
print(df["llm_label"].value_counts())
print("\nnext step: run human_review.py to review/fix these labels")