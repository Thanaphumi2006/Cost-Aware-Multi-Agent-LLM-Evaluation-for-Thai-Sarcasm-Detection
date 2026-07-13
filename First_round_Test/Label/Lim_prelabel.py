# -*- coding: utf-8 -*-
"""
ทางเลือก 2 - ตัวที่ 1: ให้ LLM ติดป้ายร่าง (pre-label)  [เวอร์ชัน GPT / OpenAI]

อินพุต : to_label.csv        (จากสคริปต์รอบที่ 1)
เอาต์พุต: to_label_prelabeled.csv  (เพิ่มคอลัมน์ llm_label, llm_reason)

- ใช้โมเดล GPT ติดป้ายทีละข้อ พร้อมเหตุผลสั้นๆ
- เซฟทีละข้อ (incremental) + รันซ้ำได้ (ข้ามข้อที่ติดป้ายแล้ว) กันงานหายถ้าค้าง
- ป้ายนี้เป็นแค่ "ร่าง" คนต้องมาตรวจ/แก้ในตัวที่ 2 อีกที

ติดตั้ง:  pip install openai pandas
ตั้งคีย์:  export OPENAI_API_KEY="sk-..."          (Windows: set OPENAI_API_KEY=...)
รัน:       python llm_prelabel.py
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI

# ================== ปรับได้ ==================
INPUT_CSV = "to_label.csv"
OUTPUT_CSV = "to_label_prelabeled.csv"
MODEL = "gpt-4o"          # เปลี่ยนเป็นรุ่นที่คุณมีสิทธิ์ใช้ได้ (เช่นรุ่นใหม่กว่านี้)
SLEEP_SEC = 0.3           # หน่วงกันชนลิมิต
# =============================================

client = OpenAI()  # อ่านคีย์จาก OPENAI_API_KEY อัตโนมัติ

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
        response_format={"type": "json_object"},   # บังคับให้ตอบเป็น JSON
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


# ---- โหลด (ถ้ามีไฟล์ output อยู่แล้ว = รันต่อจากเดิม) ----
if os.path.exists(OUTPUT_CSV):
    df = pd.read_csv(OUTPUT_CSV)
else:
    df = pd.read_csv(INPUT_CSV)
    df["llm_label"] = ""
    df["llm_reason"] = ""

# ---- ติดป้ายเฉพาะข้อที่ยังว่าง ----
todo = df.index[df["llm_label"].astype(str).str.len() == 0].tolist()
print(f"ต้องติดป้ายอีก {len(todo)} จากทั้งหมด {len(df)} ข้อ")

for i, idx in enumerate(todo, 1):
    lab, reason = label_one(str(df.at[idx, "text"]))
    df.at[idx, "llm_label"] = lab
    df.at[idx, "llm_reason"] = reason
    if i % 10 == 0 or i == len(todo):
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")  # เซฟทุก 10 ข้อ
        print(f"  ...{i}/{len(todo)}  (ล่าสุด: [{lab}] {reason[:40]})")
    time.sleep(SLEEP_SEC)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("\nเสร็จ! บันทึกที่:", OUTPUT_CSV)
print("การกระจายป้ายร่างจาก LLM:")
print(df["llm_label"].value_counts())
print("\nขั้นต่อไป: รัน human_review.py เพื่อตรวจ/แก้ป้ายเหล่านี้")