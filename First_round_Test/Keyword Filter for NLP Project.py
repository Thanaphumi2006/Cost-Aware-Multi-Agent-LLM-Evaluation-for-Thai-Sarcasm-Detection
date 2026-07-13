# -*- coding: utf-8 -*-
"""
สเต็ป 3 (รอบที่ 1): คัด "ผู้ต้องสงสัยว่าเป็นประชด" ด้วยคีย์เวิร์ด/แพตเทิร์น

อินพุต : raw_texts.csv  (จากสเต็ป 1 — มีคอลัมน์ text, source)
เอาต์พุต:
  - scored_texts.csv : ทุกข้อความ + คะแนนความน่าสงสัย (ไว้ตรวจ/ปรับจูน)
  - to_label.csv     : กองที่จะเอาไปติดป้ายจริง (ผสมสงสัย+ปกติ, มีช่อง label ว่าง)

หลักการ: ยิ่งมีสัญญาณประชดหลายอย่าง ยิ่งได้คะแนนสูง
สัญญาณแรงสุด = มีทั้ง "คำชม" และ "บริบทแย่" ในข้อความเดียว
วิธีรัน:  pip install pandas   แล้ว   python round1_keyword_filter.py
"""

import re
import pandas as pd

# ================== ปรับค่าได้ตรงนี้ ==================
INPUT_CSV = "raw_texts.csv"
OUTPUT_ALL = "scored_texts.csv"
OUTPUT_TO_LABEL = "to_label.csv"
HIGH_SUSPECT_THRESHOLD = 3     # คะแนน >= ค่านี้ = ผู้ต้องสงสัยสูง
N_TO_LABEL = 400               # จำนวนที่จะดึงไปติดป้าย (เผื่อคัดเหลือ gold 200-300)
SUSPECT_RATIO = 0.6            # สัดส่วนผู้ต้องสงสัยในกองติดป้าย (ที่เหลือเป็นกองปกติ)
MIN_LEN = 15                   # ตัดข้อความสั้นกว่านี้ (มักตัดสินไม่ได้)
RANDOM_SEED = 42
# =====================================================

# ---- สัญญาณ (อิงจากคู่มือติดป้าย) ----
PRAISE = ["ดี", "เยี่ยม", "สุดยอด", "เลิศ", "ปัง", "เทพ", "ฟิน", "ประทับใจ",
          "เก่ง", "ดีงาม", "คุณภาพ", "สุดๆ", "เว่อร์", "ที่สุด"]
NEG_CONTEXT = ["รอ", "นาน", "ช้า", "พัง", "เสีย", "แย่", "ผิด", "ยกเลิก", "ไม่มา",
               "หาย", "เจ๊ง", "ห่วย", "งง", "ผิดหวัง", "ไม่คุ้ม", "โกง", "ปัญหา", "ไม่ได้"]
THANKS = ["ขอบคุณ", "ขอบใจ"]
SARCASTIC_EMOJI = ["🙄", "🙃", "👏", "😑", "😏", "🤡", "✨"]


def score_text(t):
    """คืน (คะแนน, สัญญาณที่เจอ) — สัญญาณช่วยให้ผู้ติดป้ายเห็นว่าทำไมถูกดึงมา"""
    s, hits = 0, []
    praise = [w for w in PRAISE if w in t]
    if praise:
        s += 1; hits.append("ชม:" + ",".join(praise[:3]))
    emoji = [e for e in SARCASTIC_EMOJI if e in t]
    if emoji:
        s += 2; hits.append("อิโมจิ:" + "".join(emoji))
    if re.search(r"5{3,}", t):                 # หัวเราะ 555
        s += 1; hits.append("555")
    if re.search(r"(.)\1\1", t):               # ยืดตัวอักษร เช่น มากกก / เยี่ยมมม
        s += 1; hits.append("ยืดอักษร")
    if any(w in t for w in THANKS):
        s += 1; hits.append("ขอบคุณ")
    neg = [w for w in NEG_CONTEXT if w in t]
    if praise and neg:                          # ** สัญญาณแรงสุด: ชม + บริบทแย่ **
        s += 3; hits.append("ชม+แย่:" + ",".join(neg[:3]))
    return s, "; ".join(hits)


# ---- 1) โหลด + ล้างเบื้องต้น ----
df = pd.read_csv(INPUT_CSV)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= MIN_LEN].drop_duplicates("text").reset_index(drop=True)

# ---- 2) ให้คะแนน ----
scored = df["text"].apply(lambda t: pd.Series(score_text(t), index=["suspect_score", "signals"]))
df = pd.concat([df, scored], axis=1)
df["group"] = df["suspect_score"].apply(
    lambda x: "สงสัยสูง" if x >= HIGH_SUSPECT_THRESHOLD else "ปกติ"
)
df = df.sort_values("suspect_score", ascending=False).reset_index(drop=True)
df.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")

print("== สรุปการให้คะแนน ==")
print(df["group"].value_counts())
print("การกระจายคะแนน:")
print(df["suspect_score"].value_counts().sort_index())

# ---- 3) สร้างกองติดป้าย: ผสมสงสัยสูง + ปกติ (กันข้อมูลเอียง) ----
n_suspect = int(N_TO_LABEL * SUSPECT_RATIO)
n_normal = N_TO_LABEL - n_suspect
suspect_pool = df[df["group"] == "สงสัยสูง"]
normal_pool = df[df["group"] == "ปกติ"]

take_suspect = suspect_pool.sample(min(n_suspect, len(suspect_pool)), random_state=RANDOM_SEED)
take_normal = normal_pool.sample(min(n_normal, len(normal_pool)), random_state=RANDOM_SEED)

to_label = (
    pd.concat([take_suspect, take_normal])
    .sample(frac=1, random_state=RANDOM_SEED)   # สลับลำดับ ไม่ให้ติดป้ายแบบเดากอง
    .reset_index(drop=True)
)
to_label["label"] = ""     # ช่องติดป้าย: 1=ประชด, 0=ไม่ประชด, X=ตัดออก
to_label["note"] = ""      # เหตุผลเคสยาก
to_label[["text", "source", "suspect_score", "signals", "label", "note"]].to_csv(
    OUTPUT_TO_LABEL, index=False, encoding="utf-8-sig"
)

print(f"\nกองติดป้าย {len(to_label)} ข้อ (สงสัยสูง {len(take_suspect)} / ปกติ {len(take_normal)})")
print("บันทึกที่:", OUTPUT_TO_LABEL)

# หมายเหตุ:
# - คีย์เวิร์ดจับ "ผู้ต้องสงสัย" ไม่ใช่ "ประชดแน่ๆ" ตัวจริงยังต้องคนติดป้ายตัดสิน
# - ถ้ากองสงสัยสูงน้อยไป ลด HIGH_SUSPECT_THRESHOLD เป็น 2
# - ถ้าอยากได้คุณภาพดีขึ้น ค่อยเอา to_label ไปกรองซ้ำด้วย LLM (รอบที่ 2)
