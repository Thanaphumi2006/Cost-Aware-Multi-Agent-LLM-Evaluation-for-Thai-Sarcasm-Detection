# -*- coding: utf-8 -*-
"""
สเต็ป 1: โหลดข้อมูลดิบ 2 ชุด -> เอาเฉพาะตัวข้อความ -> ทิ้งป้ายเดิม -> รวมเป็นไฟล์เดียว

จุดสำคัญ:
- ทั้งสองชุดเป็น dataset แบบ "มีสคริปต์โหลด" ซึ่ง datasets>=4.0 เลิกรองรับแล้ว
  จึงต้องปักเวอร์ชัน datasets<4.0 และใส่ trust_remote_code=True
- คอลัมน์ข้อความ: wisesight = 'texts' , wongnai = 'review_body'

วิธีติดตั้ง (รันครั้งเดียว):
    pip install "datasets<4.0" pandas
"""

from datasets import load_dataset
import pandas as pd


def get_text_column(ds_split, candidates):
    """เลือกชื่อคอลัมน์ข้อความแบบยืดหยุ่น เผื่อชื่อคอลัมน์เปลี่ยน"""
    for c in candidates:
        if c in ds_split.column_names:
            return c
    raise KeyError(f"ไม่พบคอลัมน์ข้อความ ลองดูจาก: {ds_split.column_names}")


# ── 1) โหลด wisesight (ข้อความโซเชียลไทย) ──────────────────────────────
ws = load_dataset("pythainlp/wisesight_sentiment", trust_remote_code=True)
ws_texts = []
for split in ws:                                   # รวมทุก split (train/validation/test)
    col = get_text_column(ws[split], ["texts", "text"])
    ws_texts += list(ws[split][col])               # เอาเฉพาะข้อความ ไม่แตะ 'category'

# ── 2) โหลด wongnai (รีวิวร้านอาหาร) ───────────────────────────────────
wg = load_dataset("Wongnai/wongnai_reviews", trust_remote_code=True)
wg_texts = []
for split in wg:
    col = get_text_column(wg[split], ["review_body", "text"])
    wg_texts += list(wg[split][col])               # เอาเฉพาะข้อความ ไม่แตะ 'star_rating'

# ── 3) รวมเป็น DataFrame เดียว เก็บแค่ text + source (ทิ้งป้ายเดิมทั้งหมด) ──
df = pd.DataFrame(
    [{"text": t, "source": "wisesight"} for t in ws_texts]
    + [{"text": t, "source": "wongnai"} for t in wg_texts]
)

# ── 4) ล้างเบื้องต้น: ตัดช่องว่างหัวท้าย, ตัดว่าง, ตัดซ้ำ ──────────────────
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]
df = df.drop_duplicates(subset="text").reset_index(drop=True)

# ── 5) ดูผล + เซฟไว้ใช้ต่อ ─────────────────────────────────────────────
print("จำนวนข้อความทั้งหมด:", len(df))
print(df["source"].value_counts())
print("\nตัวอย่าง 5 ข้อความ:")
print(df.sample(5, random_state=0).to_string(index=False))

df.to_csv("raw_texts.csv", index=False, encoding="utf-8-sig")   # utf-8-sig เปิดใน Excel แล้วไทยไม่เพี้ยน
print("\nบันทึกแล้วที่: raw_texts.csv")

# หมายเหตุ: ถ้ายัง error เรื่องสคริปต์ (เช่นเครื่องบังคับ datasets 4.x)
# ทางแก้สำรอง: โหลดจากไฟล์ parquet ที่ HF แปลงอัตโนมัติ เช่น
#   df_ws = pd.read_parquet("hf://datasets/pythainlp/wisesight_sentiment/data/train-00000-of-00001.parquet")
# แล้วดึงคอลัมน์ 'texts' มาแทน (ต้องติดตั้ง: pip install huggingface_hub pyarrow)