# -*- coding: utf-8 -*-
"""เช็กว่า gold.csv พอวัดผลไหม: จำนวน, สัดส่วน 1/0, และคำเตือน"""
import os
import pandas as pd

# อ่าน gold.csv จากโฟลเดอร์เดียวกับสคริปต์นี้ (รันจากที่ไหนก็ได้)
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold.csv")
g = pd.read_csv(GOLD)
g["label"] = g["label"].astype(str).str.strip()

n = len(g)
n1 = (g["label"] == "1").sum()   # ประชด
n0 = (g["label"] == "0").sum()   # ไม่ประชด

print(f"ทั้งหมด: {n} ข้อ")
print(f"  ประชด (1)    : {n1}  ({n1/n*100:.0f}%)")
print(f"  ไม่ประชด (0) : {n0}  ({n0/n*100:.0f}%)")
if "source" in g.columns:
    print("\nแหล่งข้อมูล:")
    print(g["source"].value_counts().to_string())

print("\n== ประเมิน ==")
ok = True
if n1 < 30:
    print(f"⚠ ประชดมีแค่ {n1} ข้อ (ควร >=30-40) -> F1 ฝั่งประชดจะไม่เสถียร ควรเก็บประชดเพิ่ม")
    ok = False
if n0 < 30:
    print(f"⚠ ไม่ประชดมีแค่ {n0} ข้อ (ควร >=30) -> ควรเก็บเพิ่ม")
    ok = False
ratio = min(n1, n0) / max(n1, n0) if max(n1, n0) else 0
if ratio < 0.3:
    print(f"⚠ ข้อมูลเอียงมาก (สัดส่วน {ratio:.2f}) -> ตอนวัดต้องดู F1/precision/recall ไม่ใช่ accuracy")
if ok:
    print("✓ พอเริ่มทำ baseline ได้")
else:
    print("-> แนะนำ: ตรวจเพิ่มโดยเจาะข้อที่ suspect_score สูง (มีโอกาสเป็นประชด) เพื่อเพิ่มฝั่ง 1")